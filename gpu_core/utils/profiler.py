"""
GPU Profiling utilities for EV Fleet RL training.

Provides tools for measuring:
- GPU memory usage
- Computation time
- Throughput metrics
"""

import torch
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import functools


@dataclass
class TimingResult:
    """Result of a timing measurement."""
    name: str
    cpu_time_ms: float
    gpu_time_ms: float
    count: int = 1
    
    @property
    def avg_cpu_time_ms(self) -> float:
        return self.cpu_time_ms / max(self.count, 1)
    
    @property
    def avg_gpu_time_ms(self) -> float:
        return self.gpu_time_ms / max(self.count, 1)


@dataclass
class MemorySnapshot:
    """GPU memory snapshot."""
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    
    @classmethod
    def capture(cls, device: torch.device = None) -> 'MemorySnapshot':
        if not torch.cuda.is_available():
            return cls(0, 0, 0)
        
        if device is None:
            device = torch.device('cuda')
        
        return cls(
            allocated_mb=torch.cuda.memory_allocated(device) / 1024**2,
            reserved_mb=torch.cuda.memory_reserved(device) / 1024**2,
            max_allocated_mb=torch.cuda.max_memory_allocated(device) / 1024**2
        )


@dataclass
class ProfileResult:
    """Complete profiling result."""
    name: str
    timing: TimingResult
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    
    @property
    def memory_delta_mb(self) -> float:
        return self.memory_after.allocated_mb - self.memory_before.allocated_mb


class GPUProfiler:
    """
    GPU profiler for measuring performance metrics.
    
    Example:
        profiler = GPUProfiler()
        
        with profiler.profile("forward"):
            output = model(input)
        
        print(profiler.summary())
    """
    
    def __init__(self, device: str = "cuda", enabled: bool = True):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.enabled = enabled and torch.cuda.is_available()
        self.results: Dict[str, List[ProfileResult]] = {}
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
    
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code block."""
        if not self.enabled:
            yield
            return
        
        memory_before = MemorySnapshot.capture(self.device)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        cpu_start = time.perf_counter()
        start_event.record()
        
        try:
            yield
        finally:
            end_event.record()
            torch.cuda.synchronize()
            cpu_end = time.perf_counter()
            
            gpu_time_ms = start_event.elapsed_time(end_event)
            cpu_time_ms = (cpu_end - cpu_start) * 1000
            
            memory_after = MemorySnapshot.capture(self.device)
            
            result = ProfileResult(
                name=name,
                timing=TimingResult(
                    name=name,
                    cpu_time_ms=cpu_time_ms,
                    gpu_time_ms=gpu_time_ms
                ),
                memory_before=memory_before,
                memory_after=memory_after
            )
            
            if name not in self.results:
                self.results[name] = []
            self.results[name].append(result)
    
    def time_function(self, name: Optional[str] = None):
        """Decorator for timing functions."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_name = name or func.__name__
                with self.profile(func_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a profiled operation."""
        if name not in self.results:
            return {}
        
        results = self.results[name]
        gpu_times = [r.timing.gpu_time_ms for r in results]
        cpu_times = [r.timing.cpu_time_ms for r in results]
        memory_deltas = [r.memory_delta_mb for r in results]
        
        return {
            'count': len(results),
            'gpu_time_mean_ms': sum(gpu_times) / len(gpu_times),
            'gpu_time_min_ms': min(gpu_times),
            'gpu_time_max_ms': max(gpu_times),
            'cpu_time_mean_ms': sum(cpu_times) / len(cpu_times),
            'memory_delta_mean_mb': sum(memory_deltas) / len(memory_deltas)
        }
    
    def summary(self) -> str:
        """Generate profiling summary."""
        lines = ["=" * 70]
        lines.append("GPU Profiling Summary")
        lines.append("=" * 70)
        lines.append(f"{'Operation':<25} {'Count':>8} {'GPU ms':>12} {'CPU ms':>12} {'Mem MB':>10}")
        lines.append("-" * 70)
        
        for name, results in sorted(self.results.items()):
            stats = self.get_stats(name)
            lines.append(
                f"{name:<25} "
                f"{stats['count']:>8} "
                f"{stats['gpu_time_mean_ms']:>12.3f} "
                f"{stats['cpu_time_mean_ms']:>12.3f} "
                f"{stats['memory_delta_mean_mb']:>10.2f}"
            )
        
        lines.append("=" * 70)
        
        if torch.cuda.is_available():
            mem = MemorySnapshot.capture(self.device)
            lines.append(f"Current GPU Memory: {mem.allocated_mb:.1f} MB allocated, "
                        f"{mem.reserved_mb:.1f} MB reserved")
            lines.append(f"Peak GPU Memory: {mem.max_allocated_mb:.1f} MB")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all profiling data."""
        self.results.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


class ThroughputMeter:
    """
    Measures throughput metrics for RL training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.steps = 0
        self.episodes = 0
        self.samples = 0
        self.updates = 0
        
        self._step_times: List[float] = []
        self._episode_times: List[float] = []
        self._update_times: List[float] = []
    
    def record_step(self, batch_size: int = 1, time_taken: Optional[float] = None):
        """Record a simulation step."""
        self.steps += batch_size
        if time_taken:
            self._step_times.append(time_taken)
    
    def record_episode(self, time_taken: Optional[float] = None):
        """Record an episode completion."""
        self.episodes += 1
        if time_taken:
            self._episode_times.append(time_taken)
    
    def record_update(self, batch_size: int = 1, time_taken: Optional[float] = None):
        """Record a training update."""
        self.updates += 1
        self.samples += batch_size
        if time_taken:
            self._update_times.append(time_taken)
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def steps_per_second(self) -> float:
        return self.steps / max(self.elapsed_time, 1e-6)
    
    @property
    def episodes_per_hour(self) -> float:
        return self.episodes / max(self.elapsed_time / 3600, 1e-9)
    
    @property
    def updates_per_second(self) -> float:
        return self.updates / max(self.elapsed_time, 1e-6)
    
    @property
    def samples_per_second(self) -> float:
        return self.samples / max(self.elapsed_time, 1e-6)
    
    def summary(self) -> Dict[str, float]:
        return {
            'elapsed_time': self.elapsed_time,
            'total_steps': self.steps,
            'total_episodes': self.episodes,
            'total_updates': self.updates,
            'total_samples': self.samples,
            'steps_per_second': self.steps_per_second,
            'episodes_per_hour': self.episodes_per_hour,
            'updates_per_second': self.updates_per_second,
            'samples_per_second': self.samples_per_second
        }
    
    def __str__(self) -> str:
        s = self.summary()
        return (
            f"Throughput: {s['steps_per_second']:.1f} steps/s, "
            f"{s['updates_per_second']:.1f} updates/s, "
            f"{s['episodes_per_hour']:.1f} episodes/hour"
        )


class MemoryTracker:
    """
    Tracks GPU memory usage over time.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device) if torch.cuda.is_available() else None
        self.snapshots: List[Dict[str, Any]] = []
    
    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        if self.device is None:
            return
        
        self.snapshots.append({
            'label': label,
            'time': time.time(),
            'allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2,
            'max_allocated_mb': torch.cuda.max_memory_allocated(self.device) / 1024**2
        })
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if not self.snapshots:
            return 0
        return max(s['allocated_mb'] for s in self.snapshots)
    
    def plot_memory(self, save_path: Optional[str] = None):
        """Plot memory usage over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return
        
        if not self.snapshots:
            return
        
        times = [s['time'] - self.snapshots[0]['time'] for s in self.snapshots]
        allocated = [s['allocated_mb'] for s in self.snapshots]
        reserved = [s['reserved_mb'] for s in self.snapshots]
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, allocated, label='Allocated', linewidth=2)
        plt.plot(times, reserved, label='Reserved', linewidth=2, linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Memory (MB)')
        plt.title('GPU Memory Usage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def summary(self) -> str:
        if not self.snapshots:
            return "No memory snapshots recorded"
        
        latest = self.snapshots[-1]
        return (
            f"Memory: {latest['allocated_mb']:.1f} MB allocated, "
            f"{latest['reserved_mb']:.1f} MB reserved, "
            f"Peak: {latest['max_allocated_mb']:.1f} MB"
        )


def profile_model_memory(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = "cuda",
    batch_sizes: List[int] = [1, 8, 32, 64, 128]
) -> Dict[int, Dict[str, float]]:
    """
    Profile model memory usage for different batch sizes.
    
    Args:
        model: PyTorch model
        input_shape: Input shape without batch dimension
        device: Device to run on
        batch_sizes: Batch sizes to test
        
    Returns:
        Dict mapping batch size to memory stats
    """
    results = {}
    device = torch.device(device)
    model = model.to(device)
    
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(batch_size, *input_shape, device=device)
        
        with torch.no_grad():
            _ = model(x)
        
        torch.cuda.synchronize()
        
        results[batch_size] = {
            'allocated_mb': torch.cuda.memory_allocated(device) / 1024**2,
            'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(device) / 1024**2
        }
        
        del x
    
    torch.cuda.empty_cache()
    return results


def estimate_max_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = "cuda",
    target_memory_fraction: float = 0.8
) -> int:
    """
    Estimate maximum batch size that fits in GPU memory.
    
    Args:
        model: PyTorch model
        input_shape: Input shape without batch dimension
        device: Device to run on
        target_memory_fraction: Fraction of GPU memory to use
        
    Returns:
        Estimated maximum batch size
    """
    if not torch.cuda.is_available():
        return 32
    
    device = torch.device(device)
    
    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_memory = total_memory * target_memory_fraction
    
    results = profile_model_memory(
        model, input_shape, device=str(device),
        batch_sizes=[1, 8, 16, 32, 64]
    )
    
    batch_sizes = sorted(results.keys())
    if len(batch_sizes) < 2:
        return 32
    
    b1, b2 = batch_sizes[0], batch_sizes[1]
    m1, m2 = results[b1]['max_allocated_mb'], results[b2]['max_allocated_mb']
    
    mem_per_sample = (m2 - m1) / (b2 - b1)
    base_memory = m1 - mem_per_sample * b1
    
    target_memory_mb = target_memory / 1024**2
    max_batch = int((target_memory_mb - base_memory) / max(mem_per_sample, 0.1))
    
    max_batch = max(1, min(max_batch, 4096))
    
    return max_batch
