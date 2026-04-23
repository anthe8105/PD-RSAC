"""
Visualization utilities for EV Fleet RL training.

Provides tools for visualizing:
- Training curves (reward, loss, etc.)
- Fleet state
- Spatial distributions
- Performance metrics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class TrainingMetrics:
    """Container for training metrics over time."""
    episodes: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    actor_losses: List[float] = field(default_factory=list)
    critic_losses: List[float] = field(default_factory=list)
    alpha_values: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    
    trips_served: List[int] = field(default_factory=list)
    avg_soc: List[float] = field(default_factory=list)
    steps_per_episode: List[int] = field(default_factory=list)
    
    eval_rewards: List[float] = field(default_factory=list)
    eval_episodes: List[int] = field(default_factory=list)
    
    timestamps: List[float] = field(default_factory=list)
    
    def add_episode(
        self,
        episode: int,
        reward: float,
        actor_loss: Optional[float] = None,
        critic_loss: Optional[float] = None,
        alpha: Optional[float] = None,
        entropy: Optional[float] = None,
        trips: Optional[int] = None,
        soc: Optional[float] = None,
        steps: Optional[int] = None,
        timestamp: Optional[float] = None
    ):
        self.episodes.append(episode)
        self.rewards.append(reward)
        
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
        if alpha is not None:
            self.alpha_values.append(alpha)
        if entropy is not None:
            self.entropies.append(entropy)
        if trips is not None:
            self.trips_served.append(trips)
        if soc is not None:
            self.avg_soc.append(soc)
        if steps is not None:
            self.steps_per_episode.append(steps)
        if timestamp is not None:
            self.timestamps.append(timestamp)
    
    def add_eval(self, episode: int, reward: float):
        self.eval_episodes.append(episode)
        self.eval_rewards.append(reward)
    
    def save(self, path: str):
        data = {
            'episodes': self.episodes,
            'rewards': self.rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'alpha_values': self.alpha_values,
            'entropies': self.entropies,
            'trips_served': self.trips_served,
            'avg_soc': self.avg_soc,
            'steps_per_episode': self.steps_per_episode,
            'eval_rewards': self.eval_rewards,
            'eval_episodes': self.eval_episodes,
            'timestamps': self.timestamps
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingMetrics':
        with open(path, 'r') as f:
            data = json.load(f)
        
        metrics = cls()
        metrics.episodes = data.get('episodes', [])
        metrics.rewards = data.get('rewards', [])
        metrics.actor_losses = data.get('actor_losses', [])
        metrics.critic_losses = data.get('critic_losses', [])
        metrics.alpha_values = data.get('alpha_values', [])
        metrics.entropies = data.get('entropies', [])
        metrics.trips_served = data.get('trips_served', [])
        metrics.avg_soc = data.get('avg_soc', [])
        metrics.steps_per_episode = data.get('steps_per_episode', [])
        metrics.eval_rewards = data.get('eval_rewards', [])
        metrics.eval_episodes = data.get('eval_episodes', [])
        metrics.timestamps = data.get('timestamps', [])
        
        return metrics


class TrainingVisualizer:
    """
    Visualizer for training metrics.
    
    Can use matplotlib or output to TensorBoard-compatible format.
    """
    
    def __init__(
        self,
        metrics: Optional[TrainingMetrics] = None,
        output_dir: str = 'plots'
    ):
        self.metrics = metrics or TrainingMetrics()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._has_matplotlib = False
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self._has_matplotlib = True
            self._plt = plt
        except ImportError:
            pass
    
    def plot_rewards(
        self,
        window: int = 100,
        save_path: Optional[str] = None
    ):
        """Plot episode rewards with smoothing."""
        if not self._has_matplotlib:
            print("matplotlib not available")
            return
        
        plt = self._plt
        
        episodes = self.metrics.episodes
        rewards = self.metrics.rewards
        
        if len(rewards) == 0:
            print("No reward data to plot")
            return
        
        smoothed = self._smooth(rewards, window)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        ax.plot(episodes[:len(smoothed)], smoothed, color='blue', linewidth=2, label=f'Smoothed (w={window})')
        
        if self.metrics.eval_episodes:
            ax.scatter(
                self.metrics.eval_episodes,
                self.metrics.eval_rewards,
                color='red', s=50, zorder=5, label='Eval'
            )
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        path = save_path or self.output_dir / 'rewards.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved reward plot to: {path}")
    
    def plot_losses(self, save_path: Optional[str] = None):
        """Plot actor and critic losses."""
        if not self._has_matplotlib:
            print("matplotlib not available")
            return
        
        plt = self._plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        if self.metrics.critic_losses:
            episodes = list(range(len(self.metrics.critic_losses)))
            smoothed = self._smooth(self.metrics.critic_losses, 50)
            axes[0].plot(episodes, self.metrics.critic_losses, alpha=0.3, color='orange')
            axes[0].plot(episodes[:len(smoothed)], smoothed, color='orange', linewidth=2)
            axes[0].set_ylabel('Critic Loss')
            axes[0].set_title('Training Losses')
            axes[0].grid(True, alpha=0.3)
        
        if self.metrics.actor_losses:
            episodes = list(range(len(self.metrics.actor_losses)))
            smoothed = self._smooth(self.metrics.actor_losses, 50)
            axes[1].plot(episodes, self.metrics.actor_losses, alpha=0.3, color='green')
            axes[1].plot(episodes[:len(smoothed)], smoothed, color='green', linewidth=2)
            axes[1].set_ylabel('Actor Loss')
            axes[1].set_xlabel('Update')
            axes[1].grid(True, alpha=0.3)
        
        path = save_path or self.output_dir / 'losses.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved loss plot to: {path}")
    
    def plot_alpha_entropy(self, save_path: Optional[str] = None):
        """Plot alpha (temperature) and entropy."""
        if not self._has_matplotlib:
            print("matplotlib not available")
            return
        
        plt = self._plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        if self.metrics.alpha_values:
            episodes = list(range(len(self.metrics.alpha_values)))
            axes[0].plot(episodes, self.metrics.alpha_values, color='purple', linewidth=2)
            axes[0].set_ylabel('Alpha (Temperature)')
            axes[0].set_title('SAC Temperature and Entropy')
            axes[0].grid(True, alpha=0.3)
        
        if self.metrics.entropies:
            episodes = list(range(len(self.metrics.entropies)))
            axes[1].plot(episodes, self.metrics.entropies, color='teal', linewidth=2)
            axes[1].set_ylabel('Entropy')
            axes[1].set_xlabel('Update')
            axes[1].grid(True, alpha=0.3)
        
        path = save_path or self.output_dir / 'alpha_entropy.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved alpha/entropy plot to: {path}")
    
    def plot_fleet_metrics(self, save_path: Optional[str] = None):
        """Plot fleet-specific metrics."""
        if not self._has_matplotlib:
            print("matplotlib not available")
            return
        
        plt = self._plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        if self.metrics.trips_served:
            episodes = self.metrics.episodes[:len(self.metrics.trips_served)]
            axes[0].plot(episodes, self.metrics.trips_served, color='green', linewidth=2)
            axes[0].set_ylabel('Trips Served')
            axes[0].set_title('Fleet Metrics')
            axes[0].grid(True, alpha=0.3)
        
        if self.metrics.avg_soc:
            episodes = self.metrics.episodes[:len(self.metrics.avg_soc)]
            axes[1].plot(episodes, self.metrics.avg_soc, color='blue', linewidth=2)
            axes[1].set_ylabel('Avg SOC (%)')
            axes[1].set_xlabel('Episode')
            axes[1].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Low SOC')
            axes[1].axhline(y=80, color='green', linestyle='--', alpha=0.5, label='High SOC')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        path = save_path or self.output_dir / 'fleet_metrics.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved fleet metrics plot to: {path}")
    
    def plot_all(self, prefix: str = ''):
        """Generate all plots."""
        name_prefix = f"{prefix}_" if prefix else ""
        
        self.plot_rewards(save_path=self.output_dir / f'{name_prefix}rewards.png')
        self.plot_losses(save_path=self.output_dir / f'{name_prefix}losses.png')
        self.plot_alpha_entropy(save_path=self.output_dir / f'{name_prefix}alpha_entropy.png')
        self.plot_fleet_metrics(save_path=self.output_dir / f'{name_prefix}fleet_metrics.png')
    
    def _smooth(self, values: List[float], window: int) -> List[float]:
        """Apply moving average smoothing."""
        if len(values) < window:
            return values
        
        smoothed = []
        for i in range(len(values) - window + 1):
            smoothed.append(sum(values[i:i+window]) / window)
        return smoothed


class FleetVisualizer:
    """
    Visualizer for fleet state.
    """
    
    def __init__(self, output_dir: str = 'plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._has_matplotlib = False
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self._has_matplotlib = True
            self._plt = plt
        except ImportError:
            pass
    
    def plot_vehicle_distribution(
        self,
        positions: torch.Tensor,
        num_hexes: int,
        save_path: Optional[str] = None
    ):
        """Plot vehicle distribution across hexes."""
        if not self._has_matplotlib:
            print("matplotlib not available")
            return
        
        plt = self._plt
        
        counts = torch.zeros(num_hexes)
        for pos in positions:
            counts[pos.item()] += 1
        
        counts_np = counts.numpy()
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(num_hexes), counts_np, width=1.0)
        ax.set_xlabel('Hex ID')
        ax.set_ylabel('Vehicle Count')
        ax.set_title('Vehicle Distribution')
        
        path = save_path or self.output_dir / 'vehicle_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_soc_distribution(
        self,
        soc_values: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """Plot SOC distribution histogram."""
        if not self._has_matplotlib:
            print("matplotlib not available")
            return
        
        plt = self._plt
        
        soc_np = soc_values.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(soc_np, bins=20, range=(0, 100), edgecolor='black')
        ax.axvline(x=20, color='red', linestyle='--', label='Low threshold')
        ax.axvline(x=80, color='green', linestyle='--', label='High threshold')
        ax.set_xlabel('State of Charge (%)')
        ax.set_ylabel('Vehicle Count')
        ax.set_title('SOC Distribution')
        ax.legend()
        
        path = save_path or self.output_dir / 'soc_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_status_pie(
        self,
        status_counts: Dict[int, int],
        save_path: Optional[str] = None
    ):
        """Plot vehicle status pie chart."""
        if not self._has_matplotlib:
            print("matplotlib not available")
            return
        
        plt = self._plt
        
        labels = ['Idle', 'Serving', 'Charging', 'Repositioning']
        sizes = [
            status_counts.get(0, 0),
            status_counts.get(1, 0),
            status_counts.get(2, 0),
            status_counts.get(3, 0)
        ]
        colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Vehicle Status Distribution')
        
        path = save_path or self.output_dir / 'status_pie.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()


class PerformanceVisualizer:
    """
    Visualizer for performance benchmarks.
    """
    
    def __init__(self, output_dir: str = 'plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._has_matplotlib = False
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self._has_matplotlib = True
            self._plt = plt
        except ImportError:
            pass
    
    def plot_scaling(
        self,
        vehicle_counts: List[int],
        throughputs: List[float],
        save_path: Optional[str] = None
    ):
        """Plot throughput scaling with fleet size."""
        if not self._has_matplotlib:
            print("matplotlib not available")
            return
        
        plt = self._plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(vehicle_counts, throughputs, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Vehicles')
        ax.set_ylabel('Steps per Second')
        ax.set_title('Throughput Scaling')
        ax.grid(True, alpha=0.3)
        
        ax.set_xscale('log')
        
        path = save_path or self.output_dir / 'scaling.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_memory_scaling(
        self,
        batch_sizes: List[int],
        memory_mbs: List[float],
        save_path: Optional[str] = None
    ):
        """Plot memory scaling with batch size."""
        if not self._has_matplotlib:
            print("matplotlib not available")
            return
        
        plt = self._plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(batch_sizes, memory_mbs, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('GPU Memory (MB)')
        ax.set_title('Memory Scaling')
        ax.grid(True, alpha=0.3)
        
        path = save_path or self.output_dir / 'memory_scaling.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()


def create_training_report(
    metrics: TrainingMetrics,
    output_path: str,
    include_plots: bool = True
):
    """
    Create a comprehensive training report.
    
    Args:
        metrics: Training metrics
        output_path: Path for report directory
        include_plots: Whether to include plots
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'total_episodes': len(metrics.episodes),
        'final_reward': metrics.rewards[-1] if metrics.rewards else 0,
        'best_reward': max(metrics.rewards) if metrics.rewards else 0,
        'avg_reward_last_100': (
            sum(metrics.rewards[-100:]) / min(len(metrics.rewards), 100)
            if metrics.rewards else 0
        ),
        'total_trips': sum(metrics.trips_served) if metrics.trips_served else 0,
        'avg_soc': (
            sum(metrics.avg_soc) / len(metrics.avg_soc)
            if metrics.avg_soc else 0
        )
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    metrics.save(str(output_dir / 'metrics.json'))
    
    if include_plots:
        viz = TrainingVisualizer(metrics, output_dir=str(output_dir))
        viz.plot_all()
    
    report_lines = [
        "# Training Report",
        "",
        f"## Summary",
        f"- Total Episodes: {summary['total_episodes']}",
        f"- Final Reward: {summary['final_reward']:.2f}",
        f"- Best Reward: {summary['best_reward']:.2f}",
        f"- Avg Reward (last 100): {summary['avg_reward_last_100']:.2f}",
        f"- Total Trips Served: {summary['total_trips']}",
        f"- Average SOC: {summary['avg_soc']:.1f}%",
        "",
        "## Plots",
        "- rewards.png: Training rewards over episodes",
        "- losses.png: Actor and critic losses",
        "- alpha_entropy.png: SAC temperature and entropy",
        "- fleet_metrics.png: Trips served and SOC",
    ]
    
    with open(output_dir / 'report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Training report saved to: {output_dir}")
