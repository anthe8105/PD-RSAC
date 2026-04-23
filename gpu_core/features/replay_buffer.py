"""CPU-resident replay buffer for RL training (GPU-transfer on sample)."""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, NamedTuple


class Transition(NamedTuple):
    """Single transition tuple."""
    state: Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: Dict[str, torch.Tensor]
    done: torch.Tensor


@dataclass
class ReplayBatch:
    """Batch of transitions for training."""
    states: Dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: Dict[str, torch.Tensor]
    dones: torch.Tensor
    weights: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None
    # Assignment info for auxiliary loss
    serve_vehicle_idx: Optional[torch.Tensor] = None  # [batch, max_serve]
    serve_trip_idx: Optional[torch.Tensor] = None     # [batch, max_serve]
    num_served: Optional[torch.Tensor] = None         # [batch]
    charge_vehicle_idx: Optional[torch.Tensor] = None # [batch, max_charge]
    charge_station_idx: Optional[torch.Tensor] = None # [batch, max_charge]
    num_charged: Optional[torch.Tensor] = None        # [batch]
    # Scenario info for WDRO
    scenario_demand: Optional[torch.Tensor] = None    # [batch, num_hexes]
    scenario_context: Optional[torch.Tensor] = None   # [batch, context_dim]
    # Semi-MDP action durations (mean across vehicles per step) [batch]
    durations: Optional[torch.Tensor] = None
    # Fleet-level action data
    hex_allocations: Optional[torch.Tensor] = None    # [batch, H, 3]
    hex_repos_targets: Optional[torch.Tensor] = None  # [batch, H]
    hex_charge_power: Optional[torch.Tensor] = None   # [batch, H]
    vehicle_hex_ids: Optional[torch.Tensor] = None    # [batch, V]


class GPUReplayBuffer:
    """
    CPU-resident replay buffer for efficient training.

    Data is stored in CPU pinned memory to avoid consuming GPU VRAM.
    Sampled batches are transferred to `training_device` (GPU) on demand
    using non-blocking DMA, so CPU↔GPU transfer (~1-2 ms per batch) is
    fully hidden behind the GCN forward pass (~20-100 ms).

    Storage math (500k transitions, 1000 vehicles, 3985 hexes):
        GPU layout (old): 500k × 35,934 floats × 4 B × 2 = ~143 GB  ← impossible
        CPU pinned (new): same 143 GB on system RAM, batches streamed to GPU
    """

    def __init__(
        self,
        capacity: int,
        num_vehicles: int,
        vehicle_feature_dim: int,
        num_hexes: int,
        hex_feature_dim: int,
        context_dim: int,
        action_dim: int = 2,  # (action_type, target)
        prioritized: bool = True,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_annealing_steps: int = 100000,
        device: str = "cpu",          # storage device — always CPU
        training_device: Optional[str] = None,  # GPU to transfer batches to
        # Assignment storage for auxiliary loss
        max_serve_per_step: int = 200,  # Max vehicles that can serve per step
        max_charge_per_step: int = 100,  # Max vehicles that can charge per step
        num_stations: int = 150,
    ):
        self.capacity = capacity
        self.device = torch.device("cpu")   # storage always CPU
        self.training_device = torch.device(training_device) if training_device else None
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_annealing_steps = beta_annealing_steps
        
        self._current_beta = beta_start
        self._step = 0
        
        # Circular buffer position
        self._pos = 0
        self._size = 0
        
        # Pre-allocate CPU pinned-memory tensors.
        # Pinned (page-locked) memory enables fast non-blocking DMA transfers to GPU.
        # States
        self.vehicle_states = torch.zeros(
            capacity, num_vehicles, vehicle_feature_dim,
            dtype=torch.float32,
        ).pin_memory()
        self.hex_states = torch.zeros(
            capacity, num_hexes, hex_feature_dim,
            dtype=torch.float32,
        ).pin_memory()
        self.context_states = torch.zeros(
            capacity, context_dim,
            dtype=torch.float32,
        ).pin_memory()

        # Next states
        self.next_vehicle_states = torch.zeros(
            capacity, num_vehicles, vehicle_feature_dim,
            dtype=torch.float32,
        ).pin_memory()
        self.next_hex_states = torch.zeros(
            capacity, num_hexes, hex_feature_dim,
            dtype=torch.float32,
        ).pin_memory()
        self.next_context_states = torch.zeros(
            capacity, context_dim,
            dtype=torch.float32,
        ).pin_memory()

        # Actions, rewards, dones
        self.actions = torch.zeros(
            capacity, num_vehicles, action_dim,
            dtype=torch.long,
        ).pin_memory()
        self.rewards = torch.zeros(capacity, dtype=torch.float32).pin_memory()
        self.dones = torch.zeros(capacity, dtype=torch.bool)  # bool tensors cannot be pinned

        # === ASSIGNMENT STORAGE for auxiliary loss ===
        # Serve assignments: which vehicles got which trips
        self.max_serve = max_serve_per_step
        self.serve_vehicle_idx = torch.full(
            (capacity, max_serve_per_step), -1, dtype=torch.long,
        ).pin_memory()
        self.serve_trip_idx = torch.full(
            (capacity, max_serve_per_step), -1, dtype=torch.long,
        ).pin_memory()
        self.num_served = torch.zeros(capacity, dtype=torch.long).pin_memory()

        # Charge assignments: which vehicles got which stations
        self.max_charge = max_charge_per_step
        self.num_stations = num_stations
        self.charge_vehicle_idx = torch.full(
            (capacity, max_charge_per_step), -1, dtype=torch.long,
        ).pin_memory()
        self.charge_station_idx = torch.full(
            (capacity, max_charge_per_step), -1, dtype=torch.long,
        ).pin_memory()
        self.num_charged = torch.zeros(capacity, dtype=torch.long).pin_memory()

        # Priorities for PER — stays on CPU (torch.multinomial works on CPU)
        if prioritized:
            self.priorities = torch.zeros(capacity, dtype=torch.float32)
            self.max_priority = 1.0

        # === SEMI-MDP DURATION STORAGE ===
        self.durations = torch.ones(capacity, dtype=torch.float32).pin_memory()

        # === SCENARIO STORAGE for WDRO ===
        self.scenario_demand = torch.zeros(
            capacity, num_hexes, dtype=torch.float32,
        ).pin_memory()
        self.scenario_context = torch.zeros(
            capacity, context_dim, dtype=torch.float32,
        ).pin_memory()

        # === FLEET-LEVEL ACTION STORAGE ===
        self.num_hexes = num_hexes
        self.hex_allocations = torch.zeros(
            capacity, num_hexes, 3, dtype=torch.float32,
        ).pin_memory()
        self.hex_repos_targets = torch.zeros(
            capacity, num_hexes, dtype=torch.long,
        ).pin_memory()
        self.hex_charge_power = torch.zeros(
            capacity, num_hexes, dtype=torch.float32,
        ).pin_memory()
        self.fleet_vehicle_hex_ids = torch.zeros(
            capacity, num_vehicles, dtype=torch.long,
        ).pin_memory()
    
    def push(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool,
        serve_assignments: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        charge_assignments: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        scenario_demand: Optional[torch.Tensor] = None,
        scenario_context: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
    ) -> None:
        """Add transition to buffer.
        
        Args:
            state: Current state dict
            action: Actions taken
            reward: Reward received
            next_state: Next state dict
            done: Episode done flag
            serve_assignments: (vehicle_indices, trip_indices) for SERVE actions
            charge_assignments: (vehicle_indices, station_indices) for CHARGE actions
        """
        idx = self._pos

        def _c(t):
            """Move tensor to CPU if needed."""
            return t.cpu() if isinstance(t, torch.Tensor) and t.is_cuda else t

        # Store state (GPU → CPU copy if env runs on GPU)
        self.vehicle_states[idx] = _c(state["vehicle"])
        self.hex_states[idx] = _c(state["hex"])
        self.context_states[idx] = _c(state["context"])

        # Store next state
        self.next_vehicle_states[idx] = _c(next_state["vehicle"])
        self.next_hex_states[idx] = _c(next_state["hex"])
        self.next_context_states[idx] = _c(next_state["context"])

        # Store action, reward, done
        self.actions[idx] = _c(action)
        self.rewards[idx] = reward
        self.dones[idx] = done

        # Store scenario features for WDRO
        if scenario_demand is not None:
            self.scenario_demand[idx] = _c(scenario_demand)
        else:
            self.scenario_demand[idx] = _c(state["hex"][:, 0]) if "hex" in state else 0.0

        if scenario_context is not None:
            self.scenario_context[idx] = _c(scenario_context)
        else:
            ctx = state.get("context", torch.zeros(self.context_states.shape[1]))
            self.scenario_context[idx] = _c(ctx)
        
        # Store serve assignments
        if serve_assignments is not None:
            veh_idx, trip_idx = serve_assignments
            n = min(len(veh_idx), self.max_serve)
            self.serve_vehicle_idx[idx, :n] = _c(veh_idx[:n])
            self.serve_trip_idx[idx, :n] = _c(trip_idx[:n])
            self.num_served[idx] = n
            # Clear remaining
            if n < self.max_serve:
                self.serve_vehicle_idx[idx, n:] = -1
                self.serve_trip_idx[idx, n:] = -1
        else:
            self.serve_vehicle_idx[idx] = -1
            self.serve_trip_idx[idx] = -1
            self.num_served[idx] = 0

        # Store charge assignments
        if charge_assignments is not None:
            veh_idx, station_idx = charge_assignments
            n = min(len(veh_idx), self.max_charge)
            self.charge_vehicle_idx[idx, :n] = _c(veh_idx[:n])
            self.charge_station_idx[idx, :n] = _c(station_idx[:n])
            self.num_charged[idx] = n
            # Clear remaining
            if n < self.max_charge:
                self.charge_vehicle_idx[idx, n:] = -1
                self.charge_station_idx[idx, n:] = -1
        else:
            self.charge_vehicle_idx[idx] = -1
            self.charge_station_idx[idx] = -1
            self.num_charged[idx] = 0

        # Store Semi-MDP duration (mean across vehicles → scalar per transition)
        if duration is not None:
            if isinstance(duration, torch.Tensor) and duration.numel() > 1:
                self.durations[idx] = _c(duration).float().mean()
            else:
                self.durations[idx] = float(duration.item() if isinstance(duration, torch.Tensor) else duration)
        else:
            self.durations[idx] = 1.0

        # Update priority
        if self.prioritized:
            self.priorities[idx] = self.max_priority

        # Update position
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
    
    def push_fleet(
        self,
        state: Dict[str, torch.Tensor],
        hex_allocations: torch.Tensor,       # [H, 3]
        hex_repos_targets: torch.Tensor,     # [H]
        hex_charge_power: torch.Tensor,      # [H]
        vehicle_hex_ids: torch.Tensor,       # [V]
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool,
        serve_assignments: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        charge_assignments: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        scenario_demand: Optional[torch.Tensor] = None,
        scenario_context: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
    ) -> None:
        """Add fleet-level transition to buffer.

        Stores hex-level actions instead of per-vehicle actions.
        State/next_state/reward/done handling is identical to push().
        """
        idx = self._pos

        def _c(t):
            return t.cpu() if isinstance(t, torch.Tensor) and t.is_cuda else t

        # States
        self.vehicle_states[idx] = _c(state["vehicle"])
        self.hex_states[idx] = _c(state["hex"])
        self.context_states[idx] = _c(state["context"])
        self.next_vehicle_states[idx] = _c(next_state["vehicle"])
        self.next_hex_states[idx] = _c(next_state["hex"])
        self.next_context_states[idx] = _c(next_state["context"])

        # Fleet actions
        self.hex_allocations[idx] = _c(hex_allocations)
        self.hex_repos_targets[idx] = _c(hex_repos_targets)
        self.hex_charge_power[idx] = _c(hex_charge_power)
        self.fleet_vehicle_hex_ids[idx] = _c(vehicle_hex_ids)

        # Keep per-vehicle actions zeroed (not used in fleet mode)
        self.actions[idx] = 0

        self.rewards[idx] = reward
        self.dones[idx] = done

        # Scenario features
        if scenario_demand is not None:
            self.scenario_demand[idx] = _c(scenario_demand)
        else:
            self.scenario_demand[idx] = _c(state["hex"][:, 0]) if "hex" in state else 0.0
        if scenario_context is not None:
            self.scenario_context[idx] = _c(scenario_context)
        else:
            ctx = state.get("context", torch.zeros(self.context_states.shape[1]))
            self.scenario_context[idx] = _c(ctx)

        # Serve/charge assignments (still useful for logging)
        if serve_assignments is not None:
            veh_idx, trip_idx = serve_assignments
            n = min(len(veh_idx), self.max_serve)
            self.serve_vehicle_idx[idx, :n] = _c(veh_idx[:n])
            self.serve_trip_idx[idx, :n] = _c(trip_idx[:n])
            self.num_served[idx] = n
            if n < self.max_serve:
                self.serve_vehicle_idx[idx, n:] = -1
                self.serve_trip_idx[idx, n:] = -1
        else:
            self.serve_vehicle_idx[idx] = -1
            self.serve_trip_idx[idx] = -1
            self.num_served[idx] = 0

        if charge_assignments is not None:
            veh_idx, station_idx = charge_assignments
            n = min(len(veh_idx), self.max_charge)
            self.charge_vehicle_idx[idx, :n] = _c(veh_idx[:n])
            self.charge_station_idx[idx, :n] = _c(station_idx[:n])
            self.num_charged[idx] = n
            if n < self.max_charge:
                self.charge_vehicle_idx[idx, n:] = -1
                self.charge_station_idx[idx, n:] = -1
        else:
            self.charge_vehicle_idx[idx] = -1
            self.charge_station_idx[idx] = -1
            self.num_charged[idx] = 0

        # Duration
        if duration is not None:
            if isinstance(duration, torch.Tensor) and duration.numel() > 1:
                self.durations[idx] = _c(duration).float().mean()
            else:
                self.durations[idx] = float(duration.item() if isinstance(duration, torch.Tensor) else duration)
        else:
            self.durations[idx] = 1.0

        # Priority
        if self.prioritized:
            self.priorities[idx] = self.max_priority

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_batch(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: Dict[str, torch.Tensor],
        dones: torch.Tensor,
    ) -> None:
        """Add batch of transitions to buffer (vectorized)."""
        batch_size = len(rewards)

        # Calculate indices on CPU (buffer lives on CPU)
        indices = (torch.arange(batch_size) + self._pos) % self.capacity

        def _c(t):
            return t.cpu() if isinstance(t, torch.Tensor) and t.is_cuda else t

        # Vectorized copy (GPU → CPU if env tensors are on GPU)
        self.vehicle_states[indices] = _c(states["vehicle"])
        self.hex_states[indices] = _c(states["hex"])
        self.context_states[indices] = _c(states["context"])
        self.next_vehicle_states[indices] = _c(next_states["vehicle"])
        self.next_hex_states[indices] = _c(next_states["hex"])
        self.next_context_states[indices] = _c(next_states["context"])
        self.actions[indices] = _c(actions)
        self.rewards[indices] = _c(rewards)
        self.dones[indices] = _c(dones).float()

        # Update position and size
        self._pos = (self._pos + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)
    
    def sample(
        self,
        batch_size: int,
        adjacency: Optional[torch.Tensor] = None,
    ) -> ReplayBatch:
        """Sample batch of transitions."""
        if self.prioritized:
            return self._sample_prioritized(batch_size, adjacency)
        else:
            return self._sample_uniform(batch_size, adjacency)
    
    def _build_batch(
        self,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor],
        adjacency: Optional[torch.Tensor],
    ) -> ReplayBatch:
        """Gather tensors for sampled indices and move to training device."""
        td = self.training_device  # GPU or None

        def _g(t):
            """Index on CPU then transfer to training device non-blocking."""
            sliced = t[indices]
            if td is not None:
                return sliced.to(td, non_blocking=True)
            return sliced

        states = {
            "vehicle": _g(self.vehicle_states),
            "hex": _g(self.hex_states),
            "context": _g(self.context_states),
        }
        if adjacency is not None:
            states["adjacency"] = adjacency  # already on training device

        next_states = {
            "vehicle": _g(self.next_vehicle_states),
            "hex": _g(self.next_hex_states),
            "context": _g(self.next_context_states),
        }
        if adjacency is not None:
            next_states["adjacency"] = adjacency

        return ReplayBatch(
            states=states,
            actions=_g(self.actions),
            rewards=_g(self.rewards),
            next_states=next_states,
            dones=_g(self.dones),
            weights=weights.to(td, non_blocking=True) if (weights is not None and td is not None) else weights,
            indices=indices,  # keep on CPU — used by update_priorities against CPU buffer
            serve_vehicle_idx=_g(self.serve_vehicle_idx),
            serve_trip_idx=_g(self.serve_trip_idx),
            num_served=_g(self.num_served),
            charge_vehicle_idx=_g(self.charge_vehicle_idx),
            charge_station_idx=_g(self.charge_station_idx),
            num_charged=_g(self.num_charged),
            scenario_demand=_g(self.scenario_demand),
            scenario_context=_g(self.scenario_context),
            durations=_g(self.durations),
            # Fleet-level action data
            hex_allocations=_g(self.hex_allocations),
            hex_repos_targets=_g(self.hex_repos_targets),
            hex_charge_power=_g(self.hex_charge_power),
            vehicle_hex_ids=_g(self.fleet_vehicle_hex_ids),
        )

    def _sample_uniform(
        self,
        batch_size: int,
        adjacency: Optional[torch.Tensor],
    ) -> ReplayBatch:
        """Sample uniformly from buffer."""
        indices = torch.randint(0, self._size, (batch_size,))  # CPU
        return self._build_batch(indices, weights=None, adjacency=adjacency)

    def _sample_prioritized(
        self,
        batch_size: int,
        adjacency: Optional[torch.Tensor],
    ) -> ReplayBatch:
        """Sample based on priorities (all PER math stays on CPU)."""
        priorities = self.priorities[:self._size].clamp(min=1e-8)
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0 or not torch.isfinite(probs_sum):
            # Fallback: uniform sampling when priorities are degenerate
            indices = torch.randint(0, self._size, (batch_size,))
            weights = torch.ones(batch_size)
            return self._build_batch(indices, weights=weights, adjacency=adjacency)
        probs = probs / probs_sum
        probs = probs.clamp(min=0)          # guard floating-point negatives
        probs = probs / probs.sum()         # re-normalise after clamp

        indices = torch.multinomial(probs, batch_size, replacement=True)  # CPU

        weights = (self._size * probs[indices]) ** (-self._current_beta)
        weights = weights / weights.max()  # Normalize; still on CPU here

        return self._build_batch(indices, weights=weights, adjacency=adjacency)

    def update_priorities(
        self,
        indices: torch.Tensor,
        td_errors: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> None:
        """Update priorities based on TD errors.

        indices  — CPU tensor (as returned in ReplayBatch.indices)
        td_errors — may be on GPU; moved to CPU here before writing to buffer
        """
        if not self.prioritized:
            return

        priorities = (torch.abs(td_errors) + epsilon).cpu()
        cpu_indices = indices.cpu()
        self.priorities[cpu_indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())
    
    def step_beta(self) -> None:
        """Anneal beta for importance sampling."""
        self._step += 1
        progress = min(self._step / self.beta_annealing_steps, 1.0)
        self._current_beta = self.beta_start + progress * (self.beta_end - self.beta_start)
    
    def __len__(self) -> int:
        return self._size
    
    def clear(self) -> None:
        """Clear buffer."""
        self._pos = 0
        self._size = 0
        self._step = 0
        self._current_beta = self.beta_start
        if self.prioritized:
            self.priorities.fill_(0)
            self.max_priority = 1.0
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        stats = {
            "size": self._size,
            "capacity": self.capacity,
            "fill_ratio": self._size / self.capacity,
        }
        if self.prioritized:
            valid_priorities = self.priorities[:self._size]
            stats["mean_priority"] = valid_priorities.mean().item()
            stats["max_priority"] = self.max_priority
            stats["beta"] = self._current_beta
        return stats
