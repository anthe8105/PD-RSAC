"""
Enhanced Episode Collector with Semi-MDP duration tracking.

Extends EpisodeCollector to track action durations for Semi-MDP training
as described in paper Section 5.1.2.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time

from .episode_collector import EpisodeCollector, EpisodeStats
from ..simulator.environment import GPUEnvironment
from ..features.replay_buffer import GPUReplayBuffer


@dataclass
class EnhancedEpisodeStats(EpisodeStats):
    """Extended episode stats with duration tracking."""
    avg_serve_duration: float = 0.0
    avg_reposition_duration: float = 0.0
    avg_charge_duration: float = 0.0
    duration_variance: float = 0.0


@dataclass
class TransitionWithDuration:
    """Single transition with duration for Semi-MDP."""
    state: Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    next_state: Dict[str, torch.Tensor]
    done: bool
    duration: float  # Time until next decision (Δ in paper)
    action_type: int  # 0=SERVE, 1=CHARGE, 2=REPOSITION (IDLE removed)


class EnhancedReplayBuffer:
    """
    Replay buffer that stores durations for Semi-MDP training.
    
    Per paper Section 5.1.2:
    - Each transition stores duration Δ
    - Used to compute γ^Δ discount factor
    """
    
    def __init__(
        self,
        base_buffer: GPUReplayBuffer,
        device: str = "cuda",
        num_vehicles: int = 100
    ):
        self.base_buffer = base_buffer
        self.device = torch.device("cpu")          # storage always CPU
        self.training_device = torch.device(device) if device != "cpu" else None

        # Duration storage on CPU pinned memory — transferred to GPU on sample.
        capacity = base_buffer.capacity
        self.durations = torch.zeros((capacity, num_vehicles)).pin_memory()
        self.action_types_storage = torch.zeros((capacity, num_vehicles), dtype=torch.long).pin_memory()
        self.scenarios = [0] * capacity  # Keep as list, only used for filtering
        self.buffer_size = 0  # Track actual filled size
        
        # Duration statistics for normalization
        self.duration_stats = {
            'serve': {'sum': 0.0, 'count': 0},
            'reposition': {'sum': 0.0, 'count': 0},
            'charge': {'sum': 0.0, 'count': 0}
        }
    
    def push(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool,
        duration: torch.Tensor,
        action_types: torch.Tensor,
        scenario: int = 0,
        serve_assignments=None,
        charge_assignments=None
    ):
        """Push transition with duration info."""
        # Push to base buffer — forward duration so GPUReplayBuffer.durations
        # (mean per step) stays populated for the trainer's γ^Δ discounting.
        self.base_buffer.push(
            state, action, reward, next_state, done,
            serve_assignments=serve_assignments,
            charge_assignments=charge_assignments,
            duration=duration,
        )
        
        # Store duration info in preallocated CPU tensor (GPU → CPU if needed)
        idx = self.buffer_size % self.durations.shape[0]
        self.durations[idx] = duration.cpu() if isinstance(duration, torch.Tensor) and duration.is_cuda else duration
        self.action_types_storage[idx] = action_types.cpu() if isinstance(action_types, torch.Tensor) and action_types.is_cuda else action_types
        self.scenarios[idx] = scenario
        
        if self.buffer_size < self.durations.shape[0]:
            self.buffer_size += 1
        
        # Update stats
        self._update_duration_stats(duration, action_types)
    
    def push_fleet(
        self,
        state: Dict[str, torch.Tensor],
        hex_allocations: torch.Tensor,
        hex_repos_targets: torch.Tensor,
        hex_charge_power: torch.Tensor,
        vehicle_hex_ids: torch.Tensor,
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool,
        duration: torch.Tensor,
        action_types: torch.Tensor,
        scenario: int = 0,
        serve_assignments=None,
        charge_assignments=None
    ):
        """Push fleet-level transition with duration info."""
        self.base_buffer.push_fleet(
            state=state,
            hex_allocations=hex_allocations,
            hex_repos_targets=hex_repos_targets,
            hex_charge_power=hex_charge_power,
            vehicle_hex_ids=vehicle_hex_ids,
            reward=reward,
            next_state=next_state,
            done=done,
            serve_assignments=serve_assignments,
            charge_assignments=charge_assignments,
            duration=duration,
        )
        
        # Store duration info in preallocated CPU tensor (GPU → CPU if needed)
        idx = self.buffer_size % self.durations.shape[0]
        self.durations[idx] = duration.cpu() if isinstance(duration, torch.Tensor) and duration.is_cuda else duration
        self.action_types_storage[idx] = action_types.cpu() if isinstance(action_types, torch.Tensor) and action_types.is_cuda else action_types
        self.scenarios[idx] = scenario
        
        if self.buffer_size < self.durations.shape[0]:
            self.buffer_size += 1
        
        # Update stats
        self._update_duration_stats(duration, action_types)
    
    def _update_duration_stats(self, duration: torch.Tensor, action_types: torch.Tensor):
        """Update duration statistics for normalization."""
        # Ensure CPU for stats computation
        dur_cpu = duration.cpu() if isinstance(duration, torch.Tensor) and duration.is_cuda else duration
        act_cpu = action_types.cpu() if isinstance(action_types, torch.Tensor) and action_types.is_cuda else action_types
        for action_idx, name in [(0, 'serve'), (1, 'charge'), (2, 'reposition')]:
            mask = act_cpu == action_idx
            if mask.any():
                self.duration_stats[name]['sum'] += dur_cpu[mask].sum().item()
                self.duration_stats[name]['count'] += mask.sum().item()
    
    def sample(self, batch_size: int) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Sample batch with durations.
        
        Returns:
            base_batch: Base buffer batch
            durations: Duration tensor [batch_size, num_vehicles]
            scenarios: Scenario indices [batch_size]
        """
        # Get indices from base buffer
        if self.buffer_size < batch_size:
            batch_size = self.buffer_size

        indices = torch.randint(0, self.buffer_size, (batch_size,))  # CPU

        # Get base batch (already moved to training_device by GPUReplayBuffer._build_batch)
        base_batch = self.base_buffer.sample(batch_size)

        # Index on CPU then move to training device
        durations = self.durations[indices]
        scenarios = torch.tensor([self.scenarios[i.item()] for i in indices])
        if self.training_device is not None:
            durations = durations.to(self.training_device, non_blocking=True)
            scenarios = scenarios.to(self.training_device, non_blocking=True)

        return base_batch, durations, scenarios
    
    def sample_by_scenario(self, batch_size: int, scenario: int) -> Tuple[Dict, torch.Tensor]:
        """Sample transitions from specific scenario (for WDRO)."""
        # All indexing on CPU; move results to training device at the end
        scenarios_tensor = torch.tensor(self.scenarios[:self.buffer_size])  # CPU
        scenario_mask = scenarios_tensor == scenario
        scenario_indices = torch.nonzero(scenario_mask, as_tuple=False).squeeze(-1)

        if len(scenario_indices) < batch_size:
            batch_size = len(scenario_indices)

        if batch_size == 0:
            return None, None

        sample_positions = torch.randint(0, len(scenario_indices), (batch_size,))  # CPU
        actual_indices = scenario_indices[sample_positions]

        base_batch = self.base_buffer.sample(batch_size)
        durations = self.durations[actual_indices]  # CPU
        if self.training_device is not None:
            durations = durations.to(self.training_device, non_blocking=True)

        return base_batch, durations
    
    def get_avg_durations(self) -> Dict[str, float]:
        """Get average durations by action type."""
        result = {}
        for name, stats in self.duration_stats.items():
            if stats['count'] > 0:
                result[name] = stats['sum'] / stats['count']
            else:
                result[name] = 1.0  # Default
        return result
    
    def __len__(self) -> int:
        return len(self.durations)


class EnhancedEpisodeCollector(EpisodeCollector):
    """
    Episode collector with duration tracking for Semi-MDP.
    
    Per paper Section 5.1.2:
    - Tracks action durations for γ^Δ discounting
    - Estimates duration based on action type and trip distance
    
    Duration estimates (from NYC data analysis):
    - IDLE: 1 timestep (waiting)
    - SERVE: trip_duration (from data) + pickup_distance/speed
    - CHARGE: SOC_gained / charging_rate
    - REPOSITION: distance / speed
    """
    
    # Default duration parameters (can be tuned from data)
    DURATION_PARAMS = {
        'serve_steps': 1.0,          # SERVE duration in steps (1 step minimum)
        'charge_steps': 1.0,         # CHARGE duration in steps
        'reposition_steps': 1.0,     # REPOSITION duration in steps
        'min_serve_duration': 3.0,   # Minimum serve duration (steps)
        'min_charge_duration': 2.0,
        'pickup_speed_hex_per_step': 1.0,
        'charge_rate_soc_per_step': 4.0,
        'reposition_speed_hex_per_step': 1.0,
    }
    
    def __init__(
        self,
        env: GPUEnvironment,
        replay_buffer: GPUReplayBuffer,
        device: str = "cuda",
        use_milp: bool = False,
        track_durations: bool = True
    ):
        super().__init__(env, replay_buffer, device, use_milp)
        
        self.track_durations = track_durations
        
        # Create enhanced buffer wrapper
        if track_durations:
            # Handle both v1 (env.fleet.num_vehicles) and v2 (env.num_vehicles)
            num_vehicles = getattr(env, 'num_vehicles', None) or env.fleet.num_vehicles
            self.enhanced_buffer = EnhancedReplayBuffer(replay_buffer, device, num_vehicles)
        
        # Duration estimation helpers
        self._distance_matrix = None
        if hasattr(env, 'hex_grid') and hasattr(env.hex_grid, 'distance_matrix'):
            self._distance_matrix = env.hex_grid.distance_matrix._distances
    
    def collect_episode(
        self,
        agent: Any,
        exploration_noise: float = 0.1,
        seed: Optional[int] = None,
        render: bool = False,
        temperature: float = 1.0,
        deterministic: bool = False,
        scenario: int = 0  # For WDRO: which demand scenario
    ) -> EnhancedEpisodeStats:
        """
        Collect episode with duration tracking.
        
        Args:
            agent: RL agent
            exploration_noise: Exploration noise
            seed: Random seed
            render: Whether to render
            temperature: Softmax temperature
            scenario: Demand scenario index (for WDRO)
        
        Returns:
            Enhanced episode statistics
        """
        start_time = time.time()
        state = self.env.reset(seed=seed, fleet_state_only=True)
        
        stats = EnhancedEpisodeStats(episode_id=self.total_episodes)
        done = False
        step_count = 0
        
        # Duration tracking
        total_durations = {'serve': 0.0, 'reposition': 0.0, 'charge': 0.0}
        duration_counts = {'serve': 0, 'reposition': 0, 'charge': 0}
        
        while not done:
            step_temperature = max(0.5, temperature * (1.0 - 0.3 * step_count / self.env.episode_steps))
            
            # Get vehicle states before action (for duration estimation)
            prev_positions = self.env.fleet_state.positions.clone()
            prev_socs = self.env.fleet_state.socs.clone()
            
            with torch.no_grad():
                # Use MILP assignment if enabled
                if self.use_milp and hasattr(self, '_assigner'):
                    action_type, reposition_target, selected_trip = self._select_action_milp(
                        agent, state, temperature=step_temperature, deterministic=deterministic
                    )
                else:
                    action_type, reposition_target, selected_trip = self._select_action(
                        agent, state, exploration_noise, temperature=step_temperature, deterministic=deterministic
                    )
            
            # Track action distribution (only for AVAILABLE vehicles)
            # BUSY vehicles are not counted to avoid inflating IDLE count
            # OPTIMIZED: Use tensor operations instead of Python loop
            action_names = ['serve', 'charge', 'reposition']  # IDLE removed
            available_mask = self.env.get_available_actions()
            is_available = available_mask.any(dim=1)  # [num_vehicles]
            
            # Vectorized action counting: mask invalid actions, then bincount
            action_flat = action_type.view(-1)
            valid_actions = action_flat[(action_flat >= 0) & (action_flat < 3) & is_available]
            if len(valid_actions) > 0:
                counts = torch.bincount(valid_actions, minlength=3)
                for i, name in enumerate(action_names):
                    stats.action_counts[name] += counts[i].item()
            
            step_count += 1
            
            next_state, reward, done_tensor, info = self.env.step(
                action_type, reposition_target, selected_trip
            )
            done = done_tensor.item()
            
            # Estimate durations for this step
            durations = self._estimate_durations(
                action_type=action_type,
                prev_positions=prev_positions,
                new_positions=self.env.fleet_state.positions,
                prev_socs=prev_socs,
                new_socs=self.env.fleet_state.socs,
                reposition_targets=reposition_target,
                info=info
            )
            
            # Update duration stats
            self._update_duration_tracking(durations, action_type, total_durations, duration_counts)
            
            action = self._encode_action(action_type, reposition_target)
            state_dict = self._flat_to_dict_state(state)
            next_state_dict = self._flat_to_dict_state(next_state)
            reward_scalar = reward.item() if isinstance(reward, torch.Tensor) else reward
            
            # Extract assignment info from EnvInfo
            serve_assignments = getattr(info, 'serve_assignments', None)
            charge_assignments = getattr(info, 'charge_assignments', None)
            
            # Route through enhanced_buffer when tracking durations so that:
            #   1. enhanced_buffer stores per-vehicle durations (for future use)
            #   2. enhanced_buffer.push() forwards duration to base GPUReplayBuffer
            #      so ReplayBatch.durations (mean per step) is populated for trainer
            # When not tracking durations (plain SAC), push directly to base buffer.
            if self.track_durations:
                self.enhanced_buffer.push(
                    state=state_dict,
                    action=action,
                    reward=reward_scalar,
                    next_state=next_state_dict,
                    done=done,
                    duration=durations,
                    action_types=action_type,
                    scenario=scenario,
                    serve_assignments=serve_assignments,
                    charge_assignments=charge_assignments,
                )
            else:
                self.replay_buffer.push(
                    state=state_dict,
                    action=action,
                    reward=reward_scalar,
                    next_state=next_state_dict,
                    done=done,
                    serve_assignments=serve_assignments,
                    charge_assignments=charge_assignments,
                )
            
            state = next_state
            stats.total_reward += reward_scalar
            stats.steps += 1
            
            if render:
                self.env.render()
        
        # Finalize stats
        stats.trips_served = info.trips_served
        stats.trips_dropped = info.trips_dropped
        stats.trips_loaded = info.trips_loaded  # FIX: was missing, causing Trips: X/0
        stats.avg_soc = info.avg_soc
        stats.revenue = info.revenue
        stats.driving_cost = getattr(info, 'driving_cost', 0.0)
        stats.energy_cost = info.energy_cost
        stats.profit = stats.revenue - stats.driving_cost - stats.energy_cost
        stats.collection_time = time.time() - start_time
        
        # Duration stats
        for action_name in ['serve', 'reposition', 'charge']:
            if duration_counts[action_name] > 0:
                avg_duration = total_durations[action_name] / duration_counts[action_name]
                setattr(stats, f'avg_{action_name}_duration', avg_duration)
        
        # Compute duration variance
        if sum(duration_counts.values()) > 0:
            all_durations = sum(total_durations.values())
            all_counts = sum(duration_counts.values())
            stats.duration_variance = all_durations / all_counts  # Simplified
        
        self.total_episodes += 1
        self.total_steps += stats.steps
        
        return stats
    
    def _estimate_durations(
        self,
        action_type: torch.Tensor,
        prev_positions: torch.Tensor,
        new_positions: torch.Tensor,
        prev_socs: torch.Tensor,
        new_socs: torch.Tensor,
        reposition_targets: torch.Tensor,
        info: Any
    ) -> torch.Tensor:
        """
        Estimate action durations for Semi-MDP.
        
        Per paper Section 5.1.2:
        - Duration varies by action type
        - SERVE: depends on trip distance
        - CHARGE: depends on SOC gained
        - REPOSITION: depends on distance traveled
        
        Returns:
            durations: [num_vehicles] tensor of estimated durations
        """
        num_vehicles = action_type.shape[0]
        durations = torch.ones(num_vehicles, device=self.device)
        
        params = self.DURATION_PARAMS
        
        # Action mappings for V2: 0=SERVE, 1=CHARGE, 2=REPOSITION
        
        # SERVE (action 0): trip duration + pickup time
        serve_mask = action_type == 0
        if serve_mask.any():
            # Estimate based on position change (using distance matrix if available)
            if self._distance_matrix is not None:
                distances = self._distance_matrix[
                    prev_positions[serve_mask],
                    new_positions[serve_mask]
                ]
                serve_duration = distances / params['pickup_speed_hex_per_step'] + params['min_serve_duration']
            else:
                # Fallback: use position difference as proxy
                pos_diff = (prev_positions[serve_mask] != new_positions[serve_mask]).float()
                serve_duration = pos_diff * 5.0 + params['min_serve_duration']
            
            durations[serve_mask] = serve_duration
        
        # CHARGE (action 1): based on SOC gained
        charge_mask = action_type == 1
        if charge_mask.any():
            soc_gained = (new_socs[charge_mask] - prev_socs[charge_mask]).clamp(min=0)
            charge_duration = soc_gained / params['charge_rate_soc_per_step']
            charge_duration = charge_duration.clamp(min=params['min_charge_duration'])
            durations[charge_mask] = charge_duration
        
        # REPOSITION (action 2): match environment travel-time physics
        reposition_mask = action_type == 2
        if reposition_mask.any():
            if self._distance_matrix is not None:
                distances = self._distance_matrix[
                    prev_positions[reposition_mask],
                    reposition_targets[reposition_mask]
                ]
                repo_duration = self.env.time_dynamics.distance_to_steps(distances).to(torch.float32)
            else:
                repo_duration = torch.ones(reposition_mask.sum(), device=self.device)

            durations[reposition_mask] = repo_duration.clamp(min=1.0)
        
        return durations
    
    def _update_duration_tracking(
        self,
        durations: torch.Tensor,
        action_type: torch.Tensor,
        total_durations: Dict[str, float],
        duration_counts: Dict[str, int]
    ):
        """Update duration tracking for stats."""
        # Fix: Mappings for V2 (0=SERVE, 1=CHARGE, 2=REPOSITION)
        action_names = {0: 'serve', 1: 'charge', 2: 'reposition'}
        
        for action_idx, name in action_names.items():
            mask = action_type == action_idx
            if mask.any():
                total_durations[name] += durations[mask].sum().item()
                duration_counts[name] += mask.sum().item()

    def collect_episode_fleet(
        self,
        agent: Any,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
        scenario: int = 0
    ) -> EnhancedEpisodeStats:
        """Collect a single episode using fleet-level actor with duration tracking."""
        start_time = time.time()
        state = self.env.reset(seed=seed, fleet_state_only=True)
        
        stats = EnhancedEpisodeStats(episode_id=self.total_episodes)
        done = False
        step_count = 0
        
        # Duration tracking
        total_durations = {'serve': 0.0, 'reposition': 0.0, 'charge': 0.0}
        duration_counts = {'serve': 0, 'reposition': 0, 'charge': 0}
        
        while not done:
            step_temperature = max(
                0.1, temperature * (1.0 - 0.3 * step_count / self.env.episode_steps)
            )

            # Get vehicle states before action (for duration estimation and replay consistency)
            prev_positions = self.env.fleet_state.positions.clone()
            prev_socs = self.env.fleet_state.socs.clone()
            pre_step_vehicle_hex_ids = self.env.fleet_state.positions.long().clone()

            with torch.no_grad():
                fleet_out = self._select_action_fleet(
                    agent, state,
                    temperature=step_temperature,
                    deterministic=deterministic,
                )
                if self.use_milp and hasattr(self, '_assigner'):
                    fleet_out = self._project_fleet_action_milp(fleet_out)
            
            action_type = fleet_out.action_type          # [V]
            reposition_target = fleet_out.reposition_target  # [V]
            stats.forced_charge_count += int(getattr(fleet_out, 'forced_charge_count', 0))
            stats.forced_charge_total_idle += int(getattr(fleet_out, 'forced_charge_total_idle', 0))
            
            # Track action distribution for available vehicles
            action_names = ['serve', 'charge', 'reposition']
            available_mask = self.env.get_available_actions()
            is_available = available_mask.any(dim=1)
            action_flat = action_type.view(-1)
            valid_actions = action_flat[(action_flat >= 0) & (action_flat < 3) & is_available]
            if len(valid_actions) > 0:
                counts = torch.bincount(valid_actions, minlength=3)
                for i, name in enumerate(action_names):
                    stats.action_counts[name] += counts[i].item()
            
            step_count += 1

            # Pass MILP serve assignments if available, otherwise greedy matching
            milp_trips = getattr(fleet_out, 'milp_serve_trip_ids', None)
            next_state, reward, done_tensor, info = self.env.step(
                action_type, reposition_target,
                vehicle_charge_power=fleet_out.vehicle_charge_power,
                fleet_state_only=True,
                milp_serve_trip_ids=milp_trips,
            )
            done = done_tensor.item()

            # Estimate durations for this step
            durations = self._estimate_durations(
                action_type=action_type,
                prev_positions=prev_positions,
                new_positions=self.env.fleet_state.positions,
                prev_socs=prev_socs,
                new_socs=self.env.fleet_state.socs,
                reposition_targets=reposition_target,
                info=info
            )
            
            # Update duration stats
            self._update_duration_tracking(durations, action_type, total_durations, duration_counts)
            
            # Extract assignment info for logging
            serve_assignments = getattr(info, 'serve_assignments', None)
            charge_assignments = getattr(info, 'charge_assignments', None)
            
            reward_scalar = reward.item() if isinstance(reward, torch.Tensor) else reward
            
            # Store fleet-level transition
            if self.track_durations:
                self.enhanced_buffer.push_fleet(
                    state=state,
                    hex_allocations=fleet_out.allocation_probs,
                    hex_repos_targets=fleet_out.repos_sampled_targets,
                    hex_charge_power=fleet_out.charge_power,
                    vehicle_hex_ids=pre_step_vehicle_hex_ids,
                    reward=reward_scalar,
                    next_state=next_state,
                    done=done,
                    duration=durations,
                    action_types=action_type,
                    scenario=scenario,
                    serve_assignments=serve_assignments,
                    charge_assignments=charge_assignments,
                )
            else:
                self.replay_buffer.push_fleet(
                    state=state,
                    hex_allocations=fleet_out.allocation_probs,
                    hex_repos_targets=fleet_out.repos_sampled_targets,
                    hex_charge_power=fleet_out.charge_power,
                    vehicle_hex_ids=pre_step_vehicle_hex_ids,
                    reward=reward_scalar,
                    next_state=next_state,
                    done=done,
                    duration=durations,
                    serve_assignments=serve_assignments,
                    charge_assignments=charge_assignments,
                )
            
            state = next_state
            stats.total_reward += reward_scalar
            stats.steps += 1
            
        # Finalize stats
        stats.trips_served = info.trips_served
        stats.trips_dropped = info.trips_dropped
        stats.trips_loaded = info.trips_loaded
        stats.avg_soc = info.avg_soc
        stats.revenue = info.revenue
        stats.driving_cost = getattr(info, 'driving_cost', 0.0)
        stats.energy_cost = info.energy_cost
        stats.profit = stats.revenue - stats.driving_cost - stats.energy_cost
        stats.collection_time = time.time() - start_time
        
        # Duration stats
        for action_name in ['serve', 'reposition', 'charge']:
            if duration_counts[action_name] > 0:
                avg_duration = total_durations[action_name] / duration_counts[action_name]
                setattr(stats, f'avg_{action_name}_duration', avg_duration)
        
        # Compute duration variance
        if sum(duration_counts.values()) > 0:
            all_durations = sum(total_durations.values())
            all_counts = sum(duration_counts.values())
            stats.duration_variance = all_durations / all_counts  # Simplified
        
        self.total_episodes += 1
        self.total_steps += stats.steps
        
        return stats
    
    def get_enhanced_buffer(self) -> EnhancedReplayBuffer:
        """Get the enhanced replay buffer with duration info."""
        return self.enhanced_buffer
    
    def get_duration_stats(self) -> Dict[str, float]:
        """Get average durations by action type."""
        if hasattr(self, 'enhanced_buffer'):
            return self.enhanced_buffer.get_avg_durations()
        return {}


def create_enhanced_collector(
    env: GPUEnvironment,
    replay_buffer: GPUReplayBuffer,
    device: str = "cuda",
    use_milp: bool = False,
) -> EnhancedEpisodeCollector:
    """
    Factory function to create enhanced episode collector.
    
    Args:
        env: GPU environment
        replay_buffer: Base replay buffer
        device: Device to use
    Returns:
        EnhancedEpisodeCollector instance
    """
    return EnhancedEpisodeCollector(
        env=env,
        replay_buffer=replay_buffer,
        device=device,
        use_milp=use_milp,
        track_durations=True
    )
