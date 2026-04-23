"""
Episode collector for GPU-native RL training.

Collects experience from environment using the agent.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time

from ..simulator.environment import GPUEnvironment, BatchedGPUEnvironment, EnvInfo
from ..features.replay_buffer import GPUReplayBuffer
from ..features.builder import FeatureBuilder
from ..networks.sac import FleetSACAgent, FleetSACOutput
try:
    from .milp_assignment import MILPAssignment
except ImportError:
    MILPAssignment = None


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode_id: int = 0
    total_reward: float = 0.0
    steps: int = 0
    trips_served: int = 0
    trips_dropped: int = 0
    trips_loaded: int = 0  # Total trips that appeared this episode
    avg_soc: float = 0.0
    revenue: float = 0.0
    driving_cost: float = 0.0
    energy_cost: float = 0.0
    profit: float = 0.0
    collection_time: float = 0.0
    # Action distribution tracking
    action_counts: Dict[str, int] = field(default_factory=lambda: {'serve': 0, 'charge': 0, 'reposition': 0})
    forced_charge_count: int = 0
    forced_charge_total_idle: int = 0
    num_serve_attempted: int = 0
    num_serve_success: int = 0
    next_value: float = 0.0  # Bootstrap V(s_{T+1}) for PPO GAE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'total_reward': self.total_reward,
            'steps': self.steps,
            'trips_served': self.trips_served,
            'trips_dropped': self.trips_dropped,
            'trips_loaded': self.trips_loaded,
            'avg_soc': self.avg_soc,
            'revenue': self.revenue,
            'driving_cost': self.driving_cost,
            'energy_cost': self.energy_cost,
            'profit': self.profit,
            'collection_time': self.collection_time
        }


@dataclass
class CollectionMetrics:
    """Metrics for experience collection."""
    episodes_collected: int = 0
    total_steps: int = 0
    total_time: float = 0.0
    steps_per_second: float = 0.0
    avg_episode_reward: float = 0.0
    avg_episode_length: float = 0.0
    episode_stats: List[EpisodeStats] = field(default_factory=list)


class EpisodeCollector:
    """
    Collects episodes from environment using agent policy.
    
    Features:
    - Batched collection for efficiency
    - GPU-native experience storage
    - Integration with replay buffer
    - Optional MILP-based action projection
    """
    
    def __init__(
        self,
        env: GPUEnvironment,
        replay_buffer: GPUReplayBuffer,
        device: str = "cuda",
        use_milp: bool = False              # Enable MILP assignment
    ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.device = torch.device(device)
        self.use_milp = use_milp
        
        self.total_episodes = 0
        self.total_steps = 0
        
        # Get dimensions for state conversion
        self._vehicle_feature_dim = env._vehicle_feature_dim
        self._hex_feature_dim = env._hex_feature_dim
        self._context_dim = env._context_dim
        self._num_vehicles = env.num_vehicles
        self._num_hexes = env.num_hexes
        
        # Initialize optional MILP assignment module
        if use_milp and MILPAssignment is not None:
            _cfg      = getattr(env, 'config', None)
            _step_min = getattr(getattr(_cfg, 'episode',     None), 'step_duration_minutes', 5.0)  if _cfg else 5.0
            _num_sta  = getattr(getattr(_cfg, 'environment', None), 'num_stations',           150) if _cfg else 50
            _delta_t  = _step_min / 60.0  # convert minutes → hours
            _max_pkup = getattr(env, 'max_pickup_distance', 5.0)
            # Extract real station positions and config from environment
            _sta_hex = None
            _num_ports = 10
            _p_max_s = 50.0
            _e_max = 100.0
            _eta_drv = 0.20
            if hasattr(env, 'station_state') and env.station_state is not None:
                _sta_hex = env.station_state.hex_ids.cpu().numpy()
                _num_sta = len(_sta_hex)
                if hasattr(env.station_state, 'ports'):
                    _num_ports = int(env.station_state.ports[0].item()) if env.station_state.ports.numel() > 0 else 10
            if _cfg:
                _phys = getattr(_cfg, 'physics', None)
                if _phys:
                    _p_max_s = getattr(_phys, 'charge_power_kw', 50.0)
                    _e_max = getattr(_phys, 'max_soc', 100.0)
                    _eta_drv = getattr(_phys, 'energy_per_km', 0.20)
            self._assigner = MILPAssignment(
                num_vehicles=env.num_vehicles,
                num_hexes=env.num_hexes,
                num_stations=_num_sta,
                device=device,
                delta_t=_delta_t,
                mu=4.0,
                max_pickup_distance=_max_pkup,
                station_positions=_sta_hex,
                port_capacity=_num_ports,
                p_max_s=_p_max_s,
                e_max_kwh=_e_max,
                eta_drv=_eta_drv,
                lambda_power=0.02,
            )
            env.set_feeder_power_limit(float(self._assigner.p_max_feed))
            print(f"[Assignment] MILP assigner active (V={env.num_vehicles}, S={_num_sta}, delta_t={_delta_t:.4f}h, mu=4.0, max_pkup={_max_pkup:.1f}km)")
        elif use_milp and MILPAssignment is None:
            print("[Assignment] WARNING: --milp requested but gurobipy import failed. Using raw policy output.")
        
        # Initialize feature builder for EnhancedActor
        self._feature_builder = FeatureBuilder(
            hex_grid=env.hex_grid,
            num_vehicles=env.num_vehicles,
            device=device
        )
    
    def _get_reposition_mask(self, vehicle_hex_ids: torch.Tensor) -> torch.Tensor:
        """Create reposition mask: restrict targets to local neighborhood (paper Eq 3).
        
        Args:
            vehicle_hex_ids: [num_vehicles] current positions
            
        Returns:
            reposition_mask: [num_vehicles, num_hexes] True for valid hexes
        """
        num_vehicles = vehicle_hex_ids.shape[0]
        
        # Try to use distance matrix if available to restrict reposition targets
        # The paper restricts to N(h_{i,t}), which we approximate as hexes within max_pickup_distance
        if hasattr(self.env, 'hex_grid') and hasattr(self.env.hex_grid, 'distance_matrix') and self.env.hex_grid.distance_matrix is not None:
            try:
                # Use same distance threshold as max_pickup_distance (typically 5.0 km)
                threshold = getattr(self.env, 'max_pickup_distance', 5.0)
                
                # Most memory efficient approach: Create target tensor and use get_distances_batch
                # This avoids slicing a 1000x3985 float tensor (~15MB per step per environment)
                all_hexes = torch.arange(self._num_hexes, device=self.device).unsqueeze(0).expand(num_vehicles, -1)
                veh_hexes_expanded = vehicle_hex_ids.unsqueeze(1).expand(-1, self._num_hexes)
                
                # Memory efficient batch distance calculation
                veh_distances = self.env.hex_grid.distance_matrix.get_distances_batch(
                    veh_hexes_expanded.reshape(-1), 
                    all_hexes.reshape(-1)
                ).view(num_vehicles, self._num_hexes)
                
                reposition_mask = veh_distances <= threshold
                
                # Prevent repositioning to the exact same hex (force CHARGE instead since IDLE removed)
                # When repos target == current position, use CHARGE as placeholder
                reposition_mask.scatter_(1, vehicle_hex_ids.unsqueeze(1), False)
                return reposition_mask
                
            except AttributeError:
                # Fallback if get_distances_batch is missing, though less memory efficient
                distance_matrix = self.env.hex_grid.distance_matrix._distances
                if distance_matrix is not None:
                    veh_distances = distance_matrix[vehicle_hex_ids]
                    threshold = getattr(self.env, 'max_pickup_distance', 5.0)
                    reposition_mask = veh_distances <= threshold
                    reposition_mask.scatter_(1, vehicle_hex_ids.unsqueeze(1), False)
                    return reposition_mask
                
        # Fallback if no distance matrix (should not happen in real training run)
        reposition_mask = torch.ones(num_vehicles, self._num_hexes, 
                                     dtype=torch.bool, device=self.device)
        return reposition_mask
    
    def _build_per_vehicle_trip_mask(
        self,
        vehicle_hex_ids: torch.Tensor,  # [num_vehicles]
        max_trips: int
    ) -> torch.Tensor:
        """Build a [num_vehicles, max_trips] reachability mask.
        
        Entry [v, t] is True iff trip t's pickup hex is within max_pickup_distance
        of vehicle v's current position.  This lets the GCN trip_selection_head only
        sample reachable trips, eliminating ServeFail from random out-of-range choices.
        
        Vehicles that have NO trip within range get all available trips unmasked as a
        fallback (they should choose CHARGE/REPOSITION via the action mask instead).
        """
        num_vehicles = vehicle_hex_ids.shape[0]
        trip_mask = torch.zeros(num_vehicles, max_trips, dtype=torch.bool, device=self.device)
        
        if not hasattr(self.env, 'trip_state'):
            return trip_mask
        
        unassigned_mask = self.env.trip_state.get_unassigned_mask()
        if not unassigned_mask.any():
            return trip_mask
        
        available_indices = unassigned_mask.nonzero(as_tuple=True)[0]
        num_available = min(len(available_indices), max_trips)
        
        if num_available == 0:
            return trip_mask
        
        # Pickup hexes for available trips: [num_available]
        pickup_hexes = self.env.trip_state.pickup_hex[available_indices[:num_available]]
        
        if hasattr(self.env, 'hex_grid') and hasattr(self.env.hex_grid, 'distance_matrix') \
                and self.env.hex_grid.distance_matrix is not None:
            dist_matrix = self.env.hex_grid.distance_matrix._distances
            # Direct [V, T] gather via broadcasting — no intermediate [V, H] tensor.
            # Cost: V×T = 1000×500 = 500K lookups (6× cheaper than the existing
            # reposition mask which already does V×H = 1000×3000 = 3M lookups/step).
            within = dist_matrix[vehicle_hex_ids[:, None],     # [V, 1] → broadcast
                                  pickup_hexes[None, :]]       # [1, T] → broadcast
            within = within <= getattr(self.env, 'max_pickup_distance', 5.0)  # [V, T] bool
        else:
            # No distance matrix: allow all available trips
            within = torch.ones(num_vehicles, num_available, dtype=torch.bool, device=self.device)
        
        trip_mask[:, :num_available] = within
        
        # Fallback: vehicles with zero reachable trips get all available trips unmasked
        # (action_mask should prevent them from choosing SERVE anyway; this avoids all-inf softmax)
        no_reachable = ~trip_mask.any(dim=1)  # [num_vehicles]
        if no_reachable.any():
            trip_mask[no_reachable, :num_available] = True
        
        return trip_mask


    def _flat_to_dict_state(self, flat_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert flat state tensor to dict format for replay buffer."""
        vehicle_size = self._num_vehicles * self._vehicle_feature_dim
        hex_size = self._num_hexes * self._hex_feature_dim
        
        vehicle_features = flat_state[:vehicle_size].view(self._num_vehicles, self._vehicle_feature_dim)
        hex_features = flat_state[vehicle_size:vehicle_size + hex_size].view(self._num_hexes, self._hex_feature_dim)
        context_features = flat_state[vehicle_size + hex_size:]
        
        return {
            'vehicle': vehicle_features,
            'hex': hex_features,
            'context': context_features
        }
    
    def collect_episode(
        self,
        agent: Any,
        exploration_noise: float = 0.1,
        seed: Optional[int] = None,
        render: bool = False,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> EpisodeStats:
        """
        Collect a single episode of experience.
        
        Args:
            agent: RL agent with select_action method
            exploration_noise: Noise for exploration
            seed: Random seed for episode
            render: Whether to render environment
            temperature: Temperature for annealed softmax (paper Eq. 16)
            deterministic: If True, use argmax (greedy) action selection for evaluation
            
        Returns:
            Episode statistics
        """
        start_time = time.time()
        
        state = self.env.reset(seed=seed)
        
        stats = EpisodeStats(episode_id=self.total_episodes)
        done = False
        step_count = 0
        
        while not done:
            # Temperature annealing within episode (optional - can also be done across episodes)
            # Start warmer, cool down as episode progresses
            step_temperature = max(0.1, temperature * (1.0 - 0.3 * step_count / self.env.episode_steps))
            
            with torch.no_grad():
                # Use MILP action projection if enabled
                if self.use_milp and hasattr(self, '_assigner'):
                    action_type, reposition_target, selected_trip = self._select_action_milp(
                        agent, state, temperature=step_temperature, deterministic=deterministic
                    )
                else:
                    action_type, reposition_target, selected_trip = self._select_action(
                        agent, state, exploration_noise, temperature=step_temperature, deterministic=deterministic
                    )
            
            # Track action distribution (only for AVAILABLE vehicles)
            # BUSY vehicles have action=-1 from _select_action
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
            
            # Use step_with_preferences if agent has EnhancedActor (serve_scores, charge_scores)
            if hasattr(agent, 'use_enhanced_actor') and agent.use_enhanced_actor:
                # Build trip and station features for actor
                with torch.no_grad():
                    # Build features from environment state
                    actor_module = agent.actor.module if hasattr(agent.actor, 'module') else agent.actor
                    trip_features, trip_mask = self._feature_builder.build_trip_features(
                        self.env.trip_state,
                        self.env.fleet_state,
                        max_trips=actor_module.max_trips  # From Actor config (use parameter!)
                    )
                    station_features, station_mask = self._feature_builder.build_station_features(
                        self.env.station_state,
                        self.env.fleet_state,
                        current_step=step_count
                    )
                    
                    # Add batch dimension
                    trip_features = trip_features.unsqueeze(0)  # [1, max_trips, 8]
                    trip_mask = trip_mask.unsqueeze(0)  # [1, max_trips]
                    station_features = station_features.unsqueeze(0)  # [1, num_stations, 6]
                    station_mask = station_mask.unsqueeze(0)  # [1, num_stations]
                    
                    # Get serve/charge scores from actor with features
                    state_input = state.unsqueeze(0) if state.dim() == 1 else state
                    actor_output = agent.actor(
                        state_input,
                        trip_features=trip_features,
                        trip_mask=trip_mask,
                        station_features=station_features,
                        station_mask=station_mask
                    )
                    serve_scores = actor_output.serve_scores if actor_output.serve_scores is not None else None
                    charge_scores = actor_output.charge_scores if actor_output.charge_scores is not None else None
                
                next_state, reward, done_tensor, info = self.env.step_with_preferences(
                    action_type, reposition_target, serve_scores, charge_scores
                )
            else:
                next_state, reward, done_tensor, info = self.env.step(
                    action_type, reposition_target, selected_trip  # NEW: pass selected_trip
                )
            done = done_tensor.item()
            
            action = self._encode_action(action_type, reposition_target)
            
            # Convert flat states to dict format for replay buffer
            state_dict = self._flat_to_dict_state(state)
            next_state_dict = self._flat_to_dict_state(next_state)
            
            # Extract assignment info from EnvInfo
            serve_assignments = getattr(info, 'serve_assignments', None)
            charge_assignments = getattr(info, 'charge_assignments', None)
            
            # Store in replay buffer with assignment info
            reward_scalar = reward.item() if isinstance(reward, torch.Tensor) else reward
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
        
        stats.trips_served = info.trips_served
        stats.trips_dropped = info.trips_dropped
        stats.trips_loaded = info.trips_loaded
        stats.avg_soc = info.avg_soc
        stats.revenue = info.revenue
        stats.driving_cost = getattr(info, 'driving_cost', 0.0)
        stats.energy_cost = info.energy_cost
        stats.profit = stats.revenue - stats.driving_cost - stats.energy_cost
        stats.collection_time = time.time() - start_time
        
        self.total_episodes += 1
        self.total_steps += stats.steps
        
        return stats
    
    def collect_episodes(
        self,
        agent: Any,
        num_episodes: int,
        exploration_noise: float = 0.1,
        progress_callback: Optional[callable] = None
    ) -> CollectionMetrics:
        """
        Collect multiple episodes.
        
        Args:
            agent: RL agent
            num_episodes: Number of episodes to collect
            exploration_noise: Exploration noise
            progress_callback: Optional callback for progress updates
            
        Returns:
            Collection metrics
        """
        metrics = CollectionMetrics()
        start_time = time.time()
        
        for i in range(num_episodes):
            episode_stats = self.collect_episode(
                agent,
                exploration_noise=exploration_noise,
                seed=self.total_episodes
            )
            
            metrics.episode_stats.append(episode_stats)
            metrics.total_steps += episode_stats.steps
            
            if progress_callback:
                progress_callback(i + 1, num_episodes, episode_stats)
        
        metrics.episodes_collected = num_episodes
        metrics.total_time = time.time() - start_time
        metrics.steps_per_second = metrics.total_steps / max(metrics.total_time, 1e-6)
        
        if num_episodes > 0:
            metrics.avg_episode_reward = sum(
                s.total_reward for s in metrics.episode_stats
            ) / num_episodes
            metrics.avg_episode_length = sum(
                s.steps for s in metrics.episode_stats
            ) / num_episodes
        
        return metrics
    
    def _select_action(
        self,
        agent: Any,
        state: torch.Tensor,
        exploration_noise: float,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Select action using GCN actor with per-vehicle decisions.
        
        Returns:
            action_type, reposition_target, selected_trip
        """
        available_mask = self.env.get_available_actions()  # [num_vehicles, action_dim]
        
        # Always use GCN actor (EnhancedActor removed)
        return self._select_action_gcn(agent, state, available_mask, temperature, deterministic=deterministic)
    
    def _select_action_gcn(
        self,
        agent: Any,
        state: torch.Tensor,
        available_mask: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # NOW returns 3 values
        """Select action using GCN actor (per-vehicle decisions).
        
        Returns:
            action_type, reposition_target, selected_trip
        """
        with torch.no_grad():
            # Get vehicle positions (hex IDs)
            vehicle_hex_ids = self.env.fleet_state.positions.long()
            
            # Get reposition mask (vehicles can't reposition to hex they're already in)
            reposition_mask = self._get_reposition_mask(vehicle_hex_ids)
            
            # Per-vehicle trip reachability mask: [num_vehicles, max_trips]
            # Only trips whose pickup hex is within max_pickup_distance are unmasked.
            # This prevents the GCN from selecting unreachable trips, eliminating ServeFail.
            trip_mask = None
            if hasattr(self.env, 'trip_state'):
                actor_module = agent.actor.module if hasattr(agent.actor, 'module') else agent.actor
                max_trips = actor_module.max_trips
                trip_mask = self._build_per_vehicle_trip_mask(vehicle_hex_ids, max_trips)
            
            # Use agent's select_action_gcn method
            output = agent.select_action_gcn(
                state=state,
                action_mask=available_mask,
                reposition_mask=reposition_mask,
                trip_mask=trip_mask,  # NEW: tell GCN which trips are valid
                vehicle_hex_ids=vehicle_hex_ids,
                temperature=temperature,
                deterministic=deterministic
            )
            
            action_type = output.action_type  # [num_vehicles]
            reposition_target = output.reposition_target  # [num_vehicles]
            selected_trip = output.selected_trip  # NEW: [num_vehicles] trip selections
            
            # Squeeze batch dimension if present
            if action_type.dim() > 1:
                action_type = action_type.squeeze(0)
            if reposition_target.dim() > 1:
                reposition_target = reposition_target.squeeze(0)
            if selected_trip is not None and selected_trip.dim() > 1:
                selected_trip = selected_trip.squeeze(0)
            
            # SOC constraints per paper Section 2.2 (Eq. 3)
            socs = self.env.fleet_state.socs
            
            # SOC guard: vehicles with SOC < 20% must CHARGE
            low_soc_mask = socs < 20.0
            action_type = torch.where(
                low_soc_mask, 
                torch.tensor(1, device=self.device),  # CHARGE=1 (was 2, indices shifted)
                action_type
            )
            
            # High SOC vehicles (>90%) should not CHARGE (index 1)
            high_soc_mask = socs > 90.0
            charging_high_soc = (action_type == 1) & high_soc_mask  # CHARGE=1
            serve_available = available_mask[:, 0]  # SERVE=0
            redirect_action = torch.where(
                serve_available,
                torch.tensor(0, device=self.device),  # SERVE=0
                torch.tensor(2, device=self.device)   # REPOSITION=2 (IDLE removed)
            )
            action_type = torch.where(charging_high_soc, redirect_action, action_type)
            
        return action_type, reposition_target, selected_trip
    
    def _select_action_fleet(
        self,
        agent: FleetSACAgent,
        state: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> FleetSACOutput:
        """Select action using the fleet actor from live environment state."""
        del state
        with torch.no_grad():
            policy_inputs = self.env.get_fleet_policy_inputs()
            # Training/eval collection should follow the original policy behavior
            # (low-SOC forced charge only). Disable long-idle override here.
            idle_steps = policy_inputs.get('idle_steps', None)
            zero_idle_steps = torch.zeros_like(idle_steps) if idle_steps is not None else None
            select_kwargs = dict(
                hex_features=policy_inputs['hex_features'],
                hex_vehicle_summary=policy_inputs['hex_vehicle_summary'],
                context_features=policy_inputs['context_features'],
                vehicle_hex_ids=policy_inputs['vehicle_hex_ids'],
                vehicle_socs=policy_inputs['vehicle_socs'],
                vehicle_status=policy_inputs['vehicle_status'],
                temperature=temperature,
                deterministic=deterministic,
            )
            if zero_idle_steps is not None:
                try:
                    output = agent.select_action_fleet(idle_steps=zero_idle_steps, **select_kwargs)
                except TypeError as exc:
                    if "idle_steps" not in str(exc):
                        raise
                    output = agent.select_action_fleet(**select_kwargs)
            else:
                output = agent.select_action_fleet(**select_kwargs)

        return output

    def _project_fleet_action_milp(
        self,
        fleet_out: FleetSACOutput,
    ) -> FleetSACOutput:
        """Project current fleet actor output through the MILP assigner.

        The fleet actor remains the source of the per-hex SAC/WDRO outputs stored in
        replay. MILP only replaces the executable per-vehicle actions used by the
        simulator.
        """
        if not (self.use_milp and hasattr(self, '_assigner')):
            return fleet_out

        vehicle_positions = self.env.fleet_state.positions.long()
        vehicle_socs = self.env.fleet_state.socs
        vehicle_status = self.env.fleet_state.status
        available_mask = self.env.fleet_state.get_available_mask(self.env.current_step)

        unassigned = self.env.trip_state.get_unassigned_mask()
        if unassigned.any():
            trip_indices = unassigned.nonzero(as_tuple=True)[0]
            trip_pickups = self.env.trip_state.pickup_hex[trip_indices]
            trip_dropoffs = self.env.trip_state.dropoff_hex[trip_indices]
            trip_fares = self.env.trip_state.fare[trip_indices]
        else:
            trip_pickups = torch.empty(0, dtype=torch.long, device=self.device)
            trip_dropoffs = torch.empty(0, dtype=torch.long, device=self.device)
            trip_fares = torch.empty(0, dtype=torch.float32, device=self.device)

        distance_matrix = self.env.hex_grid.distance_matrix._distances

        if hasattr(self._assigner, 'max_pickup_distance'):
            self._assigner.max_pickup_distance = getattr(self.env, 'max_pickup_distance', 5.0)

        result = self._assigner.assign_from_fleet(
            vehicle_positions=vehicle_positions,
            vehicle_socs=vehicle_socs,
            vehicle_status=vehicle_status,
            trip_pickups=trip_pickups,
            trip_dropoffs=trip_dropoffs,
            trip_fares=trip_fares,
            distance_matrix=distance_matrix,
            allocation_probs=fleet_out.allocation_probs,
            repos_sampled_targets=fleet_out.repos_sampled_targets,
            charge_power=fleet_out.charge_power,
            current_vehicle_charge_power=self.env.fleet_state.charge_power,
            available_mask=available_mask,
            current_step=self.env.current_step,
            episode_steps=self.env.episode_steps,
        )

        # Convert MILP serve_targets (indices into the unassigned trip pool)
        # to actual trip-state IDs so the environment can use them directly.
        V_full = vehicle_positions.shape[0]
        milp_serve_trip_ids = torch.full((V_full,), -1, dtype=torch.long, device=vehicle_positions.device)
        serve_mask = result.action_types == 0
        if serve_mask.any() and unassigned.any():
            trip_indices = unassigned.nonzero(as_tuple=True)[0]
            serve_vehicle_indices = serve_mask.nonzero(as_tuple=True)[0]
            milp_trip_pool_indices = result.serve_targets[serve_mask]
            # Clamp to valid range (safety guard)
            valid = (milp_trip_pool_indices >= 0) & (milp_trip_pool_indices < len(trip_indices))
            if valid.any():
                milp_serve_trip_ids[serve_vehicle_indices[valid]] = trip_indices[milp_trip_pool_indices[valid]]

        return FleetSACOutput(
            action_type=result.action_types,
            reposition_target=result.reposition_targets,
            vehicle_charge_power=(
                result.vehicle_charge_power
                if result.vehicle_charge_power is not None
                else torch.zeros_like(fleet_out.vehicle_charge_power)
            ),
            allocation_probs=fleet_out.allocation_probs,
            allocation_log_probs=fleet_out.allocation_log_probs,
            repos_sampled_targets=fleet_out.repos_sampled_targets,
            charge_power=fleet_out.charge_power,
            charge_power_log_prob=fleet_out.charge_power_log_prob,
            allocation_entropy=fleet_out.allocation_entropy,
            active_hex_mask=fleet_out.active_hex_mask,
            hex_embeddings=fleet_out.hex_embeddings,
            forced_charge_count=fleet_out.forced_charge_count,
            forced_charge_total_idle=fleet_out.forced_charge_total_idle,
            milp_serve_trip_ids=milp_serve_trip_ids,
        )

    def collect_episode_fleet(
        self,
        agent: FleetSACAgent,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> EpisodeStats:
        """Collect a single episode using fleet-level actor.

        Uses push_fleet() to store hex-level actions in the replay buffer.
        """
        start_time = time.time()
        state = self.env.reset(seed=seed, fleet_state_only=True)
        stats = EpisodeStats(episode_id=self.total_episodes)
        done = False
        step_count = 0

        while not done:
            step_temperature = max(
                0.1, temperature * (1.0 - 0.3 * step_count / self.env.episode_steps)
            )

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
            pre_step_vehicle_hex_ids = self.env.fleet_state.positions.long().clone()
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

            # Extract assignment info for logging
            serve_assignments = getattr(info, 'serve_assignments', None)
            charge_assignments = getattr(info, 'charge_assignments', None)

            # Duration (for semi-MDP)
            duration = getattr(info, 'duration', None)

            reward_scalar = reward.item() if isinstance(reward, torch.Tensor) else reward

            # Store fleet-level transition
            self.replay_buffer.push_fleet(
                state=state,
                hex_allocations=fleet_out.allocation_probs,
                hex_repos_targets=fleet_out.repos_sampled_targets,
                hex_charge_power=fleet_out.charge_power,
                vehicle_hex_ids=pre_step_vehicle_hex_ids,
                reward=reward_scalar,
                next_state=next_state,
                done=done,
                serve_assignments=serve_assignments,
                charge_assignments=charge_assignments,
                duration=duration,
            )

            state = next_state
            stats.total_reward += reward_scalar
            stats.steps += 1

        stats.trips_served = info.trips_served
        stats.trips_dropped = info.trips_dropped
        stats.trips_loaded = info.trips_loaded
        stats.avg_soc = info.avg_soc
        stats.revenue = info.revenue
        stats.driving_cost = getattr(info, 'driving_cost', 0.0)
        stats.energy_cost = info.energy_cost
        stats.profit = stats.revenue - stats.driving_cost - stats.energy_cost
        stats.collection_time = time.time() - start_time

        self.total_episodes += 1
        self.total_steps += stats.steps

        return stats

    def _encode_action(
        self,
        action_type: torch.Tensor,
        reposition_target: torch.Tensor
    ) -> torch.Tensor:
        """Encode action for replay buffer storage."""
        # Replay buffer expects [num_vehicles, action_dim] with long dtype
        action = torch.zeros(self._num_vehicles, 2, dtype=torch.long, device=self.device)
        action[:, 0] = action_type.long()
        action[:, 1] = reposition_target.long()
        return action

    def _select_action_milp(
        self,
        agent: Any,
        state: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action using the MILP assignment module."""
        # Get raw policy probabilities from the GCN actor
        with torch.no_grad():
            available_mask = self.env.get_available_actions()
            vehicle_hex_ids = self.env.fleet_state.positions.long()
            reposition_mask = self._get_reposition_mask(vehicle_hex_ids)
            
            # Per-vehicle reachability mask (same helper as _select_action_gcn)
            trip_mask = None
            if hasattr(self.env, 'trip_state'):
                actor_module = agent.actor.module if hasattr(agent.actor, 'module') else agent.actor
                max_trips = actor_module.max_trips
                trip_mask = self._build_per_vehicle_trip_mask(vehicle_hex_ids, max_trips)
            
            output = agent.select_action_gcn(
                state=state,
                action_mask=available_mask,
                reposition_mask=reposition_mask,
                trip_mask=trip_mask,
                vehicle_hex_ids=vehicle_hex_ids,
                temperature=temperature,
                deterministic=deterministic
            )
            
            raw_action_probs = output.action_probs  # Assume we can extract raw action logits/probs
            if raw_action_probs is not None and raw_action_probs.dim() > 2:
                raw_action_probs = raw_action_probs.squeeze(0)
            if not hasattr(output, 'action_probs') or raw_action_probs is None:
                # If the forward pass does not return raw probs, re-run forward pass to get them
                fwd_out = agent.actor(
                     state.unsqueeze(0) if state.dim() == 1 else state,
                     trip_mask=trip_mask.unsqueeze(0) if trip_mask is not None else None
                )
                raw_action_probs = fwd_out.action_probs.squeeze(0) if fwd_out.action_probs.dim() > 2 else fwd_out.action_probs

        # Extract environment states needed for MILP assignment
        vehicle_positions = self.env.fleet_state.positions
        vehicle_socs = self.env.fleet_state.socs
        vehicle_status = self.env.fleet_state.status
        V_full = len(vehicle_positions)

        # ── Only run MILP for AVAILABLE vehicles ─────────────────────────────
        # Busy vehicles would steal trip assignments in trip_once_j constraints
        # yet their actions are discarded by available_mask in env.step().
        avail_mask_1d = self.env.fleet_state.get_available_mask(self.env.current_step)
        avail_idx = avail_mask_1d.nonzero(as_tuple=True)[0]  # [V_avail]

        unassigned = self.env.trip_state.get_unassigned_mask()
        if unassigned.any():
            trip_indices = unassigned.nonzero(as_tuple=True)[0]
            trip_pickups = self.env.trip_state.pickup_hex[trip_indices]
            trip_dropoffs = self.env.trip_state.dropoff_hex[trip_indices]
            trip_fares = self.env.trip_state.fare[trip_indices]
        else:
            trip_pickups = torch.tensor([], device=self.device)
            trip_dropoffs = torch.tensor([], device=self.device)
            trip_fares = torch.tensor([], device=self.device)

        distance_matrix = self.env.hex_grid.distance_matrix._distances

        # Default full-fleet output (busy vehicles get repos to pos[0], serve=-1)
        full_action_types    = torch.zeros(V_full, dtype=torch.long, device=self.device)
        full_serve_targets   = torch.full((V_full,), -1, dtype=torch.long, device=self.device)
        full_charge_targets  = torch.full((V_full,), -1, dtype=torch.long, device=self.device)
        full_repos_targets   = vehicle_positions.clone().long()

        if len(avail_idx) > 0:
            probs_avail = raw_action_probs[avail_idx] if raw_action_probs is not None else None

            # Keep MILP pickup radius in sync with curriculum schedule
            if hasattr(self._assigner, 'max_pickup_distance'):
                self._assigner.max_pickup_distance = getattr(self.env, 'max_pickup_distance', 5.0)

            # Run MILP assignment
            result = self._assigner.assign(
                vehicle_positions=vehicle_positions[avail_idx],
                vehicle_socs=vehicle_socs[avail_idx],
                vehicle_status=vehicle_status[avail_idx],
                trip_pickups=trip_pickups,
                trip_dropoffs=trip_dropoffs,
                trip_fares=trip_fares,
                distance_matrix=distance_matrix,
                policy_probs=probs_avail,
                current_step=self.env.current_step,
                episode_steps=self.env.episode_steps
            )

            full_action_types[avail_idx]   = result.action_types
            full_serve_targets[avail_idx]  = result.serve_targets
            full_charge_targets[avail_idx] = result.charge_targets
            full_repos_targets[avail_idx]  = result.reposition_targets

            # Keep learned reposition targets for vehicles MILP routes to REPOSITION.
            gcn_repos = getattr(output, 'reposition_target', None)
            if gcn_repos is not None:
                if gcn_repos.dim() > 1:
                    gcn_repos = gcn_repos.squeeze(0)
                repos_local_mask = result.action_types == 2  # REPOS=2 in new scheme
                if repos_local_mask.any():
                    repos_global_idx = avail_idx[repos_local_mask]
                    full_repos_targets[repos_global_idx] = gcn_repos[repos_global_idx]

        return full_action_types, full_repos_targets, full_serve_targets
