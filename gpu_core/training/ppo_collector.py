import torch
from typing import Optional

from gpu_core.networks.ppo_agent import PPOAgent
from gpu_core.features.ppo_buffer import PPORolloutBuffer
from gpu_core.simulator.environment import GPUEnvironment
from gpu_core.training.episode_collector import EpisodeStats
from gpu_core.spatial.neighbors import HexNeighbors


class PPOCollector:
    """Episode Collector tailored for On-Policy PPO Training."""

    def __init__(
        self,
        env: GPUEnvironment,
        replay_buffer: PPORolloutBuffer,
        device: str = "cuda",
        use_smart_assignment: bool = True,
        repos_khop: int = 4,
        use_khop_candidates: bool = False,
    ):
        self.env = env
        self.buffer = replay_buffer
        self.device = device
        self.use_smart_assignment = use_smart_assignment
        self.episode_count = 0
        self.repos_khop = max(1, int(repos_khop))
        self.use_khop_candidates = bool(use_khop_candidates)
        self._repos_khop_mask: Optional[torch.Tensor] = None
        self._khop_neighbor_indices: Optional[torch.Tensor] = None
        self._khop_neighbor_mask: Optional[torch.Tensor] = None
        self._max_k_neighbors: int = 0

    def _build_per_vehicle_trip_mask(
        self,
        vehicle_hex_ids: torch.Tensor,
        max_trips: int,
    ) -> torch.Tensor:
        """Build [num_vehicles, max_trips] reachability mask for trip selection."""
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

        pickup_hexes = self.env.trip_state.pickup_hex[available_indices[:num_available]]

        if hasattr(self.env, 'hex_grid') and hasattr(self.env.hex_grid, 'distance_matrix') \
                and self.env.hex_grid.distance_matrix is not None:
            dist_matrix = self.env.hex_grid.distance_matrix._distances
            pickup_distance = dist_matrix[vehicle_hex_ids[:, None], pickup_hexes[None, :]]
            within = pickup_distance <= getattr(self.env, 'max_pickup_distance', 5.0)

            if hasattr(self.env, 'trip_state') and hasattr(self.env.trip_state, 'distance_km') \
                    and hasattr(self.env, 'energy_dynamics') and self.env.energy_dynamics is not None \
                    and hasattr(self.env, 'fleet_state') and hasattr(self.env.fleet_state, 'socs'):
                trip_distance = self.env.trip_state.distance_km[available_indices[:num_available]]
                total_distance = pickup_distance + trip_distance.unsqueeze(0)
                energy_needed = self.env.energy_dynamics.compute_consumption(total_distance)
                available_energy = (self.env.fleet_state.socs - self.env.energy_dynamics.min_soc_reserve).clamp(min=0.0)
                energy_ok = available_energy[:, None] >= energy_needed
                within = within & energy_ok
        else:
            within = torch.ones(num_vehicles, num_available, dtype=torch.bool, device=self.device)

        trip_mask[:, :num_available] = within

        return trip_mask

    def _ensure_reposition_khop_mask(self) -> Optional[torch.Tensor]:
        """Build and cache [H, H] k-hop reposition feasibility mask."""
        if self._repos_khop_mask is not None:
            return self._repos_khop_mask

        adjacency = getattr(self.env, '_adjacency_matrix', None)
        if adjacency is None and hasattr(self.env, 'adjacency_matrix'):
            adjacency = self.env.adjacency_matrix
        if adjacency is None:
            return None

        self._repos_khop_mask = HexNeighbors.compute_khop_mask(adjacency, k=self.repos_khop)
        return self._repos_khop_mask

    def _build_reposition_mask(self, vehicle_hex_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Build [num_vehicles, num_hexes] reposition mask from cached k-hop map."""
        khop_mask = self._ensure_reposition_khop_mask()
        if khop_mask is None:
            return None
        return khop_mask[vehicle_hex_ids]

    def _ensure_khop_neighbors(self) -> None:
        """Build and cache k-hop neighbor candidate indices/masks [H, K]."""
        if self._khop_neighbor_indices is not None and self._khop_neighbor_mask is not None:
            return

        adjacency = getattr(self.env, '_adjacency_matrix', None)
        if adjacency is None and hasattr(self.env, 'adjacency_matrix'):
            adjacency = self.env.adjacency_matrix
        if adjacency is None:
            raise RuntimeError('PPOCollector requires adjacency matrix for k-hop reposition candidates')

        khop_mask_hh = HexNeighbors.compute_khop_mask(adjacency, k=self.repos_khop)
        khop_indices, _, max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh)
        self._khop_neighbor_indices = khop_indices.to(self.device)
        self._khop_neighbor_mask = (khop_indices != -1).to(self.device)
        self._max_k_neighbors = int(max_k)

    def collect_episode(
        self,
        agent: PPOAgent,
        rollout_steps: int = 2048,
        exploration_noise: float = 0.0,
        seed: Optional[int] = None,
        deterministic: bool = False,
        **kwargs,
    ) -> EpisodeStats:
        """Collect a segment of experience defined by rollout_steps."""
        del exploration_noise, kwargs

        if seed is not None:
            torch.manual_seed(seed)
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)

        state = self.env.reset(seed=seed)

        total_reward = 0.0
        steps = 0
        done = False
        info = None
        action_counts = {"idle": 0, "serve": 0, "charge": 0, "reposition": 0}
        total_serve_attempted = 0
        total_serve_success = 0

        if self.use_khop_candidates:
            self._ensure_khop_neighbors()

        while steps < rollout_steps and not done:
            state_dict = self.env._build_state_dict(state)
            available_mask = self.env.get_available_actions()
            num_vehicles = available_mask.shape[0]

            vehicle_hex_ids = self.env.fleet_state.positions.long()
            reposition_mask = self._build_reposition_mask(vehicle_hex_ids)

            trip_mask = None
            trip_active_count = None
            if hasattr(self.env, 'trip_state'):
                unassigned_mask = self.env.trip_state.get_unassigned_mask()
                if unassigned_mask.any():
                    trip_active_count = min(int(unassigned_mask.sum().item()), agent._max_trips)
                trip_mask = self._build_per_vehicle_trip_mask(vehicle_hex_ids, agent._max_trips)

            # MAPPO-only SERVE feasibility gate: disable SERVE when no reachable trip.
            if hasattr(self.env, 'trip_state'):
                if trip_mask is None:
                    available_mask[:, 0] = False
                else:
                    available_mask[:, 0] = available_mask[:, 0] & trip_mask.any(dim=1)

            with torch.no_grad():
                ppo_out = agent.select_action(
                    state=state,
                    action_mask=available_mask,
                    reposition_mask=reposition_mask,
                    trip_mask=trip_mask,
                    deterministic=deterministic,
                    khop_neighbor_indices=self._khop_neighbor_indices if self.use_khop_candidates else None,
                    khop_neighbor_mask=self._khop_neighbor_mask if self.use_khop_candidates else None,
                    vehicle_hex_ids=vehicle_hex_ids,
                )

            action_type = ppo_out.action_type
            reposition_target = ppo_out.reposition_target
            selected_trip = ppo_out.selected_trip

            if action_type.dim() > 1:
                action_type = action_type.squeeze(0)
                reposition_target = reposition_target.squeeze(0)
                if selected_trip is not None:
                    selected_trip = selected_trip.squeeze(0)

            selected_trip_for_env = selected_trip if getattr(agent, 'use_trip_head', True) else None
            vehicle_charge_power_for_env = None
            if getattr(agent, 'learn_charge_power', False) and ppo_out.vehicle_charge_power is not None:
                vehicle_charge_power_for_env = ppo_out.vehicle_charge_power
                if vehicle_charge_power_for_env.dim() > 1:
                    vehicle_charge_power_for_env = vehicle_charge_power_for_env.squeeze(0)

            next_state, reward, done_tensor, info = self.env.step(
                action_type,
                reposition_target,
                selected_trip_for_env,
                vehicle_charge_power=vehicle_charge_power_for_env,
            )
            done = done_tensor.item()
            duration = getattr(info, 'duration', 1.0)
            total_serve_attempted += int(getattr(info, 'num_serve_attempted', 0))
            total_serve_success += int(getattr(info, 'num_serve_success', 0))

            available_1d = available_mask.any(dim=1)
            valid_actions = action_type[(action_type >= 0) & (action_type < 3) & available_1d]
            if valid_actions.numel() > 0:
                counts = torch.bincount(valid_actions, minlength=3)
                action_counts["serve"] += int(counts[0].item())
                action_counts["charge"] += int(counts[1].item())
                action_counts["reposition"] += int(counts[2].item())

            log_prob_total = ppo_out.action_log_prob + ppo_out.reposition_log_prob
            if ppo_out.trip_log_prob is not None:
                log_prob_total = log_prob_total + ppo_out.trip_log_prob
            if log_prob_total.dim() > 1:
                log_prob_total = log_prob_total.squeeze(0)

            action_tensor = torch.zeros((num_vehicles, 3), dtype=torch.long, device=self.device)
            action_tensor[:, 0] = action_type
            action_tensor[:, 1] = reposition_target
            if ppo_out.selected_trip is not None:
                selected_trip_val = ppo_out.selected_trip
                if selected_trip_val.dim() > 1:
                    selected_trip_val = selected_trip_val.squeeze(0)
                action_tensor[:, 2] = selected_trip_val

            executed_action_tensor = action_tensor.clone()

            if info.serve_assignments is not None:
                matched_veh_idx, _ = info.serve_assignments
                attempted_serve_mask = (action_type == 0)
                success_serve_mask = torch.zeros_like(attempted_serve_mask)
                success_serve_mask[matched_veh_idx] = True
                failed_serve_mask = attempted_serve_mask & (~success_serve_mask)
                executed_action_tensor[failed_serve_mask, 0] = -1
                executed_action_tensor[failed_serve_mask, 2] = -1

            if info.charge_assignments is not None:
                matched_veh_idx, _ = info.charge_assignments
                attempted_charge_mask = (action_type == 1)
                success_charge_mask = torch.zeros_like(attempted_charge_mask)
                success_charge_mask[matched_veh_idx] = True
                failed_charge_mask = attempted_charge_mask & (~success_charge_mask)
                executed_action_tensor[failed_charge_mask, 0] = -1

            if hasattr(info, 'per_vehicle_reward') and info.per_vehicle_reward is not None:
                actor_reward = info.per_vehicle_reward.clone().detach()
            else:
                scalar_reward = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                actor_reward = torch.full(
                    (num_vehicles,),
                    float(scalar_reward),
                    dtype=torch.float32,
                    device=self.device,
                )

            self.buffer.store(
                state=state_dict,
                action=action_tensor,
                executed_action=executed_action_tensor,
                reward=actor_reward,
                duration=duration,
                done=done,
                value=ppo_out.value.clone().detach(),
                log_prob=log_prob_total.clone().detach(),
                action_mask=available_mask,
                reposition_mask=reposition_mask,
                trip_mask=trip_mask,
                trip_active_count=trip_active_count,
                vehicle_hex_ids=vehicle_hex_ids,
            )

            state = next_state
            total_reward += reward.item() if isinstance(reward, torch.Tensor) else reward
            steps += 1

        self.episode_count += 1

        next_value = 0.0
        if not done:
            with torch.no_grad():
                next_value = agent.get_value(state).detach()

        return EpisodeStats(
            episode_id=self.episode_count,
            total_reward=total_reward,
            steps=steps,
            trips_served=getattr(info, 'trips_served', getattr(self.env.fleet_state, 'trips_completed', 0)),
            trips_dropped=getattr(info, 'trips_dropped', getattr(self.env, 'total_trips', 0)),
            trips_loaded=getattr(info, 'trips_loaded', 1),
            avg_soc=self.env.fleet_state.socs.mean().item() if hasattr(self.env.fleet_state, 'socs') else 0.0,
            revenue=getattr(info, 'revenue', 0.0),
            energy_cost=getattr(info, 'energy_cost', 0.0),
            profit=getattr(info, 'revenue', 0.0) - getattr(info, 'energy_cost', 0.0) - getattr(info, 'driving_cost', 0.0),
            action_counts=action_counts,
            num_serve_attempted=total_serve_attempted,
            num_serve_success=total_serve_success,
            next_value=next_value,
        )
