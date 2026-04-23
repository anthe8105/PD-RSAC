"""
maddpg_collector.py — Episode Collector for MADDPG (NIPS 2017).

Collects off-policy transitions and pushes them to MADDPGReplayBuffer.
"""

import torch
from typing import Optional, Dict

from gpu_core.networks.maddpg_agent import MADDPGAgent
from gpu_core.features.maddpg_buffer import MADDPGReplayBuffer
from gpu_core.training.episode_collector import EpisodeStats
from gpu_core.spatial.neighbors import HexNeighbors


class MADDPGCollector:
    """Off-policy episode collector for MADDPG."""

    def __init__(
        self,
        env,
        replay_buffer: MADDPGReplayBuffer,
        device: str = 'cuda',
        repos_khop: int = 4,
    ):
        self.env = env
        self.buffer = replay_buffer
        self.device = device
        self.episode_count = 0
        self.repos_khop = max(1, int(repos_khop))
        self._khop_neighbor_indices: Optional[torch.Tensor] = None
        self._khop_neighbor_mask: Optional[torch.Tensor] = None
        self._max_k_neighbors: int = 0

    def _ensure_khop_neighbors(self) -> None:
        if self._khop_neighbor_indices is not None and self._khop_neighbor_mask is not None:
            return

        adjacency = getattr(self.env, '_adjacency_matrix', None)
        if adjacency is None and hasattr(self.env, 'adjacency_matrix'):
            adjacency = self.env.adjacency_matrix
        if adjacency is None:
            raise RuntimeError('MADDPGCollector requires adjacency matrix for k-hop reposition candidates')

        khop_mask_hh = HexNeighbors.compute_khop_mask(adjacency, k=max(1, int(self.repos_khop)))
        khop_indices, _, max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh)
        self._khop_neighbor_indices = khop_indices.to(self.device)
        self._khop_neighbor_mask = (khop_indices != -1).to(self.device)
        self._max_k_neighbors = int(max_k)

    def _build_per_vehicle_trip_mask(self, vehicle_hex_ids: torch.Tensor, max_trips: int) -> torch.Tensor:
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

        if hasattr(self.env, 'hex_grid') and hasattr(self.env.hex_grid, 'distance_matrix') and self.env.hex_grid.distance_matrix is not None:
            dist_matrix = self.env.hex_grid.distance_matrix._distances
            pickup_distance = dist_matrix[vehicle_hex_ids[:, None], pickup_hexes[None, :]]
            within = pickup_distance <= getattr(self.env, 'max_pickup_distance', 5.0)

            if hasattr(self.env.trip_state, 'distance_km') and hasattr(self.env, 'energy_dynamics') and self.env.energy_dynamics is not None and hasattr(self.env, 'fleet_state') and hasattr(self.env.fleet_state, 'socs'):
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

    def _build_reposition_mask(self, vehicle_hex_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if self._khop_neighbor_indices is None or self._khop_neighbor_mask is None:
            return None

        cand_abs = self._khop_neighbor_indices[vehicle_hex_ids]
        cand_valid = self._khop_neighbor_mask[vehicle_hex_ids] & (cand_abs >= 0)
        reposition_mask = torch.zeros(vehicle_hex_ids.shape[0], self.env.num_hexes, dtype=torch.bool, device=self.device)
        if cand_valid.any():
            row_idx, col_idx = cand_valid.nonzero(as_tuple=True)
            reposition_mask[row_idx, cand_abs[row_idx, col_idx]] = True
        return reposition_mask
    def collect_episode(
        self,
        agent: MADDPGAgent,
        rollout_steps: int = 288,
        seed: Optional[int] = None,
        deterministic: bool = False,
        exploration_eps: float = 0.0,
        **kwargs,
    ) -> EpisodeStats:
        del kwargs

        if seed is not None:
            torch.manual_seed(seed)

        state = self.env.reset(seed=seed, episode_idx=self.episode_count)
        self._ensure_khop_neighbors()

        total_reward = 0.0
        steps = 0
        done = False
        info = None
        action_counts = {"serve": 0, "charge": 0, "reposition": 0}
        total_serve_attempted = 0
        total_serve_success = 0
        total_charge_attempted = 0
        total_charge_success = 0
        total_reposition_attempted = 0
        total_reposition_success = 0

        agent.actor.eval()

        while steps < rollout_steps and not done:
            state_dict = self._build_state_dict(state)
            available_mask = self.env.get_available_actions()

            vehicle_hex_ids = self.env.fleet_state.positions.long()
            reposition_mask = self._build_reposition_mask(vehicle_hex_ids)

            trip_mask = None
            if hasattr(self.env, 'trip_state'):
                trip_mask = self._build_per_vehicle_trip_mask(vehicle_hex_ids, agent.actor.max_trips)
                available_mask[:, 0] = available_mask[:, 0] & trip_mask.any(dim=1)

            reposition_candidate_hexes = None
            with torch.no_grad():
                maddpg_out = agent.select_action(
                    state=state,
                    action_mask=available_mask,
                    reposition_mask=reposition_mask,
                    trip_mask=None,
                    deterministic=deterministic,
                    khop_neighbor_indices=self._khop_neighbor_indices,
                    khop_neighbor_mask=self._khop_neighbor_mask,
                    vehicle_hex_ids=vehicle_hex_ids,
                )
                if self._khop_neighbor_indices is not None:
                    reposition_candidate_hexes = self._khop_neighbor_indices[vehicle_hex_ids]

            action_type = maddpg_out.action_type
            reposition_target = maddpg_out.reposition_target
            selected_trip = maddpg_out.selected_trip

            if exploration_eps > 0.0:
                rnd_mask = torch.rand(action_type.shape, device=self.device) < exploration_eps
                rnd_action = torch.randint(0, agent.action_dim, action_type.shape, device=self.device)
                rnd_repos = torch.randint(0, agent.num_hexes, action_type.shape, device=self.device)
                action_type = torch.where(rnd_mask, rnd_action, action_type)
                reposition_target = torch.where(rnd_mask, rnd_repos, reposition_target)

            next_state, reward, done_tensor, info = self.env.step(
                action_type, reposition_target, None
            )
            done = done_tensor.item()

            vehicle_size = self.env.num_vehicles * self.env._vehicle_feature_dim
            hex_size = self.env.num_hexes * self.env._hex_feature_dim

            vf_curr = state[:vehicle_size].view(
                self.env.num_vehicles, self.env._vehicle_feature_dim
            )
            cf_curr = state[vehicle_size + hex_size:]

            vf_next = next_state[:vehicle_size].view(
                self.env.num_vehicles, self.env._vehicle_feature_dim
            )
            cf_next = next_state[vehicle_size + hex_size:]

            if hasattr(info, 'per_vehicle_reward') and info.per_vehicle_reward is not None:
                per_veh_rew = info.per_vehicle_reward.clone().detach()
            else:
                scalar_r = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                per_veh_rew = torch.full(
                    (self.env.num_vehicles,),
                    scalar_r / self.env.num_vehicles,
                    dtype=torch.float32,
                    device=self.device,
                )

            executed_action_type = action_type.clone()
            if info is not None and getattr(info, 'serve_assignments', None) is not None:
                attempted_serve = (action_type == 0)
                success_serve = torch.zeros_like(attempted_serve)
                serve_veh_idx, _ = info.serve_assignments
                if serve_veh_idx is not None and len(serve_veh_idx) > 0:
                    success_serve[serve_veh_idx] = True
                executed_action_type[attempted_serve & (~success_serve)] = -1

            if info is not None and getattr(info, 'charge_assignments', None) is not None:
                attempted_charge = (action_type == 1)
                success_charge = torch.zeros_like(attempted_charge)
                charge_veh_idx, _ = info.charge_assignments
                if charge_veh_idx is not None and len(charge_veh_idx) > 0:
                    success_charge[charge_veh_idx] = True
                executed_action_type[attempted_charge & (~success_charge)] = -1

            attempted_reposition = (action_type == 2)
            executed_actions_repos = reposition_target.clone()
            if info is not None and hasattr(info, 'extra') and isinstance(info.extra, dict):
                repos_failed_indices = info.extra.get('reposition_failed_indices', None)
                if repos_failed_indices is not None and len(repos_failed_indices) > 0:
                    executed_actions_repos[repos_failed_indices.long()] = -1
            success_reposition = attempted_reposition & (executed_actions_repos >= 0)

            self.buffer.push(
                vehicle_features=vf_curr,
                context_features=cf_curr,
                next_vehicle_features=vf_next,
                next_context_features=cf_next,
                actions_type=action_type,
                actions_repos=reposition_target,
                per_vehicle_rewards=per_veh_rew,
                done=done,
                action_mask=available_mask,
                reposition_mask=None,
                trip_mask=None,
                vehicle_hex_ids=vehicle_hex_ids,
                reposition_candidate_hexes=None,
                executed_actions_type=executed_action_type,
                executed_actions_repos=executed_actions_repos,
            )

            if info is not None:
                total_serve_attempted += int(getattr(info, 'num_serve_attempted', 0))
                total_serve_success += int(getattr(info, 'num_serve_success', 0))
                total_charge_attempted += int(getattr(info, 'num_charge_attempted', 0))
                total_charge_success += int(getattr(info, 'num_charge_success', 0))
            total_reposition_attempted += int(attempted_reposition.sum().item())
            total_reposition_success += int(success_reposition.sum().item())

            available_1d = available_mask.any(dim=1)
            valid_actions = action_type[(action_type >= 0) & (action_type < 3) & available_1d]
            if valid_actions.numel() > 0:
                counts = torch.bincount(valid_actions, minlength=3)
                action_counts["serve"] += int(counts[0].item())
                action_counts["charge"] += int(counts[1].item())
                action_counts["reposition"] += int(counts[2].item())

            state = next_state
            total_reward += reward.item() if isinstance(reward, torch.Tensor) else float(reward)
            steps += 1

        agent.actor.train()
        self.episode_count += 1

        stats = EpisodeStats(
            episode_id=self.episode_count,
            total_reward=total_reward,
            steps=steps,
            trips_served=getattr(info, 'trips_served', 0) if steps > 0 else 0,
            trips_dropped=getattr(info, 'trips_dropped', 0) if steps > 0 else 0,
            trips_loaded=getattr(info, 'trips_loaded', 1) if steps > 0 else 1,
            avg_soc=self.env.fleet_state.socs.mean().item() if hasattr(self.env, 'fleet_state') else 0.0,
            revenue=getattr(info, 'revenue', 0.0) if steps > 0 else 0.0,
            energy_cost=getattr(info, 'energy_cost', 0.0) if steps > 0 else 0.0,
            profit=(
                getattr(info, 'revenue', 0.0)
                - getattr(info, 'energy_cost', 0.0)
                - getattr(info, 'driving_cost', 0.0)
            ) if steps > 0 else 0.0,
            action_counts=action_counts,
            num_serve_attempted=total_serve_attempted,
            num_serve_success=total_serve_success,
            next_value=0.0,
        )
        setattr(stats, 'num_charge_attempted', total_charge_attempted)
        setattr(stats, 'num_charge_success', total_charge_success)
        setattr(stats, 'num_reposition_attempted', total_reposition_attempted)
        setattr(stats, 'num_reposition_success', total_reposition_success)
        return stats

    def _build_state_dict(self, flat_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        if hasattr(self.env, '_build_state_dict'):
            return self.env._build_state_dict(flat_state)
        vehicle_size = self.env.num_vehicles * self.env._vehicle_feature_dim
        hex_size = self.env.num_hexes * self.env._hex_feature_dim
        return {
            'vehicle': flat_state[:vehicle_size].view(self.env.num_vehicles, -1),
            'hex': flat_state[vehicle_size:vehicle_size + hex_size].view(self.env.num_hexes, -1),
            'context': flat_state[vehicle_size + hex_size:],
        }
