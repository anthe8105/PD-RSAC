"""
maddpg_buffer.py — Replay buffer for hex-level MADDPG.
"""

import torch
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class MADDPGBatch:
    vehicle_states: torch.Tensor
    context_states: torch.Tensor
    next_vehicle_states: torch.Tensor
    next_context_states: torch.Tensor
    actions_type: torch.Tensor
    actions_repos: torch.Tensor
    per_vehicle_rewards: torch.Tensor
    dones: torch.Tensor
    action_mask: Optional[torch.Tensor] = None
    reposition_mask: Optional[torch.Tensor] = None
    trip_mask: Optional[torch.Tensor] = None
    vehicle_hex_ids: Optional[torch.Tensor] = None
    reposition_candidate_hexes: Optional[torch.Tensor] = None
    executed_actions_type: Optional[torch.Tensor] = None
    executed_actions_repos: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None


class MADDPGReplayBuffer:
    """GPU-resident replay buffer storing hex-level transitions."""

    def __init__(
        self,
        capacity: int,
        num_vehicles: int,
        vehicle_feature_dim: int,
        context_dim: int,
        action_dim: int = 3,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        device: str = 'cuda',
    ):
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.vehicle_feature_dim = vehicle_feature_dim
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.prioritized = prioritized
        self.alpha = alpha
        self.device = torch.device(device)

        self._pos = 0
        self._size = 0

        self.vehicle_states = torch.zeros(capacity, num_vehicles, vehicle_feature_dim, dtype=torch.float32, device=self.device)
        self.context_states = torch.zeros(capacity, context_dim, dtype=torch.float32, device=self.device)
        self.next_vehicle_states = torch.zeros(capacity, num_vehicles, vehicle_feature_dim, dtype=torch.float32, device=self.device)
        self.next_context_states = torch.zeros(capacity, context_dim, dtype=torch.float32, device=self.device)
        self.actions_type = torch.zeros(capacity, num_vehicles, dtype=torch.long, device=self.device)
        self.actions_repos = torch.zeros(capacity, num_vehicles, dtype=torch.long, device=self.device)
        self.per_vehicle_rewards = torch.zeros(capacity, num_vehicles, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=self.device)

        self.action_masks: List[Optional[torch.Tensor]] = [None] * capacity
        self.reposition_masks: List[Optional[torch.Tensor]] = [None] * capacity
        self.trip_masks: List[Optional[torch.Tensor]] = [None] * capacity
        self.vehicle_hex_ids_list: List[Optional[torch.Tensor]] = [None] * capacity
        self.reposition_candidate_hexes_list: List[Optional[torch.Tensor]] = [None] * capacity
        self.executed_actions_type_list: List[Optional[torch.Tensor]] = [None] * capacity
        self.executed_actions_repos_list: List[Optional[torch.Tensor]] = [None] * capacity

        if prioritized:
            self.priorities = torch.ones(capacity, dtype=torch.float32, device=self.device)
            self.max_priority = 1.0
        self._beta = beta_start

    def push(
        self,
        vehicle_features: torch.Tensor,
        context_features: torch.Tensor,
        next_vehicle_features: torch.Tensor,
        next_context_features: torch.Tensor,
        actions_type: torch.Tensor,
        actions_repos: torch.Tensor,
        per_vehicle_rewards: torch.Tensor,
        done: bool,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
        reposition_candidate_hexes: Optional[torch.Tensor] = None,
        executed_actions_type: Optional[torch.Tensor] = None,
        executed_actions_repos: Optional[torch.Tensor] = None,
    ) -> None:
        idx = self._pos
        self.vehicle_states[idx] = vehicle_features
        self.context_states[idx] = context_features
        self.next_vehicle_states[idx] = next_vehicle_features
        self.next_context_states[idx] = next_context_features
        self.actions_type[idx] = actions_type
        self.actions_repos[idx] = actions_repos
        self.per_vehicle_rewards[idx] = per_vehicle_rewards
        self.dones[idx] = done

        self.action_masks[idx] = action_mask.detach().to(self.device).clone() if action_mask is not None else None
        self.reposition_masks[idx] = reposition_mask.detach().to(self.device).clone() if reposition_mask is not None else None
        self.trip_masks[idx] = trip_mask.detach().to(self.device).clone() if trip_mask is not None else None
        self.vehicle_hex_ids_list[idx] = vehicle_hex_ids.detach().to(self.device).clone() if vehicle_hex_ids is not None else None
        self.reposition_candidate_hexes_list[idx] = reposition_candidate_hexes.detach().to(self.device).clone() if reposition_candidate_hexes is not None else None
        self.executed_actions_type_list[idx] = executed_actions_type.detach().to(self.device).clone() if executed_actions_type is not None else None
        self.executed_actions_repos_list[idx] = executed_actions_repos.detach().to(self.device).clone() if executed_actions_repos is not None else None

        if self.prioritized:
            self.priorities[idx] = self.max_priority

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> MADDPGBatch:
        if self.prioritized:
            probs = self.priorities[:self._size] ** self.alpha
            probs = probs / probs.sum()
            indices = torch.multinomial(probs, batch_size, replacement=True)
            weights = (self._size * probs[indices]) ** (-self._beta)
            weights = weights / weights.max()
            return self._gather(indices, weights)
        indices = torch.randint(0, self._size, (batch_size,), device=self.device)
        return self._gather(indices, None)

    def _gather(self, indices: torch.Tensor, weights: Optional[torch.Tensor]) -> MADDPGBatch:
        indices_list = indices.detach().cpu().tolist()

        def _stack_optional(items: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
            selected = [items[int(i)] for i in indices_list]
            if not selected or any(x is None for x in selected):
                return None
            return torch.stack(selected, dim=0)

        return MADDPGBatch(
            vehicle_states=self.vehicle_states[indices],
            context_states=self.context_states[indices],
            next_vehicle_states=self.next_vehicle_states[indices],
            next_context_states=self.next_context_states[indices],
            actions_type=self.actions_type[indices],
            actions_repos=self.actions_repos[indices],
            per_vehicle_rewards=self.per_vehicle_rewards[indices],
            dones=self.dones[indices],
            action_mask=_stack_optional(self.action_masks),
            reposition_mask=_stack_optional(self.reposition_masks),
            trip_mask=_stack_optional(self.trip_masks),
            vehicle_hex_ids=_stack_optional(self.vehicle_hex_ids_list),
            reposition_candidate_hexes=_stack_optional(self.reposition_candidate_hexes_list),
            executed_actions_type=_stack_optional(self.executed_actions_type_list),
            executed_actions_repos=_stack_optional(self.executed_actions_repos_list),
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor, epsilon: float = 1e-6) -> None:
        if not self.prioritized:
            return
        priorities = td_errors.abs() + epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())

    def __len__(self) -> int:
        return self._size
