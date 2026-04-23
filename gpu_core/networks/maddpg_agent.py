"""
maddpg_agent.py — MADDPG Agent Wrapper (NIPS 2017, Lowe et al.).
"""

import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

from .maddpg_actor import MADDPGActor
from .maddpg_critic import MADDPGCritic


@dataclass
class MADDPGOutput:
    action_type: torch.Tensor
    reposition_target: torch.Tensor
    selected_trip: torch.Tensor
    action_type_soft: Optional[torch.Tensor] = None
    reposition_soft: Optional[torch.Tensor] = None
    trip_soft: Optional[torch.Tensor] = None


class MADDPGAgent(nn.Module):
    """MADDPG agent with shared actor/critic and target networks."""

    def __init__(
        self,
        vehicle_feature_dim: int,
        context_dim: int,
        num_vehicles: int,
        num_hexes: int,
        actor_hidden_dims: List[int] = [256, 256],
        critic_hidden_dims: List[int] = [256, 256],
        action_dim: int = 3,
        max_trips: int = 2000,
        action_embed_dim: int = 64,
        dropout: float = 0.1,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gumbel_tau: float = 1.0,
        device: str = 'cuda',
        state_dim: Optional[int] = None,
        hex_feature_dim: int = 5,
        max_k_neighbors: int = 61,
    ):
        super().__init__()

        self.num_vehicles = num_vehicles
        self.num_hexes = num_hexes
        self.vehicle_feature_dim = vehicle_feature_dim
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.gumbel_tau = gumbel_tau
        self.device_str = device
        self.hex_feature_dim = hex_feature_dim
        self.max_k_neighbors = max_k_neighbors

        self.state_dim = state_dim or (
            num_vehicles * vehicle_feature_dim + num_hexes * hex_feature_dim + context_dim
        )

        self.actor = MADDPGActor(
            vehicle_feature_dim=vehicle_feature_dim,
            context_dim=context_dim,
            num_hexes=num_hexes,
            hidden_dims=actor_hidden_dims,
            action_dim=action_dim,
            max_trips=max_trips,
            dropout=dropout,
            gumbel_tau=gumbel_tau,
            max_k_neighbors=max_k_neighbors,
        )
        self.actor_target = copy.deepcopy(self.actor)
        for p in self.actor_target.parameters():
            p.requires_grad_(False)

        self.critic = MADDPGCritic(
            vehicle_feature_dim=vehicle_feature_dim,
            context_dim=context_dim,
            num_vehicles=num_vehicles,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims,
            action_embed_dim=action_embed_dim,
            dropout=dropout,
        )
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.optimizer = self.actor_optimizer

    def _parse_state(
        self,
        state: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(state, dict):
            vf = state.get('vehicle', state.get('vehicle_features'))
            cf = state.get('context', state.get('context_features'))
        else:
            if state.dim() == 1:
                state = state.unsqueeze(0)
            B = state.size(0)
            vehicle_size = self.num_vehicles * self.vehicle_feature_dim
            vf = state[:, :vehicle_size].view(B, self.num_vehicles, self.vehicle_feature_dim)
            cf = state[:, -self.context_dim:]

        if vf.dim() == 2:
            vf = vf.unsqueeze(0)
        if cf.dim() == 1:
            cf = cf.unsqueeze(0)

        return vf, cf

    @staticmethod
    def build_joint_actions_onehot(
        action_type: torch.Tensor,
        action_dim: int,
    ) -> torch.Tensor:
        if action_type.dim() == 1:
            action_type = action_type.unsqueeze(0)
        B, N = action_type.shape
        onehot = torch.zeros(B, N, action_dim, dtype=torch.float32, device=action_type.device)
        clamped = action_type.clamp(0, action_dim - 1)
        onehot.scatter_(-1, clamped.unsqueeze(-1), 1.0)
        return onehot

    def select_action(
        self,
        state: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        khop_neighbor_indices: Optional[torch.Tensor] = None,
        khop_neighbor_mask: Optional[torch.Tensor] = None,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MADDPGOutput:
        del kwargs

        with torch.no_grad():
            vf, cf = self._parse_state(state)
            B, N, vdim = vf.shape

            vf_flat = vf.reshape(B * N, vdim)
            cf_flat = cf.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)

            am_flat = None
            if action_mask is not None:
                if action_mask.dim() == 2:
                    am_flat = action_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
                elif action_mask.dim() == 3:
                    am_flat = action_mask.reshape(B * N, -1)

            rm_flat = None
            if reposition_mask is not None:
                if reposition_mask.dim() == 2:
                    rm_flat = reposition_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
                elif reposition_mask.dim() == 3:
                    rm_flat = reposition_mask.reshape(B * N, -1)

            vh_flat = None
            if vehicle_hex_ids is not None:
                if vehicle_hex_ids.dim() == 1:
                    vh_flat = vehicle_hex_ids.unsqueeze(0).expand(B, -1).reshape(B * N)
                elif vehicle_hex_ids.dim() == 2:
                    vh_flat = vehicle_hex_ids.reshape(B * N)

            action_type, reposition_target, selected_trip = self.actor.act(
                vehicle_feature=vf_flat,
                context_feature=cf_flat,
                action_mask=am_flat,
                reposition_mask=rm_flat,
                trip_mask=trip_mask,
                deterministic=deterministic,
                khop_neighbor_indices=khop_neighbor_indices,
                khop_neighbor_mask=khop_neighbor_mask,
                vehicle_hex_ids=vh_flat,
            )

            action_type = action_type.reshape(B, N)
            reposition_target = reposition_target.reshape(B, N)
            selected_trip = selected_trip.reshape(B, N)

            if B == 1:
                action_type = action_type.squeeze(0)
                reposition_target = reposition_target.squeeze(0)
                selected_trip = selected_trip.squeeze(0)

        return MADDPGOutput(
            action_type=action_type,
            reposition_target=reposition_target,
            selected_trip=selected_trip,
        )

    def gumbel_actions(
        self,
        vehicle_features: torch.Tensor,
        context_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        khop_neighbor_indices: Optional[torch.Tensor] = None,
        khop_neighbor_mask: Optional[torch.Tensor] = None,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, vdim = vehicle_features.shape
        vf_flat = vehicle_features.reshape(B * N, vdim)
        cf_flat = context_features.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)

        am_flat = None
        if action_mask is not None:
            if action_mask.dim() == 2:
                am_flat = action_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
            elif action_mask.dim() == 3:
                am_flat = action_mask.reshape(B * N, -1)

        rm_flat = None
        if reposition_mask is not None:
            if reposition_mask.dim() == 2:
                rm_flat = reposition_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
            elif reposition_mask.dim() == 3:
                rm_flat = reposition_mask.reshape(B * N, -1)

        vh_flat = None
        if vehicle_hex_ids is not None:
            if vehicle_hex_ids.dim() == 1:
                vh_flat = vehicle_hex_ids.unsqueeze(0).expand(B, -1).reshape(B * N)
            elif vehicle_hex_ids.dim() == 2:
                vh_flat = vehicle_hex_ids.reshape(B * N)

        out = self.actor.forward(
            vehicle_feature=vf_flat,
            context_feature=cf_flat,
            action_mask=am_flat,
            reposition_mask=rm_flat,
            trip_mask=trip_mask,
            tau=tau,
            khop_neighbor_indices=khop_neighbor_indices,
            khop_neighbor_mask=khop_neighbor_mask,
            vehicle_hex_ids=vh_flat,
        )

        return {
            'action_type_soft': out['action_type_soft'].reshape(B, N, -1),
            'action_type': out['action_type'].reshape(B, N),
            'reposition_soft': out['reposition_soft'].reshape(B, N, -1),
            'reposition_target': out['reposition_target'].reshape(B, N),
            'trip_soft': out['trip_soft'].reshape(B, N, -1),
            'selected_trip': out['selected_trip'].reshape(B, N),
        }

    def soft_update_target(self, tau: Optional[float] = None) -> None:
        tau = tau if tau is not None else self.tau
        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

    def hard_update_target(self) -> None:
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
