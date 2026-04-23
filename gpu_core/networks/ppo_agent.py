"""
ppo_agent.py — Standard MAPPO Agent.

Wraps MAPPOActor + MAPPOCritic with:
- A SINGLE shared Adam optimizer (actor + critic parameters)
- select_action(): passes per-agent LOCAL observations to actor
- get_value(): passes GLOBAL state (mean vehicle + context) to critic
- evaluate_actions(): used during the PPO update
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass

from .mappo_actor import MAPPOActor
from .mappo_critic import MAPPOCritic


@dataclass
class PPOOutput:
    action_type: torch.Tensor
    reposition_target: torch.Tensor
    action_log_prob: torch.Tensor
    reposition_log_prob: torch.Tensor
    selected_trip: Optional[torch.Tensor] = None
    trip_log_prob: Optional[torch.Tensor] = None
    action_entropy: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    vehicle_charge_power: Optional[torch.Tensor] = None


class PPOAgent(nn.Module):
    """Standard MAPPO Agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_hexes: int,
        actor_hidden_dims: List[int] = [256, 256],
        critic_hidden_dims: List[int] = [256, 256],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        dropout: float = 0.1,
        device: str = 'cuda',
        num_vehicles: int = 1000,
        vehicle_feature_dim: int = 16,
        hex_feature_dim: int = 5,
        context_dim: int = 9,
        max_trips: int = 2000,
        use_trip_head: bool = False,
        learn_charge_power: bool = False,
        max_k_neighbors: int = 61,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_hexes = num_hexes
        self.num_vehicles = num_vehicles
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.device = device
        self.vehicle_feature_dim = vehicle_feature_dim
        self.hex_feature_dim = hex_feature_dim
        self.context_dim = context_dim
        self._max_trips = max_trips
        self.use_trip_head = use_trip_head
        self.learn_charge_power = learn_charge_power
        self.max_k_neighbors = max_k_neighbors
        self._value_norm_eps = 1e-5

        self.actor = MAPPOActor(
            vehicle_feature_dim=vehicle_feature_dim,
            context_dim=context_dim,
            hex_feature_dim=hex_feature_dim,
            num_hexes=num_hexes,
            hidden_dims=actor_hidden_dims,
            action_dim=action_dim,
            dropout=dropout,
            max_trips=max_trips,
            use_trip_head=use_trip_head,
            learn_charge_power=learn_charge_power,
            max_k_neighbors=max_k_neighbors,
        ).to(device)

        self.critic = MAPPOCritic(
            vehicle_feature_dim=vehicle_feature_dim,
            context_dim=context_dim,
            hex_feature_dim=hex_feature_dim,
            hidden_dims=critic_hidden_dims,
            dropout=dropout,
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.optimizer = self.actor_optimizer

        self.register_buffer('value_norm_mean', torch.zeros(1, device=device))
        self.register_buffer('value_norm_var', torch.ones(1, device=device))
        self.register_buffer('value_norm_count', torch.tensor(1e-4, device=device))

    @torch.no_grad()
    def update_value_norm_stats(self, targets: torch.Tensor) -> None:
        flat = targets.detach().reshape(-1).float()
        if flat.numel() == 0:
            return

        batch_count = torch.tensor(float(flat.numel()), device=flat.device)
        batch_mean = flat.mean()
        batch_var = flat.var(unbiased=False)

        delta = batch_mean - self.value_norm_mean
        total_count = self.value_norm_count + batch_count

        new_mean = self.value_norm_mean + delta * (batch_count / total_count)
        m_a = self.value_norm_var * self.value_norm_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (self.value_norm_count * batch_count / total_count)
        new_var = m2 / total_count

        self.value_norm_mean.copy_(new_mean)
        self.value_norm_var.copy_(torch.clamp(new_var, min=self._value_norm_eps))
        self.value_norm_count.copy_(total_count)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(torch.clamp(self.value_norm_var, min=self._value_norm_eps))
        return (values - self.value_norm_mean) / std

    def _parse_state(
        self,
        state: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract (vehicle_features, hex_features, context_features) from state."""
        if isinstance(state, dict):
            vf = state['vehicle'] if 'vehicle' in state else state['vehicle_features']
            hf = state['hex'] if 'hex' in state else state['hex_features']
            cf = state['context'] if 'context' in state else state['context_features']
        else:
            if state.dim() == 1:
                state = state.unsqueeze(0)
            B = state.size(0)
            vehicle_size = self.num_vehicles * self.vehicle_feature_dim
            hex_size = self.num_hexes * self.hex_feature_dim
            vf = state[:, :vehicle_size].view(B, self.num_vehicles, self.vehicle_feature_dim)
            hf = state[:, vehicle_size:vehicle_size + hex_size].view(B, self.num_hexes, self.hex_feature_dim)
            cf = state[:, vehicle_size + hex_size:vehicle_size + hex_size + self.context_dim]

        if vf.dim() == 2:
            vf = vf.unsqueeze(0)
        if hf.dim() == 2:
            hf = hf.unsqueeze(0)
        if cf.dim() == 1:
            cf = cf.unsqueeze(0)

        return vf, hf, cf

    def get_value(self, state: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        vf, hf, cf = self._parse_state(state)
        return self.critic(vf, cf, hf)

    def select_action(
        self,
        state: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_features: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,
        station_features: Optional[torch.Tensor] = None,
        station_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
        khop_neighbor_indices: Optional[torch.Tensor] = None,
        khop_neighbor_mask: Optional[torch.Tensor] = None,
    ) -> PPOOutput:
        del trip_features, station_features, station_mask

        with torch.no_grad():
            vf, hf, cf = self._parse_state(state)
            out = self.actor(
                vehicle_features=vf,
                context_features=cf,
                hex_features=hf,
                action_mask=action_mask,
                reposition_mask=reposition_mask,
                trip_mask=trip_mask,
                deterministic=deterministic,
                khop_neighbor_indices=khop_neighbor_indices,
                khop_neighbor_mask=khop_neighbor_mask,
                vehicle_hex_ids=vehicle_hex_ids,
            )
            value = self.get_value(state)
            if value.dim() == 2:
                value = value.squeeze(-1)

        return PPOOutput(
            action_type=out['action_type'],
            reposition_target=out['reposition_target'],
            action_log_prob=out['action_log_prob'],
            reposition_log_prob=out['reposition_log_prob'],
            selected_trip=out.get('selected_trip') if self.use_trip_head else None,
            trip_log_prob=out.get('trip_log_prob') if self.use_trip_head else None,
            action_entropy=out.get('action_entropy'),
            value=value,
            vehicle_charge_power=out.get('vehicle_charge_power') if self.learn_charge_power else None,
        )

    def evaluate_actions(
        self,
        states,
        action_type: torch.Tensor,
        reposition_target: torch.Tensor,
        selected_trip: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
        khop_neighbor_indices: Optional[torch.Tensor] = None,
        khop_neighbor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vf, hf, cf = self._parse_state(states)

        out = self.actor(
            vehicle_features=vf,
            context_features=cf,
            hex_features=hf,
            action_mask=action_mask,
            reposition_mask=reposition_mask,
            trip_mask=trip_mask,
            deterministic=False,
            eval_action_type=action_type,
            eval_reposition_target=reposition_target,
            eval_selected_trip=selected_trip if self.use_trip_head else None,
            vehicle_hex_ids=vehicle_hex_ids,
            khop_neighbor_indices=khop_neighbor_indices,
            khop_neighbor_mask=khop_neighbor_mask,
        )

        total_log_prob = out['action_log_prob'] + out['reposition_log_prob']
        if self.use_trip_head and out.get('trip_log_prob') is not None:
            total_log_prob = total_log_prob + out['trip_log_prob']

        full_entropy = out['action_entropy']
        if out.get('reposition_entropy') is not None:
            full_entropy = full_entropy + out['reposition_entropy']
        if self.use_trip_head and out.get('trip_entropy') is not None:
            full_entropy = full_entropy + out['trip_entropy']

        return total_log_prob, full_entropy
