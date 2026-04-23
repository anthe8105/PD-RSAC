"""
maddpg_critic.py — MADDPG Centralized Critic (NIPS 2017, Lowe et al.).

MADDPG Critic for agent i:
    Q_i^μ(x, a_1, ..., a_N)

where:
    x       = global state (concatenated vehicle observations + context)
    a_1..N  = actions of ALL N agents (one-hot encoded, concat'd)

With parameter sharing: all N agents share the same critic weights.
The critic is trained once per step using the joint action of all agents.
"""

import torch
import torch.nn as nn
from typing import List


class MADDPGCritic(nn.Module):
    """MADDPG centralized critic."""

    def __init__(
        self,
        vehicle_feature_dim: int,
        context_dim: int,
        num_vehicles: int,
        action_dim: int = 3,
        hidden_dims: List[int] = [256, 256],
        action_embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_vehicles = num_vehicles
        self.action_dim = action_dim
        self.vehicle_feature_dim = vehicle_feature_dim
        self.context_dim = context_dim

        joint_action_dim = num_vehicles * action_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(joint_action_dim, action_embed_dim * 4),
            nn.ReLU(),
            nn.Linear(action_embed_dim * 4, action_embed_dim),
            nn.ReLU(),
        )

        global_state_dim = vehicle_feature_dim + context_dim
        in_dim = global_state_dim + action_embed_dim
        layers = []
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.q_head = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self,
        vehicle_features: torch.Tensor,
        context_features: torch.Tensor,
        actions_onehot: torch.Tensor,
    ) -> torch.Tensor:
        B = vehicle_features.size(0)

        mean_veh = vehicle_features.mean(dim=1)
        global_state = torch.cat([mean_veh, context_features], -1)

        joint_action = actions_onehot.reshape(B, -1).float()
        action_embedding = self.action_encoder(joint_action)

        x = torch.cat([global_state, action_embedding], dim=-1)
        h = self.backbone(x)
        return self.q_head(h).squeeze(-1)

    def forward_dict(
        self,
        state_dict: dict,
        actions_onehot: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(
            vehicle_features=state_dict['vehicle_features'],
            context_features=state_dict['context_features'],
            actions_onehot=actions_onehot,
        )
