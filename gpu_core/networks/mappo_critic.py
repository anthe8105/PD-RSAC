"""
mappo_critic.py — Standard MAPPO Centralized Critic (MLP on global state).

Standard MAPPO design:
- Centralized: sees the GLOBAL state during training (CTDE)
- Global state = mean(vehicle_features) ++ context_features ++ mean(hex_features)
  (mean-pooling avoids O(N_agents) input dimension, standard for large fleets)
- MLP → scalar V(s) per timestep
- No GCN, no per-vehicle outputs
"""

import torch
import torch.nn as nn
from typing import List, Dict


class MAPPOCritic(nn.Module):
    """
    Standard MAPPO Centralized Critic — MLP on global state.

    Global state:
        mean(vehicle_features over all agents) ++ context_features ++ mean(hex_features)
        → [B, vehicle_feature_dim + context_dim + hex_feature_dim]

    Output: V(s) scalar per timestep → [B]
    """

    def __init__(
        self,
        vehicle_feature_dim: int,
        context_dim: int,
        hex_feature_dim: int,
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.1,
    ):
        super().__init__()

        global_state_dim = vehicle_feature_dim + context_dim + hex_feature_dim
        self.input_norm = nn.LayerNorm(global_state_dim)

        layers = []
        in_dim = global_state_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.Tanh(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self,
        vehicle_features: torch.Tensor,
        context_features: torch.Tensor,
        hex_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns V(s): [B]."""
        mean_veh = vehicle_features.mean(dim=1)
        mean_hex = hex_features.mean(dim=1)
        global_state = torch.cat([mean_veh, context_features, mean_hex], dim=-1)
        global_state = self.input_norm(global_state)
        h = self.backbone(global_state)
        return self.value_head(h).squeeze(-1)

    def forward_dict(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convenience wrapper accepting a state dict."""
        return self.forward(
            vehicle_features=state_dict['vehicle_features'],
            context_features=state_dict['context_features'],
            hex_features=state_dict['hex_features'],
        )
