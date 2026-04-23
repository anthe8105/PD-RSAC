"""
GCN-based Critic to match Actor's spatial structure.

FIXES:
1. Uses GCN to encode spatial hex features (like Actor)
2. Per-vehicle action encoding (no aggregation loss)
3. Consistent state representation with Actor
4. Coordination-aware Q-value estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .gcn import GCNEncoder


class GCNCritic(nn.Module):
    """
    GCN-based Critic matching Actor's spatial architecture.

    Key improvements over flat Critic:
    1. GCN backbone for hex embeddings (same as Actor)
    2. Per-vehicle Q-value computation (no aggregation before encoding)
    3. Spatial structure preservation
    4. Compatible with structured state dict
    """

    def __init__(
        self,
        num_hexes: int,
        hex_feature_dim: int,
        vehicle_feature_dim: int,
        context_dim: int,
        action_dim: int = 3,
        gcn_hidden_dim: int = 128,
        gcn_output_dim: int = 64,
        critic_hidden_dim: int = 256,
        dropout: float = 0.1,
        aggregation: str = 'mean',  # 'mean', 'sum', or 'weighted'
    ):
        """
        Initialize GCN-based Critic.

        Args:
            num_hexes: Number of hexagons
            hex_feature_dim: Hex feature dimension
            vehicle_feature_dim: Vehicle feature dimension
            context_dim: Global context dimension
            action_dim: Action space dimension
            gcn_hidden_dim: GCN hidden dimension
            gcn_output_dim: GCN output (hex embedding) dimension
            critic_hidden_dim: Critic MLP hidden dimension
            dropout: Dropout rate
            aggregation: How to aggregate per-vehicle Q-values to fleet Q
        """
        super().__init__()
        self.num_hexes = num_hexes
        self.action_dim = action_dim
        self.gcn_output_dim = gcn_output_dim
        self.aggregation = aggregation

        # ===== GCN Encoder (SHARED STRUCTURE WITH ACTOR) =====
        # Two-layer GCN with symmetric normalization
        self.gcn = GCNEncoder(
            input_dim=hex_feature_dim,
            hidden_dims=[gcn_hidden_dim],
            output_dim=gcn_output_dim,
            dropout=dropout,
            use_batch_norm=True,
            activation='silu'
        )

        # ===== Vehicle-Action Context Encoder =====
        # Combines: vehicle features + local hex context + action encoding
        # Input: vehicle_features (14) + local_hex (64) + context (9) + action_one_hot (4)
        context_input_dim = vehicle_feature_dim + gcn_output_dim + context_dim + action_dim

        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, critic_hidden_dim),
            nn.LayerNorm(critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.LayerNorm(critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ===== Q-Value Head =====
        # Outputs Q-value per vehicle
        self.q_head = nn.Sequential(
            nn.Linear(critic_hidden_dim, critic_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim // 2, 1)
        )

        # ===== Weighted Aggregation (optional) =====
        if aggregation == 'weighted':
            self.importance_head = nn.Sequential(
                nn.Linear(critic_hidden_dim, critic_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(critic_hidden_dim // 2, 1)
            )

    def forward(
        self,
        hex_features: torch.Tensor,  # [batch, num_hexes, hex_feature_dim]
        vehicle_features: torch.Tensor,  # [batch, num_vehicles, vehicle_feature_dim]
        vehicle_hex_ids: torch.Tensor,  # [batch, num_vehicles] hex indices
        context_features: torch.Tensor,  # [batch, context_dim]
        adjacency: torch.Tensor,  # [num_hexes, num_hexes]
        actions: Optional[torch.Tensor],  # [batch, num_vehicles] action types (or None to evaluate all)
        hex_embeddings: Optional[torch.Tensor] = None,  # pre-computed [batch, num_hexes, gcn_out_dim]
        return_per_vehicle: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with structured state.

        Args:
            hex_features: Hex spatial features
            vehicle_features: Per-vehicle features
            vehicle_hex_ids: Vehicle positions (hex indices)
            context_features: Global context
            adjacency: Hex graph adjacency matrix
            actions: Per-vehicle action types
            return_per_vehicle: If True, return [batch, V] per-vehicle Q instead of [batch] fleet Q

        Returns:
            q_values: [batch] fleet Q-values or [batch, V] per-vehicle Q-values
        """
        batch_size, num_vehicles = vehicle_features.shape[0], vehicle_features.shape[1]

        # ===== 1. GCN Encoding (SAME AS ACTOR) =====
        if hex_embeddings is None:
            hex_embeddings = self.gcn(hex_features, adjacency)  # [batch, num_hexes, 64]

        # ===== 2. Gather Local Hex Context per Vehicle =====
        # Create batch indices for gathering
        batch_indices = torch.arange(batch_size, device=hex_features.device).unsqueeze(1).expand(-1, num_vehicles)

        # Gather local hex embeddings for each vehicle
        local_hex_context = hex_embeddings[batch_indices, vehicle_hex_ids]  # [batch, num_vehicles, 64]

        # ===== 3. Expand Global Context =====
        context_expanded = context_features.unsqueeze(1).expand(-1, num_vehicles, -1)  # [batch, num_vehicles, 9]

        # ===== 4. Encode Actions (handle None = all actions) =====
        if actions is not None:
            action_one_hot = F.one_hot(actions.long(), self.action_dim).float()  # [batch, num_vehicles, action_dim]

            # ===== 5. Combine All Features =====
            vehicle_action_context = torch.cat([
                vehicle_features,      # [batch, num_vehicles, 14]
                local_hex_context,     # [batch, num_vehicles, 64] - FROM GCN!
                context_expanded,      # [batch, num_vehicles, 9]
                action_one_hot        # [batch, num_vehicles, action_dim]
            ], dim=-1)  # [batch, num_vehicles, context_dim]

            # ===== 6. Encode Context (single action) =====
            encoded_context = self.context_encoder(vehicle_action_context)  # [batch, V, hidden]

            # ===== 7. Compute Per-Vehicle Q-Values =====
            q_per_vehicle = self.q_head(encoded_context).squeeze(-1)  # [batch, V]

        else:
            # Evaluate ALL actions efficiently:
            # Instead of a 4D [B, V, A, F] tensor through the MLP (slow, large activations),
            # we reshape to [B*V*A, F] — a single 2D matmul that CUDA fuses perfectly.
            # This gives identical results but is ~3-8x faster and uses far less VRAM.
            A = self.action_dim
            BV = batch_size * num_vehicles

            # Base features: [batch, V, F_base]
            base_feat = torch.cat([vehicle_features, local_hex_context, context_expanded], dim=-1)
            # → [B*V, F_base]
            base_flat = base_feat.reshape(BV, -1)
            # Repeat A times: [B*V*A, F_base]
            base_expanded = base_flat.unsqueeze(1).expand(-1, A, -1).reshape(BV * A, -1)

            # Action one-hot: [A, A] → tile to [B*V*A, A]
            a_onehot = torch.eye(A, device=hex_features.device)           # [A, A]
            a_expanded = a_onehot.unsqueeze(0).expand(BV, -1, -1).reshape(BV * A, A)  # [B*V*A, A]

            combined_flat = torch.cat([base_expanded, a_expanded], dim=-1)  # [B*V*A, F_total]

            # Single fused MLP forward on 2D tensor (fully CUDA-optimized)
            encoded_flat = self.context_encoder(combined_flat)  # [B*V*A, hidden]
            q_flat = self.q_head(encoded_flat).squeeze(-1)      # [B*V*A]

            # Reshape back: [batch, V, A]
            q_per_vehicle = q_flat.reshape(batch_size, num_vehicles, A)  # [batch, V, A]

        if return_per_vehicle:
            return q_per_vehicle  # [batch, V]

        # ===== 8. Aggregate to Fleet Q-Value =====
        if self.aggregation == 'mean':
            q_fleet = q_per_vehicle.mean(dim=1)  # [batch]
        elif self.aggregation == 'sum':
            q_fleet = q_per_vehicle.sum(dim=1)  # [batch]
        elif self.aggregation == 'weighted':
            # Importance-weighted aggregation
            importance_logits = self.importance_head(encoded_context).squeeze(-1)  # [batch, num_vehicles]
            importance_weights = F.softmax(importance_logits, dim=1)  # [batch, num_vehicles]
            q_fleet = (q_per_vehicle * importance_weights).sum(dim=1)  # [batch]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return q_fleet

    def forward_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience method accepting state dict directly.

        Args:
            state_dict: Dict with keys: hex_features, vehicle_features,
                        vehicle_hex_ids, context_features, adjacency
            actions: [batch, num_vehicles] action types

        Returns:
            q_values: [batch] fleet Q-values
        """
        return self.forward(
            hex_features=state_dict['hex_features'],
            vehicle_features=state_dict['vehicle_features'],
            vehicle_hex_ids=state_dict['vehicle_hex_ids'],
            context_features=state_dict['context_features'],
            adjacency=state_dict['adjacency'],
            actions=actions
        )


class GCNTwinCritic(nn.Module):
    """
    Twin GCN Critics for SAC (reduce overestimation bias).
    """

    def __init__(
        self,
        num_hexes: int,
        hex_feature_dim: int,
        vehicle_feature_dim: int,
        context_dim: int,
        action_dim: int = 3,
        gcn_hidden_dim: int = 128,
        gcn_output_dim: int = 64,
        critic_hidden_dim: int = 256,
        dropout: float = 0.1,
        aggregation: str = 'mean',
    ):
        super().__init__()

        # Twin critics
        self.critic1 = GCNCritic(
            num_hexes=num_hexes,
            hex_feature_dim=hex_feature_dim,
            vehicle_feature_dim=vehicle_feature_dim,
            context_dim=context_dim,
            action_dim=action_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            gcn_output_dim=gcn_output_dim,
            critic_hidden_dim=critic_hidden_dim,
            dropout=dropout,
            aggregation=aggregation
        )

        self.critic2 = GCNCritic(
            num_hexes=num_hexes,
            hex_feature_dim=hex_feature_dim,
            vehicle_feature_dim=vehicle_feature_dim,
            context_dim=context_dim,
            action_dim=action_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            gcn_output_dim=gcn_output_dim,
            critic_hidden_dim=critic_hidden_dim,
            dropout=dropout,
            aggregation=aggregation
        )

    def forward(
        self,
        hex_features: torch.Tensor,
        vehicle_features: torch.Tensor,
        vehicle_hex_ids: torch.Tensor,
        context_features: torch.Tensor,
        adjacency: torch.Tensor,
        actions: Optional[torch.Tensor],
        hex_embeddings_q1: Optional[torch.Tensor] = None,  # pre-computed for critic1
        hex_embeddings_q2: Optional[torch.Tensor] = None,  # pre-computed for critic2
        return_per_vehicle: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both critics.

        Returns:
            q1, q2: [batch] Q-values from both critics (or [batch, V] if return_per_vehicle)
        """
        q1 = self.critic1(hex_features, vehicle_features, vehicle_hex_ids,
                         context_features, adjacency, actions,
                         hex_embeddings=hex_embeddings_q1,
                         return_per_vehicle=return_per_vehicle)
        q2 = self.critic2(hex_features, vehicle_features, vehicle_hex_ids,
                         context_features, adjacency, actions,
                         hex_embeddings=hex_embeddings_q2,
                         return_per_vehicle=return_per_vehicle)
        return q1, q2

    def forward_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method with state dict."""
        q1 = self.critic1.forward_dict(state_dict, actions)
        q2 = self.critic2.forward_dict(state_dict, actions)
        return q1, q2

    def q1(
        self,
        hex_features: torch.Tensor,
        vehicle_features: torch.Tensor,
        vehicle_hex_ids: torch.Tensor,
        context_features: torch.Tensor,
        adjacency: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Get Q-value from first critic only."""
        return self.critic1(hex_features, vehicle_features, vehicle_hex_ids,
                           context_features, adjacency, actions)

    def min_q(
        self,
        hex_features: torch.Tensor,
        vehicle_features: torch.Tensor,
        vehicle_hex_ids: torch.Tensor,
        context_features: torch.Tensor,
        adjacency: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Get minimum Q-value from twin critics."""
        q1, q2 = self.forward(hex_features, vehicle_features, vehicle_hex_ids,
                             context_features, adjacency, actions)
        return torch.min(q1, q2)

    def min_q_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Get min Q-value with state dict."""
        q1, q2 = self.forward_dict(state_dict, actions)
        return torch.min(q1, q2)


class FleetGCNCritic(nn.Module):
    """Fleet-level critic: evaluates Q(s, hex_allocations, charge_power, repos summary).

    Takes soft per-hex allocation probs [B, H, 3] and continuous charge_power
    [B, H] as action input. Outputs fleet-level Q-value via vehicle-count
    weighted mean of per-hex Q contributions.
    """

    def __init__(
        self,
        num_hexes: int,
        gcn_output_dim: int = 64,
        hex_vehicle_agg_dim: int = 8,
        context_dim: int = 9,
        action_dim: int = 3,
        critic_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_hexes = num_hexes
        self.gcn_output_dim = gcn_output_dim

        # GCN Encoder (own copy — not shared with actor for target network independence)
        from .gcn import GCNEncoder
        self.gcn = GCNEncoder(
            input_dim=5,  # hex_feature_dim (from builder: 5 features)
            hidden_dims=[128],
            output_dim=gcn_output_dim,
            dropout=dropout,
            use_batch_norm=True,
            activation='silu',
        )

        # Hex Q-Encoder
        # Input: hex_emb + veh_summary + context + allocation + power
        q_input_dim = gcn_output_dim + hex_vehicle_agg_dim + context_dim + action_dim + 1
        self.q_encoder = nn.Sequential(
            nn.Linear(q_input_dim, critic_hidden_dim),
            nn.LayerNorm(critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim, critic_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim // 2, 1),
        )

    def forward(
        self,
        hex_features: torch.Tensor,          # [B, H, hex_feature_dim]
        hex_vehicle_summary: torch.Tensor,    # [B, H, 8]
        context_features: torch.Tensor,       # [B, 9]
        hex_allocations: torch.Tensor,        # [B, H, 3] soft allocation probs
        charge_power: torch.Tensor,           # [B, H] continuous charging power
        adjacency: torch.Tensor,              # [H, H] normalized adjacency
        vehicle_counts: Optional[torch.Tensor] = None,  # [B, H] vehicles per hex
        hex_embeddings: Optional[torch.Tensor] = None,   # [B, H, 64] pre-computed
    ) -> torch.Tensor:
        """Compute fleet-level Q-value.

        Args:
            hex_features: Spatial features for GCN.
            hex_vehicle_summary: Per-hex vehicle aggregates.
            context_features: Global context.
            hex_allocations: Soft action allocation [B, H, 3].
            charge_power: Continuous charging power [B, H].
            adjacency: Normalized adjacency for GCN.
            vehicle_counts: [B, H] vehicle count per hex (for weighted mean).
            hex_embeddings: Pre-computed GCN embeddings (skip GCN if provided).

        Returns:
            q_fleet: [B] scalar Q-value per batch.
        """
        B, H = hex_features.shape[0], hex_features.shape[1]

        # GCN encoding
        if hex_embeddings is None:
            hex_embeddings = self.gcn(hex_features, adjacency)  # [B, H, 64]

        # Expand context
        context_expanded = context_features.unsqueeze(1).expand(-1, H, -1)  # [B, H, 9]
        # Concatenate all inputs
        q_input = torch.cat([
            hex_embeddings,                           # [B, H, 64]
            hex_vehicle_summary,                      # [B, H, 8]
            context_expanded,                         # [B, H, 9]
            hex_allocations,                          # [B, H, 3]
            charge_power.unsqueeze(-1),               # [B, H, 1]
        ], dim=-1)

        # Per-hex Q contribution
        q_per_hex = self.q_encoder(q_input).squeeze(-1)  # [B, H]

        # Fleet Q: vehicle-count weighted mean
        if vehicle_counts is not None:
            total_vehicles = vehicle_counts.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
            weights = vehicle_counts / total_vehicles  # [B, H]
            q_fleet = (q_per_hex * weights).sum(dim=1)  # [B]
        else:
            # Fallback: uniform mean
            q_fleet = q_per_hex.mean(dim=1)  # [B]

        return q_fleet


class FleetGCNTwinCritic(nn.Module):
    """Twin fleet critics for SAC (reduce overestimation bias)."""

    def __init__(self, num_hexes: int, **kwargs):
        super().__init__()
        self.critic1 = FleetGCNCritic(num_hexes=num_hexes, **kwargs)
        self.critic2 = FleetGCNCritic(num_hexes=num_hexes, **kwargs)

    def forward(
        self,
        hex_features: torch.Tensor,
        hex_vehicle_summary: torch.Tensor,
        context_features: torch.Tensor,
        hex_allocations: torch.Tensor,
        charge_power: torch.Tensor,
        adjacency: torch.Tensor,
        vehicle_counts: Optional[torch.Tensor] = None,
        hex_embeddings_q1: Optional[torch.Tensor] = None,
        hex_embeddings_q2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through both critics.

        Returns:
            q1, q2: [B] Q-values from both critics.
        """
        q1 = self.critic1(
            hex_features, hex_vehicle_summary, context_features,
            hex_allocations, charge_power, adjacency,
            vehicle_counts=vehicle_counts, hex_embeddings=hex_embeddings_q1,
        )
        q2 = self.critic2(
            hex_features, hex_vehicle_summary, context_features,
            hex_allocations, charge_power, adjacency,
            vehicle_counts=vehicle_counts, hex_embeddings=hex_embeddings_q2,
        )
        return q1, q2

    def min_q(
        self,
        hex_features: torch.Tensor,
        hex_vehicle_summary: torch.Tensor,
        context_features: torch.Tensor,
        hex_allocations: torch.Tensor,
        charge_power: torch.Tensor,
        adjacency: torch.Tensor,
        vehicle_counts: Optional[torch.Tensor] = None,
        hex_embeddings_q1: Optional[torch.Tensor] = None,
        hex_embeddings_q2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get minimum Q-value from twin critics."""
        q1, q2 = self.forward(
            hex_features, hex_vehicle_summary, context_features,
            hex_allocations, charge_power, adjacency,
            vehicle_counts=vehicle_counts,
            hex_embeddings_q1=hex_embeddings_q1,
            hex_embeddings_q2=hex_embeddings_q2,
        )
        return torch.min(q1, q2)
