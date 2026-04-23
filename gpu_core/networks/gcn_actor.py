"""
GCN-based Actor per paper Section 5.1.

Implements:
- Two-layer GCN encoder over hex graph (paper Eq. 15)
- Context encoder for vehicle features
- Masked softmax with temperature annealing (paper Eq. 16)
- Continuous charging power head (paper Eq. 17-18)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Optional, Tuple, List, Dict

from .gcn import GCNEncoder


class FleetGCNActor(nn.Module):
    """Fleet-level actor: outputs per-hex action allocations + continuous charging power.

    Instead of per-vehicle decisions, this actor outputs:
      - allocation_probs [B, H, 3]: P(serve), P(charge), P(repos) per hex
      - repos_target_probs [B, H, max_K]: reposition target distribution per hex
      - charge_power [B, H]: continuous charging power in (0,1) via squashed Gaussian

    A deterministic HexVehicleAssigner maps these to per-vehicle actions.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        num_hexes: int,
        hex_feature_dim: int = 5,
        hex_vehicle_agg_dim: int = 8,
        context_dim: int = 9,
        gcn_hidden_dim: int = 128,
        gcn_output_dim: int = 64,
        hex_decision_hidden_dim: int = 256,
        action_dim: int = 3,          # SERVE=0, CHARGE=1, REPOSITION=2
        max_K_neighbors: int = 61,    # From K=4 hop neighborhood
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_hexes = num_hexes
        self.action_dim = action_dim
        self.gcn_output_dim = gcn_output_dim
        self.max_K_neighbors = max_K_neighbors

        # ===== GCN Encoder (same as per-vehicle actor) =====
        self.gcn = GCNEncoder(
            input_dim=hex_feature_dim,
            hidden_dims=[gcn_hidden_dim],
            output_dim=gcn_output_dim,
            dropout=dropout,
            use_batch_norm=True,
            activation='silu',
        )

        # ===== Hex Decision Encoder =====
        # Input: hex_embedding (64) + hex_vehicle_summary (8) + context (9) = 81
        encoder_input_dim = gcn_output_dim + hex_vehicle_agg_dim + context_dim
        self.hex_decision_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hex_decision_hidden_dim),
            nn.LayerNorm(hex_decision_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hex_decision_hidden_dim, hex_decision_hidden_dim),
            nn.LayerNorm(hex_decision_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ===== Action Allocation Head =====
        # Per-hex: logits for [SERVE, CHARGE, REPOSITION]
        self.allocation_head = nn.Linear(hex_decision_hidden_dim, action_dim)

        # ===== Reposition Target Head =====
        # Per-hex: logits over K-hop neighbors
        self.repos_target_head = nn.Linear(hex_decision_hidden_dim, max_K_neighbors)

        # ===== Charging Power Head (Squashed Gaussian) =====
        # Outputs (mu, log_sigma) per hex for continuous power in (0, 1)
        self.charge_power_head = nn.Sequential(
            nn.Linear(hex_decision_hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),  # (mu, log_sigma)
        )

        # Store adjacency for GCN
        self._adj_cache = None

    def set_adjacency(self, adj: torch.Tensor):
        """Set adjacency matrix for GCN. Expects pre-normalized adjacency (sparse or dense)."""
        self._adj_cache = adj

    def _sample_squashed_gaussian(
        self,
        mu: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from squashed Gaussian: sigmoid(z), z ~ N(mu, sigma).

        Returns:
            power: in (0, 1) — continuous charging power fraction.
            log_prob: Jacobian-corrected log probability.
        """
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            z = mu
        else:
            eps = torch.randn_like(mu)
            z = mu + std * eps

        power = torch.sigmoid(z)  # in (0, 1)

        # Log prob: log N(z; mu, sigma) - log |d sigmoid / dz|
        # d sigmoid/dz = sigmoid(z) * (1 - sigmoid(z)) = power * (1 - power)
        log_prob_z = (
            -0.5 * ((z - mu) / (std + 1e-8)).pow(2)
            - log_std
            - 0.5 * torch.log(torch.tensor(2.0 * 3.141592653589793, device=mu.device))
        )
        # Jacobian correction for sigmoid squashing
        log_prob = log_prob_z - torch.log(power * (1.0 - power) + 1e-6)

        return power, log_prob

    def forward(
        self,
        hex_features: torch.Tensor,           # [B, H, hex_feature_dim]
        hex_vehicle_summary: torch.Tensor,     # [B, H, hex_vehicle_agg_dim]
        context_features: torch.Tensor,        # [B, context_dim]
        adj: Optional[torch.Tensor] = None,    # [H, H] normalized adjacency
        active_hex_mask: Optional[torch.Tensor] = None,  # [B, H] bool: hexes with idle vehicles
        khop_neighbor_indices: Optional[torch.Tensor] = None,  # [H, max_K] padded with -1
        khop_neighbor_mask: Optional[torch.Tensor] = None,     # [H, max_K] bool: valid neighbors
        temperature: float = 1.0,
        deterministic: bool = False,
        hex_embeddings: Optional[torch.Tensor] = None,  # [B, H, gcn_output_dim] pre-computed
        compute_repos: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Fleet-level forward pass.

        Args:
            hex_features: Spatial hex features for GCN.
            hex_vehicle_summary: Pre-computed vehicle aggregates per hex.
            context_features: Global context vector.
            adj: Normalized adjacency (uses cached if None).
            active_hex_mask: Which hexes have idle vehicles to assign.
            khop_neighbor_indices: [H, max_K] K-hop neighbor hex indices (-1 padded).
            khop_neighbor_mask: [H, max_K] bool mask for valid neighbors.
            temperature: Softmax temperature for exploration.
            deterministic: If True, use argmax / mean instead of sampling.

        Returns:
            Dict with allocation_probs, repos_target_probs, charge_power, etc.
        """
        # Handle unbatched input
        squeeze_output = False
        if hex_features.dim() == 2:
            hex_features = hex_features.unsqueeze(0)
            hex_vehicle_summary = hex_vehicle_summary.unsqueeze(0)
            context_features = context_features.unsqueeze(0)
            if active_hex_mask is not None and active_hex_mask.dim() == 1:
                active_hex_mask = active_hex_mask.unsqueeze(0)
            squeeze_output = True

        B, H = hex_features.shape[0], hex_features.shape[1]
        adj_matrix = adj if adj is not None else self._adj_cache

        # ===== 1. GCN Encoding =====
        # Use pre-computed embeddings if provided (avoids redundant passes within one training step)
        if hex_embeddings is None:
            hex_embeddings = self.gcn(hex_features, adj_matrix)  # [B, H, 64]

        # ===== 2. Build Hex Decision Input =====
        context_expanded = context_features.unsqueeze(1).expand(-1, H, -1)  # [B, H, 9]
        hex_input = torch.cat([
            hex_embeddings,          # [B, H, 64]
            hex_vehicle_summary,     # [B, H, 8]
            context_expanded,        # [B, H, 9]
        ], dim=-1)  # [B, H, 81]

        # ===== 3. Hex Decision Encoder =====
        hex_context = self.hex_decision_encoder(hex_input)  # [B, H, 256]

        # ===== 4. Action Allocation Head =====
        alloc_logits = self.allocation_head(hex_context)  # [B, H, 3]
        alloc_logits = alloc_logits / max(temperature, 1e-6)

        # Mask inactive hexes (no idle vehicles)
        if active_hex_mask is not None:
            inactive = ~active_hex_mask  # [B, H]
            alloc_logits = alloc_logits.masked_fill(inactive.unsqueeze(-1), float('-inf'))

        # Stable softmax
        alloc_log_probs = F.log_softmax(alloc_logits, dim=-1)  # [B, H, 3]
        allocation_probs = alloc_log_probs.exp()                # [B, H, 3]

        # Fully masked rows can produce NaNs; zero them out explicitly.
        if active_hex_mask is not None:
            inactive_rows = ~active_hex_mask
            alloc_log_probs = torch.where(
                inactive_rows.unsqueeze(-1),
                torch.zeros_like(alloc_log_probs),
                torch.nan_to_num(alloc_log_probs, nan=0.0, neginf=0.0, posinf=0.0),
            )
            allocation_probs = torch.where(
                inactive_rows.unsqueeze(-1),
                torch.zeros_like(allocation_probs),
                torch.nan_to_num(allocation_probs, nan=0.0, neginf=0.0, posinf=0.0),
            )
        else:
            alloc_log_probs = torch.nan_to_num(alloc_log_probs, nan=0.0, neginf=0.0, posinf=0.0)
            allocation_probs = torch.nan_to_num(allocation_probs, nan=0.0, neginf=0.0, posinf=0.0)

        # Per-hex entropy: H = -sum p*log(p)
        allocation_entropy = -(allocation_probs * alloc_log_probs).sum(dim=-1)  # [B, H]
        # Zero entropy for inactive hexes
        if active_hex_mask is not None:
            allocation_entropy = allocation_entropy * active_hex_mask.float()

        # ===== 5. Reposition Target Head =====
        if compute_repos:
            repos_logits = self.repos_target_head(hex_context)  # [B, H, max_K]

            # Mask invalid neighbors
            if khop_neighbor_mask is not None:
                invalid_neighbors = ~khop_neighbor_mask  # [H, max_K]
                if invalid_neighbors.dim() == 2:
                    invalid_neighbors = invalid_neighbors.unsqueeze(0).expand(B, -1, -1)
                repos_logits = repos_logits.masked_fill(invalid_neighbors, float('-inf'))

            repos_log_probs = F.log_softmax(repos_logits, dim=-1)  # [B, H, max_K]
            repos_target_probs = repos_log_probs.exp()              # [B, H, max_K]

            if khop_neighbor_mask is not None:
                valid_neighbor_rows = khop_neighbor_mask
                if valid_neighbor_rows.dim() == 2:
                    valid_neighbor_rows = valid_neighbor_rows.unsqueeze(0).expand(B, -1, -1)
                valid_any = valid_neighbor_rows.any(dim=-1, keepdim=True)
                # Use -1e9 for neginf so masked (invalid) neighbors stay unattractive
                # during Gumbel sampling. Using 0.0 made them the MOST attractive targets.
                repos_log_probs = torch.where(
                    valid_any,
                    torch.nan_to_num(repos_log_probs, nan=0.0, neginf=-1e9, posinf=0.0),
                    torch.zeros_like(repos_log_probs),
                )
                repos_target_probs = torch.where(
                    valid_any,
                    torch.nan_to_num(repos_target_probs, nan=0.0, neginf=0.0, posinf=0.0),
                    torch.zeros_like(repos_target_probs),
                )
            else:
                repos_log_probs = torch.nan_to_num(repos_log_probs, nan=0.0, neginf=-1e9, posinf=0.0)
                repos_target_probs = torch.nan_to_num(repos_target_probs, nan=0.0, neginf=0.0, posinf=0.0)

            # Sample or argmax reposition targets
            if deterministic:
                repos_sampled_idx = repos_target_probs.argmax(dim=-1)  # [B, H]
            else:
                # Gumbel-max sampling
                u = torch.rand_like(repos_log_probs).clamp(1e-8, 1.0 - 1e-8)
                gumbel = -torch.log(-torch.log(u))
                repos_sampled_idx = (repos_log_probs + gumbel).argmax(dim=-1)  # [B, H]

            # Convert K-hop index to actual hex index
            if khop_neighbor_indices is not None:
                kni = khop_neighbor_indices.unsqueeze(0).expand(B, -1, -1)  # [B, H, max_K]
                repos_sampled_targets = kni.gather(
                    2, repos_sampled_idx.unsqueeze(-1)
                ).squeeze(-1)  # [B, H]
            else:
                repos_sampled_targets = repos_sampled_idx

            # Log prob of sampled repos target
            repos_target_log_prob = repos_log_probs.gather(
                2, repos_sampled_idx.unsqueeze(-1)
            ).squeeze(-1)  # [B, H]
        else:
            # Skip repos computation to save ~2.4 GB of autograd saved tensors
            repos_target_probs = torch.zeros(B, H, self.max_K_neighbors, device=hex_features.device)
            repos_log_probs = torch.zeros_like(repos_target_probs)
            repos_sampled_targets = torch.zeros(B, H, dtype=torch.long, device=hex_features.device)
            repos_target_log_prob = torch.zeros(B, H, device=hex_features.device)

        # ===== 6. Charging Power Head (Squashed Gaussian) =====
        power_params = self.charge_power_head(hex_context)  # [B, H, 2]
        power_mu = power_params[..., 0]       # [B, H]
        power_log_std = power_params[..., 1]  # [B, H]

        charge_power, charge_power_log_prob = self._sample_squashed_gaussian(
            power_mu, power_log_std, deterministic=deterministic
        )  # both [B, H]

        # ===== Build result =====
        result: Dict[str, torch.Tensor] = {
            'allocation_probs': allocation_probs,             # [B, H, 3]
            'allocation_log_probs': alloc_log_probs,          # [B, H, 3]
            'repos_target_probs': repos_target_probs,         # [B, H, max_K]
            'repos_target_log_probs': repos_target_log_prob,  # [B, H]
            'repos_sampled_targets': repos_sampled_targets,   # [B, H] actual hex indices
            'charge_power': charge_power,                     # [B, H] in (0, 1)
            'charge_power_log_prob': charge_power_log_prob,   # [B, H]
            'hex_embeddings': hex_embeddings,                 # [B, H, 64]
            'hex_context': hex_context,                       # [B, H, 256]
            'active_hex_mask': active_hex_mask if active_hex_mask is not None
                               else torch.ones(B, H, dtype=torch.bool, device=hex_features.device),
            'allocation_entropy': allocation_entropy,         # [B, H]
        }

        if squeeze_output:
            for key in result:
                if isinstance(result[key], torch.Tensor) and result[key].dim() > 1:
                    result[key] = result[key].squeeze(0)

        return result
