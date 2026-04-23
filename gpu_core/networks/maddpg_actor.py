"""
maddpg_actor.py — MADDPG Decentralized Actor (NIPS 2017).

Each agent i has an actor μ_i that maps LOCAL observation o_i → deterministic action.

For DISCRETE action space (SERVE/CHARGE/REPOSITION + hex target + trip):
  - Forward pass returns Gumbel-Softmax "soft" actions for critic backprop.
  - act() returns hard argmax actions for environment stepping.

Architecture follows the paper (Lowe et al. 2017) with hierarchical discrete heads:
  action_type → SERVE (trip selection) / CHARGE / REPOSITION (hex target)

Parameter sharing: all N vehicles share the same actor weights (weight sharing).
During execution each vehicle i provides its local obs [vehicle_features_i, context].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class MADDPGActor(nn.Module):
    """MADDPG decentralized actor."""

    def __init__(
        self,
        vehicle_feature_dim: int,
        context_dim: int,
        num_hexes: int,
        hidden_dims: List[int] = [256, 256],
        action_dim: int = 3,
        max_trips: int = 2000,
        dropout: float = 0.1,
        gumbel_tau: float = 1.0,
        max_k_neighbors: int = 61,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.num_hexes = num_hexes
        self.max_trips = max_trips
        self.gumbel_tau = gumbel_tau
        self.max_k_neighbors = max_k_neighbors

        local_obs_dim = vehicle_feature_dim + context_dim

        layers = []
        in_dim = local_obs_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        hidden_dim = hidden_dims[-1]

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.reposition_head = nn.Linear(hidden_dim, max_k_neighbors)
        self.trip_head = nn.Linear(hidden_dim, max_trips)

    def _encode(
        self,
        vehicle_feature: torch.Tensor,
        context_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Return hidden tensor [B, hidden_dim]."""
        local_obs = torch.cat([vehicle_feature, context_feature], dim=-1)
        return self.backbone(local_obs)

    @staticmethod
    def _gumbel_softmax(
        logits: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

    @staticmethod
    def _resolve_khop_inputs(
        khop_neighbor_indices: Optional[torch.Tensor],
        khop_neighbor_mask: Optional[torch.Tensor],
        vehicle_hex_ids: Optional[torch.Tensor],
        batch_agents: int,
        K: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if khop_neighbor_indices is None or vehicle_hex_ids is None:
            return None, None

        if vehicle_hex_ids.dim() == 2:
            vehicle_hex_ids = vehicle_hex_ids.reshape(-1)
        elif vehicle_hex_ids.dim() != 1:
            raise ValueError(f'vehicle_hex_ids must be rank-1 or rank-2, got dim={vehicle_hex_ids.dim()}')

        if vehicle_hex_ids.numel() != batch_agents:
            raise ValueError(
                f'vehicle_hex_ids length mismatch: got {vehicle_hex_ids.numel()}, expected {batch_agents}'
            )

        gather_idx = vehicle_hex_ids.to(torch.long)
        cand_abs = khop_neighbor_indices[gather_idx]
        if cand_abs.shape[-1] > K:
            cand_abs = cand_abs[..., :K]
        elif cand_abs.shape[-1] < K:
            pad = torch.full((batch_agents, K - cand_abs.shape[-1]), -1, dtype=torch.long, device=cand_abs.device)
            cand_abs = torch.cat([cand_abs, pad], dim=-1)

        if khop_neighbor_mask is not None:
            cand_mask = khop_neighbor_mask[gather_idx]
            if cand_mask.shape[-1] > K:
                cand_mask = cand_mask[..., :K]
            elif cand_mask.shape[-1] < K:
                pad = torch.zeros((batch_agents, K - cand_mask.shape[-1]), dtype=torch.bool, device=cand_mask.device)
                cand_mask = torch.cat([cand_mask, pad], dim=-1)
        else:
            cand_mask = cand_abs >= 0

        return cand_abs.to(device=device, dtype=torch.long), cand_mask.to(device=device, dtype=torch.bool)

    def act(
        self,
        vehicle_feature: torch.Tensor,
        context_feature: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        khop_neighbor_indices: Optional[torch.Tensor] = None,
        khop_neighbor_mask: Optional[torch.Tensor] = None,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if vehicle_feature.dim() == 1:
            vehicle_feature = vehicle_feature.unsqueeze(0)
            context_feature = context_feature.unsqueeze(0)

        h = self._encode(vehicle_feature, context_feature)

        action_logits = self.action_head(h)
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=20.0, neginf=-20.0)

        if deterministic:
            action_type = action_logits.argmax(dim=-1)
        else:
            action_type = F.gumbel_softmax(
                action_logits,
                tau=self.gumbel_tau,
                hard=True,
                dim=-1,
            ).argmax(dim=-1)

        repos_logits = self.reposition_head(h)
        repos_logits = torch.nan_to_num(repos_logits, nan=0.0, posinf=20.0, neginf=-20.0)

        cand_abs, cand_mask = self._resolve_khop_inputs(
            khop_neighbor_indices=khop_neighbor_indices,
            khop_neighbor_mask=khop_neighbor_mask,
            vehicle_hex_ids=vehicle_hex_ids,
            batch_agents=repos_logits.shape[0],
            K=self.max_k_neighbors,
            device=repos_logits.device,
        )
        if cand_abs is None:
            fallback_abs = torch.arange(self.max_k_neighbors, device=repos_logits.device, dtype=torch.long)
            fallback_abs = fallback_abs.clamp(max=self.num_hexes - 1)
            cand_abs = fallback_abs.unsqueeze(0).expand(repos_logits.shape[0], -1)
            cand_mask = torch.ones_like(cand_abs, dtype=torch.bool)

        slot_mask = cand_mask
        if reposition_mask is not None:
            rm_abs = reposition_mask.to(dtype=torch.bool)
            cand_valid = cand_abs >= 0
            rm_gather = torch.zeros_like(slot_mask, dtype=torch.bool)
            if cand_valid.any():
                rm_gather = rm_abs.gather(1, cand_abs.clamp(min=0))
            slot_mask = slot_mask.to(dtype=torch.bool) & rm_gather & cand_valid

        repos_logits = repos_logits.masked_fill(~slot_mask, float('-inf'))
        repos_lsm = F.log_softmax(repos_logits, dim=-1)
        repos_probs = repos_lsm.exp()

        all_masked_r = repos_probs.isnan().any(dim=-1) | (repos_probs.sum(dim=-1) < 1e-8)
        if all_masked_r.any():
            uniform_lp = -torch.log(torch.tensor(float(self.max_k_neighbors), device=repos_logits.device))
            repos_lsm = torch.where(
                all_masked_r.unsqueeze(-1),
                torch.full_like(repos_lsm, uniform_lp),
                repos_lsm,
            )
            repos_probs = torch.where(
                all_masked_r.unsqueeze(-1),
                torch.full_like(repos_probs, 1.0 / self.max_k_neighbors),
                repos_probs,
            )

        if deterministic:
            reposition_slot = repos_probs.argmax(dim=-1)
        else:
            reposition_slot = F.gumbel_softmax(
                repos_lsm,
                tau=self.gumbel_tau,
                hard=True,
                dim=-1,
            ).argmax(dim=-1)

        reposition_target = cand_abs.gather(1, reposition_slot.unsqueeze(-1)).squeeze(-1).clamp(min=0)

        max_active = self.max_trips
        if trip_mask is not None and trip_mask.any():
            max_active = max(1, int(trip_mask.nonzero(as_tuple=False)[:, -1].max().item()) + 1)
            max_active = min(max_active, self.max_trips)

        w = self.trip_head.weight[:max_active, :]
        b = self.trip_head.bias[:max_active] if self.trip_head.bias is not None else None
        trip_logits = F.linear(h, w, b)

        if trip_mask is not None:
            tm = trip_mask[..., :max_active]
            if tm.dim() == 1:
                trip_logits = trip_logits.masked_fill(~tm.unsqueeze(0).expand_as(trip_logits), float('-inf'))
            else:
                trip_logits = trip_logits.masked_fill(~tm, float('-inf'))
        trip_logits = torch.nan_to_num(trip_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        selected_trip = trip_logits.argmax(dim=-1)

        return action_type, reposition_target, selected_trip

    def forward(
        self,
        vehicle_feature: torch.Tensor,
        context_feature: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        khop_neighbor_indices: Optional[torch.Tensor] = None,
        khop_neighbor_mask: Optional[torch.Tensor] = None,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if vehicle_feature.dim() == 1:
            vehicle_feature = vehicle_feature.unsqueeze(0)
            context_feature = context_feature.unsqueeze(0)

        tau = tau if tau is not None else self.gumbel_tau

        h = self._encode(vehicle_feature, context_feature)

        action_logits = self.action_head(h)
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        action_type_soft = self._gumbel_softmax(action_logits, tau=tau)
        action_type = action_type_soft.argmax(dim=-1)

        repos_logits = self.reposition_head(h)
        repos_logits = torch.nan_to_num(repos_logits, nan=0.0, posinf=20.0, neginf=-20.0)

        cand_abs, cand_mask = self._resolve_khop_inputs(
            khop_neighbor_indices=khop_neighbor_indices,
            khop_neighbor_mask=khop_neighbor_mask,
            vehicle_hex_ids=vehicle_hex_ids,
            batch_agents=repos_logits.shape[0],
            K=self.max_k_neighbors,
            device=repos_logits.device,
        )
        if cand_abs is None:
            fallback_abs = torch.arange(self.max_k_neighbors, device=repos_logits.device, dtype=torch.long)
            fallback_abs = fallback_abs.clamp(max=self.num_hexes - 1)
            cand_abs = fallback_abs.unsqueeze(0).expand(repos_logits.shape[0], -1)
            cand_mask = torch.ones_like(cand_abs, dtype=torch.bool)

        slot_mask = cand_mask
        if reposition_mask is not None:
            rm_abs = reposition_mask.to(dtype=torch.bool)
            cand_valid = cand_abs >= 0
            rm_gather = torch.zeros_like(slot_mask, dtype=torch.bool)
            if cand_valid.any():
                rm_gather = rm_abs.gather(1, cand_abs.clamp(min=0))
            slot_mask = slot_mask.to(dtype=torch.bool) & rm_gather & cand_valid

        reposition_soft = self._gumbel_softmax(repos_logits, tau=tau, mask=slot_mask)
        reposition_slot = reposition_soft.argmax(dim=-1)
        reposition_target = cand_abs.gather(1, reposition_slot.unsqueeze(-1)).squeeze(-1).clamp(min=0)

        max_active = self.max_trips
        if trip_mask is not None and trip_mask.any():
            max_active = max(1, int(trip_mask.nonzero(as_tuple=False)[:, -1].max().item()) + 1)
            max_active = min(max_active, self.max_trips)

        w = self.trip_head.weight[:max_active, :]
        b = self.trip_head.bias[:max_active] if self.trip_head.bias is not None else None
        trip_logits = F.linear(h, w, b)

        tm_mask = None
        if trip_mask is not None:
            tm_mask = trip_mask[..., :max_active]

        trip_soft = self._gumbel_softmax(trip_logits, tau=tau, mask=tm_mask)
        selected_trip = trip_soft.argmax(dim=-1)

        return {
            'action_type_soft': action_type_soft,
            'action_type': action_type,
            'reposition_soft': reposition_soft,
            'reposition_target': reposition_target,
            'trip_soft': trip_soft,
            'selected_trip': selected_trip,
            'hidden': h,
        }
