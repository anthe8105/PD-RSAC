"""
mappo_actor.py — Standard MAPPO Actor (MLP, parameter sharing).

Standard MAPPO design:
- Each agent acts on its LOCAL observation only: [vehicle_features_i, context_features]
- All agents share one MLP (parameter sharing via batched processing [B*V, local_obs_dim])
- No GCN, no hex features, no global mean pooling → strict CTDE compliance
- Hierarchical action: action_type → reposition_target (REPOSITION) or trip (SERVE)
- Masking preserved for environment constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class MAPPOActor(nn.Module):
    """
    Standard MAPPO Actor — shared MLP on per-agent local observation.

    Local observation per agent i:
        [vehicle_features_i, context_features]  →  shape [B, V, vehicle_dim + context_dim]

    All V agents share the same MLP weights (parameter sharing).
    Processing is fully batched: reshape to [B*V, local_obs_dim] → MLP → reshape back.
    """

    def __init__(
        self,
        vehicle_feature_dim: int,
        context_dim: int,
        hex_feature_dim: int,
        num_hexes: int,
        hidden_dims: List[int] = [256, 256],
        action_dim: int = 3,
        dropout: float = 0.1,
        max_trips: int = 2000,
        use_trip_head: bool = False,
        learn_charge_power: bool = False,
        max_k_neighbors: int = 61,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.num_hexes = num_hexes
        self.max_trips = max_trips
        self.use_trip_head = use_trip_head
        self.learn_charge_power = learn_charge_power
        self.hex_feature_dim = hex_feature_dim
        self.max_k_neighbors = max_k_neighbors

        local_obs_dim = vehicle_feature_dim + context_dim + hex_feature_dim

        layers = []
        in_dim = local_obs_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.Tanh(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        hidden_dim = hidden_dims[-1]

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.reposition_head = nn.Linear(hidden_dim, max_k_neighbors)
        self.trip_head = nn.Linear(hidden_dim, max_trips) if use_trip_head else None
        self.charge_power_head = nn.Linear(hidden_dim, 1) if learn_charge_power else None

    def _encode(
        self,
        vehicle_features: torch.Tensor,
        context_features: torch.Tensor,
        hex_features: torch.Tensor,
        reposition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return encoded hidden features [B, V, hidden_dim]."""
        B, V, _ = vehicle_features.shape
        _, H, _ = hex_features.shape
        ctx = context_features.unsqueeze(1).expand(-1, V, -1)

        local_hex_ctx = None
        if reposition_mask is not None:
            if reposition_mask.dim() == 2:
                if reposition_mask.shape[0] != V:
                    raise ValueError(
                        f"reposition_mask vehicle dim mismatch: got {reposition_mask.shape[0]}, expected {V}"
                    )
                rm = reposition_mask.unsqueeze(0).expand(B, -1, -1)
            elif reposition_mask.dim() == 3:
                if reposition_mask.shape[1] != V:
                    raise ValueError(
                        f"reposition_mask vehicle dim mismatch: got {reposition_mask.shape[1]}, expected {V}"
                    )
                if reposition_mask.shape[0] == B:
                    rm = reposition_mask
                elif reposition_mask.shape[0] == 1:
                    rm = reposition_mask.expand(B, -1, -1)
                else:
                    raise ValueError(
                        f"reposition_mask batch dim mismatch: got {reposition_mask.shape[0]}, expected {B}"
                    )
            else:
                raise ValueError(f"reposition_mask must have dim 2 or 3, got {reposition_mask.dim()}")

            if rm.shape[-1] > H:
                rm = rm[..., :H]
            elif rm.shape[-1] < H:
                pad_shape = list(rm.shape)
                pad_shape[-1] = H - rm.shape[-1]
                rm = torch.cat([rm, torch.zeros(pad_shape, dtype=torch.bool, device=rm.device)], dim=-1)

            rm_f = rm.float()
            neighbor_count = rm_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            local_hex_ctx = torch.matmul(rm_f, hex_features) / neighbor_count

        if local_hex_ctx is None:
            hex_summary = hex_features.mean(dim=1)
            local_hex_ctx = hex_summary.unsqueeze(1).expand(-1, V, -1)

        local_obs = torch.cat([vehicle_features, ctx, local_hex_ctx], dim=-1)
        flat = local_obs.reshape(B * V, -1)
        encoded = self.backbone(flat)
        return encoded.reshape(B, V, -1)

    @staticmethod
    def _resolve_khop_inputs(
        khop_neighbor_indices: Optional[torch.Tensor],
        khop_neighbor_mask: Optional[torch.Tensor],
        vehicle_hex_ids: Optional[torch.Tensor],
        B: int,
        V: int,
        K: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if khop_neighbor_indices is None or vehicle_hex_ids is None:
            return None, None

        if vehicle_hex_ids.dim() == 1:
            vehicle_hex_ids = vehicle_hex_ids.unsqueeze(0).expand(B, -1)
        elif vehicle_hex_ids.dim() == 2 and vehicle_hex_ids.shape[0] == 1 and B > 1:
            vehicle_hex_ids = vehicle_hex_ids.expand(B, -1)

        if vehicle_hex_ids.shape[0] != B or vehicle_hex_ids.shape[1] != V:
            raise ValueError(
                f"vehicle_hex_ids shape mismatch: got {tuple(vehicle_hex_ids.shape)}, expected ({B}, {V})"
            )

        gather_idx = vehicle_hex_ids.to(torch.long)
        cand_abs = khop_neighbor_indices[gather_idx]
        if cand_abs.shape[-1] > K:
            cand_abs = cand_abs[..., :K]
        elif cand_abs.shape[-1] < K:
            pad = torch.full((B, V, K - cand_abs.shape[-1]), -1, dtype=torch.long, device=cand_abs.device)
            cand_abs = torch.cat([cand_abs, pad], dim=-1)

        if khop_neighbor_mask is not None:
            cand_mask = khop_neighbor_mask[gather_idx]
            if cand_mask.shape[-1] > K:
                cand_mask = cand_mask[..., :K]
            elif cand_mask.shape[-1] < K:
                pad = torch.zeros((B, V, K - cand_mask.shape[-1]), dtype=torch.bool, device=cand_mask.device)
                cand_mask = torch.cat([cand_mask, pad], dim=-1)
        else:
            cand_mask = cand_abs >= 0

        cand_abs = cand_abs.to(device=device, dtype=torch.long)
        cand_mask = cand_mask.to(device=device, dtype=torch.bool)
        return cand_abs, cand_mask

    def forward(
        self,
        vehicle_features: torch.Tensor,
        context_features: torch.Tensor,
        hex_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        eval_action_type: Optional[torch.Tensor] = None,
        eval_reposition_target: Optional[torch.Tensor] = None,
        eval_selected_trip: Optional[torch.Tensor] = None,
        khop_neighbor_indices: Optional[torch.Tensor] = None,
        khop_neighbor_mask: Optional[torch.Tensor] = None,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute action distributions and either sample or evaluate given actions."""
        squeeze_output = False
        if vehicle_features.dim() == 2:
            vehicle_features = vehicle_features.unsqueeze(0)
            context_features = context_features.unsqueeze(0)
            hex_features = hex_features.unsqueeze(0)
            squeeze_output = True

        B, V, _ = vehicle_features.shape
        device = vehicle_features.device

        encoded = self._encode(
            vehicle_features,
            context_features,
            hex_features,
            reposition_mask=reposition_mask,
        )

        action_logits = self.action_head(encoded)
        action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=20.0, neginf=-20.0)

        if action_mask is not None:
            if action_mask.dim() == 2:
                action_mask = action_mask.unsqueeze(0).expand(B, -1, -1)
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))

        action_lsm = F.log_softmax(action_logits, dim=-1)
        action_probs = action_lsm.exp()

        all_masked = action_probs.isnan().any(dim=-1) | (action_probs.sum(dim=-1) < 1e-8)
        if all_masked.any():
            k = float(self.action_dim)
            fb_lsm = torch.full_like(action_lsm, -torch.log(torch.tensor(k, device=device)))
            fb_p = torch.ones_like(action_probs) / k
            action_lsm = torch.where(all_masked.unsqueeze(-1), fb_lsm, action_lsm)
            action_probs = torch.where(all_masked.unsqueeze(-1), fb_p, action_probs)

        if eval_action_type is not None:
            action_type = eval_action_type if eval_action_type.dim() == 2 else eval_action_type.unsqueeze(0)
        elif deterministic:
            action_type = action_probs.argmax(dim=-1)
        else:
            u = torch.rand_like(action_lsm)
            gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            action_type = (action_lsm + gumbel).argmax(dim=-1)

        action_log_prob = torch.gather(action_lsm, -1, action_type.unsqueeze(-1)).squeeze(-1)
        action_log_prob = action_log_prob.clamp(min=-20.0, max=0.0)
        action_entropy = -(action_probs * action_lsm.clamp(min=-20.0)).sum(dim=-1)

        reposition_target = torch.zeros(B, V, dtype=torch.long, device=device)
        reposition_log_prob = torch.zeros(B, V, dtype=torch.float32, device=device)
        reposition_entropy = torch.zeros(B, V, dtype=torch.float32, device=device)

        candidate_abs, candidate_mask = self._resolve_khop_inputs(
            khop_neighbor_indices=khop_neighbor_indices,
            khop_neighbor_mask=khop_neighbor_mask,
            vehicle_hex_ids=vehicle_hex_ids,
            B=B,
            V=V,
            K=self.max_k_neighbors,
            device=device,
        )
        if candidate_abs is None:
            fallback_abs = torch.arange(self.max_k_neighbors, device=device, dtype=torch.long)
            fallback_abs = fallback_abs.clamp(max=self.num_hexes - 1)
            candidate_abs = fallback_abs.unsqueeze(0).unsqueeze(0).expand(B, V, -1)
            candidate_mask = torch.ones(B, V, self.max_k_neighbors, dtype=torch.bool, device=device)

        repos_mask_action = (action_type == 2)
        if repos_mask_action.any():
            flat_idx = repos_mask_action.nonzero(as_tuple=False)
            b_idx, v_idx = flat_idx[:, 0], flat_idx[:, 1]

            r_logits = self.reposition_head(encoded[b_idx, v_idx])
            r_logits = torch.nan_to_num(r_logits, nan=0.0, posinf=20.0, neginf=-20.0)

            r_slot_mask = candidate_mask[b_idx, v_idx]
            if reposition_mask is not None:
                if reposition_mask.dim() == 2:
                    rm_abs = reposition_mask[v_idx]
                elif reposition_mask.dim() == 3:
                    rm_abs = reposition_mask[b_idx, v_idx]
                else:
                    rm_abs = None
                if rm_abs is not None:
                    cand_abs = candidate_abs[b_idx, v_idx]
                    valid_abs = cand_abs >= 0
                    rm_gather = torch.zeros_like(r_slot_mask)
                    if valid_abs.any():
                        gather_src = rm_abs.to(dtype=torch.bool)
                        gather_idx = cand_abs.clamp(min=0)
                        rm_gather = gather_src.gather(1, gather_idx)
                    r_slot_mask = r_slot_mask.to(dtype=torch.bool) & rm_gather.to(dtype=torch.bool) & valid_abs

            r_logits = r_logits.masked_fill(~r_slot_mask, float('-inf'))
            r_lsm = F.log_softmax(r_logits, dim=-1)
            r_p = r_lsm.exp()

            all_masked_r = r_p.isnan().any(dim=-1) | (r_p.sum(dim=-1) < 1e-8)
            if all_masked_r.any():
                u_lp = -torch.log(torch.tensor(float(self.max_k_neighbors), device=device))
                r_lsm = torch.where(
                    all_masked_r.unsqueeze(-1),
                    torch.full_like(r_lsm, u_lp),
                    r_lsm,
                )
                r_p = torch.where(
                    all_masked_r.unsqueeze(-1),
                    torch.full_like(r_p, 1.0 / self.max_k_neighbors),
                    r_p,
                )

            if eval_reposition_target is not None:
                eval_abs = eval_reposition_target[b_idx, v_idx]
                cand_abs = candidate_abs[b_idx, v_idx]
                match_mask = cand_abs == eval_abs.unsqueeze(-1)
                has_match = match_mask.any(dim=-1)
                slot_from_eval = match_mask.float().argmax(dim=-1).long()
                fallback_slot = r_p.argmax(dim=-1)
                r_slot = torch.where(has_match, slot_from_eval, fallback_slot)
            elif deterministic:
                r_slot = r_p.argmax(dim=-1)
            else:
                u = torch.rand_like(r_lsm)
                gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
                r_slot = (r_lsm + gumbel).argmax(dim=-1)

            r_lp = torch.gather(r_lsm, -1, r_slot.unsqueeze(-1)).squeeze(-1).clamp(min=-20.0, max=0.0)
            r_ent = -(r_p * r_lsm.clamp(min=-20.0)).sum(dim=-1)

            mapped_abs = candidate_abs[b_idx, v_idx, r_slot].clamp(min=0)
            reposition_target[b_idx, v_idx] = mapped_abs
            reposition_log_prob[b_idx, v_idx] = r_lp
            reposition_entropy[b_idx, v_idx] = r_ent

        vehicle_charge_power = None
        if self.learn_charge_power:
            charge_power = torch.sigmoid(self.charge_power_head(encoded).squeeze(-1))
            charge_power = torch.nan_to_num(charge_power, nan=0.5, posinf=1.0, neginf=0.0)
            vehicle_charge_power = charge_power.clamp(min=0.0, max=1.0)

        if self.use_trip_head:
            selected_trip = torch.zeros(B, V, dtype=torch.long, device=device)
            trip_log_prob = torch.zeros(B, V, dtype=torch.float32, device=device)
            trip_entropy = torch.zeros(B, V, dtype=torch.float32, device=device)

            serve_mask_action = (action_type == 0)
            if serve_mask_action.any():
                flat_idx_s = serve_mask_action.nonzero(as_tuple=False)
                b_idx_s, v_idx_s = flat_idx_s[:, 0], flat_idx_s[:, 1]
                serve_hidden = encoded[b_idx_s, v_idx_s]

                max_active = 1
                if trip_mask is not None and trip_mask.any():
                    max_active = max(max_active, int(trip_mask.nonzero(as_tuple=False)[:, -1].max().item()) + 1)
                if eval_selected_trip is not None and eval_selected_trip.numel() > 0:
                    max_active = max(max_active, int(eval_selected_trip.max().item()) + 1)
                max_active = min(max_active, self.max_trips)

                w = self.trip_head.weight[:max_active, :]
                b_bias = self.trip_head.bias[:max_active] if self.trip_head.bias is not None else None
                t_logits = F.linear(serve_hidden, w, b_bias)
                t_logits = torch.nan_to_num(t_logits, nan=0.0, posinf=20.0, neginf=-20.0)

                if trip_mask is not None:
                    tm = trip_mask[..., :max_active]
                    if tm.dim() == 1:
                        t_logits = t_logits.masked_fill(~tm.unsqueeze(0).expand_as(t_logits), float('-inf'))
                    elif tm.dim() == 2:
                        idx = b_idx_s if tm.shape[0] == B else v_idx_s
                        t_logits = t_logits.masked_fill(~tm[idx], float('-inf'))

                t_lsm = F.log_softmax(t_logits, dim=-1)
                t_p = t_lsm.exp()

                all_masked_t = t_p.isnan().any(dim=-1) | (t_p.sum(dim=-1) < 1e-8)
                if all_masked_t.any():
                    u_lp = -torch.log(torch.tensor(float(max_active), device=device))
                    t_lsm = torch.where(
                        all_masked_t.unsqueeze(-1),
                        torch.full_like(t_lsm, u_lp),
                        t_lsm,
                    )
                    t_p = torch.where(
                        all_masked_t.unsqueeze(-1),
                        torch.full_like(t_p, 1.0 / max_active),
                        t_p,
                    )

                if eval_selected_trip is not None:
                    t_tgt = eval_selected_trip[b_idx_s, v_idx_s]
                elif deterministic:
                    t_tgt = t_p.argmax(dim=-1)
                else:
                    u = torch.rand_like(t_lsm)
                    gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
                    t_tgt = (t_lsm + gumbel).argmax(dim=-1)

                t_lp = torch.gather(t_lsm, -1, t_tgt.unsqueeze(-1)).squeeze(-1).clamp(min=-20.0, max=0.0)
                t_ent = -(t_p * t_lsm.clamp(min=-20.0)).sum(dim=-1)

                selected_trip[b_idx_s, v_idx_s] = t_tgt
                trip_log_prob[b_idx_s, v_idx_s] = t_lp
                trip_entropy[b_idx_s, v_idx_s] = t_ent
        else:
            selected_trip = torch.full((B, V), -1, dtype=torch.long, device=device)
            trip_log_prob = torch.zeros(B, V, dtype=torch.float32, device=device)
            trip_entropy = torch.zeros(B, V, dtype=torch.float32, device=device)

        result = {
            'action_type': action_type,
            'action_log_prob': action_log_prob,
            'action_entropy': action_entropy,
            'reposition_target': reposition_target,
            'reposition_log_prob': reposition_log_prob,
            'reposition_entropy': reposition_entropy,
            'selected_trip': selected_trip,
            'trip_log_prob': trip_log_prob,
            'trip_entropy': trip_entropy,
            'vehicle_charge_power': vehicle_charge_power,
        }

        if squeeze_output:
            for k, v in result.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    result[k] = v.squeeze(0)

        return result
