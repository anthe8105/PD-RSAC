import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import copy

from .gcn_actor import GCNActor, FleetGCNActor
from .critic import TwinCritic  # Legacy flat critic
from .gcn_critic import GCNTwinCritic, FleetGCNTwinCritic
from ..assignment import HexVehicleAssigner


@dataclass
class SACOutput:
    action_type: torch.Tensor
    reposition_target: torch.Tensor
    action_log_prob: torch.Tensor
    reposition_log_prob: torch.Tensor
    selected_trip: Optional[torch.Tensor] = None  # NEW: for GCN trip selection
    trip_log_prob: Optional[torch.Tensor] = None  # NEW: log prob of trip selection
    serve_scores: Optional[torch.Tensor] = None  # For assignment
    charge_scores: Optional[torch.Tensor] = None  # For assignment
    q1: Optional[torch.Tensor] = None
    q2: Optional[torch.Tensor] = None
    action_entropy: Optional[torch.Tensor] = None  # For GCN actor
    hex_embeddings: Optional[torch.Tensor] = None  # For GCN actor
    action_probs: Optional[torch.Tensor] = None         # [batch, V, 3] full distribution
    action_log_probs_all: Optional[torch.Tensor] = None  # [batch, V, 3] log probs all actions
    _encoded_context: Optional[torch.Tensor] = None      # [batch, V, hidden] for aux loss
    _global_hex_context: Optional[torch.Tensor] = None   # [batch, gcn_out] for aux loss


@dataclass
class FleetSACOutput:
    """Output from FleetSACAgent action selection."""
    # Per-vehicle (for simulator)
    action_type: torch.Tensor              # [V]
    reposition_target: torch.Tensor        # [V]
    vehicle_charge_power: torch.Tensor     # [V]
    # Per-hex (for replay buffer / training)
    allocation_probs: torch.Tensor         # [H, 3]
    allocation_log_probs: torch.Tensor     # [H, 3]
    repos_sampled_targets: torch.Tensor    # [H]
    charge_power: torch.Tensor             # [H]
    charge_power_log_prob: torch.Tensor    # [H]
    allocation_entropy: torch.Tensor       # [H]
    active_hex_mask: torch.Tensor          # [H]
    hex_embeddings: Optional[torch.Tensor] = None  # [H, 64]
    forced_charge_count: int = 0
    forced_charge_total_idle: int = 0
    forced_reposition_count: int = 0
    milp_serve_trip_ids: Optional[torch.Tensor] = None  # [V] actual trip-state IDs from MILP, -1 = none


class FleetSACAgent(nn.Module):
    """Fleet-level SAC Agent.

    Uses FleetGCNActor (per-hex allocations) + FleetGCNTwinCritic (hex-level Q)
    + HexVehicleAssigner (deterministic hex→vehicle mapping).

    The actor outputs soft allocation probs which flow directly into the critic
    for clean gradient computation. No action enumeration needed.
    """

    def __init__(
        self,
        num_hexes: int,
        num_vehicles: int,
        vehicle_feature_dim: int = 13,
        hex_feature_dim: int = 5,
        hex_vehicle_agg_dim: int = 8,
        context_dim: int = 9,
        action_dim: int = 3,
        max_K_neighbors: int = 61,
        gcn_hidden_dim: int = 128,
        gcn_output_dim: int = 64,
        hex_decision_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        lr_actor: float = 5e-5,
        lr_critic: float = 1e-4,
        lr_alpha: float = 5e-5,
        dropout: float = 0.1,
        device: str = 'cuda',
        min_alpha: float = 0.05,
        max_alpha: float = 1.0,
        repos_aux_weight: float = 0.1,
        soc_low_threshold: float = 20.0,
        assignment_soc_priority: bool = True,
        use_semi_mdp: bool = True,
    ):
        super().__init__()
        self.num_hexes = num_hexes
        self.num_vehicles = num_vehicles
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.device = device
        self.use_semi_mdp = use_semi_mdp
        self.repos_aux_weight = repos_aux_weight

        # Store feature dims for state parsing
        self.hex_feature_dim = hex_feature_dim
        self.hex_vehicle_agg_dim = hex_vehicle_agg_dim
        self.context_dim = context_dim
        self.vehicle_feature_dim = vehicle_feature_dim

        # Fleet Actor
        self.actor = FleetGCNActor(
            num_hexes=num_hexes,
            hex_feature_dim=hex_feature_dim,
            hex_vehicle_agg_dim=hex_vehicle_agg_dim,
            context_dim=context_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            gcn_output_dim=gcn_output_dim,
            hex_decision_hidden_dim=hex_decision_hidden_dim,
            action_dim=action_dim,
            max_K_neighbors=max_K_neighbors,
            dropout=dropout,
        ).to(device)

        # Fleet Twin Critic
        self.critic = FleetGCNTwinCritic(
            num_hexes=num_hexes,
            gcn_output_dim=gcn_output_dim,
            hex_vehicle_agg_dim=hex_vehicle_agg_dim,
            context_dim=context_dim,
            action_dim=action_dim,
            critic_hidden_dim=critic_hidden_dim,
            dropout=dropout,
        ).to(device)

        # Target critic (frozen copy)
        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Assignment layer (no learnable params)
        self.assigner = HexVehicleAssigner(
            soc_low_threshold=soc_low_threshold,
            soc_priority=assignment_soc_priority,
        )

        # Adjacency and K-hop data (set by environment)
        self._adjacency_matrix: Optional[torch.Tensor] = None
        self._khop_neighbor_indices: Optional[torch.Tensor] = None
        self._khop_neighbor_mask: Optional[torch.Tensor] = None

        # Entropy tuning
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        if auto_alpha:
            if target_entropy is None:
                target_entropy = -0.5 * torch.log(torch.tensor(float(action_dim))).item()
            self.target_entropy = target_entropy
            self.log_alpha = nn.Parameter(torch.tensor(alpha).log())
        else:
            self.register_buffer('log_alpha', torch.tensor(alpha).log())
            self.target_entropy = None

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        if auto_alpha:
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        # Reward normalization
        self.reward_mean = -200.0
        self.reward_std = 150.0
        self.reward_count = 1000

    @property
    def alpha(self) -> torch.Tensor:
        raw_alpha = self.log_alpha.exp()
        return torch.clamp(raw_alpha, min=self.min_alpha, max=self.max_alpha)

    def set_adjacency(self, adj: torch.Tensor):
        """Set adjacency matrix for GCN."""
        self._adjacency_matrix = adj
        actor_module = self.actor.module if hasattr(self.actor, 'module') else self.actor
        actor_module.set_adjacency(adj)

    def set_khop_data(
        self,
        khop_neighbor_indices: torch.Tensor,
        khop_neighbor_mask: torch.Tensor,
    ):
        """Set K-hop neighborhood data for reposition target masking."""
        self._khop_neighbor_indices = khop_neighbor_indices
        self._khop_neighbor_mask = khop_neighbor_mask

    def _parse_flat_state(
        self, state: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flat state into components.

        Returns:
            hex_features [B, H, 5], vehicle_features [B, V, F_vehicle],
            vehicle_hex_ids [B, V], context_features [B, 9]
        """
        if isinstance(state, dict):
            vehicle_features = state['vehicle']
            hex_features = state['hex']
            context_features = state['context']
            if vehicle_features.dim() == 2:
                vehicle_features = vehicle_features.unsqueeze(0)
            if hex_features.dim() == 2:
                hex_features = hex_features.unsqueeze(0)
            if context_features.dim() == 1:
                context_features = context_features.unsqueeze(0)
        else:
            if state.dim() == 1:
                state = state.unsqueeze(0)
            B = state.size(0)
            vs = self.num_vehicles * self.vehicle_feature_dim
            hs = self.num_hexes * self.hex_feature_dim
            vehicle_features = state[:, :vs].view(B, self.num_vehicles, self.vehicle_feature_dim)
            hex_features = state[:, vs:vs + hs].view(B, self.num_hexes, self.hex_feature_dim)
            context_features = state[:, vs + hs:]

        vehicle_hex_ids = (vehicle_features[:, :, 0] * self.num_hexes).long().clamp(0, self.num_hexes - 1)
        return hex_features, vehicle_features, vehicle_hex_ids, context_features

    def _build_hex_vehicle_summary_from_features(
        self,
        vehicle_features: torch.Tensor,  # [B, V, F_vehicle]
        vehicle_hex_ids: torch.Tensor,   # [B, V]
    ) -> torch.Tensor:
        """Build hex vehicle summary from stored vehicle features (for training).

        Fully vectorized over the batch dimension — no Python for-loop.
        Uses batched scatter_add_ across [B, H] tensors.

        Returns: [B, H, 8]
        """
        B, V = vehicle_features.shape[0], vehicle_features.shape[1]
        H = self.num_hexes
        device = vehicle_features.device

        socs = vehicle_features[:, :, 1] * 100.0          # [B, V]
        idle_indicator = vehicle_features[:, :, 2]         # [B, V]
        low_soc = (socs < 20.0).float()                    # [B, V]
        high_soc = (socs > 60.0).float()                   # [B, V]
        time_feat = vehicle_features[:, :, 7]              # [B, V]
        charging = vehicle_features[:, :, 8]               # [B, V]
        ones = torch.ones(B, V, device=device)             # [B, V]

        # Batched scatter_add_: accumulate over vehicle dim → hex dim
        def scatter_sum(src):  # src: [B, V]
            out = torch.zeros(B, H, device=device)
            out.scatter_add_(1, vehicle_hex_ids, src)
            return out

        veh_count  = scatter_sum(ones)                     # [B, H]
        idle_count = scatter_sum(idle_indicator)            # [B, H]
        soc_sum    = scatter_sum(socs)                     # [B, H]
        low_count  = scatter_sum(low_soc)                  # [B, H]
        high_count = scatter_sum(high_soc)                 # [B, H]
        time_sum   = scatter_sum(time_feat)                # [B, H]
        charge_count = scatter_sum(charging)               # [B, H]

        vc_safe = veh_count.clamp(min=1.0)                 # [B, H]
        empty = (veh_count == 0)                           # [B, H]

        mean_soc   = soc_sum / vc_safe / 100.0            # [B, H]

        # Stack features: [B, H, 8]
        summary = torch.stack([
            veh_count / max(V, 1),
            idle_count / vc_safe,
            mean_soc,
            torch.zeros(B, H, device=device),              # min_soc placeholder
            low_count  / vc_safe,
            high_count / vc_safe,
            time_sum   / vc_safe,
            charge_count / vc_safe,
        ], dim=-1)  # [B, H, 8]

        summary[empty] = 0.0
        return summary

    def _build_vehicle_counts(
        self,
        vehicle_hex_ids: torch.Tensor,  # [B, V]
    ) -> torch.Tensor:
        """Build vehicle count per hex. Returns [B, H]. Fully vectorized."""
        B, V = vehicle_hex_ids.shape
        device = vehicle_hex_ids.device
        counts = torch.zeros(B, self.num_hexes, device=device)
        counts.scatter_add_(1, vehicle_hex_ids, torch.ones(B, V, device=device))
        return counts

    def _compute_total_entropy(
        self,
        allocation_entropy: torch.Tensor,
        charge_power_log_prob: torch.Tensor,
        active_hex_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate fleet policy entropy consistently across SAC losses.

        Returns: total_entropy, mean_alloc_entropy, mean_charge_entropy, each [B].
        """
        active_mask = active_hex_mask.float()
        n_active = active_mask.sum(dim=1).clamp(min=1.0)
        mean_alloc_entropy = (allocation_entropy * active_mask).sum(dim=1) / n_active
        mean_charge_entropy = -(charge_power_log_prob * active_mask).sum(dim=1) / n_active
        total_entropy = mean_alloc_entropy + mean_charge_entropy
        return total_entropy, mean_alloc_entropy, mean_charge_entropy

    def _build_active_hex_mask(
        self,
        hex_vehicle_summary: torch.Tensor,
    ) -> torch.Tensor:
        """Hexes with at least one idle vehicle available for assignment."""
        veh_count = hex_vehicle_summary[..., 0]
        idle_frac = hex_vehicle_summary[..., 1]
        return (veh_count > 0.0) & (idle_frac > 1e-6)

    @torch.no_grad()
    def select_action_fleet(
        self,
        hex_features: torch.Tensor,          # [H, 5]
        hex_vehicle_summary: torch.Tensor,    # [H, 8]
        context_features: torch.Tensor,       # [9]
        vehicle_hex_ids: torch.Tensor,        # [V]
        vehicle_socs: torch.Tensor,           # [V]
        vehicle_status: torch.Tensor,         # [V]
        idle_steps: torch.Tensor = None,      # [V]
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> FleetSACOutput:
        """Select actions at episode collection time.

        Returns both hex-level (for buffer) and per-vehicle (for simulator) actions.
        """
        expected_hexes = int(self.num_hexes)
        observed_hexes = int(hex_features.shape[0])
        if observed_hexes != expected_hexes:
            raise ValueError(
                f"Hex feature size mismatch in select_action_fleet: got {observed_hexes}, expected {expected_hexes}"
            )
        if self._adjacency_matrix is None:
            raise ValueError("Adjacency matrix is not set for FleetSACAgent")
        if self._adjacency_matrix.shape[0] != observed_hexes or self._adjacency_matrix.shape[1] != observed_hexes:
            raise ValueError(
                f"Adjacency size mismatch in select_action_fleet: adj={tuple(self._adjacency_matrix.shape)}, hex={observed_hexes}"
            )
        if self._khop_neighbor_indices is not None and self._khop_neighbor_indices.shape[0] != observed_hexes:
            raise ValueError(
                f"K-hop index size mismatch in select_action_fleet: khop={tuple(self._khop_neighbor_indices.shape)}, hex={observed_hexes}"
            )
        if self._khop_neighbor_mask is not None and self._khop_neighbor_mask.shape[0] != observed_hexes:
            raise ValueError(
                f"K-hop mask size mismatch in select_action_fleet: khop_mask={tuple(self._khop_neighbor_mask.shape)}, hex={observed_hexes}"
            )

        # Forward fleet actor
        actor_out = self.actor(
            hex_features=hex_features,
            hex_vehicle_summary=hex_vehicle_summary,
            context_features=context_features,
            adj=self._adjacency_matrix,
            active_hex_mask=self._build_active_hex_mask(hex_vehicle_summary.unsqueeze(0)).squeeze(0),
            khop_neighbor_indices=self._khop_neighbor_indices,
            khop_neighbor_mask=self._khop_neighbor_mask,
            temperature=temperature,
            deterministic=deterministic,
        )

        # Squeeze batch dim (collection is unbatched)
        alloc_probs = actor_out['allocation_probs']       # [H, 3]
        repos_targets = actor_out['repos_sampled_targets'] # [H]
        charge_pow = actor_out['charge_power']             # [H]

        # Assignment: hex→vehicle
        action_type, repos_target, veh_charge_power = self.assigner.assign(
            allocation_probs=alloc_probs,
            repos_sampled_targets=repos_targets,
            charge_power=charge_pow,
            vehicle_hex_ids=vehicle_hex_ids,
            vehicle_socs=vehicle_socs,
            vehicle_status=vehicle_status,
            idle_steps=idle_steps,
        )

        idle_mask = (vehicle_status == 0)
        forced_charge_mask = idle_mask & (vehicle_socs < self.assigner.soc_low_threshold)
        forced_reposition_mask = idle_mask & (
            (idle_steps >= self.assigner.idle_force_charge_steps) if idle_steps is not None else torch.zeros_like(idle_mask)
        ) & (~forced_charge_mask)
        forced_charge_count = int(forced_charge_mask.sum().item())
        forced_charge_total_idle = int(idle_mask.sum().item())
        forced_reposition_count = int(forced_reposition_mask.sum().item())

        return FleetSACOutput(
            action_type=action_type,
            reposition_target=repos_target,
            vehicle_charge_power=veh_charge_power,
            allocation_probs=alloc_probs,
            allocation_log_probs=actor_out['allocation_log_probs'],
            repos_sampled_targets=repos_targets,
            charge_power=charge_pow,
            charge_power_log_prob=actor_out['charge_power_log_prob'],
            allocation_entropy=actor_out['allocation_entropy'],
            active_hex_mask=actor_out['active_hex_mask'],
            hex_embeddings=actor_out.get('hex_embeddings'),
            forced_charge_count=forced_charge_count,
            forced_charge_total_idle=forced_charge_total_idle,
            forced_reposition_count=forced_reposition_count,
        )

    def compute_critic_loss(
        self,
        states: Union[torch.Tensor, Dict[str, torch.Tensor]],
        hex_allocations: torch.Tensor,       # [B, H, 3]
        charge_power: torch.Tensor,          # [B, H]
        rewards: torch.Tensor,               # [B]
        next_states: Union[torch.Tensor, Dict[str, torch.Tensor]],
        dones: torch.Tensor,                 # [B]
        vehicle_hex_ids: Optional[torch.Tensor] = None,  # [B, V]
        durations: Optional[torch.Tensor] = None,  # [B] for semi-MDP
    ) -> torch.Tensor:
        """Compute critic loss with fleet-level actions.

        No action enumeration — actor outputs soft probs, critic evaluates directly.

        Target: r + gamma * (1-done) * [Q_target(s', pi(s')) + alpha * H(pi(s'))]
        """
        # Parse states
        hex_features, vehicle_features, parsed_hex_ids, context = self._parse_flat_state(states)
        next_hex, next_veh, next_hex_ids, next_ctx = self._parse_flat_state(next_states)

        if vehicle_hex_ids is None:
            vehicle_hex_ids = parsed_hex_ids

        # Build hex vehicle summaries
        hex_veh_summary = self._build_hex_vehicle_summary_from_features(vehicle_features, vehicle_hex_ids)
        next_hex_veh_summary = self._build_hex_vehicle_summary_from_features(next_veh, next_hex_ids)
        next_active_hex_mask = self._build_active_hex_mask(next_hex_veh_summary)

        # Vehicle counts for weighted aggregation
        veh_counts = self._build_vehicle_counts(vehicle_hex_ids)
        next_veh_counts = self._build_vehicle_counts(next_hex_ids)

        # --- Next-state value via actor + target critic ---
        with torch.no_grad():
            next_actor_out = self.actor(
                hex_features=next_hex,
                hex_vehicle_summary=next_hex_veh_summary,
                context_features=next_ctx,
                adj=self._adjacency_matrix,
                active_hex_mask=next_active_hex_mask,
                khop_neighbor_indices=self._khop_neighbor_indices,
                khop_neighbor_mask=self._khop_neighbor_mask,
                deterministic=False,
            )
            next_alloc = next_actor_out['allocation_probs']      # [B, H, 3]
            next_charge_pow = next_actor_out['charge_power']     # [B, H]
            next_alloc_entropy = next_actor_out['allocation_entropy']   # [B, H]
            next_charge_log_prob = next_actor_out['charge_power_log_prob']  # [B, H]
            # Target Q
            next_q1, next_q2 = self.critic_target(
                hex_features=next_hex,
                hex_vehicle_summary=next_hex_veh_summary,
                context_features=next_ctx,
                hex_allocations=next_alloc,
                charge_power=next_charge_pow,
                adjacency=self._adjacency_matrix,
                vehicle_counts=next_veh_counts,
            )
            next_q = torch.min(next_q1, next_q2)  # [B]

            # Soft value uses the same total entropy as actor/V_phi training
            total_entropy, _, _ = self._compute_total_entropy(
                allocation_entropy=next_alloc_entropy,
                charge_power_log_prob=next_charge_log_prob,
                active_hex_mask=next_actor_out['active_hex_mask'],
            )

            # V(s') = Q(s', pi(s')) + alpha * H_total(pi(s'))
            next_v = next_q + self.alpha.detach() * total_entropy

            # Discount
            if self.use_semi_mdp and durations is not None:
                discount = self.gamma ** durations.clamp(min=1.0)
            else:
                discount = self.gamma
            target_q = rewards + (1.0 - dones.float()) * discount * next_v

        # --- Current Q ---
        q1, q2 = self.critic(
            hex_features=hex_features,
            hex_vehicle_summary=hex_veh_summary,
            context_features=context,
            hex_allocations=hex_allocations,
            charge_power=charge_power,
            adjacency=self._adjacency_matrix,
            vehicle_counts=veh_counts,
        )

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        return critic_loss

    def compute_actor_loss(
        self,
        states: Union[torch.Tensor, Dict[str, torch.Tensor]],
        vehicle_hex_ids: Optional[torch.Tensor] = None,
        actor_hex_embeddings: Optional[torch.Tensor] = None,
        critic_hex_embeddings_q1: Optional[torch.Tensor] = None,
        critic_hex_embeddings_q2: Optional[torch.Tensor] = None,
        hex_features: Optional[torch.Tensor] = None,
        vehicle_features: Optional[torch.Tensor] = None,
        context_features: Optional[torch.Tensor] = None,
        hex_vehicle_summary: Optional[torch.Tensor] = None,
        vehicle_counts: Optional[torch.Tensor] = None,
        active_hex_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Compute actor loss: maximize Q + alpha * entropy."""
        if hex_features is None or vehicle_features is None or context_features is None:
            hex_features, vehicle_features, parsed_hex_ids, parsed_context = self._parse_flat_state(states)
            if vehicle_hex_ids is None:
                vehicle_hex_ids = parsed_hex_ids
            context = parsed_context
        else:
            hex_features = hex_features
            vehicle_features = vehicle_features
            context = context_features
            if vehicle_hex_ids is None:
                _, _, parsed_hex_ids, _ = self._parse_flat_state(states)
                vehicle_hex_ids = parsed_hex_ids

        if hex_vehicle_summary is None:
            hex_veh_summary = self._build_hex_vehicle_summary_from_features(vehicle_features, vehicle_hex_ids)
        else:
            hex_veh_summary = hex_vehicle_summary
        if vehicle_counts is None:
            veh_counts = self._build_vehicle_counts(vehicle_hex_ids)
        else:
            veh_counts = vehicle_counts
        if active_hex_mask is None:
            active_hex_mask = self._build_active_hex_mask(hex_veh_summary)

        # Forward actor — skip GCN if embeddings provided
        actor_out = self.actor(
            hex_features=hex_features,
            hex_vehicle_summary=hex_veh_summary,
            context_features=context,
            adj=self._adjacency_matrix,
            active_hex_mask=active_hex_mask,
            khop_neighbor_indices=self._khop_neighbor_indices,
            khop_neighbor_mask=self._khop_neighbor_mask,
            deterministic=False,
            hex_embeddings=actor_hex_embeddings,  # bypass GCN when provided
        )
        alloc_probs = actor_out['allocation_probs']          # [B, H, 3]
        charge_pow = actor_out['charge_power']               # [B, H]
        alloc_entropy = actor_out['allocation_entropy']      # [B, H]
        charge_log_prob = actor_out['charge_power_log_prob'] # [B, H]

        # Q-value of current policy — skip GCN if embeddings provided
        q1, q2 = self.critic(
            hex_features=hex_features,
            hex_vehicle_summary=hex_veh_summary,
            context_features=context,
            hex_allocations=alloc_probs,
            charge_power=charge_pow,
            adjacency=self._adjacency_matrix,
            vehicle_counts=veh_counts,
            hex_embeddings_q1=critic_hex_embeddings_q1,
            hex_embeddings_q2=critic_hex_embeddings_q2,
        )
        q_fleet = torch.min(q1, q2)  # [B]

        total_entropy, mean_alloc_entropy, mean_charge_entropy = self._compute_total_entropy(
            allocation_entropy=alloc_entropy,
            charge_power_log_prob=charge_log_prob,
            active_hex_mask=actor_out['active_hex_mask'],
        )

        active_mask_f = actor_out['active_hex_mask'].float()
        active_hex_denom = active_mask_f.sum().clamp(min=1.0)
        serve_frac_active = (alloc_probs[:, :, 0] * active_mask_f).sum() / active_hex_denom
        charge_frac_active = (alloc_probs[:, :, 1] * active_mask_f).sum() / active_hex_denom
        repos_frac_active = (alloc_probs[:, :, 2] * active_mask_f).sum() / active_hex_denom

        # Actor loss: minimize -Q - alpha * entropy
        actor_loss = (-q_fleet - self.alpha.detach() * total_entropy).mean()

        # Auxiliary reposition loss: encourage repos targets toward high demand
        repos_aux_loss = torch.tensor(0.0, device=hex_features.device)
        if self.repos_aux_weight > 0 and self._khop_neighbor_indices is not None:
            # hex demand is feature index 2 (norm_demand from builder)
            demand = hex_features[:, :, 2]  # [B, H]
            B, H = demand.shape
            max_K = self._khop_neighbor_indices.shape[1]
            kni = self._khop_neighbor_indices.unsqueeze(0).expand(B, -1, -1)  # [B, H, max_K]
            
            # Gather demand of K-hop neighbors: [B, H, max_K]
            demand_flat = demand.unsqueeze(1).expand(-1, H, -1)  # [B, H, H]
            neighbor_demand = demand_flat.gather(2, kni.clamp(min=0))  # [B, H, max_K]
            
            # Mask out invalid (-1) neighbors
            valid_mask = (kni != -1)
            neighbor_demand = neighbor_demand.masked_fill(~valid_mask, float('-inf'))
            # Prevent all -inf which causes NaN in softmax
            valid_counts = valid_mask.sum(dim=-1, keepdim=True)
            empty_rows = (valid_counts == 0)
            neighbor_demand = neighbor_demand.masked_fill(empty_rows.expand_as(neighbor_demand), 0.0)
            
            # Soft target over the max_K neighbors
            demand_target = F.softmax(neighbor_demand * 10.0, dim=-1)  # [B, H, max_K]
            
            # The network output to optimize
            repos_probs = actor_out['repos_target_probs']  # [B, H, max_K]

            # Cross entropy over neighbors: - sum( p_true * log(p_pred) )
            # Add 1e-8 to prevent PyTorch `0 * inf` gradient trap during backward pass
            cx_loss = -(demand_target.detach() * (repos_probs + 1e-8).log().clamp(min=-10)).sum(dim=-1)  # [B, H]
            
            active_mask = actor_out['active_hex_mask'] & (~empty_rows.squeeze(-1))
            cx_loss = cx_loss.masked_fill(~active_mask, 0.0)
            denom = active_mask.float().sum(dim=1).clamp(min=1.0)
            repos_aux_loss = (cx_loss.sum(dim=1) / denom).mean()
            actor_loss = actor_loss + self.repos_aux_weight * repos_aux_loss

        neg_entropy = -total_entropy.mean()  # For alpha update

        info = {
            'q_mean': q_fleet.mean().item(),
            'q_fleet_mean': q_fleet.mean().item(),
            'alloc_entropy': mean_alloc_entropy.mean().item(),
            'charge_entropy': mean_charge_entropy.mean().item(),
            'repos_aux_loss': repos_aux_loss.item() if isinstance(repos_aux_loss, torch.Tensor) else repos_aux_loss,
            'serve_frac': serve_frac_active.item(),
            'charge_frac': charge_frac_active.item(),
            'repos_frac': repos_frac_active.item(),
        }

        return actor_loss, neg_entropy, info

    def compute_alpha_loss(self, neg_entropy: torch.Tensor) -> torch.Tensor:
        """Compute alpha loss for automatic entropy tuning."""
        if not self.auto_alpha:
            return torch.tensor(0.0, device=neg_entropy.device)
        alpha_loss = -(self.log_alpha * (neg_entropy - self.target_entropy).detach()).mean()
        return alpha_loss

    def soft_update_target(self):
        """Polyak averaging: target ← tau * critic + (1-tau) * target."""
        critic_module = self.critic.module if hasattr(self.critic, 'module') else self.critic
        target_module = self.critic_target.module if hasattr(self.critic_target, 'module') else self.critic_target
        for p, tp in zip(critic_module.parameters(), target_module.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running statistics."""
        batch_mean = rewards.mean().item()
        batch_std = rewards.std().item() + 1e-8
        batch_size = rewards.numel()
        new_count = self.reward_count + batch_size
        delta = batch_mean - self.reward_mean
        self.reward_mean += delta * batch_size / new_count
        self.reward_std = 0.9 * self.reward_std + 0.1 * batch_std
        self.reward_count = new_count
        normalized = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        return torch.clamp(normalized, -5.0, 5.0)

    def save(self, path: str):
        actor_m = self.actor.module if hasattr(self.actor, 'module') else self.actor
        critic_m = self.critic.module if hasattr(self.critic, 'module') else self.critic
        target_m = self.critic_target.module if hasattr(self.critic_target, 'module') else self.critic_target
        torch.save({
            'actor': actor_m.state_dict(),
            'critic': critic_m.state_dict(),
            'critic_target': target_m.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_alpha else None,
        }, path)

    def load(self, path: str):
        actor_m = self.actor.module if hasattr(self.actor, 'module') else self.actor
        critic_m = self.critic.module if hasattr(self.critic, 'module') else self.critic
        target_m = self.critic_target.module if hasattr(self.critic_target, 'module') else self.critic_target
        ckpt = torch.load(path, map_location=self.device)
        actor_m.load_state_dict(ckpt['actor'])
        critic_m.load_state_dict(ckpt['critic'])
        target_m.load_state_dict(ckpt['critic_target'])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt['log_alpha'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        if self.auto_alpha and ckpt['alpha_optimizer'] is not None:
            self.alpha_optimizer.load_state_dict(ckpt['alpha_optimizer'])
