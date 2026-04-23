"""
Wasserstein Distributionally Robust Optimization (WDRO) per paper Section 4.

Implements:
- Graph-Aligned Mahalanobis (MAG) metric (paper Eq. 14)
- Wasserstein-1 ambiguity set with KR dual (paper Eq. 19-20)
- Robust backup with projected gradient ascent (paper Section 4.3)
- Primal-dual risk-budget tracking (paper Eq. 23-24)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass


@dataclass
class WDROConfig:
    """Configuration for WDRO."""
    rho: float = 1.0                 # Ambiguity radius
    metric: str = "mag"             # Ground metric: "mag" or "euclidean"
    rho_target: float = 0.5          # Target risk budget
    lambda_init: float = 1.0         # Initial dual variable
    lambda_lr: float = 0.01          # Learning rate for lambda update
    inner_steps: int = 5             # Steps for inner maximization
    inner_lr: float = 0.02           # Step size for inner gradient ascent
    beta: float = 0.3                # Graph Laplacian weight
    epsilon: float = 1e-6            # Numerical stability
    support_radius: float = 10.0     # Radius of support set B


class MAGMetric:
    
    def __init__(
        self,
        feature_dim,
        beta=0.3,
        epsilon=1e-6,
        device="cuda",
        metric: str = "mag",
    ):
        self.feature_dim = feature_dim
        self.beta = beta
        self.epsilon = epsilon
        self.device = torch.device(device)
        self.metric = metric

        self.running_var = torch.ones(feature_dim, device=device)
        self.sum = torch.zeros(feature_dim, device=device)
        self.sq_sum = torch.zeros(feature_dim, device=device)
        self.count = 0
        self.num_hexes = None

        self.Q_graph = None

        self.Q = None
        self._Q_sqrt = None
        self._Q_inv_sqrt = None
        self.L_spatial = torch.eye(feature_dim)  # CPU — only used in _build_Q
        self.L_od = torch.eye(feature_dim)       # CPU — only used in _build_Q
    

    def update_statistics(self, features, rebuild_every: int = 2):
        # features: [batch, dim] — may be float16 under autocast, cast to float32
        features = features.float()

        self.sum += features.sum(dim=0)
        self.sq_sum += (features**2).sum(dim=0)
        self.count += features.size(0)
        self._stats_calls = getattr(self, '_stats_calls', 0) + 1

        mean = self.sum / self.count
        var = self.sq_sum / self.count - mean**2

        self.running_var = var.clamp(min=self.epsilon)

        # Rebuild Q periodically — Cholesky on [H,H] is O(H³), too expensive every step
        if self._stats_calls % rebuild_every == 1 or self._stats_calls == 1:
            self._build_Q()


    def set_spatial_graph(self, adj, already_normalized: bool = True):
        """Set spatial graph Laplacian.

        Args:
            adj: Adjacency matrix [H, H]. If ``already_normalized`` (default),
                 expects the symmetrically-normalized adjacency D^{-1/2} A D^{-1/2}
                 as produced by the environment.  Otherwise applies normalization.
        """
        adj_cpu = adj.cpu().float()

        if not already_normalized:
            deg = adj_cpu.sum(dim=1)
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-6))
            adj_cpu = D_inv_sqrt @ adj_cpu @ D_inv_sqrt

        L = torch.eye(adj_cpu.size(0)) - adj_cpu
        self.L_spatial = L  # stays on CPU — transferred to GPU in _build_Q

    def build_od_graph(self, OD):
        # OD: [H,H] — compute on CPU, store on CPU
        OD_cpu = OD.cpu().float()

        # cosine similarity
        OD_norm = F.normalize(OD_cpu, dim=1)
        sim = OD_norm @ OD_norm.T

        # threshold
        sim[sim < 0.2] = 0

        deg = sim.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-6))

        L = torch.eye(sim.size(0)) - D_inv_sqrt @ sim @ D_inv_sqrt

        self.L_od = L  # stays on CPU — transferred to GPU in _build_Q
    

    def _build_Q(self):
        # Force float32 — Cholesky and triangular solve are numerically unstable in float16.
        # This method can be called from update_statistics() which runs under autocast.
        with torch.cuda.amp.autocast(enabled=False):
            if self.metric == "euclidean":
                self.is_diagonal = True
                self.Q = torch.ones(self.feature_dim, device=self.device)
                self._Q_sqrt = torch.ones(self.feature_dim, device=self.device)
                self._Q_inv_sqrt = torch.ones(self.feature_dim, device=self.device)
                return

            w = 1.0 / (self.running_var.float() + self.epsilon)
            Q_diag = torch.diag(w)

            # Transfer Laplacians from CPU to GPU for this computation only
            Q_graph = (self.L_spatial + self.L_od).to(self.device).float()
            use_graph = Q_graph.shape[0] == Q_diag.shape[0]
            self.is_diagonal = not use_graph

            if self.is_diagonal:
                self.Q = w + self.epsilon
                self._Q_sqrt = torch.sqrt(self.Q)
                self._Q_inv_sqrt = 1.0 / self._Q_sqrt
                return

            Q = Q_diag + self.beta * Q_graph

            # Normalize by H so d_Q is a per-hex-average distance, not a
            # sum-over-hex norm.  Raw d_Q scales as √H (H≈4000), while
            # rewards/V/Q are per-vehicle-average scalars.  Without this,
            # λ d_Q dominates the robust target by ~60× and the distance
            # subgradient overwhelms ∇V in the inner loop.
            H = Q.size(0)
            Q = Q / H

            # Regularize (after normalization so ε stays at fixed absolute scale)
            Q = Q + self.epsilon * torch.eye(H, device=self.device)

            self.Q = Q

            # Cholesky: Q = L L^T  (L lower triangular)
            L = torch.linalg.cholesky(Q)

            # Q^{1/2} is L in column convention
            self._Q_sqrt = L

            # L^{-1} via triangular solve, then store L^{-T}
            L_inv = torch.linalg.solve_triangular(
                L, torch.eye(L.shape[0], device=self.device), upper=False
            )
            self._Q_inv_sqrt = L_inv.T
    

    def distance(self, xi: torch.Tensor, xi_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute MAG distance d_Q(ξ, ξ').
        
        Args:
            xi: [batch, feature_dim] - perturbed scenario
            xi_hat: [batch, feature_dim] - nominal scenario
            
        Returns:
            distance: [batch] - d_Q(ξ, ξ')
        """
        # Lazy initialization: build Q from unit variance if never updated
        if self._Q_sqrt is None:
            self._build_Q()
        diff = xi - xi_hat
        if getattr(self, 'is_diagonal', False):
            z = diff * self._Q_sqrt
        else:
            # ||diff||_Q = ||diff @ L||  (row-vector convention, Q = L L^T)
            z = diff @ self._Q_sqrt
        return torch.norm(z, dim=-1)
    

    def subgradient(self, xi: torch.Tensor, xi_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute subgradient of d_Q w.r.t. ξ per paper Section 4.3.
        
        g_d(ξ) = Q(ξ - ξ̂) / max(ε, ||ξ - ξ̂||_Q)
        """
        if self.Q is None:
            self._build_Q()
        diff = xi - xi_hat
        dist = self.distance(xi, xi_hat).unsqueeze(-1)
        if getattr(self, 'is_diagonal', False):
            grad = diff * self.Q
        else:
            grad = diff @ self.Q
        return grad / dist.clamp(min=self.epsilon)
    

    def project_to_ball(
        self,
        xi: torch.Tensor,
        center: torch.Tensor,
        radius: float
    ) -> torch.Tensor:
        """
        Project ξ onto Q-ball of given radius centered at center.
        
        Per paper Eq. 21:
        u = Q^{1/2}(ξ - center)
        u ← R * u / max(R, ||u||)
        ξ_proj = center + Q^{-1/2} u
        """
        if self._Q_sqrt is None:
            self._build_Q()
        diff = xi - center

        if getattr(self, 'is_diagonal', False):
            u = diff * self._Q_sqrt
            norm = torch.norm(u, dim=-1, keepdim=True)
            u = radius * u / norm.clamp(min=radius)
            diff_proj = u * self._Q_inv_sqrt
        else:
            # u = diff @ L  (whitened coords, same convention as distance())
            u = diff @ self._Q_sqrt
            norm = torch.norm(u, dim=-1, keepdim=True)
            u = radius * u / norm.clamp(min=radius)
            # Recover: diff_proj = u @ L^{-1} = u @ _Q_inv_sqrt.T  (since _Q_inv_sqrt = L^{-T})
            diff_proj = u @ self._Q_inv_sqrt.T

        return center + diff_proj

    def get_state(self) -> Dict[str, Any]:
        """Serialize MAG running statistics needed to resume WDRO updates."""
        return {
            'running_var': self.running_var.detach().clone(),
            'sum': self.sum.detach().clone(),
            'sq_sum': self.sq_sum.detach().clone(),
            'count': int(self.count),
            'num_hexes': self.num_hexes,
            'stats_calls': int(getattr(self, '_stats_calls', 0)),
        }

    def load_state(self, state: Optional[Dict[str, Any]]):
        """Restore MAG running statistics and rebuild Q on the current device."""
        if not state:
            return

        self.running_var = state.get('running_var', self.running_var).to(self.device)
        self.sum = state.get('sum', self.sum).to(self.device)
        self.sq_sum = state.get('sq_sum', self.sq_sum).to(self.device)
        self.count = int(state.get('count', self.count))
        self.num_hexes = state.get('num_hexes', self.num_hexes)
        self._stats_calls = int(state.get('stats_calls', getattr(self, '_stats_calls', 0)))

        if self.count > 0:
            self._build_Q()

class ValueNetwork(nn.Module):
    """Lightweight V_φ approximating Eq. 27: V(s) = E_a~π[min_k Q_k(s,a) - α log π(a|s)].

    Shares the critic's GCN encoder (paper Section VI-C).  Takes pre-computed
    hex embeddings as input — no duplicate GCN.  Only the MLP head is trained
    by the V_φ supervised loss; GCN gradients are owned by the critic loss.

    In the WDRO inner loop the critic's GCN is re-run WITH gradient so that
    ∇_ξ V flows through: ξ → hex_features → critic.GCN → hex_emb → V_φ → V.
    """

    def __init__(
        self,
        gcn_output_dim: int = 64,
        vehicle_feature_dim: int = 16,
        context_dim: int = 9,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Vehicle feature aggregation
        self.vehicle_proj = nn.Linear(vehicle_feature_dim, 32)

        # MLP value head: hex_emb(pooled) + vehicle_proj + context → scalar
        input_dim = gcn_output_dim + 32 + context_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim // 2, 1),
        )

    def forward(
        self,
        hex_embeddings: torch.Tensor,     # [batch, H, gcn_output_dim] from critic GCN
        vehicle_features: torch.Tensor,   # [batch, V, vehicle_feature_dim]
        context_features: torch.Tensor,   # [batch, context_dim]
    ) -> torch.Tensor:
        """Returns V: [batch]."""
        hex_agg = hex_embeddings.mean(dim=1)  # [batch, gcn_output_dim]
        veh_agg = self.vehicle_proj(vehicle_features.mean(dim=1))  # [batch, 32]
        x = torch.cat([hex_agg, veh_agg, context_features], dim=-1)
        return self.mlp(x).squeeze(-1)  # [batch]


class WDROAdversary(nn.Module):
    """
    WDRO adversary that finds worst-case scenarios per paper Section 4.

    Solves the paper's practical inner maximization (Eq. 21):
    max_ξ∈B [γ^Δ(ξ) V(s'(ξ)) - λ d_Q(ξ, ξ̂)]

    using projected gradient ascent.

    Two-phase V evaluation:
      Phase 1 (use_learned_value=False): V computed exactly from the active
        fleet actor + twin critics via Eq. 27.
      Phase 2 (use_learned_value=True): V_φ (lightweight ValueNetwork) used
        in the inner loop.
    """

    def __init__(
        self,
        scenario_dim: int,
        config: WDROConfig,
        device: str = "cuda",
        adjacency_matrix: Optional[torch.Tensor] = None,
        critic: Optional[nn.Module] = None,
        actor: Optional[nn.Module] = None,
        build_hex_vehicle_summary_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        build_vehicle_counts_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        khop_neighbor_indices: Optional[torch.Tensor] = None,
        khop_neighbor_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.scenario_dim = scenario_dim
        self.config = config
        self.device = torch.device(device)

        # Active fleet actor/critic for exact Eq. 27 evaluation
        self.critic = critic
        self.actor = actor
        self.adjacency = adjacency_matrix
        self.build_hex_vehicle_summary_fn = build_hex_vehicle_summary_fn
        self.build_vehicle_counts_fn = build_vehicle_counts_fn
        self.khop_neighbor_indices = khop_neighbor_indices
        self.khop_neighbor_mask = khop_neighbor_mask

        # Learned V_φ — set by trainer after construction
        self.value_network: Optional[ValueNetwork] = None
        self.use_learned_value: bool = False
        self.last_debug: Dict[str, float] = {}

        # MAG metric
        self.mag_metric = MAGMetric(
            feature_dim=scenario_dim,
            beta=config.beta,
            device=device,
            metric=config.metric,
        )

        # Initialize graph-aligned metric if adjacency is provided
        if adjacency_matrix is not None and config.metric == "mag":
            self.mag_metric.set_spatial_graph(adjacency_matrix)

        # Dual variable λ for Lagrangian
        self.log_lambda = nn.Parameter(torch.tensor(config.lambda_init).log())

        # Running statistics for primal-dual update
        self.running_rho_hat = 0.0
        self.update_count = 0

    @property
    def lambda_(self) -> torch.Tensor:
        """Get current λ value (positive)."""
        return self.log_lambda.exp()

    def _ensure_finite(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite WDRO tensor detected in {name}")
        return tensor

    def _eval_value_exact(
        self,
        state_components: Dict[str, torch.Tensor],
        action_probs: torch.Tensor,
        action_log_probs: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Compute Eq. 27 exactly using the fleet actor and fleet twin critics."""
        del action_probs, action_log_probs

        if self.actor is None or self.critic is None:
            raise RuntimeError("WDRO exact value evaluation requires both actor and critic.")
        if self.build_hex_vehicle_summary_fn is None or self.build_vehicle_counts_fn is None:
            raise RuntimeError("WDRO exact value evaluation requires fleet summary helper functions.")

        hex_features = state_components['hex_features'].float()
        vehicle_features = state_components['vehicle_features'].float()
        context_features = state_components['context_features'].float()
        vehicle_hex_ids = state_components['vehicle_hex_ids']

        actor_module = self.actor.module if hasattr(self.actor, 'module') else self.actor
        critic_module = self.critic.module if hasattr(self.critic, 'module') else self.critic

        hex_vehicle_summary = self.build_hex_vehicle_summary_fn(vehicle_features, vehicle_hex_ids)
        vehicle_counts = self.build_vehicle_counts_fn(vehicle_hex_ids)

        actor_out = actor_module(
            hex_features=hex_features,
            hex_vehicle_summary=hex_vehicle_summary,
            context_features=context_features,
            adj=self.adjacency,
            active_hex_mask=(hex_vehicle_summary[:, :, 0] > 0.0) & (hex_vehicle_summary[:, :, 1] > 1e-6),
            khop_neighbor_indices=self.khop_neighbor_indices,
            khop_neighbor_mask=self.khop_neighbor_mask,
            deterministic=False,
            compute_repos=False,  # repos not used for V computation — saves ~2.4 GB
        )

        alloc_probs = actor_out['allocation_probs']
        charge_power = actor_out['charge_power']
        alloc_entropy = actor_out['allocation_entropy']
        charge_log_prob = actor_out['charge_power_log_prob']
        active_mask = actor_out['active_hex_mask'].float()
        n_active = active_mask.sum(dim=1).clamp(min=1.0)
        mean_alloc_entropy = (alloc_entropy * active_mask).sum(dim=1) / n_active
        mean_charge_entropy = -(charge_log_prob * active_mask).sum(dim=1) / n_active
        total_entropy = mean_alloc_entropy + mean_charge_entropy
        self._ensure_finite('exact_alloc_entropy', mean_alloc_entropy)
        self._ensure_finite('exact_charge_entropy', mean_charge_entropy)
        self._ensure_finite('exact_total_entropy', total_entropy)

        q_value = critic_module.min_q(
            hex_features=hex_features,
            hex_vehicle_summary=hex_vehicle_summary,
            context_features=context_features,
            hex_allocations=alloc_probs,
            charge_power=charge_power,
            adjacency=self.adjacency,
            vehicle_counts=vehicle_counts,
        )
        self._ensure_finite('exact_q_value', q_value)
        value = q_value + float(alpha) * total_entropy
        return self._ensure_finite('exact_value', value)

    def _eval_value_approx(
        self,
        state_components: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute V ≈ V_φ(s) via the learned value network.

        Used in the WDRO inner loop during Phase 2.
        Runs critic1's GCN (shared encoder) WITH gradient, then V_φ MLP.
        Gradient flows: ξ → hex_features → critic.GCN → hex_emb → V_φ → V.
        """
        hex_features = state_components['hex_features'].float()
        critic_module = self.critic.module if hasattr(self.critic, 'module') else self.critic
        hex_emb = critic_module.critic1.gcn(hex_features, self.adjacency)
        value = self.value_network(
            hex_emb,
            state_components['vehicle_features'].float(),
            state_components['context_features'].float(),
        )
        return self._ensure_finite('approx_value', value)

    def find_adversarial_scenario(
        self,
        xi_hat: torch.Tensor,        # [batch, scenario_dim] - nominal scenario
        next_state_fn,               # Function: scenario -> dict of structured state components
        gamma: float = 0.99,
        duration_fn=None,            # Optional function: scenario -> duration
        action_probs: torch.Tensor = None,   # [batch, A] detached policy probs (Phase 1 only)
        action_log_probs: torch.Tensor = None,  # [batch, A] detached log probs (Phase 1 only)
        alpha: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find worst-case scenario ξ* via projected gradient ascent (paper Eq. 21).

        Phase 1 (use_learned_value=False): uses _eval_value_exact (critics).
        Phase 2 (use_learned_value=True):  uses _eval_value_approx (V_φ).

        Returns:
            xi_star, worst_value, discount_star, distance_star
        """
        use_approx = self.use_learned_value and self.value_network is not None

        # CRITICAL: Force float32 for the entire inner loop.
        with torch.cuda.amp.autocast(enabled=False):
            xi_hat = xi_hat.float()
            if action_probs is not None:
                action_probs = action_probs.float()
            if action_log_probs is not None:
                action_log_probs = action_log_probs.float()

            xi = xi_hat.detach().clone().requires_grad_(True)

            for k in range(self.config.inner_steps):
                state_components = next_state_fn(xi)

                if use_approx:
                    value = self._eval_value_approx(state_components)
                else:
                    value = self._eval_value_exact(
                        state_components, action_probs, action_log_probs, alpha)

                if duration_fn is not None:
                    duration = duration_fn(xi)
                    discount = torch.pow(gamma, duration)
                else:
                    discount = gamma

                value_term = discount * value
                grad_v = torch.autograd.grad(value_term.sum(), xi, create_graph=False)[0]
                g_d = self.mag_metric.subgradient(xi, xi_hat)

                xi_new = xi.detach() - self.config.inner_lr * (grad_v + self.lambda_ * g_d)
                xi_new = self.mag_metric.project_to_ball(
                    xi_new, xi_hat.detach(), self.config.support_radius
                )
                # Keep attacked demand in the same normalized domain as env features.
                xi_new = xi_new.clamp(min=0.0, max=1.0)
                self._ensure_finite('projected_scenario', xi_new)
                xi = xi_new.detach().clone().requires_grad_(True)

            # Final value at ξ* — no gradient needed (target is .detach()ed by caller)
            with torch.no_grad():
                state_star = next_state_fn(xi)
                if use_approx:
                    worst_value = self._eval_value_approx(state_star)
                else:
                    worst_value = self._eval_value_exact(
                        state_star, action_probs, action_log_probs, alpha)

                if duration_fn is not None:
                    duration_star = duration_fn(xi)
                    discount_star = torch.pow(gamma, duration_star)
                else:
                    discount_star = gamma

                distance_star = self.mag_metric.distance(xi, xi_hat)
                self._ensure_finite('worst_value', worst_value)
                self._ensure_finite('distance_star', distance_star)
                if isinstance(discount_star, torch.Tensor):
                    self._ensure_finite('discount_star', discount_star)

        return xi.detach(), worst_value, discount_star, distance_star
    
    def compute_robust_target(
        self,
        rewards: torch.Tensor,
        xi_hat: torch.Tensor,
        next_state_fn,
        gamma: float = 0.99,
        duration_fn=None,
        dones: Optional[torch.Tensor] = None,
        action_probs: torch.Tensor = None,
        action_log_probs: torch.Tensor = None,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute robust backup target per paper Eq. 21.

        y_t^rob = r_t - λρ + min_ξ∈B {γ^Δ(ξ) V(s'(ξ)) + λ d_Q(ξ, ξ̂)}

        Phase 2: V_φ used in inner loop (action_probs can be None).
        Phase 1: V from the fleet actor + critics via Eq. 27.
        """
        xi_star, worst_value, discount, distance = self.find_adversarial_scenario(
            xi_hat, next_state_fn, gamma, duration_fn,
            action_probs=action_probs,
            action_log_probs=action_log_probs,
            alpha=alpha,
        )

        # Force float32 for target arithmetic (called under autocast from trainer)
        with torch.cuda.amp.autocast(enabled=False):
            rewards = rewards.float()
            worst_value = worst_value.float()
            distance = distance.float()
            if isinstance(discount, torch.Tensor):
                discount = discount.float()

            # Bootstrap term from Eq. 21: γ^Δ V(ξ*) + λ d_Q(ξ*, ξ̂)
            # Zero out for terminal transitions so we don't bootstrap past episode end.
            robust_term = discount * worst_value + self.lambda_ * distance
            if dones is not None:
                robust_term = (1.0 - dones.float()) * robust_term

            target = rewards - self.lambda_ * self.config.rho + robust_term
            self._ensure_finite('robust_target', target)

        target_abs_max = target.abs().amax().item()
        self.last_debug = {
            'wdro_lambda': float(self.lambda_.item()),
            'wdro_distance_mean': float(distance.mean().item()),
            'wdro_distance_max': float(distance.max().item()),
            'wdro_worst_value_mean': float(worst_value.mean().item()),
            'wdro_target_mean': float(target.mean().item()),
            'wdro_target_abs_max': float(target_abs_max),
        }
        if target_abs_max > 1e4:
            print(
                f"[WDRO] Large robust target detected: max|target|={target_abs_max:.2f}, "
                f"lambda={self.lambda_.item():.4f}, rho_hat={distance.mean().item():.4f}"
            )

        # Update running ρ_hat for primal-dual
        rho_hat = distance.mean().item()
        if self.update_count == 0:
            # First sample: use directly to avoid bias from 0.0 initialisation
            self.running_rho_hat = rho_hat
        else:
            alpha = min(0.1, 1.0 / (self.update_count + 1))
            self.running_rho_hat = (1 - alpha) * self.running_rho_hat + alpha * rho_hat
        self.update_count += 1

        return target
    
    def get_runtime_state(self) -> Dict[str, Any]:
        """Serialize non-parameter WDRO state so checkpoint resume keeps risk tracking continuous."""
        return {
            'running_rho_hat': float(self.running_rho_hat),
            'update_count': int(self.update_count),
            'use_learned_value': bool(self.use_learned_value),
            'mag_metric': self.mag_metric.get_state(),
        }

    def load_runtime_state(self, state: Optional[Dict[str, Any]]):
        """Restore non-parameter WDRO runtime state from a checkpoint if present."""
        if not state:
            return

        self.running_rho_hat = float(state.get('running_rho_hat', self.running_rho_hat))
        self.update_count = int(state.get('update_count', self.update_count))
        self.use_learned_value = bool(state.get('use_learned_value', self.use_learned_value))
        self.mag_metric.load_state(state.get('mag_metric'))

    def update_lambda(self):
        """
        Update dual variable λ via primal-dual per paper Eq. 24.
        
        λ_{t+1} = [λ_t + η_λ (ρ̂_t - ρ_target)]_+
        """
        # Compute step size with decay
        eta = self.config.lambda_lr / (self.update_count ** 0.5 + 1)
        
        # Gradient: ρ̂ - ρ_target
        grad = self.running_rho_hat - self.config.rho_target
        
        # Update in log space for positivity
        with torch.no_grad():
            new_lambda = (self.lambda_ + eta * grad).clamp(min=1e-6)
            self.log_lambda.data = new_lambda.log()


