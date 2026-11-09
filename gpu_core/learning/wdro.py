"""Graph-aligned Wasserstein robustness for per-vehicle transitions (workstream E7; F08, F09,
F21, F22, F37, F38; decisions Q5-Q7).

- Scenario xi = this epoch's arrivals by origin cell and by destination cell (2m numbers, scaled
  by the mean vehicles per cell). xi-hat of a transition is the data at its next epoch t + Delta.
- s'(xi): the next state's arrival channels are replaced by xi, which changes the vehicle's
  GCN embedding and its 1-ring summary; V_phi is evaluated on that surrogate. Only the 2-hop
  neighbourhood of the next cell influences V (two GCN layers, 1-ring summary), so the adversary
  moves those coordinates. The coordinates outside are free to follow at no effect on V, so the
  transport cost of a local move is its cheapest completion: the Schur complement
  S_c = Q_ii - Q_io Q_oo^-1 Q_oi = ((Q^-1)_ii)^-1 of the block (round 8). This is the exact
  restriction of the full problem without the box on the outside coordinates; ``schur=False``
  keeps the plain block Q_ii (rounds <= 7).
- Metric (17): d(x, y) = sqrt((x - y)^T Q (x - y) / q), q = 2m, Q = diag(w) + beta * Q_graph;
  w is the inverse variance of each coordinate over the training epochs, renormalised to mean 1;
  Q_graph = blockdiag(L, L) with L the sum of the normalised Laplacians of the H3 1-ring graph and
  of the training OD graph. ``metric='diag'`` sets beta = 0, ``'identity'`` uses Q = I (N11).
- Support B: a box xi in [max(0, xi-hat - k sigma), xi-hat + k sigma], independent of rho; the
  projection onto a box is exact (F38).
- Inner minimisation (21): K projected steps on f = gamma^Delta V(s'(xi)) + lambda d(xi, xi-hat),
  keeping the best iterate. ``step_rule='metric'`` (round 8) steps along the steepest descent
  direction of the metric, p = S_c^-1 grad f, with length ``step_size_d`` measured in d_Q, so the
  metric shapes the scenario the adversary finds; ``'sigma'`` (rounds <= 7) steps along
  sigma * grad f / max|grad f|, the same search for every metric. Then the box projection.
- Target (21) with the terminal mask on the whole continuation, including -lambda rho (F37);
  dual update (22)-(23) with the current batch's realised radius (F22).
"""

import math
from typing import Optional

import numpy as np
import torch

from .features import ARR_D, ARR_O, Geometry
from .networks import local_from_n2, with_arrivals


def _norm_laplacian(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float64).copy()
    np.fill_diagonal(A, 0.0)
    d = A.sum(1)
    inv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    L = np.eye(len(A)) - inv[:, None] * A * inv[None, :]
    L[d == 0] = 0.0
    L[:, d == 0] = 0.0
    return L


class ScenarioMetric:

    def __init__(self, geo: Geometry, arrivals: np.ndarray, od: Optional[np.ndarray], beta: float = 0.3,
                 metric: str = "graph", support_sigmas: float = 3.0, schur: bool = True):
        """``arrivals`` [n_epochs, 2H] scaled arrivals of the training split; ``od`` [H, H] counts."""
        H = geo.H
        var = arrivals.var(0)
        sigma = np.sqrt(var)
        med = np.median(var[var > 0]) if (var > 0).any() else 1.0
        w = 1.0 / np.clip(var, 0.1 * med, 10.0 * med)
        w = w / w.mean()
        if metric == "identity":
            Q = np.eye(2 * H)
        else:
            Q = np.diag(w)
            if metric == "graph" and beta > 0:
                L = _norm_laplacian(_ring_adj(geo))
                if od is not None:
                    L = L + _norm_laplacian(od + od.T)
                Q = Q + beta * np.kron(np.eye(2), L)
            elif metric not in ("graph", "diag"):
                raise ValueError(f"unknown metric {metric}")
        self.metric = metric
        self.schur = bool(schur)
        self.Q = torch.as_tensor(Q, dtype=torch.float32, device=geo.device)
        self.S, self.S_inv = self._local_blocks(geo, Q)              # [H, 2 N2, 2 N2] each
        self.sigma = torch.as_tensor(np.maximum(sigma, 1e-3), dtype=torch.float32, device=geo.device)   # [2H]
        self.k = support_sigmas
        self.q = 2 * H
        self.geo = geo

    def _local_blocks(self, geo: Geometry, Q: np.ndarray):
        """Per cell, the metric restricted to its 2-hop coordinates and its inverse (zero on padding).
        Schur: S_c = ((Q^-1)_ii)^-1 and S_c^-1 = (Q^-1)_ii; plain block: Q_ii and its inverse."""
        H = geo.H
        idx, msk = geo.n2_idx.cpu().numpy(), geo.n2_mask.cpu().numpy()
        N2 = idx.shape[1]
        Qinv = np.linalg.inv(Q) if self.schur else None
        S = np.zeros((H, 2 * N2, 2 * N2))
        S_inv = np.zeros((H, 2 * N2, 2 * N2))
        for c in range(H):
            valid = np.nonzero(msk[c])[0]
            pos = np.concatenate([valid, N2 + valid])                      # positions in [origin | destination]
            coord = np.concatenate([idx[c][valid], idx[c][valid] + H])     # coordinates of xi
            if self.schur:
                inv_b = Qinv[np.ix_(coord, coord)]
                blk = np.linalg.inv(inv_b)
            else:
                blk = Q[np.ix_(coord, coord)]
                inv_b = np.linalg.inv(blk)
            S[c][np.ix_(pos, pos)] = blk
            S_inv[c][np.ix_(pos, pos)] = inv_b
        return (torch.as_tensor(S, dtype=torch.float32, device=geo.device),
                torch.as_tensor(S_inv, dtype=torch.float32, device=geo.device))

    def local(self, cell):
        """Coordinates, metric block (Schur complement or plain Q_ii), sigma and padding mask of the
        2-hop neighbourhood of ``cell``."""
        g = self.geo
        idx = g.n2_idx[cell]                                    # [B, N2]
        m = g.n2_mask[cell]
        H = g.H
        coord = torch.cat([idx, idx + H], 1)                     # origin block, destination block
        mask = torch.cat([m, m], 1)
        return coord, self.S[cell], self.sigma[coord] * mask, mask

    def local_inverse(self, cell):
        """Inverse of the local metric block (zero on padding): the preconditioner of the metric step."""
        return self.S_inv[cell]

    def distance(self, delta, Qb):
        quad = torch.einsum("bi,bij,bj->b", delta, Qb, delta)
        return torch.sqrt(quad.clamp(min=0) / self.q + 1e-12)


def arrivals_from_dataset(dataset, geo: Geometry, step_s: int):
    """Scaled arrivals [n_epochs, 2H] of every epoch of the split and its OD counts [H, H]."""
    t = dataset.pickup_s - dataset.start_s
    epoch = np.ceil(t / step_s).astype(np.int64)                 # pickup in (t_k - dt, t_k] -> epoch k
    n = int(epoch.max()) + 1 if len(epoch) else 1
    H = geo.H
    pu = dataset.pickup_idx.numpy()
    do = dataset.dropoff_idx.numpy()
    o = np.bincount(epoch * H + pu, minlength=n * H).reshape(n, H)
    d = np.bincount(epoch * H + do, minlength=n * H).reshape(n, H)
    od = np.bincount(pu * H + do, minlength=H * H).reshape(H, H).astype(np.float64)
    return np.concatenate([o, d], 1) / geo.count_scale, od


def arrivals_from_replay(replay, geo: Geometry):
    xs = [e.xi[:e.hex.shape[0] - 1] for e in replay.episodes]
    return np.concatenate(xs) / geo.count_scale if xs else np.zeros((1, 2 * geo.H))


def _ring_adj(geo: Geometry) -> np.ndarray:
    H = geo.H
    A = np.zeros((H, H))
    idx, m = geo.ring_idx.cpu().numpy(), geo.ring_mask.cpu().numpy()
    for h in range(H):
        A[h, idx[h][m[h]]] = 1.0
    return A


class Adversary:
    """Inner loop of (21) and the primal-dual radius tracking of (22)-(23)."""

    def __init__(self, metric: ScenarioMetric, rho: float = 0.3, rho_target: float = 0.2, eta0: float = 0.01,
                 steps: int = 5, step_size: float = 0.5, lam: float = 1.0, step_rule: str = "metric",
                 step_size_d: float = 0.1):
        if step_rule not in ("metric", "sigma"):
            raise ValueError(f"unknown wdro.step_rule {step_rule}")
        self.metric, self.rho, self.rho_target, self.eta0 = metric, rho, rho_target, eta0
        self.steps, self.step_size = steps, step_size             # step_size: sigma units ('sigma' rule)
        self.step_rule, self.step_size_d = step_rule, step_size_d  # step_size_d: d_Q units ('metric' rule)
        self.lam = lam
        self.t = 0
        self.last_nominal = None          # V(s'(xi-hat)) of the last call (diagnostics)

    def worst_case(self, value_fn, X2, cell, discount):
        """xi* in B minimising discount * V(s'(xi)) + lambda d(xi, xi-hat), per transition."""
        coord, Qb, sigma, mask = self.metric.local(cell)
        B, N2 = X2.shape[0], X2.shape[1]
        xi_hat = torch.cat([X2[..., ARR_O], X2[..., ARR_D]], 1)            # [B, 2 N2] (scaled)
        lo = torch.clamp(xi_hat - self.metric.k * sigma, min=0.0) - xi_hat
        hi = self.metric.k * sigma
        def objective(delta, grad):
            xi = (xi_hat + delta) * mask
            Xp = with_arrivals(X2, torch.stack([xi[:, :N2], xi[:, N2:]], -1))
            v = value_fn(Xp)
            d = self.metric.distance(delta, Qb)
            return discount * v + self.lam * d, v, d

        P = self.metric.local_inverse(cell) if self.step_rule == "metric" else None
        delta = torch.zeros_like(xi_hat)
        with torch.no_grad():
            best_f, best_v, best_d = objective(delta, False)          # xi = xi-hat is feasible
        self.last_nominal = best_v.detach()
        for _ in range(self.steps):
            delta.requires_grad_(True)
            f, _, _ = objective(delta, True)
            g, = torch.autograd.grad(f.sum(), delta)
            g = g * mask
            if P is not None:                                  # steepest descent of the metric, fixed d_Q length
                p = torch.einsum("bij,bj->bi", P, g) * mask
                step = self.step_size_d * p / self.metric.distance(p, Qb).clamp(min=1e-6)[:, None]
            else:                                              # rounds <= 7: same sigma-scaled search for every metric
                step = self.step_size * sigma * g / g.abs().amax(1, keepdim=True).clamp(min=1e-12)
            delta = torch.maximum(torch.minimum(delta.detach() - step, hi), lo) * mask
            with torch.no_grad():
                f, v, d = objective(delta, False)
                better = f < best_f                                    # keep the best iterate per sample
                best_f = torch.where(better, f, best_f)
                best_v = torch.where(better, v, best_v)
                best_d = torch.where(better, d, best_d)
        return best_v, best_d

    def update_dual(self, rho_hat: float):
        self.t += 1
        self.lam = max(0.0, self.lam + self.eta0 / math.sqrt(self.t) * (rho_hat - self.rho_target))

    def state_dict(self):
        return {"lam": self.lam, "t": self.t}

    def load_state_dict(self, d):
        self.lam, self.t = float(d["lam"]), int(d["t"])
