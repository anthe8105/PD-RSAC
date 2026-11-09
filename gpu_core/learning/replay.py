"""Two-tier replay with per-vehicle semi-MDP transitions (workstream E5, E9; Q2, Q17).

Tier 1 stores one ``Snapshot`` per decision epoch. Tier 2 stores one row per vehicle decision:

    (episode, t, t + Delta_i, context at t and at t + Delta_i, intention, executed action, R_i, done)

with R_i = sum_{k < Delta_i} gamma^k r_{i, t+k} (PD-RSAC family; the fixed-gamma families sum the
rewards undiscounted and bootstrap with gamma). A vehicle's transition closes at its next decision
epoch, which is exactly t + Delta_i of the executed action (checked). Rows live in a window of the
last ``window`` episodes; sampling favours recent episodes (weight ``recency ** age``).
"""

from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch

from .features import F_GLOB, F_HEX, N_CODES, Context, Snapshot

CTX_FIELDS = ("cell", "energy", "charging", "own_station", "serve_ok", "n_orders")


class _Episode:

    def __init__(self, steps, H, S, cap):
        self.hex = np.zeros((steps + 1, H, F_HEX), np.float32)
        self.xi = np.zeros((steps + 1, 2 * H), np.float32)
        self.free = np.zeros((steps + 1, S), np.float32)
        self.glob = np.zeros((steps + 1, F_GLOB), np.float32)
        self.exec_counts = np.zeros((steps + 1, H, N_CODES), np.float32)
        self.share = np.zeros(steps + 1, np.float32)
        self.serve_intent = np.zeros((steps + 1, H), np.float32)
        self.rows: Dict[str, np.ndarray] = {}
        self.n = 0
        self.cap = cap

    def add_rows(self, rows: Dict[str, np.ndarray]):
        k = len(next(iter(rows.values())))
        if k == 0:
            return
        if not self.rows:
            self.rows = {name: np.zeros((self.cap,) + v.shape[1:], v.dtype) for name, v in rows.items()}
        if self.n + k > self.cap:
            self.cap = max(2 * self.cap, self.n + k)
            for name in self.rows:
                grown = np.zeros((self.cap,) + self.rows[name].shape[1:], self.rows[name].dtype)
                grown[:self.n] = self.rows[name][:self.n]
                self.rows[name] = grown
        for name, v in rows.items():
            self.rows[name][self.n:self.n + k] = v
        self.n += k


class ReplayBuffer:

    def __init__(self, steps_per_episode: int, num_hexes: int, num_stations: int, window: int = 20,
                 recency: float = 0.8, rows_hint: int = 4096):
        self.steps, self.H, self.S = steps_per_episode, num_hexes, num_stations
        self.window, self.recency = window, recency
        self.episodes: deque = deque(maxlen=window)
        self.rows_hint = rows_hint

    # ---------------------------------------------------------------- writing
    def new_episode(self):
        self.episodes.append(_Episode(self.steps, self.H, self.S, self.rows_hint))

    @property
    def current(self) -> _Episode:
        return self.episodes[-1]

    def put_snapshot(self, t: int, snap: Snapshot):
        ep = self.current
        ep.hex[t] = snap.hex[0].cpu().numpy()
        ep.xi[t] = snap.xi[0].cpu().numpy()
        ep.free[t] = snap.free_ports[0].cpu().numpy()
        ep.glob[t] = snap.glob[0].cpu().numpy()
        ep.share[t] = float(snap.feeder_share[0])

    def put_exec_counts(self, t: int, counts: torch.Tensor, serve_intent: torch.Tensor):
        self.current.exec_counts[t] = counts.cpu().numpy()
        self.current.serve_intent[t] = serve_intent.cpu().numpy()

    def add_transitions(self, rows: Dict[str, np.ndarray]):
        self.current.add_rows(rows)

    def __len__(self):
        return sum(e.n for e in self.episodes)

    # ---------------------------------------------------------------- sampling
    def sample(self, batch: int, rng: np.random.Generator, device) -> Optional[Dict]:
        eps = [e for e in self.episodes if e.n > 0]
        if not eps or sum(e.n for e in eps) < batch:
            return None
        ages = np.arange(len(eps))[::-1]
        w = np.array([e.n for e in eps], np.float64) * self.recency ** ages
        which = rng.choice(len(eps), size=batch, p=w / w.sum())
        order = np.argsort(which, kind="stable")
        picks, pos = [], 0
        for e_i in np.unique(which):
            n_take = int((which == e_i).sum())
            r = rng.integers(0, eps[e_i].n, size=n_take)
            picks.append((e_i, eps[e_i], r, order[pos:pos + n_take]))
            pos += n_take
        return self._assemble(picks, batch, device)

    def current_batches(self, batch: int, rng: np.random.Generator, device, extra: Optional[Dict[str, np.ndarray]] = None):
        """Shuffled minibatches covering every row of the current episode once (on-policy
        learners). ``extra`` adds per-row arrays (e.g. advantages) to ``rows``."""
        ep = self.current
        perm = rng.permutation(ep.n)
        for lo in range(0, ep.n, batch):
            r = perm[lo:lo + batch]
            b = self._assemble([(0, ep, r, np.arange(len(r)))], len(r), device)
            if extra:
                for k, v in extra.items():
                    b["rows"][k] = torch.as_tensor(v[r], device=device)
            yield b

    def current_rows(self, device, chunk: int = 4096):
        """The current episode's rows in order, in chunks (for value passes over a rollout)."""
        ep = self.current
        for lo in range(0, ep.n, chunk):
            r = np.arange(lo, min(ep.n, lo + chunk))
            yield r, self._assemble([(0, ep, r, np.arange(len(r)))], len(r), device)

    @staticmethod
    def _assemble(picks, batch: int, device) -> Dict:
        out: Dict[str, np.ndarray] = {}
        snaps, snap_keys = [], {}
        cur_ids, nxt_ids = np.zeros(batch, np.int64), np.zeros(batch, np.int64)
        for e_i, ep, r, sl in picks:
            for name, arr in ep.rows.items():
                buf = out.setdefault(name, np.zeros((batch,) + arr.shape[1:], arr.dtype))
                buf[sl] = arr[r]
            for j, (t, tn) in zip(sl, zip(ep.rows["t"][r], ep.rows["t_next"][r])):
                for key, tt, target in (((e_i, int(t)), t, cur_ids), ((e_i, int(tn)), tn, nxt_ids)):
                    if key not in snap_keys:
                        snap_keys[key] = len(snaps)
                        snaps.append((ep, int(tt)))
                    target[j] = snap_keys[key]
        snap = Snapshot(
            hex=torch.as_tensor(np.stack([e.hex[t] for e, t in snaps]), device=device),
            xi=torch.as_tensor(np.stack([e.xi[t] for e, t in snaps]), device=device),
            free_ports=torch.as_tensor(np.stack([e.free[t] for e, t in snaps]), device=device),
            glob=torch.as_tensor(np.stack([e.glob[t] for e, t in snaps]), device=device),
            exec_counts=torch.as_tensor(np.stack([e.exec_counts[t] for e, t in snaps]), device=device),
            feeder_share=torch.as_tensor(np.array([e.share[t] for e, t in snaps]), device=device),
            serve_intent=torch.as_tensor(np.stack([e.serve_intent[t] for e, t in snaps]), device=device))
        t = {k: torch.as_tensor(v, device=device) for k, v in out.items()}
        cur = _ctx(t, "", torch.as_tensor(cur_ids, device=device))
        nxt = _ctx(t, "n_", torch.as_tensor(nxt_ids, device=device))
        return dict(snap=snap, cur=cur, nxt=nxt, rows=t)


def _ctx(t, prefix, ids) -> Context:
    return Context(snap=ids, cell=t[prefix + "cell"].long(), energy=t[prefix + "energy"].float(),
                   charging=t[prefix + "charging"].bool(), own_station=t[prefix + "own_station"].long(),
                   serve_ok=t[prefix + "serve_ok"].bool(), n_orders=t[prefix + "n_orders"].float(),
                   order=t[prefix + "order"].float())


def ctx_rows(ctx: Context, prefix: str) -> Dict[str, np.ndarray]:
    d = {prefix + f: getattr(ctx, f).cpu().numpy() for f in CTX_FIELDS}
    d[prefix + "order"] = ctx.order.cpu().numpy()
    return d


class TransitionTracker:
    """Open per-vehicle transitions: rewards accumulate until the vehicle decides again."""

    def __init__(self, num_vehicles: int, gamma: float, semi_mdp: bool):
        self.V, self.gamma, self.semi_mdp = num_vehicles, gamma, semi_mdp
        self.open = np.zeros(num_vehicles, bool)
        self.k = np.zeros(num_vehicles, np.int64)
        self.R = np.zeros(num_vehicles, np.float64)
        self.delta = np.zeros(num_vehicles, np.int64)
        self.data: Dict[str, np.ndarray] = {}

    def reset(self):
        self.open[:] = False
        self.data = {}

    def start(self, vehicles: np.ndarray, t: int, rows: Dict[str, np.ndarray], delta: np.ndarray):
        if not self.data:
            self.data = {k: np.zeros((self.V,) + v.shape[1:], v.dtype) for k, v in rows.items()}
        for k, v in rows.items():
            self.data[k][vehicles] = v
        self.data["t"][vehicles] = t
        self.open[vehicles] = True
        self.k[vehicles] = 0
        self.R[vehicles] = 0.0
        self.delta[vehicles] = delta

    def reward(self, r: np.ndarray):
        o = self.open
        disc = self.gamma ** self.k[o] if self.semi_mdp else 1.0
        self.R[o] += disc * r[o]
        self.k[o] += 1

    def close(self, vehicles: np.ndarray, t_next: int, next_rows: Dict[str, np.ndarray], done: bool):
        vehicles = vehicles[self.open[vehicles]]
        if len(vehicles) == 0:
            return {}
        known = self.delta[vehicles] > 0          # 0 = charging session, length known only at its end
        if not done and (known & (self.k[vehicles] != self.delta[vehicles])).any():
            bad = vehicles[known & (self.k[vehicles] != self.delta[vehicles])]
            raise RuntimeError(f"vehicle {bad[:5]} decided after {self.k[bad[:5]]} epochs, executed Delta was {self.delta[bad[:5]]}")
        rows = {k: v[vehicles].copy() for k, v in self.data.items()}
        rows["t_next"] = np.full(len(vehicles), t_next, np.int64)
        rows["R"] = self.R[vehicles].astype(np.float32)
        rows["delta"] = self.k[vehicles].astype(np.int64)
        rows["done"] = np.full(len(vehicles), done, bool)
        for k, v in next_rows.items():
            rows[k] = v
        self.open[vehicles] = False
        return rows
