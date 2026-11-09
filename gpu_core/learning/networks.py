"""Networks of the vehicle-level agent (workstream E1, E3, E4, E6).

- ``GCNEncoder``: Eq. (24), H1 = SiLU(A X W0 + 1 b0^T), E = SiLU(A H1 W1 + 1 b1^T), no batch norm,
  no dropout (F20, F50, F56). ``local`` evaluates the embedding of one cell per sample from its
  2-hop neighbourhood only, which is exactly the full-graph value; the WDRO adversary perturbs
  those inputs. One encoder is shared; only the critic loss updates it (actor and V use a
  detached embedding).
- ``Actor``: shared over vehicles, Eq. (7): mode {serve, reposition, charge}; pointer over the 91
  reposition slots (candidate features + a learned embedding per slot offset); 6 charging options
  (5 stations + continue); power p = P_max (1 + tanh u) / 2, Eq. (26).
- ``Critic``: Q(o_i, a_i, m_i, g) on the executed action; m_i (1-ring supply/demand, mean-field
  executed actions, free ports at the candidate stations) can be switched off (IAC ablation).
- ``ValueNet``: V(o_i, m_i, g) with the state part of m_i.
LayerNorm is kept inside the MLPs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import ARR_D, ARR_O, F_GLOB, F_HEX, F_STATION, F_VEH, N_CODES, N_STATION_OPTIONS


def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class GCNEncoder(nn.Module):

    def __init__(self, f_in=F_HEX, hidden=128, out=64):
        super().__init__()
        self.W0 = nn.Linear(f_in, hidden, bias=False)
        self.b0 = nn.Parameter(torch.zeros(hidden))
        self.W1 = nn.Linear(hidden, out, bias=False)
        self.b1 = nn.Parameter(torch.zeros(out))
        self.out_dim = out

    def forward(self, X, A):
        """Full graph: X [..., H, F], A [H, H] -> [..., H, out]."""
        h1 = F.silu(A @ self.W0(X) + self.b0)
        return F.silu(A @ self.W1(h1) + self.b1)

    def local(self, X2, geo, cell):
        """Embedding of ``cell`` [B] from its 2-hop features X2 [B, N2, F] (same value as forward)."""
        pos, w1 = geo.l1_pos[cell], geo.l1_w[cell]                     # [B, R, R]
        B, R, _ = pos.shape
        Xn = torch.gather(X2, 1, pos.reshape(B, R * R, 1).expand(-1, -1, X2.shape[-1])).reshape(B, R, R, X2.shape[-1])
        agg1 = (w1[..., None] * Xn).sum(2)                             # [B, R, F] = (A X)[ring]
        h1 = F.silu(self.W0(agg1) + self.b0)                            # [B, R, hidden]
        w2 = geo.l2_w[cell]                                             # [B, R]
        return F.silu(self.W1((w2[..., None] * h1).sum(1)) + self.b1)


def with_arrivals(X2, xi_loc):
    """Replace the arrival channels of 2-hop features by the scenario xi_loc [B, N2, 2] (scaled)."""
    X = X2.clone()
    X[..., ARR_O] = xi_loc[..., 0]
    X[..., ARR_D] = xi_loc[..., 1]
    return X


def local_from_n2(X2, geo, cell):
    """1-ring mean of the hex features computed from the 2-hop block (differentiable in X2)."""
    pos, m = geo.ring_pos_n2[cell], geo.ring_mask[cell].float()
    R = torch.gather(X2, 1, pos[..., None].expand(-1, -1, X2.shape[-1]))
    return (R * m[..., None]).sum(1) / m.sum(1, keepdim=True)


class Actor(nn.Module):

    def __init__(self, emb_dim, hidden=256, key_dim=64, num_slots=91):
        super().__init__()
        self.trunk = mlp(F_VEH + emb_dim + F_HEX + 5 + F_GLOB, hidden, hidden)
        self.mode = nn.Linear(hidden, 3)
        self.query = nn.Linear(hidden, key_dim)
        self.slot_key = mlp(F_HEX + 2, key_dim, key_dim, layers=1)
        self.slot_emb = nn.Parameter(torch.randn(num_slots, key_dim) * 0.1)
        self.station = mlp(hidden + F_STATION, 64, 1, layers=1)
        self.power = nn.Linear(hidden, 2)
        self.key_dim = key_dim

    def forward(self, inp, emb):
        h = self.trunk(torch.cat([inp["veh"], emb, inp["local"], inp["ports"], inp["glob"]], -1))
        neg = torch.finfo(h.dtype).min / 4
        mode = self.mode(h).masked_fill(~inp["mode_ok"], neg)
        keys = self.slot_key(inp["slot_feat"]) + self.slot_emb                   # [B, K, d]
        slot = (keys @ self.query(h)[..., None]).squeeze(-1) / self.key_dim ** 0.5
        slot = slot.masked_fill(~inp["slot_ok"], neg)
        hs = h[:, None].expand(-1, N_STATION_OPTIONS, -1)
        st = self.station(torch.cat([hs, inp["opt_feat"]], -1)).squeeze(-1)
        st = st.masked_fill(~inp["opt_ok"], neg)
        mu, log_std = self.power(h).unbind(-1)
        return dict(mode=mode, slot=slot, station=st, mu=mu, log_std=log_std.clamp(-5.0, 2.0))


class ActionEmbedding(nn.Module):
    """Executed action -> vector: code one-hot, slot (offset embedding + target cell features),
    charging option features, power, served order. Accepts soft one-hots (straight-through)."""

    def __init__(self, num_slots=91, dim=32):
        super().__init__()
        self.slot_emb = nn.Parameter(torch.randn(num_slots, dim) * 0.1)
        self.slot_feat = nn.Linear(F_HEX + 2, dim)
        self.out_dim = N_CODES + dim + F_STATION + 1 + 3

    def forward(self, code, slot_oh, opt_oh, power, order, inp):
        moved = code[:, 1:3].sum(1, keepdim=True)                                 # reposition or stay
        slot = (slot_oh[..., None] * (self.slot_emb + self.slot_feat(inp["slot_feat"]))).sum(1) * moved
        charged = code[:, 3:5].sum(1, keepdim=True)
        st = (opt_oh[..., None] * inp["opt_feat"]).sum(1) * charged
        return torch.cat([code, slot, st, power[:, None] * charged, order * code[:, :1]], -1)


class Critic(nn.Module):

    def __init__(self, emb_dim, hidden=256, num_slots=91, population=True):
        super().__init__()
        self.population = population
        self.act = ActionEmbedding(num_slots)
        d_m = F_HEX + 5 + N_CODES
        self.q = mlp(F_VEH + emb_dim + d_m + F_GLOB + self.act.out_dim, hidden, 1)

    def forward(self, inp, emb, code, slot_oh, opt_oh, power, order):
        m = torch.cat([inp["local"], inp["ports"], inp["meanfield"]], -1)
        if not self.population:
            m = torch.zeros_like(m)
        a = self.act(code, slot_oh, opt_oh, power, order, inp)
        return self.q(torch.cat([inp["veh"], emb, m, inp["glob"], a], -1)).squeeze(-1)


class ValueNet(nn.Module):

    def __init__(self, emb_dim, hidden=256, population=True):
        super().__init__()
        self.population = population
        self.v = mlp(F_VEH + emb_dim + F_HEX + 5 + F_GLOB, hidden, 1)

    def forward(self, veh, emb, local, ports, glob):
        m = torch.cat([local, ports], -1)
        if not self.population:
            m = torch.zeros_like(m)
        return self.v(torch.cat([veh, emb, m, glob], -1)).squeeze(-1)
