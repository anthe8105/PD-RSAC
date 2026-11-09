"""Policy distribution, entropies and the local projection surrogate Pi-hat (E3, E8, E10; F19, F32).

The intention factorises as Eq. (7): mode, then the target of that mode (a reposition slot or a
charging option), then the power when charging. Discrete parts use Gumbel-softmax with the
straight-through estimator (Eq. 25): the forward pass uses the hard sample, gradients flow
through the relaxed sample at temperature tau. Power is a tanh-squashed Gaussian (Eq. 26).

Entropy follows the factorisation: H = H(mode) + pi(repos) H(slot) + pi(charge) (H(option) + H(power)),
each part with its own coefficient (Q18).

Pi-hat is the projection seen by the actor loss and the soft value (Q21): reposition and
charging intentions are already restricted to feasible candidates by the masks, serve becomes the
nearest feasible order, and the power is clipped to what the executor in use allows
(p_min when projected, battery headroom, feeder share for continuing chargers). The executed
action replaces the intention in the forward pass and the gradient passes straight through.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .features import CONTINUE, N_CODES

LOG_EPS = 1e-6


@dataclass
class Sample:
    mode: torch.Tensor      # [B, 3] one-hot (straight-through when relaxed)
    slot: torch.Tensor      # [B, K]
    opt: torch.Tensor       # [B, 6]
    p: torch.Tensor         # [B] power fraction in [0, 1]
    log_p: torch.Tensor     # [B] log density of the power
    u: torch.Tensor         # [B] pre-squash value


def _masked_logp(logits):
    return F.log_softmax(logits, -1)


def entropies(out):
    """Analytic entropies of the discrete parts and the mode probabilities."""
    res = {}
    for k in ("mode", "slot", "station"):
        lp = _masked_logp(out[k])
        p = lp.exp()
        res[k] = -(p * torch.where(p > 0, lp, torch.zeros_like(lp))).sum(-1)
    res["pm"] = _masked_logp(out["mode"]).exp()
    return res


def gumbel_st(logits, tau, hard_only=False, generator=None):
    g = -torch.log(-torch.log(torch.rand(logits.shape, generator=generator, device=logits.device).clamp(1e-10, 1 - 1e-10)))
    y = logits + g
    idx = y.argmax(-1)
    hard = F.one_hot(idx, logits.shape[-1]).to(logits.dtype)
    if hard_only:
        return hard
    soft = F.softmax(y / tau, -1)
    return hard + soft - soft.detach()


def map_action(out):
    """Most probable joint discrete action: argmax over pi(serve), pi(repos) pi(slot k) and
    pi(charge) pi(option o). Taking each factor's argmax separately is not the joint mode: the
    reposition mode pools 91 slots, so 'argmax mode, then argmax slot' turns into 'stay' far more
    often than the policy ever samples it."""
    lm, ls, lo = (F.log_softmax(out[k], -1) for k in ("mode", "slot", "station"))
    best_slot, best_opt = ls.max(-1), lo.max(-1)
    joint = torch.stack([lm[:, 0], lm[:, 1] + best_slot.values, lm[:, 2] + best_opt.values], 1)
    mode = F.one_hot(joint.argmax(-1), 3).float()
    return {"mode": mode, "slot": F.one_hot(best_slot.indices, ls.shape[-1]).float(),
            "station": F.one_hot(best_opt.indices, lo.shape[-1]).float()}


def sample(out, tau, deterministic=False, relaxed=True, generator=None) -> Sample:
    if deterministic:
        oh = map_action(out)
        u = out["mu"]
    else:
        oh = {k: gumbel_st(out[k], tau, hard_only=not relaxed, generator=generator) for k in ("mode", "slot", "station")}
        eps = torch.randn(out["mu"].shape, generator=generator, device=out["mu"].device)
        u = out["mu"] + out["log_std"].exp() * eps
    t = torch.tanh(u)
    p = (1.0 + t) / 2.0
    normal = torch.distributions.Normal(out["mu"], out["log_std"].exp())
    log_p = normal.log_prob(u) - torch.log((1.0 - t * t) / 2.0 + LOG_EPS)
    return Sample(oh["mode"], oh["slot"], oh["station"], p, log_p, u)


def codes(mode, slot, opt):
    """Executed-action code one-hot [B, 5] from (possibly relaxed) intention one-hots."""
    s0, cont = slot[:, :1], opt[:, CONTINUE:CONTINUE + 1]
    return torch.cat([mode[:, :1], mode[:, 1:2] * (1 - s0), mode[:, 1:2] * s0,
                      mode[:, 2:3] * (1 - cont), mode[:, 2:3] * cont], 1)


def project_hat(s: Sample, inp, ctx, p_max_kw):
    """Pi-hat with straight-through gradients. Returns (code, slot, opt, charge share, order).
    The continuous action of a charge is the share of the free battery capacity to add in the
    session (round 6); it is executed as intended, so it passes through unchanged."""
    code = codes(s.mode, s.slot, s.opt)
    order = ctx.order * ctx.serve_ok[:, None].float()
    return code, s.slot, s.opt, s.p, order


def fallback_action(s: Sample, inp, ctx, p_max_kw, projected: bool = False):
    """What a SERVE intention executes when no order is left. Raw executor: the no-op (stay, or
    keep charging). MILP projection: stay for every vehicle, a charger included (round 5: the
    L1 term then prefers staying to continuing, which also pays electricity)."""
    B = s.mode.shape[0]
    dev = s.mode.device
    chg = ctx.charging.float()[:, None] * (0.0 if projected else 1.0)
    code = torch.zeros(B, N_CODES, device=dev)
    code[:, 2] = 1.0 - chg[:, 0]
    code[:, 4] = chg[:, 0]
    slot = torch.zeros_like(s.slot)
    slot[:, 0] = 1.0
    opt = torch.zeros_like(s.opt)
    opt[:, CONTINUE] = 1.0
    p_kw = torch.minimum(torch.maximum(s.p * p_max_kw, inp["opt_lo"][:, CONTINUE]), inp["opt_ub"][:, CONTINUE])
    return code, slot, opt, (p_kw / p_max_kw).detach(), torch.zeros(B, 3, device=dev)


def target_entropy(inp, pm, ratio_mode, ratio_target):
    """Per-sample entropy targets: a ratio of log(#valid candidates) for mode and targets (Q18)."""
    n_mode = inp["mode_ok"].sum(1).float()
    n_slot = inp["slot_ok"].sum(1).float()
    n_opt = inp["opt_ok"].sum(1).float().clamp(min=1)
    h_mode = ratio_mode * torch.log(n_mode)
    h_target = ratio_target * (pm[:, 1] * torch.log(n_slot) + pm[:, 2] * torch.log(n_opt))
    return h_mode, h_target


def onehot_exec(code_idx, slot_idx, opt_idx, num_slots):
    """Hard one-hots of stored executed actions (index -1 = not applicable)."""
    code = F.one_hot(code_idx, N_CODES).float()
    slot = F.one_hot(slot_idx.clamp(min=0), num_slots).float() * (slot_idx >= 0)[:, None]
    opt = F.one_hot(opt_idx.clamp(min=0), 6).float() * (opt_idx >= 0)[:, None]
    return code, slot, opt


def joint_log_prob(out, mode_idx, slot_idx, opt_idx, u):
    """log pi(a) of an intention under Eq. (7): log pi(mode) + [repos] log pi(slot) +
    [charge] (log pi(option) + log N(u; mu, std)). The power density is taken in u-space, which
    leaves the PPO ratio unchanged (the tanh Jacobian cancels)."""
    lm, ls, lo = (F.log_softmax(out[k], -1) for k in ("mode", "slot", "station"))
    lp = lm.gather(1, mode_idx[:, None]).squeeze(1)
    lp_slot = ls.gather(1, slot_idx.clamp(min=0)[:, None]).squeeze(1)
    lp_opt = lo.gather(1, opt_idx.clamp(min=0)[:, None]).squeeze(1)
    lp_u = torch.distributions.Normal(out["mu"], out["log_std"].exp()).log_prob(u)
    return lp + (mode_idx == 1).float() * lp_slot + (mode_idx == 2).float() * (lp_opt + lp_u)


def policy_entropy(out):
    """H(mode) + pi(repos) H(slot) + pi(charge) (H(option) + H(power)) with H(power) the entropy
    of the pre-squash Gaussian (used by MAPPO's entropy bonus)."""
    ent = entropies(out)
    pm = ent["pm"]
    h_u = out["log_std"] + 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi, device=pm.device)))
    return ent["mode"] + pm[:, 1] * ent["slot"] + pm[:, 2] * (ent["station"] + h_u)
