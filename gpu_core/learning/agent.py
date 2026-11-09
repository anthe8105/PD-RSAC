"""Vehicle-level PD-RSAC / SAC agent (workstream E).

Losses for a batch of per-vehicle transitions (s, a_exec, R, s', Delta, done):

- critics (28): (Q_k(s, a_exec) - y)^2 with, before the robust phase (or without WDRO),
  y = R + (1 - done) g [min_k Qbar_k(s', Pi-hat(a')) + alpha . H(s')],  a' ~ pi(s')        (11)
  and in the robust phase
  y = R + (1 - done) [-lambda rho + min_{xi in B} (g V_phi(s'(xi)) + lambda d_Q(xi, xi-hat))]  (21)
  where g = gamma^Delta (semi-MDP) or gamma (fixed-gamma families). Only this loss updates the
  shared GCN encoder (and its Polyak target);
- value (27): V_phi(s) -> min_k Qbar_k(s, Pi-hat(a)) + alpha . H(s), a ~ pi(s);
- actor (29): -[min_k Q_k(s, Pi-hat(a)) + alpha . H(s)], a = Gumbel-ST / reparameterised sample;
- entropy coefficients: three log-alphas with the standard SAC loss log(alpha) (H - H_target).
"""

import copy
from typing import Dict, Optional

import numpy as np
import torch

from .features import CONTINUE, Context, Geometry, Snapshot, build_inputs
from .networks import Actor, Critic, GCNEncoder, ValueNet, local_from_n2
from .policy import entropies, fallback_action, onehot_exec, project_hat, sample, target_entropy


class VehicleSAC:
    algo = "sac"
    on_policy = False

    def __init__(self, env, config, device, projected: bool):
        self.cfg, self.device = config, torch.device(device)
        a, tr = config.agent, config.training
        self.geo = Geometry(env, self.device)
        self.projected = projected
        self.p_max = env.physics.p_max_kw
        K = self.geo.K
        self.enc = GCNEncoder(hidden=a.gcn_hidden, out=a.emb_dim).to(self.device)
        self.actor = Actor(a.emb_dim, a.hidden_dim, a.key_dim, K).to(self.device)
        self.q1 = Critic(a.emb_dim, a.hidden_dim, K, a.critic_population).to(self.device)
        self.q2 = Critic(a.emb_dim, a.hidden_dim, K, a.critic_population).to(self.device)
        self.value = ValueNet(a.emb_dim, a.hidden_dim, a.critic_population).to(self.device)
        self.enc_t, self.q1_t, self.q2_t = (copy.deepcopy(m) for m in (self.enc, self.q1, self.q2))
        for m in (self.enc_t, self.q1_t, self.q2_t):
            m.requires_grad_(False)
        lr = tr.learning_rate
        self.opt_critic = torch.optim.Adam(list(self.enc.parameters()) + list(self.q1.parameters())
                                           + list(self.q2.parameters()), lr=lr.critic)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr.actor)
        self.opt_value = torch.optim.Adam(self.value.parameters(), lr=a.lr_value)
        self.log_alpha = torch.full((3,), float(np.log(a.init_alpha)), device=self.device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr.alpha)
        self.gamma, self.polyak = float(tr.gamma), float(tr.tau)
        self.semi_mdp = bool(a.semi_mdp)
        self.max_grad_norm = float(tr.max_grad_norm)
        self.adversary = None             # set by the trainer when the robust phase starts
        self.updates = 0
        self.tau = a.tau_start

    # ------------------------------------------------------------------ helpers
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def inputs(self, snap: Snapshot, ctx: Context) -> Dict[str, torch.Tensor]:
        return build_inputs(snap, ctx, self.geo, self.projected)

    def embed(self, enc, inp):
        return enc.local(inp["n2_hex"], self.geo, inp["cell"])

    def q_projected(self, inp, emb, ctx, s, critics):
        """Expected min_k Q under Pi-hat: a SERVE intention gets an order with probability p_serve
        (contention in the 1-ring at that epoch), otherwise it is a no-op."""
        q_a = torch.stack([c(inp, emb, *project_hat(s, inp, ctx, self.p_max)) for c in critics]).min(0).values
        q_f = torch.stack([c(inp, emb, *fallback_action(s, inp, ctx, self.p_max, self.projected)) for c in critics]).min(0).values
        w = s.mode[:, 0] * (1.0 - inp["p_serve"])
        return q_a + w * (q_f - q_a)

    def _soft_q(self, inp, emb_actor, emb_q, ctx, critics):
        """E[min_k Q(s, Pi-hat(a))] + alpha . H(s) with a ~ pi(s) (no gradient)."""
        out = self.actor(inp, emb_actor)
        s = sample(out, self.tau, relaxed=False)
        return self.q_projected(inp, emb_q, ctx, s, critics) + self._entropy_bonus(out, s, self.alpha)

    def _entropy_bonus(self, out, s, alpha):
        """alpha . H for the factorised policy. The conditional entropies are weighted by the mode
        probabilities with the gradient stopped: otherwise the mode with the most candidates
        (91 reposition slots) collects an entropy reward larger than any Q difference and the
        mode choice drifts to it regardless of the critic."""
        ent = entropies(out)
        pm = ent["pm"].detach()
        h_t = pm[:, 1] * ent["slot"] + pm[:, 2] * ent["station"]
        h_p = pm[:, 2] * (-s.log_p)
        return alpha[0] * ent["mode"] + alpha[1] * h_t + alpha[2] * h_p

    def discount(self, delta):
        return self.gamma ** delta.float() if self.semi_mdp else torch.full_like(delta, self.gamma, dtype=torch.float32)

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, snap: Snapshot, ctx: Context, deterministic=False, generator=None):
        inp = self.inputs(snap, ctx)
        out = self.actor(inp, self.embed(self.enc, inp))
        return sample(out, self.tau, deterministic=deterministic, relaxed=False, generator=generator), inp, out

    # ------------------------------------------------------------------ learning
    def update(self, batch, robust: bool = False) -> Dict[str, float]:
        snap, cur, nxt, rows = batch["snap"], batch["cur"], batch["nxt"], batch["rows"]
        ci, ni = self.inputs(snap, cur), self.inputs(snap, nxt)
        R, done = rows["R"].float(), rows["done"].float()
        disc = self.discount(rows["delta"])
        code, slot, opt = onehot_exec(rows["code"].long(), rows["slot"].long(), rows["opt"].long(), self.geo.K)
        power, order = rows["power"].float(), rows["ex_order"].float()
        stats = {}

        # ---- critic target
        robust = robust and self.adversary is not None
        if not robust:
            with torch.no_grad():
                emb_n_online, emb_n_target = self.embed(self.enc, ni), self.embed(self.enc_t, ni)
                cont = self._soft_q(ni, emb_n_online, emb_n_target, nxt, (self.q1_t, self.q2_t))
                y = R + (1.0 - done) * disc * cont
        else:
            def value_fn(Xp):          # V_phi(s'(xi)); parameters frozen, gradient only w.r.t. xi
                emb = self.enc.local(Xp, self.geo, nxt.cell)
                return self.value(ni["veh"], emb, local_from_n2(Xp, self.geo, nxt.cell), ni["ports"], ni["glob"])
            frozen = list(self.enc.parameters()) + list(self.value.parameters())
            for p in frozen:
                p.requires_grad_(False)
            try:
                v_star, d_star = self.adversary.worst_case(value_fn, ni["n2_hex"], nxt.cell, disc)
            finally:
                for p in frozen:
                    p.requires_grad_(True)
            lam, rho = self.adversary.lam, self.adversary.rho
            y = R + (1.0 - done) * (-lam * rho + disc * v_star + lam * d_star)
            live = 1.0 - done
            rho_hat = float((d_star * live).sum() / live.sum().clamp(min=1.0))
            self.adversary.update_dual(rho_hat)
            # robust minus nominal target, R + disc V(s'(xi-hat)), on the live transitions (<= 0)
            gap = (-lam * rho + disc * (v_star - self.adversary.last_nominal) + lam * d_star) * live
            stats.update(wdro_lambda=lam, wdro_rho_hat=rho_hat, wdro_v_star=float(v_star.mean()),
                         wdro_v_nom=float(self.adversary.last_nominal.mean()),
                         wdro_gap=float(gap.sum() / live.sum().clamp(min=1.0)))

        # ---- critics (+ encoder)
        emb = self.embed(self.enc, ci)
        q1 = self.q1(ci, emb, code, slot, opt, power, order)
        q2 = self.q2(ci, emb, code, slot, opt, power, order)
        loss_q = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
        self.opt_critic.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.opt_critic.param_groups[0]["params"], self.max_grad_norm)
        self.opt_critic.step()

        emb_d = emb.detach()
        # ---- value (27)
        with torch.no_grad():
            v_target = self._soft_q(ci, emb_d, self.embed(self.enc_t, ci), cur, (self.q1_t, self.q2_t))
        v = self.value(ci["veh"], emb_d, ci["local"], ci["ports"], ci["glob"])
        loss_v = ((v - v_target) ** 2).mean()
        self.opt_value.zero_grad()
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        self.opt_value.step()

        # ---- actor (29) through Pi-hat with straight-through gradients
        out = self.actor(ci, emb_d)
        s = sample(out, self.tau, relaxed=True)
        q_pi = self.q_projected(ci, emb_d, cur, s, (self.q1, self.q2))
        alpha = self.alpha.detach()
        loss_pi = -(q_pi + self._entropy_bonus(out, s, alpha)).mean()
        self.opt_actor.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.opt_actor.step()

        # ---- entropy coefficients
        ent = entropies(out)
        pm = ent["pm"].detach()
        h_mode_t, h_target_t = target_entropy(ci, pm, self.cfg.agent.entropy_ratio_mode, self.cfg.agent.entropy_ratio_target)
        h_t = pm[:, 1] * ent["slot"] + pm[:, 2] * ent["station"]
        h_p = pm[:, 2] * (-s.log_p)
        gaps = torch.stack([(ent["mode"] - h_mode_t).mean(), (h_t - h_target_t).mean(),
                            (h_p - pm[:, 2] * self.cfg.agent.entropy_target_power).mean()]).detach()
        loss_a = (self.log_alpha * gaps).sum()
        self.opt_alpha.zero_grad()
        loss_a.backward()
        self.opt_alpha.step()

        # ---- Polyak averaging of the target critics and encoder
        with torch.no_grad():
            for src, dst in ((self.enc, self.enc_t), (self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, pt in zip(src.parameters(), dst.parameters()):
                    pt.mul_(1.0 - self.polyak).add_(p, alpha=self.polyak)
        self.updates += 1
        al = self.alpha.detach().cpu().numpy()
        stats.update(loss_q=loss_q.item(), loss_v=loss_v.item(), loss_pi=loss_pi.item(), q_mean=float(q1.detach().mean()),
                     y_mean=float(y.mean()), v_mean=float(v.detach().mean()), alpha_mode=float(al[0]), alpha_target=float(al[1]),
                     alpha_power=float(al[2]), h_mode=float(ent["mode"].detach().mean()), pi_serve=float(pm[:, 0].mean()),
                     pi_repos=float(pm[:, 1].mean()), pi_charge=float(pm[:, 2].mean()))
        return stats

    def eval_modules(self):
        return (self.enc, self.actor, self.q1, self.q2, self.value)

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self):
        return {
            "enc": self.enc.state_dict(), "enc_t": self.enc_t.state_dict(), "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(), "q1_t": self.q1_t.state_dict(),
            "q2_t": self.q2_t.state_dict(), "value": self.value.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
            "opt": {k: getattr(self, k).state_dict() for k in ("opt_critic", "opt_actor", "opt_value", "opt_alpha")},
            "adversary": self.adversary.state_dict() if self.adversary is not None else None,
            "updates": self.updates, "tau": self.tau,
        }

    def load_state_dict(self, d, with_optim=True):
        for k in ("enc", "enc_t", "actor", "q1", "q2", "q1_t", "q2_t", "value"):
            getattr(self, k).load_state_dict(d[k])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device))
        if with_optim:
            for k, v in d["opt"].items():
                getattr(self, k).load_state_dict(v)
        if d.get("adversary") is not None and self.adversary is not None:
            self.adversary.load_state_dict(d["adversary"])
        self.updates, self.tau = int(d["updates"]), float(d["tau"])
