"""Vehicle-level MADDPG baseline (workstream G; plan section 7, Q12).

Same encoder, actor network, candidates, critic input and Pi-hat as PD-RSAC; what changes is the
learning rule:

- deterministic actor: the discrete parts are Gumbel-softmax straight-through samples of the
  actor's logits (the MADDPG treatment of discrete actions), the power is (1 + tanh mu) / 2 with
  Gaussian exploration noise in u-space while collecting; evaluation uses the joint mode and mu;
- one critic Q(o_i, a_i, m_i, g) on the executed action (local population critic; the
  'centralised' information is m_i and g), target y = R + (1 - done) d Qbar(s', Pi-hat(mu_bar(s')))
  with the target actor and target critic; d = gamma per decision (the presets, Q12), or
  gamma^Delta with ``agent.semi_mdp``;
- actor loss -Q(s, Pi-hat(mu(s))) through Pi-hat with straight-through gradients, no entropy;
- Polyak tau = training.tau (0.005) for the encoder, critic and actor (Q11).
"""

import copy
from typing import Dict

import torch

from .agent import VehicleSAC
from .features import Context, Snapshot
from .policy import Sample, map_action, onehot_exec, gumbel_st


def ddpg_sample(out, tau, *, noise_std=0.0, deterministic=False, relaxed=True, generator=None) -> Sample:
    if deterministic:
        oh = map_action(out)
    else:
        oh = {k: gumbel_st(out[k], tau, hard_only=not relaxed, generator=generator) for k in ("mode", "slot", "station")}
    u = out["mu"]
    if noise_std > 0 and not deterministic:
        u = u + noise_std * torch.randn(u.shape, generator=generator, device=u.device)
    p = (1.0 + torch.tanh(u)) / 2.0
    return Sample(oh["mode"], oh["slot"], oh["station"], p, torch.zeros_like(p), u)


class VehicleMADDPG(VehicleSAC):
    algo = "maddpg"

    def __init__(self, env, config, device, projected: bool):
        super().__init__(env, config, device, projected)
        a, tr = config.agent, config.training
        self.q2 = self.q2_t = self.value = self.opt_value = None     # one critic, no V_phi, no entropy
        self.actor_t = copy.deepcopy(self.actor).requires_grad_(False)
        self.opt_critic = torch.optim.Adam(list(self.enc.parameters()) + list(self.q1.parameters()), lr=tr.learning_rate.critic)
        self.noise_std = float(a.explore_std)

    @torch.no_grad()
    def act(self, snap: Snapshot, ctx: Context, deterministic=False, generator=None):
        inp = self.inputs(snap, ctx)
        out = self.actor(inp, self.embed(self.enc, inp))
        s = ddpg_sample(out, self.tau, noise_std=self.noise_std, deterministic=deterministic, relaxed=False,
                        generator=generator)
        return s, inp, out

    def update(self, batch, robust: bool = False) -> Dict[str, float]:
        snap, cur, nxt, rows = batch["snap"], batch["cur"], batch["nxt"], batch["rows"]
        ci, ni = self.inputs(snap, cur), self.inputs(snap, nxt)
        R, done = rows["R"].float(), rows["done"].float()
        disc = self.discount(rows["delta"])
        code, slot, opt = onehot_exec(rows["code"].long(), rows["slot"].long(), rows["opt"].long(), self.geo.K)
        power, order = rows["power"].float(), rows["ex_order"].float()

        with torch.no_grad():
            emb_t = self.embed(self.enc_t, ni)
            s_next = ddpg_sample(self.actor_t(ni, emb_t), self.tau, relaxed=False)
            y = R + (1.0 - done) * disc * self.q_projected(ni, emb_t, nxt, s_next, (self.q1_t,))

        emb = self.embed(self.enc, ci)
        q = self.q1(ci, emb, code, slot, opt, power, order)
        loss_q = ((q - y) ** 2).mean()
        self.opt_critic.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.opt_critic.param_groups[0]["params"], self.max_grad_norm)
        self.opt_critic.step()

        emb_d = emb.detach()
        out = self.actor(ci, emb_d)
        s = ddpg_sample(out, self.tau, relaxed=True)
        loss_pi = -self.q_projected(ci, emb_d, cur, s, (self.q1,)).mean()
        self.opt_actor.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.opt_actor.step()

        with torch.no_grad():
            for src, dst in ((self.enc, self.enc_t), (self.q1, self.q1_t), (self.actor, self.actor_t)):
                for p, pt in zip(src.parameters(), dst.parameters()):
                    pt.mul_(1.0 - self.polyak).add_(p, alpha=self.polyak)
        self.updates += 1
        pm = torch.softmax(out["mode"].detach(), -1)
        return dict(loss_q=loss_q.item(), loss_pi=loss_pi.item(), q_mean=float(q.detach().mean()), y_mean=float(y.mean()),
                    pi_serve=float(pm[:, 0].mean()), pi_repos=float(pm[:, 1].mean()), pi_charge=float(pm[:, 2].mean()))

    def state_dict(self):
        return {
            "enc": self.enc.state_dict(), "enc_t": self.enc_t.state_dict(), "actor": self.actor.state_dict(),
            "actor_t": self.actor_t.state_dict(), "q1": self.q1.state_dict(), "q1_t": self.q1_t.state_dict(),
            "opt": {k: getattr(self, k).state_dict() for k in ("opt_critic", "opt_actor")},
            "updates": self.updates, "tau": self.tau,
        }

    def load_state_dict(self, d, with_optim=True):
        for k in ("enc", "enc_t", "actor", "actor_t", "q1", "q1_t"):
            getattr(self, k).load_state_dict(d[k])
        if with_optim:
            for k, v in d["opt"].items():
                getattr(self, k).load_state_dict(v)
        self.updates, self.tau = int(d["updates"]), float(d["tau"])

    def eval_modules(self):
        return (self.enc, self.actor, self.q1)
