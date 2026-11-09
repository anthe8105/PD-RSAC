"""Vehicle-level MAPPO baseline (workstream G; plan section 7, Q12).

Same encoder, actor network, candidates and executor / projection as PD-RSAC. On-policy:

- one episode is collected with the stochastic policy (exact categorical samples, Gaussian power),
  then ``update_epochs`` passes of minibatch PPO over every per-vehicle transition of it;
- value V(o_i, m_i, g) (local value; ``ValueNet``), trained with the shared encoder;
- per-vehicle transitions close when the vehicle decides again; GAE(lambda) along each vehicle's
  chain of decisions with the discount of each transition: gamma per decision (fixed gamma, Q12;
  R = undiscounted rewards over the transition) or, with ``agent.semi_mdp``, gamma^Delta for a
  transition of Delta epochs (R = sum_k gamma^k r_{t+k}, as PD-RSAC); the episode end is a
  truncation, so the last transition is bootstrapped;
- clipped surrogate on the joint log-probability of the intention (Eq. 7), entropy bonus
  ``ent_coef`` x H (mode, conditional target and power), clipped value loss.
With the MILP the projection is part of the environment: the ratio is on the intention.
"""

import copy
from typing import Dict

import numpy as np
import torch

from .features import Context, Geometry, Snapshot, build_inputs
from .networks import Actor, GCNEncoder, ValueNet
from .policy import joint_log_prob, policy_entropy, sample


def gae(R, done, v_cur, v_nxt, veh, t, gamma, lam, steps=None):
    """GAE(lambda) along each vehicle's chain of decisions (transitions sorted by start epoch).
    With d_j the discount of transition j (gamma, or gamma^Delta_j when ``steps`` holds the
    Delta of each transition, semi-MDP):
    delta_j = R_j + d_j (1 - done_j) V(s'_j) - V(s_j);  A_j = delta_j + d_j lam (1 - done_j) A_{j+1}."""
    R, done = np.asarray(R, np.float64), np.asarray(done, np.float64)
    d = np.full(len(R), float(gamma)) if steps is None else float(gamma) ** np.asarray(steps, np.float64)
    delta = R + d * (1.0 - done) * np.asarray(v_nxt, np.float64) - np.asarray(v_cur, np.float64)
    adv = np.zeros(len(R), np.float64)
    for v in np.unique(veh):
        idx = np.nonzero(veh == v)[0]
        idx = idx[np.argsort(t[idx], kind="stable")]
        run = 0.0
        for j in idx[::-1]:
            run = delta[j] + d[j] * lam * (1.0 - done[j]) * run
            adv[j] = run
    return adv


class VehicleMAPPO:
    algo = "mappo"
    on_policy = True

    def __init__(self, env, config, device, projected: bool):
        self.cfg, self.device = config, torch.device(device)
        a, tr = config.agent, config.training
        self.geo = Geometry(env, self.device)
        self.projected = projected
        self.p_max = env.physics.p_max_kw
        self.enc = GCNEncoder(hidden=a.gcn_hidden, out=a.emb_dim).to(self.device)
        self.actor = Actor(a.emb_dim, a.hidden_dim, a.key_dim, self.geo.K).to(self.device)
        self.value = ValueNet(a.emb_dim, a.hidden_dim, a.critic_population).to(self.device)
        lr = tr.learning_rate
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr.actor)
        self.opt_value = torch.optim.Adam(list(self.enc.parameters()) + list(self.value.parameters()), lr=lr.critic)
        self.gamma, self.lam = float(tr.gamma), float(tr.gae_lambda)
        self.clip, self.vf_clip = float(tr.clip_eps), float(tr.vf_clip_eps)
        self.vf_coef, self.ent_coef = float(tr.vf_coef), float(tr.ent_coef)
        self.epochs, self.batch = int(tr.update_epochs), int(tr.batch_size)
        self.max_grad_norm = float(tr.max_grad_norm)
        self.semi_mdp = bool(a.semi_mdp)            # gamma^Delta per transition; False = fixed gamma (Q12)
        self.adversary = None
        self.updates = 0
        self.tau = a.tau_start                      # unused by exact sampling; kept for the loop / trainer

    def inputs(self, snap: Snapshot, ctx: Context) -> Dict[str, torch.Tensor]:
        return build_inputs(snap, ctx, self.geo, self.projected)

    def embed(self, enc, inp):
        return enc.local(inp["n2_hex"], self.geo, inp["cell"])

    @torch.no_grad()
    def act(self, snap: Snapshot, ctx: Context, deterministic=False, generator=None):
        inp = self.inputs(snap, ctx)
        out = self.actor(inp, self.embed(self.enc, inp))
        return sample(out, 1.0, deterministic=deterministic, relaxed=False, generator=generator), inp, out

    def _v(self, inp, ctx_emb=None):
        emb = self.embed(self.enc, inp) if ctx_emb is None else ctx_emb
        return self.value(inp["veh"], emb, inp["local"], inp["ports"], inp["glob"]), emb

    def update(self, batch, robust: bool = False):
        raise RuntimeError("MAPPO is on-policy: use update_rollout after each episode")

    # ------------------------------------------------------------------ PPO on one episode
    def update_rollout(self, replay, rng: np.random.Generator) -> Dict[str, float]:
        ep = replay.current
        n = ep.n
        if n == 0:
            return {}
        rows = ep.rows
        v_cur, v_nxt = np.zeros(n, np.float32), np.zeros(n, np.float32)
        with torch.no_grad():
            for r, b in replay.current_rows(self.device):
                v_cur[r] = self._v(self.inputs(b["snap"], b["cur"]))[0].cpu().numpy()
                v_nxt[r] = self._v(self.inputs(b["snap"], b["nxt"]))[0].cpu().numpy()
        adv = gae(rows["R"][:n], rows["done"][:n], v_cur, v_nxt, rows["veh"][:n], rows["t"][:n], self.gamma, self.lam,
                  steps=rows["delta"][:n] if self.semi_mdp else None)
        ret = (adv + v_cur).astype(np.float32)
        adv_n = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)
        extra = {"adv": adv_n, "ret": ret, "v_old": v_cur}
        stats = {"loss_pi": [], "loss_v": [], "entropy": [], "clip_frac": [], "approx_kl": []}
        for _ in range(self.epochs):
            for b in replay.current_batches(self.batch, rng, self.device, extra=extra):
                rw = b["rows"]
                ci = self.inputs(b["snap"], b["cur"])
                v, emb = self._v(ci)
                v_old, ret_b = rw["v_old"].float(), rw["ret"].float()
                v_clip = v_old + (v - v_old).clamp(-self.vf_clip, self.vf_clip)
                loss_v = torch.max((v - ret_b) ** 2, (v_clip - ret_b) ** 2).mean()
                self.opt_value.zero_grad()
                (self.vf_coef * loss_v).backward()
                torch.nn.utils.clip_grad_norm_(self.opt_value.param_groups[0]["params"], self.max_grad_norm)
                self.opt_value.step()

                out = self.actor(ci, emb.detach())
                logp = joint_log_prob(out, rw["mode"].long(), rw["i_slot"].long(), rw["i_opt"].long(), rw["i_u"].float())
                ratio = torch.exp(logp - rw["logp"].float())
                a_b = rw["adv"].float()
                surr = torch.min(ratio * a_b, ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * a_b)
                ent = policy_entropy(out)
                loss_pi = -(surr + self.ent_coef * ent).mean()
                self.opt_actor.zero_grad()
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()
                self.updates += 1
                with torch.no_grad():
                    stats["loss_pi"].append(loss_pi.item())
                    stats["loss_v"].append(loss_v.item())
                    stats["entropy"].append(float(ent.mean()))
                    stats["clip_frac"].append(float(((ratio - 1.0).abs() > self.clip).float().mean()))
                    stats["approx_kl"].append(float((rw["logp"].float() - logp).mean()))
        res = {k: float(np.mean(v)) for k, v in stats.items() if v}
        res.update(v_mean=float(v_cur.mean()), adv_raw_std=float(adv.std()), transitions=int(n))
        return res

    def eval_modules(self):
        return (self.enc, self.actor, self.value)

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self):
        return {"enc": self.enc.state_dict(), "actor": self.actor.state_dict(), "value": self.value.state_dict(),
                "opt": {k: getattr(self, k).state_dict() for k in ("opt_actor", "opt_value")},
                "updates": self.updates, "tau": self.tau}

    def load_state_dict(self, d, with_optim=True):
        for k in ("enc", "actor", "value"):
            getattr(self, k).load_state_dict(d[k])
        if with_optim:
            for k, v in d["opt"].items():
                getattr(self, k).load_state_dict(v)
        self.updates, self.tau = int(d["updates"]), float(d["tau"])
