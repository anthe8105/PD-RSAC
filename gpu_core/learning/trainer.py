"""Training driver of the vehicle-level agent (workstream E, Algorithm 2; E11-E13).

- Gumbel temperature tau anneals linearly from ``tau_start`` to ``tau_end`` over
  ``tau_anneal_fraction`` of the episodes and is shared by sampling and losses (Q16);
- the first ``training.warmup_steps`` decision epochs use uniform intentions through the same
  projection / executor (E12);
- with WDRO, the first ``wdro.warmup_episodes`` use the non-robust target, then the robust target
  with the scenario metric fitted on the training split (or on the replay for synthetic runs) (Q7);
- checkpoints carry the resolved config, the spatial manifest hash, the agent (networks, targets,
  optimisers, entropy coefficients, WDRO dual) and the training counters (E13).
"""

import os
import time
from dataclasses import asdict
from typing import Callable, List, Optional

import numpy as np
import torch

from .agent import VehicleSAC
from .loop import EpisodeReport, run_episode
from .replay import ReplayBuffer
from .wdro import Adversary, ScenarioMetric, arrivals_from_dataset, arrivals_from_replay


def tau_at(cfg, episode: int, episodes: int) -> float:
    a = cfg.agent
    frac = min(1.0, episode / max(1.0, a.tau_anneal_fraction * episodes))
    return a.tau_start + (a.tau_end - a.tau_start) * frac


def build_adversary(cfg, agent: VehicleSAC, env, replay: ReplayBuffer, dataset=None) -> Adversary:
    w = cfg.wdro
    if dataset is not None:
        arrivals, od = arrivals_from_dataset(dataset, agent.geo, env._trip_manager.step_s)
    else:
        arrivals, od = arrivals_from_replay(replay, agent.geo), None
    metric = ScenarioMetric(agent.geo, arrivals, od, beta=w.beta, metric=w.metric, support_sigmas=w.support_sigmas,
                            schur=w.schur)
    return Adversary(metric, rho=w.rho, rho_target=w.rho_target, eta0=w.eta0, steps=w.inner_steps,
                     step_size=w.step_size, lam=w.lambda_init, step_rule=w.step_rule, step_size_d=w.step_size_d)


class Trainer:

    def __init__(self, cfg, env, agent: VehicleSAC, projection=None, dataset=None, seed: int = 0,
                 manifest_hash: Optional[str] = None, run_record: Optional[dict] = None):
        self.cfg, self.env, self.agent, self.projection, self.dataset = cfg, env, agent, projection, dataset
        on_policy = getattr(agent, "on_policy", False)          # MAPPO learns from its last episode only
        self.replay = ReplayBuffer(env.episode_steps, env.num_hexes, env.num_stations,
                                   window=1 if on_policy else cfg.agent.replay_window, recency=cfg.agent.replay_recency)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.episode = 0
        self.warmup_left = int(cfg.training.warmup_steps)
        self.history: List[EpisodeReport] = []
        self.manifest_hash = manifest_hash
        self.best_val = -float("inf")        # best validation profit so far (kept across resumes)
        self.run_record = run_record or {}

    def robust_now(self) -> bool:
        w = self.cfg.wdro
        return bool(w.enabled) and self.episode >= w.warmup_episodes

    def train_episode(self, episodes_total: int) -> EpisodeReport:
        self.agent.tau = tau_at(self.cfg, self.episode, episodes_total)
        robust = self.robust_now()
        if robust and self.agent.adversary is None:
            self.agent.adversary = build_adversary(self.cfg, self.agent, self.env, self.replay, self.dataset)
        rep = run_episode(self.env, self.agent, replay=self.replay, projection=self.projection,
                          seed=self.seed * 100003 + self.episode, warmup_steps=self.warmup_left, train=True,
                          robust=robust, rng=self.rng)
        self.warmup_left = max(0, self.warmup_left - rep.steps)
        if getattr(self.agent, "on_policy", False):
            t0 = time.perf_counter()
            before = self.agent.updates
            rep.update_stats = self.agent.update_rollout(self.replay, self.rng)
            rep.updates = self.agent.updates - before
            rep.timing["rollout_update_s"] = time.perf_counter() - t0
        self.episode += 1
        self.history.append(rep)
        return rep

    def evaluate(self, env, projection=None, episodes: int = 1, seed: int = 10_000, episode_start_s=None) -> List[EpisodeReport]:
        out = []
        for k in range(episodes):
            out.append(run_episode(env, self.agent, projection=projection, seed=seed + k, deterministic=True,
                                   episode_start_s=episode_start_s))
        return out

    # ---------------------------------------------------------------- checkpoints (E13)
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        from ..utils.run_record import checkpoint_fields
        torch.save({
            **checkpoint_fields(self),
            "algo": "vehicle_sac",                 # vehicle-level stack; the learner is agent_algo
            "agent_algo": self.agent.algo,
            "manifest_infra_hash": self.manifest_hash,
            "agent": self.agent.state_dict(),
            "episode": self.episode, "warmup_left": self.warmup_left, "best_val": self.best_val,
            "wdro_active": self.agent.adversary is not None,
            "history": [asdict(h) for h in self.history[-50:]],
        }, path)

    def load(self, path: str, with_optim: bool = True) -> None:
        ck = torch.load(path, map_location=self.agent.device, weights_only=False)
        if self.manifest_hash is not None and ck.get("manifest_infra_hash") not in (None, self.manifest_hash):
            raise ValueError("checkpoint was trained on a different spatial manifest (cells, stations or ports)")
        if ck.get("wdro_active") and self.agent.adversary is None:
            self.agent.adversary = build_adversary(self.cfg, self.agent, self.env, self.replay, self.dataset)
        self.agent.load_state_dict(ck["agent"], with_optim=with_optim)
        self.episode, self.warmup_left = int(ck["episode"]), int(ck["warmup_left"])
        self.best_val = float(ck.get("best_val", -float("inf")))
