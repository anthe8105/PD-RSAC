#!/usr/bin/env python3
"""Policy-collapse check on a scaled-down real environment (workstream E diagnostics).

The full experiment is 1000 vehicles on a 12 % trip sample (data.sample_ratio) with 35 stations x 5
ports and a 5500 kW feeder. This script keeps every ratio per vehicle: N vehicles see a (12 % x N / 1000)
sample, round(175 x N / 1000) single-port stations (the first ones of the cover-then-fill order)
and a feeder of 5.5 x N kW. It trains the vehicle-level agent and prints, per episode, what the
deciding vehicles intend and execute, how often they choose SERVE when an order is reachable,
whether drained vehicles charge, and how many end the episode stranded.

Columns: ``service_pct`` = orders accepted / orders that appeared (%); ``i_*`` = share of the
deciding vehicles whose *intention* (the actor's sample) was serve / reposition / charge;
``x_*`` = share whose *executed* action was serve (got an order), reposition, stay (incl. serve
intentions left without an order), charge (new port) or continue (kept charging).

    python gpu_core/scripts/collapse_check.py --vehicles 100 --episodes 40            # raw executor
    python gpu_core/scripts/collapse_check.py --vehicles 100 --episodes 40 --milp     # projection
"""

import argparse
from types import SimpleNamespace
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import ConfigLoader  # noqa: E402
from gpu_core.data.factory import build_env, load_manifest  # noqa: E402
from gpu_core.learning.factory import make_agent  # noqa: E402
from gpu_core.learning.trainer import Trainer  # noqa: E402


def scaled_manifest(manifest, stations: int, ports: int):
    m = dataclasses.replace(manifest, station_cells=list(manifest.station_cells[:stations]), ports_per_station=ports)
    m.spatial_hash, m.infra_hash = m.compute_hashes()
    return m


def scaled_config(config, vehicles: int, full_vehicles: int = 1000, full_sample: float = 0.2):
    f = vehicles / full_vehicles
    config.environment.num_vehicles = vehicles
    config.data.sample_ratio = full_sample * f
    total_ports = config.environment.num_stations * config.station.num_ports
    config.environment.num_stations = max(1, round(total_ports * f))
    config.station.num_ports = 1
    config.station.feeder_kw = config.station.feeder_kw * f
    return config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="gpu_core/scripts/config_wdro.yaml")
    ap.add_argument("--vehicles", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--batch", type=int, default=None, help="default: 2048 x vehicles / 1000")
    ap.add_argument("--warmup-episodes", type=int, default=2)
    ap.add_argument("--milp", action="store_true")
    ap.add_argument("--wdro", action="store_true", help="enable the robust target after wdro.warmup_episodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates-per-step", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=5, help="deterministic evaluation episode every N episodes (0 = off)")
    ap.add_argument("--eval-hours", type=float, default=None,
                    help="length of the evaluation window (default: the training episode); e.g. 72 to test that the "
                         "fleet sustains its energy over several days, like the validation split")
    ap.add_argument("--wdro-warmup", type=int, default=None, help="override wdro.warmup_episodes")
    ap.add_argument("--full-sample", type=float, default=None,
                    help="trip sample of the full-scale experiment the scaled run mirrors (default: data.sample_ratio)")
    ap.add_argument("--final-eval-split", default=None,
                    help="after training, one deterministic episode over this whole split (e.g. val: 3 days, like the "
                         "main run's checkpoint selection), with a per-day table")
    ap.add_argument("--gamma", type=float, default=None, help="override training.gamma")
    ap.add_argument("--rho", type=float, default=None, help="override wdro.rho (robust target budget)")
    ap.add_argument("--rho-target", type=float, default=None, help="override wdro.rho_target (dual update budget)")
    ap.add_argument("--eta0", type=float, default=None, help="override wdro.eta0")
    ap.add_argument("--wdro-metric", default=None, help="override wdro.metric (graph | diag | identity)")
    ap.add_argument("--step-rule", default=None, help="override wdro.step_rule (metric | sigma)")
    ap.add_argument("--schur", type=int, default=None, help="override wdro.schur (1 | 0)")
    ap.add_argument("--step-size-d", type=float, default=None, help="override wdro.step_size_d")
    ap.add_argument("--semi-mdp", type=int, default=None, help="override agent.semi_mdp (1 | 0)")
    ap.add_argument("--feeder-kw", type=float, default=None, help="override station.feeder_kw of the full-scale run (scaled)")
    ap.add_argument("--out", default=None, help="CSV of per-episode diagnostics")
    ap.add_argument("--save", default=None, help="checkpoint path written at the end")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    cfg = ConfigLoader.from_yaml(a.config)
    if a.feeder_kw is not None:
        cfg.station.feeder_kw = a.feeder_kw
    full_ports = cfg.environment.num_stations * cfg.station.num_ports
    cfg = scaled_config(cfg, a.vehicles, full_sample=a.full_sample if a.full_sample is not None else cfg.data.sample_ratio)
    cfg.training.batch_size = a.batch or max(64, round(2048 * a.vehicles / 1000))
    cfg.training.warmup_steps = a.warmup_episodes * cfg.episode.steps_per_episode
    cfg.wdro.enabled = a.wdro
    if a.wdro_warmup is not None:
        cfg.wdro.warmup_episodes = a.wdro_warmup
    cfg.milp.enabled = a.milp
    for key, val in (("training.gamma", a.gamma), ("wdro.rho", a.rho), ("wdro.rho_target", a.rho_target), ("wdro.eta0", a.eta0),
                     ("wdro.metric", a.wdro_metric), ("wdro.step_rule", a.step_rule),
                     ("wdro.schur", None if a.schur is None else bool(a.schur)), ("wdro.step_size_d", a.step_size_d),
                     ("agent.semi_mdp", None if a.semi_mdp is None else bool(a.semi_mdp))):
        if val is not None:
            sec, name = key.split(".")
            setattr(getattr(cfg, sec), name, val)
    if a.updates_per_step:
        cfg.agent.updates_per_step = a.updates_per_step
    manifest = scaled_manifest(load_manifest(cfg), cfg.environment.num_stations, 1)
    env, train_data, _ = build_env(cfg, torch.device("cpu"), split="train", manifest=manifest, verbose=False)
    projection = None
    if a.milp:
        from gpu_core.simulator.projection import make_projection
        projection = make_projection(env)
    if cfg.training.algo in ("mappo", "ppo"):
        cfg.training.warmup_steps = 0          # on-policy: no uniform warm-up
    agent = make_agent(env, cfg, "cpu", projected=a.milp)
    eval_env, eval_proj = env, projection
    if a.eval_hours and a.eval_hours != cfg.episode.duration_hours:
        import copy
        ecfg = copy.deepcopy(cfg)
        ecfg.episode.duration_hours = a.eval_hours
        eval_env, _, _ = build_env(ecfg, torch.device("cpu"), split="train", manifest=manifest, verbose=False)
        if a.milp:
            from gpu_core.simulator.projection import make_projection
            eval_proj = make_projection(eval_env)
    trainer = Trainer(cfg, env, agent, projection=projection, dataset=train_data, seed=a.seed)
    print(f"[collapse] {a.vehicles} vehicles, sample {cfg.data.sample_ratio:.3%}, {cfg.environment.num_stations} x 1 ports "
          f"(full run {full_ports}), feeder {cfg.station.feeder_kw:.0f} kW, {train_data.total_trips:,} train trips, "
          f"batch {cfg.training.batch_size}, {'MILP' if a.milp else 'raw executor'}, WDRO {'on' if a.wdro else 'off'}, "
          f"learner {agent.algo}, gamma {cfg.training.gamma}, semi-MDP {cfg.agent.semi_mdp}, "
          f"rho {cfg.wdro.rho} / {cfg.wdro.rho_target}, eta0 {cfg.wdro.eta0}, metric {cfg.wdro.metric}, "
          f"step {cfg.wdro.step_rule}, schur {cfg.wdro.schur}")
    rows = []
    for ep in range(a.episodes):
        r = trainer.train_episode(a.episodes)
        u = r.update_stats
        row = dict(episode=ep + 1, warm=int(trainer.warmup_left > 0 or ep < a.warmup_episodes), profit=round(r.profit, 1),
                   served=r.served, loaded=r.loaded, dropped=r.dropped, service_pct=round(100.0 * r.served / max(r.loaded, 1), 1),
                   i_serve=r.intention_share.get("serve", 0), i_repos=r.intention_share.get("reposition", 0),
                   i_charge=r.intention_share.get("charge", 0),
                   x_serve=r.executed_share.get("serve", 0), x_repos=r.executed_share.get("reposition", 0),
                   x_stay=r.executed_share.get("stay", 0), x_charge=r.executed_share.get("charge", 0),
                   x_continue=r.executed_share.get("continue", 0),
                   serve_available=r.serve_available_share, serve_when_available=r.serve_when_available,
                   low_energy_stay=r.low_energy_idle_share, energy_end=r.mean_energy_end, stuck_end=r.stuck_vehicles_end,
                   pi_hat_match=r.pi_hat_match, h_mode=u.get("h_mode", np.nan), pi_serve=u.get("pi_serve", np.nan),
                   alpha_mode=u.get("alpha_mode", np.nan), loss_q=u.get("loss_q", np.nan), q_mean=u.get("q_mean", np.nan),
                   lam=u.get("wdro_lambda", np.nan), rho_hat=u.get("wdro_rho_hat", np.nan),
                   gap=u.get("wdro_gap", np.nan), delta=round(r.mean_delta, 2),
                   tau=round(agent.tau, 3), seconds=round(r.seconds, 1))
        rows.append(row)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if a.eval_every and (ep + 1) % a.eval_every == 0:
            ev = trainer.evaluate(eval_env, eval_proj, episodes=1, seed=99_000 + ep)[0]
            print(f"EVAL episode={ep + 1} hours={eval_env.episode_steps * 5 / 60:g} profit={ev.profit:.1f} served={ev.served} loaded={ev.loaded} "
                  f"service_pct={100.0 * ev.served / max(ev.loaded, 1):.1f} "
                  f"i_serve={ev.intention_share.get('serve', 0):.3f} i_charge={ev.intention_share.get('charge', 0):.3f} "
                  f"x_serve={ev.executed_share.get('serve', 0):.3f} x_stay={ev.executed_share.get('stay', 0):.3f} "
                  f"x_charge={ev.executed_share.get('charge', 0):.3f} x_continue={ev.executed_share.get('continue', 0):.3f} "
                  f"serve_when_available={ev.serve_when_available:.3f} low_energy_stay={ev.low_energy_idle_share:.3f} "
                  f"energy_end={ev.mean_energy_end:.3f} stuck_end={ev.stuck_vehicles_end}", flush=True)
    if a.final_eval_split:
        import copy
        from gpu_core.data.factory import split_hours
        from gpu_core.evaluation.protocol import evaluate_agent_on_split, print_result
        fcfg = copy.deepcopy(cfg)
        fcfg.episode.duration_hours = split_hours(fcfg, a.final_eval_split)
        fenv, _, _ = build_env(fcfg, torch.device("cpu"), split=a.final_eval_split, manifest=manifest, verbose=False)
        fproj = None
        if a.milp:
            from gpu_core.simulator.projection import make_projection
            fproj = make_projection(fenv)
        frun = SimpleNamespace(env=fenv, projection=fproj)
        res, _ = evaluate_agent_on_split(frun, agent, split=a.final_eval_split, label="collapse", seed=99_999, progress=True)
        print_result(res)
    if a.save:
        trainer.save(a.save)
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
