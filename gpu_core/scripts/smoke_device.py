#!/usr/bin/env python3
"""
Quick end-to-end check of every learner on the device the full runs will use (GPU by default).

For PD-RSAC (MILP + WDRO, so the robust target runs), MAPPO (MILP) and MADDPG (raw executor):
a few short training episodes on the real train split with a reduced fleet, a deterministic
evaluation on nye, and a checkpoint save / load round trip; then the greedy heuristic.
Prints the device, timings and OK per learner; any device mismatch shows up as a traceback.

    python gpu_core/scripts/smoke_device.py                  # cuda if available
    python gpu_core/scripts/smoke_device.py --device cpu
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config.presets import resolve_preset  # noqa: E402
from gpu_core.data.factory import SIM_DEVICE, build_run, config_for_evaluation  # noqa: E402
from gpu_core.evaluation.protocol import evaluate_agent_on_split, evaluate_policy_on_split  # noqa: E402
from gpu_core.learning.evaluation import load_vehicle_agent  # noqa: E402
from gpu_core.learning.factory import make_agent  # noqa: E402
from gpu_core.learning.trainer import Trainer  # noqa: E402
from gpu_core.utils.run_record import build_run_record  # noqa: E402


def check(preset, device, vehicles, out):
    cfg = resolve_preset(preset)
    cfg.environment.num_vehicles = vehicles
    cfg.episode.duration_hours = 1.0                        # 12 epochs per episode
    cfg.training.batch_size, cfg.training.warmup_steps = 256, 6
    cfg.wdro.warmup_episodes = 1                            # robust target from the second episode
    t0 = time.perf_counter()
    run = build_run(cfg, SIM_DEVICE, split="train", verbose=False)
    agent = make_agent(run.env, cfg, device, projected=run.projection is not None)
    tr = Trainer(cfg, run.env, agent, projection=run.projection, dataset=run.dataset, seed=0,
                 manifest_hash=run.manifest.infra_hash, run_record=build_run_record(cfg, SimpleNamespace(seed=0), preset=preset))
    for _ in range(3):
        rep = tr.train_episode(3)
    assert rep.updates > 0, "no learner update ran"
    path = os.path.join(out, f"{preset}.pt")
    tr.save(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ecfg = config_for_evaluation(ck)
    ecfg.episode.duration_hours = 24.0
    erun = build_run(ecfg, SIM_DEVICE, split="nye", verbose=False)
    loaded = load_vehicle_agent(ck, ecfg, erun.env, device, projected=erun.projection is not None)
    res, _ = evaluate_agent_on_split(erun, loaded, split="nye", label=preset, max_steps=12)
    dev = next(agent.actor.parameters()).device
    robust = "robust target on" if agent.adversary is not None else ""
    print(f"[ok] {preset:12s} learner {agent.algo:6s} on {dev} | {'MILP' if run.projection else 'raw'} | "
          f"{rep.updates} updates in the last episode {robust} | eval nye 12 epochs profit ${res['profit']:,.0f}, "
          f"infeasible {res['infeasible_executed']} | {time.perf_counter() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vehicles", type=int, default=200)
    ap.add_argument("--presets", nargs="+", default=["full", "mappo_milp", "maddpg"])
    a = ap.parse_args(argv)
    device = torch.device(a.device)
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), cuda available {torch.cuda.is_available()}, "
          f"testing on {device}", flush=True)
    with tempfile.TemporaryDirectory() as out:
        for p in a.presets:
            check(p, device, a.vehicles, out)
    cfg = resolve_preset("greedy")
    cfg.environment.num_vehicles = a.vehicles
    cfg.episode.duration_hours = 24.0
    run = build_run(cfg, SIM_DEVICE, split="nye", verbose=False)
    from gpu_core.baselines.greedy import GreedyPolicy
    res, _ = evaluate_policy_on_split(run, GreedyPolicy(run.env), split="nye", label="greedy", max_steps=12)
    print(f"[ok] greedy       12 epochs on nye, profit ${res['profit']:,.0f}, infeasible {res['infeasible_executed']}")
    if device.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB")
    print("all checks passed")


if __name__ == "__main__":
    main()
