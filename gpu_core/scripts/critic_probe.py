#!/usr/bin/env python3
"""Probe a trained vehicle-level critic: what does it think each option is worth?

Runs the deterministic policy of a checkpoint (from collapse_check.py --save) on the same scaled
environment and, for every deciding vehicle, evaluates min_k Q(s, a) for each option the vehicle
has: SERVE (the nearest feasible order, if any), stay, the policy's reposition slot, booking the
nearest free station, and continue charging (vehicles at a charger). Results are grouped by
battery level and charging status, together with the actor's mode probabilities.

    python gpu_core/scripts/critic_probe.py --checkpoint collapse.pt --vehicles 100 --steps 60
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import ConfigLoader  # noqa: E402
from gpu_core.data.factory import build_env, load_manifest  # noqa: E402
from gpu_core.learning.agent import VehicleSAC  # noqa: E402
from gpu_core.learning.features import CONTINUE, N_CODES, contexts_from_env, snapshot_from_env  # noqa: E402
from gpu_core.learning.policy import entropies  # noqa: E402
from gpu_core.scripts.collapse_check import scaled_config, scaled_manifest  # noqa: E402


def option_q(agent, inp, emb, ctx, code_idx, slot_idx, opt_idx, power):
    B = len(ctx.cell)
    code = F.one_hot(torch.full((B,), code_idx), N_CODES).float()
    slot = F.one_hot(slot_idx.clamp(min=0), agent.geo.K).float()
    opt = F.one_hot(opt_idx.clamp(min=0), 6).float()
    order = ctx.order * (code_idx == 0)
    with torch.no_grad():
        return torch.min(agent.q1(inp, emb, code, slot, opt, power, order), agent.q2(inp, emb, code, slot, opt, power, order))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="gpu_core/scripts/config_wdro.yaml")
    ap.add_argument("--vehicles", type=int, default=100)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    cfg = scaled_config(ConfigLoader.from_yaml(a.config), a.vehicles)
    m = scaled_manifest(load_manifest(cfg), cfg.environment.num_stations, 1)
    env, _, _ = build_env(cfg, torch.device("cpu"), split="train", manifest=m, verbose=False)
    agent = VehicleSAC(env, cfg, "cpu", projected=False)
    agent.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=False)["agent"], with_optim=False)
    env.reset(seed=a.seed)
    recs = []
    for _ in range(a.steps):
        snap = snapshot_from_env(env, agent.geo)
        ctx, dec = contexts_from_env(env, agent.geo)
        s, inp, out = agent.act(snap, ctx, deterministic=True)
        if len(dec):
            emb = agent.embed(agent.enc, inp)
            B = len(dec)
            p_half = torch.full((B,), 0.5)
            q = {
                "serve": option_q(agent, inp, emb, ctx, 0, torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long), p_half),
                "stay": option_q(agent, inp, emb, ctx, 2, torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long), p_half),
                "repos": option_q(agent, inp, emb, ctx, 1, s.slot.argmax(1), torch.zeros(B, dtype=torch.long), p_half),
                "charge": option_q(agent, inp, emb, ctx, 3, torch.zeros(B, dtype=torch.long), inp["opt_ok"][:, :5].float().argmax(1), torch.ones(B)),
                "continue": option_q(agent, inp, emb, ctx, 4, torch.zeros(B, dtype=torch.long), torch.full((B,), CONTINUE), torch.ones(B)),
            }
            pm = entropies(out)["pm"]
            e = (ctx.energy / env.physics.battery_kwh).numpy()
            for j in range(B):
                recs.append(dict(e=e[j], charging=bool(ctx.charging[j]), serve_ok=bool(ctx.serve_ok[j]),
                                 charge_ok=bool(inp["opt_ok"][j, :5].any()), **{k: float(v[j]) for k, v in q.items()},
                                 p_serve=float(pm[j, 0]), p_repos=float(pm[j, 1]), p_charge=float(pm[j, 2])))
        # deterministic policy actions through the raw executor (same mapping as the loop)
        V = env.num_vehicles
        mode_i, slot_i, opt_i = s.mode.argmax(1), s.slot.argmax(1), s.opt.argmax(1)
        full_mode = torch.full((V,), -1, dtype=torch.long)
        full_target = env.fleet_state.positions.clone().long()
        full_station = torch.full((V,), -1, dtype=torch.long)
        if len(dec):
            target = inp["slot_cells"].gather(1, slot_i[:, None]).squeeze(1)
            target = torch.where(target >= 0, target, ctx.cell)
            stations = torch.cat([inp["stations"], ctx.own_station[:, None]], 1).gather(1, opt_i[:, None]).squeeze(1)
            full_mode[dec] = torch.tensor([0, 2, 1])[mode_i]
            full_target[dec] = target
            full_station[dec] = torch.where(mode_i == 2, stations, torch.full_like(stations, -1))
        env.step(full_mode, full_target, vehicle_charge_power=torch.ones(V), charge_station=full_station)
    import pandas as pd
    df = pd.DataFrame(recs)
    df["bin"] = pd.cut(df.e, [0, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0])
    cols = ["serve", "stay", "repos", "charge", "continue", "p_serve", "p_repos", "p_charge"]
    pd.set_option("display.width", 200)
    for charging in (False, True):
        sub = df[df.charging == charging]
        if len(sub):
            print(f"\n=== {'at a charger' if charging else 'idle'}: {len(sub)} decisions (Q x 100 = dollars-equivalent) ===")
            g = sub.groupby("bin", observed=True)
            t = g[cols].mean()
            t[cols[:5]] *= 100
            t.insert(0, "n", g.size())
            t["serve_ok"] = g.serve_ok.mean()
            print(t.round(3).to_string())


if __name__ == "__main__":
    main()
