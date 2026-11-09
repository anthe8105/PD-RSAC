#!/usr/bin/env python3
"""
Charging power drawn from the grid over one split or a run of consecutive days (paper Fig. 6; F5, F48).

    python gpu_core/scripts/powergrid_evaluate.py --checkpoint checkpoint/full/best.pt --split nye \
        --out-csv powergrid.csv --out-json powergrid.json
    python gpu_core/scripts/powergrid_evaluate.py --checkpoint checkpoint/full/best.pt --start-day 2009-01-25 --days 3 \
        --out-csv powergrid.csv --out-json powergrid.json
    python gpu_core/scripts/powergrid_evaluate.py --preset greedy --start-day 2009-01-27 --out-csv greedy.csv

``--start-day`` with ``--days N`` runs one continuous episode over N calendar days from 00:00 of
the start day (the fleet carries its state from one day to the next); it overrides ``--split``.

Any vehicle-level checkpoint (PD-RSAC, SAC, MAPPO, MADDPG) runs with its own pipeline (MILP as
trained; ``--milp`` / ``--no-milp`` override); ``--preset greedy`` runs the heuristic. The power
of an epoch is the energy actually drawn during it divided by the epoch length, at the epoch's
wall-clock time.

Outputs are what Fig. 6 plots, one run (method) per file:
- CSV, one row per epoch: calendar date and day number of the run, time of day (hh:mm, minute,
  decimal hour for a daily x axis), hours since the start (for a multi-day x axis), the charging
  load, the feeder limit, the load as a share of the limit and the part above it (the violation
  area), whether the epoch is at the limit, vehicles charging and ports held;
- JSON ``summary``: over the whole run, the limit, the peak (kW, date and time, % of the limit,
  kW above the limit), the mean, violation epochs / minutes / energy above the limit, the share of
  epochs at the limit, energy, profit and service; ``per_day`` has the same numbers for each
  calendar day; ``timeseries`` repeats the CSV rows.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.data.factory import SIM_DEVICE, add_data_args, apply_data_args, build_run, config_for_evaluation, split_hours  # noqa: E402

FIELDS = ["label", "day", "date", "step", "timestamp_hhmm", "minute_of_day", "hour_of_day", "hours_from_start",
          "total_charging_power_kw", "feeder_limit_kw", "load_pct_of_limit", "over_limit_kw", "feeder_violation", "at_limit",
          "num_vehicles_charging", "mean_power_per_vehicle_kw", "ports_held"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="vehicle-level checkpoint")
    p.add_argument("--preset", default=None, help="greedy: the heuristic instead of a checkpoint")
    p.add_argument("--config", default=None, help="YAML instead of the config embedded in the checkpoint")
    p.add_argument("--label", default=None)
    p.add_argument("--milp", dest="milp", action="store_true", default=None, help="force the MILP projection")
    p.add_argument("--no-milp", dest="milp", action="store_false", help="force the raw executor")
    p.add_argument("--start-day", default=None, help="first calendar day YYYY-MM-DD of a configured dataset; overrides --split")
    p.add_argument("--days", type=int, default=1, help="with --start-day: number of consecutive days in one continuous episode")
    add_data_args(p, default_split="nye")
    p.add_argument("--num-vehicles", type=int, default=None, help="with --preset greedy")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-json", default=None)
    return p.parse_args(argv)


def timeseries(rows: List[Dict], limit_kw: float, step_minutes: float) -> List[Dict[str, Any]]:
    out, dates = [], []
    for r in rows:
        if r["date"] not in dates:
            dates.append(r["date"])
        n, load = r["vehicles_charging"], r["feeder_load_kw"]
        minute = int((r["clock_s"] % 86400) // 60)
        out.append({"label": r["label"], "day": len(dates), "date": r["date"], "step": r["step"], "timestamp_hhmm": r["time"],
                    "minute_of_day": minute, "hour_of_day": round(minute / 60.0, 4),
                    "hours_from_start": round(r["step"] * step_minutes / 60.0, 4), "total_charging_power_kw": load,
                    "feeder_limit_kw": limit_kw, "load_pct_of_limit": 100.0 * load / limit_kw if limit_kw else 0.0,
                    "over_limit_kw": max(0.0, load - limit_kw), "feeder_violation": r["feeder_violation"],
                    "at_limit": r["feeder_tight"], "num_vehicles_charging": n,
                    "mean_power_per_vehicle_kw": load / n if n else 0.0, "ports_held": r["ports_held"]})
    return out


def load_stats(rows: List[Dict[str, Any]], limit: float, step_minutes: float) -> Dict[str, Any]:
    """Fig. 6 numbers of a stretch of epochs: peak (date, time, % of the limit, excess), mean, violations."""
    peak = max(rows, key=lambda r: r["total_charging_power_kw"]) if rows else None
    loads = [r["total_charging_power_kw"] for r in rows]
    viol = [r for r in rows if r["feeder_violation"]]
    when = lambda r: f"{r['date']} {r['timestamp_hhmm']}"      # noqa: E731
    return {
        "num_steps": len(rows), "feeder_limit_kw": limit,
        "peak_charging_power_kw": peak["total_charging_power_kw"] if peak else 0.0,
        "peak_time": when(peak) if peak else None,
        "peak_hour_of_day": peak["hour_of_day"] if peak else None,
        "peak_pct_of_limit": peak["load_pct_of_limit"] if peak else 0.0,
        "peak_over_limit_kw": peak["over_limit_kw"] if peak else 0.0,
        "average_charging_power_kw": sum(loads) / len(loads) if loads else 0.0,
        "average_pct_of_limit": 100.0 * sum(loads) / len(loads) / limit if loads and limit else 0.0,
        "feeder_violation_steps": len(viol),
        "feeder_violation_minutes": len(viol) * step_minutes,
        "first_violation_time": when(viol[0]) if viol else None,
        "last_violation_time": when(viol[-1]) if viol else None,
        "energy_above_limit_kwh": sum(r["over_limit_kw"] for r in rows) * step_minutes / 60.0,
        "feeder_tight_share": sum(r["at_limit"] for r in rows) / len(rows) if rows else 0.0,
    }


def grid_summary(rows: List[Dict[str, Any]], res: Dict, label: str, window: str, step_minutes: float) -> Dict[str, Any]:
    """Whole-run numbers plus the same numbers for each calendar day (``per_day``)."""
    g = res["grid"]
    limit = float(g["feeder_limit_kw"])
    days = {d["date"]: d for d in res["days"]}
    per_day = []
    for date in dict.fromkeys(r["date"] for r in rows):
        d = days.get(date, {})
        per_day.append({"date": date, **load_stats([r for r in rows if r["date"] == date], limit, step_minutes),
                        "total_charged_energy_kwh": d.get("energy_charged_kwh"), "profit": d.get("profit"),
                        "service_rate": d.get("service_rate")})
    return {
        "label": label, "window": window, "milp_enabled": res["milp"], "num_days": len(per_day),
        "step_duration_minutes": step_minutes, **load_stats(rows, limit, step_minutes),
        "days_with_violation": sum(d["feeder_violation_steps"] > 0 for d in per_day),
        "total_charged_energy_kwh": g["energy_charged_kwh"], "total_bought_energy_kwh": g["energy_bought_kwh"],
        "profit": res["profit"], "service_rate": res["service_rate"], "per_day": per_day,
    }


def main(argv=None) -> Dict:
    args = parse_args(argv)
    from gpu_core.evaluation.protocol import evaluate_agent_on_split, evaluate_policy_on_split
    if bool(args.checkpoint) == bool(args.preset):
        raise SystemExit("give exactly one of --checkpoint or --preset greedy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = config_for_evaluation(checkpoint, args.config)
        if args.milp is not None:
            config.milp.enabled = bool(args.milp)
        label = args.label or Path(args.checkpoint).stem
    else:
        from gpu_core.config.presets import resolve_preset
        if args.preset != "greedy":
            raise SystemExit("only --preset greedy runs without a checkpoint")
        config = resolve_preset("greedy")
        if args.num_vehicles is not None:
            config.environment.num_vehicles = args.num_vehicles
        label = args.label or "greedy"
    apply_data_args(config, args)
    split = args.split
    if args.start_day:
        import pandas as pd
        if args.days < 1:
            raise SystemExit("--days must be >= 1")
        end = (pd.Timestamp(args.start_day) + pd.Timedelta(days=args.days)).strftime("%Y-%m-%d")
        split = f"days:{args.start_day}..{end}"
        config.data.splits[split] = [args.start_day, end]
    config.episode.duration_hours = 24.0 if args.synthetic else split_hours(config, split)
    run = build_run(config, SIM_DEVICE, split=split, synthetic=args.synthetic)
    if checkpoint is not None:
        from gpu_core.learning.evaluation import load_vehicle_agent
        agent = load_vehicle_agent(checkpoint, config, run.env, device, projected=run.projection is not None)
        res, rec = evaluate_agent_on_split(run, agent, split=split, label=label, seed=args.seed, max_steps=args.max_steps,
                                           progress=True)
    else:
        from gpu_core.baselines.greedy import GreedyPolicy
        res, rec = evaluate_policy_on_split(run, GreedyPolicy(run.env), split=split, label=label, seed=args.seed,
                                            max_steps=args.max_steps, progress=True)
    step_minutes = float(config.episode.step_duration_minutes)
    rows = timeseries(rec.rows, float(res["grid"]["feeder_limit_kw"]), step_minutes)
    s = summary = grid_summary(rows, res, label, split, step_minutes)
    over = f" (+{s['peak_over_limit_kw']:,.0f} kW over)" if s["peak_over_limit_kw"] > 0 else ""
    span = f", {s['first_violation_time']} .. {s['last_violation_time']}" if s["feeder_violation_steps"] else ""
    print(f"[Power grid | {label} | {split}] {s['num_steps']} epochs over {s['num_days']} day(s), "
          f"{'MILP' if s['milp_enabled'] else 'raw executor'}, feeder limit {s['feeder_limit_kw']:,.0f} kW")
    print(f"    peak        {s['peak_charging_power_kw']:,.0f} kW at {s['peak_time']} = {s['peak_pct_of_limit']:.1f} % of the limit{over}")
    print(f"    mean        {s['average_charging_power_kw']:,.0f} kW = {s['average_pct_of_limit']:.1f} % of the limit; "
          f"at the limit {100 * s['feeder_tight_share']:.1f} % of epochs")
    print(f"    violations  {s['feeder_violation_steps']} epochs ({s['feeder_violation_minutes']:.0f} min{span}) on "
          f"{s['days_with_violation']}/{s['num_days']} days, {s['energy_above_limit_kwh']:,.0f} kWh above the limit")
    print(f"    energy      {s['total_charged_energy_kwh']:,.0f} kWh stored, {s['total_bought_energy_kwh']:,.0f} kWh bought | "
          f"profit ${s['profit']:,.0f}, service {100 * s['service_rate']:.1f} %")
    if s["num_days"] > 1:
        print("    per day    date         peak kW   at     % limit  violations  mean kW   stored kWh")
        for d in s["per_day"]:
            print(f"               {d['date']}  {d['peak_charging_power_kw']:>7,.0f}   {d['peak_time'][-5:]}  {d['peak_pct_of_limit']:>6.1f}  "
                  f"{d['feeder_violation_steps']:>10}  {d['average_charging_power_kw']:>7,.0f}   {d['total_charged_energy_kwh'] or 0:>10,.0f}")
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "timeseries": rows}, f, indent=2, default=float)
    return summary


if __name__ == "__main__":
    main()
