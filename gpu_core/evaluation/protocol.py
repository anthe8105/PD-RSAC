"""Evaluation protocol (workstream F3, F5-F7).

- one split = one continuous episode from its first epoch (test week 25-31 Jan: 2016 epochs;
  ``nye`` 31 Dec 2009: 288 epochs), deterministic policy, fixed seed;
- the pipeline is the checkpoint's: its embedded config decides MILP or raw executor (F2, F39),
  through the shared ``build_run`` (F1);
- every result carries the economic totals, service, feeder and station metrics (F5), the D11
  projection statistics, per-epoch timings (F6) and a per-day table;
- the suite (round 11) is the test week and New Year's Eve 2009 (``nye``, the OOD day) (F7).
"""

import copy
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .recorder import StepRecorder

SUITE_SPLITS = ("test", "nye")
TOTAL_KEYS = ("profit", "revenue", "driving_cost", "energy_cost", "wait_penalty", "drop_penalty",
              "trips_loaded", "trips_served", "trips_completed", "trips_dropped", "trips_out_of_domain",
              "service_rate", "infeasible_executed", "serve_unmatched", "charge_no_port")
GRID_KEYS = ("feeder_peak_kw", "feeder_mean_kw", "feeder_violation_steps", "feeder_tight_share", "energy_charged_kwh",
             "port_utilisation_mean", "port_utilisation_max_station", "stations_full_share_mean", "charge_denied_rate")


def episode_totals(env) -> Dict:
    i = env.episode_info
    out = {k: getattr(i, k) for k in TOTAL_KEYS if k not in ("profit", "service_rate")}
    out["profit"] = i.profit
    out["service_rate"] = i.trips_served / max(i.trips_loaded, 1)
    return out


def evaluate_agent_on_split(run, agent, *, split: str, label: str = "", seed: int = 10_000,
                            deterministic: bool = True, max_steps: Optional[int] = None,
                            tight_fraction: float = 0.98, progress: bool = False):
    """One deterministic episode of a vehicle-level agent over the whole split (or ``max_steps``).
    Returns (result dict, recorder with the per-epoch rows). ``progress`` prints every 10 %."""
    from ..learning.loop import run_episode
    from ..utils.console import EpisodeProgress
    rec = StepRecorder(run.env, tight_fraction=tight_fraction, label=label, projection=run.projection)
    hook = (EpisodeProgress(run.env, f"{label} | {split}", total=min(run.env.episode_steps, max_steps or 10**9),
                            inner=rec.on_step) if progress else rec.on_step)
    rep = run_episode(run.env, agent, projection=run.projection, seed=seed, deterministic=deterministic,
                      on_step=hook, max_steps=max_steps)
    policy = {"intention_share": rep.intention_share, "executed_share": rep.executed_share,
              "serve_when_available": rep.serve_when_available, "low_energy_stay_share": rep.low_energy_idle_share,
              "mean_energy_end": rep.mean_energy_end, "stuck_vehicles_end": rep.stuck_vehicles_end,
              "pi_hat_match": rep.pi_hat_match}
    return result_record(run, rec, split=split, label=label, seed=seed, deterministic=deterministic,
                         projection=rep.projection, timing=rep.timing, policy=policy, seconds=rep.seconds), rec


def result_record(run, rec: StepRecorder, *, split: str, label: str, seed: int, deterministic: bool = True,
                  projection: Optional[Dict] = None, timing: Optional[Dict] = None, policy: Optional[Dict] = None,
                  seconds: float = 0.0) -> Dict:
    env = run.env
    return {
        "label": label, "split": split, "seed": seed, "deterministic": deterministic,
        "milp": run.projection is not None, "vehicles": env.num_vehicles, "stations": env.num_stations,
        "ports_per_station": int(env.config.station.num_ports), "steps": rec.steps,
        "start": f"{rec.rows[0]['date']} {rec.rows[0]['time']}" if rec.rows else None,
        **episode_totals(env), "grid": rec.summary(), "days": rec.day_table(),
        "projection": projection or {}, "timing": timing or {}, "policy": policy or {}, "seconds": seconds,
    }


def summary_row(res: Dict) -> Dict:
    """Flat row of one (algorithm, split) result for the comparison table."""
    row = {"label": res["label"], "split": res["split"], "milp": res["milp"], "steps": res["steps"]}
    row.update({k: res[k] for k in TOTAL_KEYS})
    row.update({k: res["grid"][k] for k in GRID_KEYS})
    p = res.get("projection") or {}
    if p:
        row.update({"proj_calls": p.get("calls"), "proj_fallback_calls": p.get("n_fallback"),
                    "proj_effective_fallback": p.get("effective_fallback_rate"),
                    "proj_optimal": p.get("n_OPTIMAL"), "proj_tl_improved": p.get("n_TL_IMPROVED"),
                    "proj_tl_warmstart_only": p.get("n_TL_WARMSTART_ONLY"),
                    "proj_no_incumbent": p.get("n_FALLBACK_NO_INCUMBENT"), "proj_error": p.get("n_FALLBACK_ERROR"),
                    "proj_mean_ms": 1000.0 * p.get("mean_t_total", 0.0), "proj_max_ms": 1000.0 * p.get("max_t_total", 0.0)})
    t = res.get("timing") or {}
    if t:
        row.update({"epoch_ms": t.get("epoch_ms"), "max_epoch_ms": t.get("max_epoch_ms")})
    return row


def compare_rows(summaries: Iterable[Dict]) -> List[Dict]:
    """Rows of the cross-algorithm table from suite summaries (``summary.json`` of each run)."""
    rows = []
    for s in summaries:
        for split, res in s["results"].items():
            rows.append(summary_row(res))
        # summaries written before round 11 also carry OOD effects (e.g. xmas - ctrl)
        deltas = s.get("ood_deltas") or ({"xmas-ctrl": s["ood_delta"]} if s.get("ood_delta") else {})
        for name, d in deltas.items():
            rows.append({"label": s["label"], "split": name, **{k: v for k, v in d.items() if not k.endswith("_rel")}})
    return rows


def write_rows_csv(path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def evaluate_policy_on_split(run, policy, *, split: str, label: str = "", seed: int = 10_000,
                             max_steps: Optional[int] = None, tight_fraction: float = 0.98, progress: bool = False):
    """One episode of a non-learning policy (``policy.act(env)`` -> mode / target / station /
    power / trip) through the simulator's exact path, with the same recorder as the agents."""
    import time
    env = run.env
    rec = StepRecorder(env, tight_fraction=tight_fraction, label=label)
    t0 = time.time()
    env.reset(seed=seed)
    from ..utils.console import EpisodeProgress
    prog = EpisodeProgress(env, f"{label} | {split}", total=min(env.episode_steps, max_steps or 10**9)) if progress else None
    clock = {"policy": 0.0, "env": 0.0}
    t, done, worst = 0, False, 0.0
    while not done:
        a0 = time.perf_counter()
        a = policy.act(env)
        a1 = time.perf_counter()
        _, _, d, info = env.step(a["mode"], a["target"], vehicle_charge_power=a["power"],
                                 milp_serve_trip_ids=a["trip"], charge_station=a["station"])
        a2 = time.perf_counter()
        clock["policy"] += a1 - a0
        clock["env"] += a2 - a1
        worst = max(worst, a2 - a0)
        rec.on_step(t, info, a["mode"])
        if prog is not None:
            prog(t, info)
        t += 1
        done = bool(d) or (max_steps is not None and t >= max_steps)
    n = max(t, 1)
    timing = {f"{k}_ms": 1000.0 * v / n for k, v in clock.items()}
    timing.update(epoch_ms=sum(timing.values()), max_epoch_ms=1000.0 * worst)
    fl = env.fleet_state
    one_trip = env.physics.reserve_kwh + env.physics.energy_per_km * 3.0
    pol = {"mean_energy_end": float(fl.socs.mean()) / env.physics.battery_kwh,
           "stuck_vehicles_end": int(((fl.status == 0) & (fl.socs < one_trip)).sum())}
    return result_record(run, rec, split=split, label=label, seed=seed, timing=timing, policy=pol,
                         seconds=time.time() - t0), rec


def run_suite(config, device, evaluate_fn, *, label: str, out_dir: str, splits: Iterable[str] = SUITE_SPLITS,
              seed: int = 10_000, verbose: bool = True) -> Dict:
    """Evaluate on every split of the suite, each from its own ``build_run`` (the config decides
    the projection); ``evaluate_fn(run, split, cfg)`` returns (result, recorder). Writes
    ``<split>.json``, ``<split>_epochs.csv``, ``<split>_days.csv`` (+ ``<split>_projection.csv``),
    ``summary.json`` and ``summary.csv``."""
    from ..data.factory import SIM_DEVICE, build_run, split_hours
    from ..utils.console import fmt_duration, rule
    import time
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    splits = list(splits)
    t_suite = time.perf_counter()
    if verbose:
        print(rule(f"protocol | {label}"))
        print(f"splits {', '.join(f'{s} {config.data.splits[s]}' for s in splits)} | "
              f"{config.environment.num_vehicles} vehicles | {'MILP projection' if config.milp.enabled else 'raw executor'} | "
              f"sample {config.data.sample_ratio:.0%} | seed {seed} | output {out}")
    for n, split in enumerate(splits, 1):
        if verbose:
            print(f"[{n}/{len(splits)}] {split}: building the environment ...", flush=True)
        cfg = copy.deepcopy(config)
        cfg.episode.duration_hours = split_hours(cfg, split)
        run = build_run(cfg, SIM_DEVICE, split=split, verbose=False)
        res, rec = evaluate_fn(run, split, cfg)
        results[split] = res
        with open(out / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=float)
        write_rows_csv(out / f"{split}_epochs.csv", rec.rows)
        write_rows_csv(out / f"{split}_days.csv", res["days"])
        if run.projection is not None:
            run.projection.log.write_csv(str(out / f"{split}_projection.csv"))
        if verbose:
            print_result(res)
    summary = {"label": label, "milp": bool(config.milp.enabled), "seed": seed,
               "sample_ratio": float(config.data.sample_ratio), "results": results}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    write_rows_csv(out / "summary.csv", compare_rows([summary]))
    if verbose:
        print(f"[done] {label}: {len(splits)} splits in {fmt_duration(time.perf_counter() - t_suite)}; results in {out}")
    return summary


def run_suite_for_checkpoint(checkpoint: Dict, config, device, *, label: str, out_dir: str,
                             splits: Iterable[str] = SUITE_SPLITS, seed: int = 10_000,
                             max_steps: Optional[int] = None, verbose: bool = True) -> Dict:
    """The suite for a vehicle-level checkpoint (SAC / PD-RSAC / MAPPO / MADDPG) with its own pipeline."""
    from ..learning.evaluation import load_vehicle_agent

    def fn(run, split, cfg):
        agent = load_vehicle_agent(checkpoint, cfg, run.env, device, projected=run.projection is not None)
        return evaluate_agent_on_split(run, agent, split=split, label=label, seed=seed, max_steps=max_steps,
                                       progress=verbose)
    return run_suite(config, device, fn, label=label, out_dir=out_dir, splits=splits, seed=seed, verbose=verbose)


def run_suite_for_greedy(config, device, *, label: str = "greedy", out_dir: str, splits: Iterable[str] = SUITE_SPLITS,
                         seed: int = 10_000, max_steps: Optional[int] = None, verbose: bool = True, **greedy_kw) -> Dict:
    """The suite for the greedy heuristic (Tian et al. 2025; raw environment, exact path)."""
    from ..baselines.greedy import GreedyPolicy

    def fn(run, split, cfg):
        return evaluate_policy_on_split(run, GreedyPolicy(run.env, **greedy_kw), split=split, label=label, seed=seed,
                                        max_steps=max_steps, progress=verbose)
    return run_suite(config, device, fn, label=label, out_dir=out_dir, splits=splits, seed=seed, verbose=verbose)


def print_result(r: Dict) -> None:
    """Summary of one (algorithm, split) result: economics, service, grid and stations, projection."""
    from ..utils.console import fmt_duration
    g, t, p = r["grid"], r.get("timing") or {}, r.get("projection") or {}
    print(f"  [{r['label']} | {r['split']}] {r['steps']} epochs from {r['start']}, "
          f"{'MILP projection' if r['milp'] else 'raw executor'}, {r['vehicles']} vehicles, seed {r['seed']}")
    print(f"    economics  profit ${r['profit']:,.0f} = revenue ${r['revenue']:,.0f} - driving ${r['driving_cost']:,.0f} "
          f"- electricity ${r['energy_cost']:,.0f} (penalties: waiting ${r['wait_penalty']:,.0f}, dropped ${r['drop_penalty']:,.0f})")
    print(f"    service    {r['service_rate']:.1%} of {r['trips_loaded']:,} orders served, {r['trips_dropped']:,} dropped, "
          f"{r['trips_completed']:,} completed in the window")
    print(f"    grid       feeder peak {g['feeder_peak_kw']:,.0f} / {g['feeder_limit_kw']:,.0f} kW at {g['feeder_peak_time']}, "
          f"mean {g['feeder_mean_kw']:,.0f} kW, violations {g['feeder_violation_steps']} epochs, at the limit "
          f"{g['feeder_tight_share']:.1%}; {g['energy_charged_kwh']:,.0f} kWh stored")
    print(f"    stations   ports used {g['port_utilisation_mean']:.1%} (busiest {g['port_utilisation_max_station']:.1%}), "
          f"full {g['stations_full_share_mean']:.1%} of the time; bookings denied {g['charge_denied']:,}/{g['charge_requests']:,}")
    if p:
        n_fb = p.get("n_fallback", p["n_TL_WARMSTART_ONLY"] + p["n_FALLBACK_NO_INCUMBENT"] + p["n_FALLBACK_ERROR"])
        print(f"    projection {p['calls']} calls; fallback (Algorithm 1) used {n_fb} times = "
              f"{p['effective_fallback_rate']:.1%} (warm-start only {p['n_TL_WARMSTART_ONLY']}, no incumbent "
              f"{p['n_FALLBACK_NO_INCUMBENT']}, error {p['n_FALLBACK_ERROR']}); optimal {p['n_OPTIMAL']}, improved "
              f"{p['n_TL_IMPROVED']}; {p['mean_t_total']:.2f} s mean, {p['max_t_total']:.2f} s max")
    if t:
        print(f"    time       {t.get('epoch_ms', 0) / 1000:.2f} s/epoch (max {t.get('max_epoch_ms', 0) / 1000:.2f} s), "
              f"total {fmt_duration(r.get('seconds', 0))}")
    if len(r["days"]) > 1:
        print("    per day    date        profit      service   served/orders   feeder peak  violations  fallback")
        for d in r["days"]:
            print(f"               {d['date']}  ${d['profit']:>10,.0f}   {d['service_rate']:6.1%}   "
                  f"{d['trips_served']:>6,}/{d['trips_loaded']:<6,}  {d['feeder_peak_kw']:>8,.0f} kW  {d['feeder_violation_steps']:>5}"
                  f"       {d.get('fallback_steps', 0):>5}")
