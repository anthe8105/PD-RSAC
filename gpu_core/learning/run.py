"""Training driver of every learned method (SAC / PD-RSAC family, MAPPO, MADDPG; called by
scripts/train.py).

Console: a header with the run, data, fleet, learner and schedule; one line per logged episode
(profit, service, intentions, learning signals, MILP statistics, time per episode and ETA); one
line per validation run. Files: ``training_history.csv`` (every episode, every statistic),
``projection_history.csv`` (D11, with the MILP), checkpoints ``best.pt`` (best validation profit),
``episode_<n>.pt`` and ``final.pt``. Every ``--eval-interval`` episodes the deterministic policy
runs once over the whole validation split through the same executor / projection as training
(F2, F39).
"""

import copy
import csv
import math
import os
import time
from dataclasses import asdict

from ..data.factory import SIM_DEVICE, build_run, split_hours
from ..utils.console import RateMeter, fmt_duration, rule
from .factory import NAMES, make_agent
from .trainer import Trainer


def _flat(rep, prefix=""):
    d = {}
    for k, v in asdict(rep).items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    d[f"{prefix}{k}.{kk}"] = vv
        elif isinstance(v, (int, float)):
            d[prefix + k] = v
    return d


class _Csv:
    """Append-only CSV whose header grows when later rows bring new columns (update statistics
    start after the warm-up, WDRO statistics after its warm-up); existing rows are kept on resume."""

    def __init__(self, path):
        self.path, self.keys, self.rows = path, [], []
        if os.path.exists(path):
            with open(path, newline="") as f:
                r = csv.DictReader(f)
                self.keys = list(r.fieldnames or [])
                self.rows = list(r)

    def write(self, row):
        new = [k for k in row if k not in self.keys]
        self.rows.append(row)
        if new or not os.path.exists(self.path):
            self.keys += new
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(self.rows)
        else:
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.keys, extrasaction="ignore").writerow(row)


def _learner_name(config, agent) -> str:
    if agent.algo == "sac":
        return "PD-RSAC" if config.wdro.enabled else ("SAC (semi-MDP)" if config.agent.semi_mdp else "SAC")
    return NAMES[agent.algo]


def _header(args, config, env, agent, projection, train_data, val_env, log_dir, ckpt_dir, device):
    n_params = sum(p.numel() for m in agent.eval_modules() if m is not None for p in m.parameters())
    tr, st = config.training, config.station
    preset = getattr(args, "preset", None)
    print(rule(f"training | {preset or os.path.basename(getattr(args, 'config', '') or 'config')}"))
    print(f"run       log {log_dir} | checkpoints {ckpt_dir} | device {device} | seed {args.seed}")
    if train_data is not None:
        s, e = config.data.splits["train"]
        print(f"data      train [{s}, {e}): {train_data.total_trips:,} trips (sample {config.data.sample_ratio:.0%})")
    else:
        print("data      synthetic trips (smoke test)")
    print(f"fleet     {env.num_vehicles} vehicles, {env.num_hexes} cells, {env.num_stations} stations x {st.num_ports} ports, "
          f"feeder {st.feeder_kw:.0f} kW (metered)")
    parts = [f"semi-MDP {'on' if config.agent.semi_mdp else 'off'} (gamma {tr.gamma})"]
    if config.wdro.enabled:
        w = config.wdro
        parts.append(f"WDRO from episode {w.warmup_episodes} (metric {w.metric}, rho {w.rho}, rho_target {w.rho_target})")
    if agent.algo == "sac":
        parts.append(f"critic {'population' if config.agent.critic_population else 'IAC'}")
    if agent.algo == "mappo":
        parts.append(f"PPO epochs {tr.update_epochs}, clip {tr.clip_eps}, GAE {tr.gae_lambda}")
    print(f"learner   {_learner_name(config, agent)}: {', '.join(parts)}; {n_params:,} parameters")
    proj = (f"MILP (tau_max {config.milp.tau_max} s, mu {config.milp.mu}, warm start {'on' if config.milp.warm_start else 'off'})"
            if projection is not None else "raw executor")
    print(f"actions   {proj}")
    val = (f"validation on '{args.eval_split}' every {args.eval_interval} episodes ({val_env.episode_steps} epochs)"
           if val_env is not None else "no validation (synthetic)")
    print(f"schedule  {tr.total_episodes} episodes x {env.episode_steps} epochs ({config.episode.duration_hours:g} h), "
          f"batch {tr.batch_size}, warm-up {tr.warmup_steps} epochs, lr actor {tr.learning_rate.actor:g} critic "
          f"{tr.learning_rate.critic:g}; {val}")
    print(rule())


def _episode_line(ep, total, rep, agent, wall, eta):
    u = rep.update_stats
    i = rep.intention_share
    service = rep.served / max(rep.loaded, 1)
    learn = []
    if "loss_q" in u:
        learn.append(f"Q-loss {u['loss_q']:.4f}")
    if "alpha_mode" in u:
        learn.append(f"alpha {u['alpha_mode']:.4f}")
    if "wdro_lambda" in u:
        learn.append(f"lambda {u['wdro_lambda']:.3f} rho^ {u['wdro_rho_hat']:.3f} gap {u.get('wdro_gap', 0):+.4f}")
    if "loss_v" in u and agent.algo == "mappo":
        learn.append(f"V-loss {u['loss_v']:.4f} KL {u.get('approx_kl', 0):.4f} H {u.get('entropy', 0):.2f}")
    if not u:
        learn.append("warm-up" if rep.updates == 0 else "")
    line = (f"[ep {ep:>4}/{total} {ep / total:5.1%}] profit ${rep.profit:>9,.0f}  service {service:6.1%} "
            f"({rep.served:,}/{rep.loaded:,}, dropped {rep.dropped:,}) | intent S/R/C {i.get('serve', 0):.2f}/"
            f"{i.get('reposition', 0):.2f}/{i.get('charge', 0):.2f} | battery end {rep.mean_energy_end:.0%}, stuck "
            f"{rep.stuck_vehicles_end}, Delta {rep.mean_delta:.1f} | {' '.join(x for x in learn if x)}")
    if rep.projection:
        p = rep.projection
        line += f" | MILP {p['mean_t_total']:.2f} s, fallback {p['effective_fallback_rate']:.1%}"
    line += f" | {fmt_duration(wall)}/ep, ETA {fmt_duration(eta)}"
    print(line, flush=True)


def train_vehicle_sac(args, config, device, env, train_data, manifest, run_record, log_dir):
    projection = None
    if config.milp.enabled:
        from ..simulator.projection import make_projection
        projection = make_projection(env)
    agent = make_agent(env, config, device, projected=projection is not None)
    trainer = Trainer(config, env, agent, projection=projection, dataset=train_data, seed=args.seed,
                      manifest_hash=manifest.infra_hash if manifest is not None else None, run_record=run_record)
    val_env = val_projection = None
    if not args.synthetic:
        vcfg = copy.deepcopy(config)
        vcfg.episode.duration_hours = split_hours(config, args.eval_split)
        val = build_run(vcfg, SIM_DEVICE, split=args.eval_split, manifest=manifest, verbose=False)
        val_env, val_projection = val.env, val.projection
    ckpt_dir = config.checkpoint.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    _header(args, config, env, agent, projection, train_data, val_env, log_dir, ckpt_dir, device)
    if args.resume:
        trainer.load(args.resume)
        print(f"[resume] from {args.resume}: episode {trainer.episode}, best validation profit "
              f"{'$%s' % format(trainer.best_val, ',.0f') if trainer.best_val > -math.inf else 'none yet'}")

    writer = None
    if getattr(args, "tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir, os.path.basename(log_dir)))
        except ImportError:
            print("[tensorboard] not installed; skipping")
    hist = _Csv(os.path.join(log_dir, "training_history.csv"))
    proj_hist = _Csv(os.path.join(log_dir, "projection_history.csv"))
    total = int(config.training.total_episodes)
    eval_every = max(1, args.eval_interval)
    ep_meter, val_meter = RateMeter(window=20), RateMeter(window=3)
    t_start = time.perf_counter()
    first = trainer.episode
    while trainer.episode < total:
        t0 = time.perf_counter()
        rep = trainer.train_episode(total)
        wall = time.perf_counter() - t0
        ep_meter.add(wall)
        ep = trainer.episode
        row = {"episode": ep, "robust": int(trainer.robust_now()), "tau": agent.tau, "wall_s": wall, **_flat(rep)}
        hist.write(row)
        if rep.projection:
            proj_hist.write({"episode": ep, **rep.projection})
        if writer is not None:
            for k, v in row.items():
                writer.add_scalar(k.replace(".", "/"), v, ep)

        # ETA: remaining training episodes + remaining validation runs (first one estimated from
        # the policy / projection / env time per epoch of the last episode)
        n_val_left = (total // eval_every - ep // eval_every) if val_env is not None else 0
        per_val = val_meter.per_unit
        if per_val is None and val_env is not None:
            t = rep.timing
            per_val = (t.get("policy_ms", 0) + t.get("projection_ms", 0) + t.get("env_ms", 0)
                       + t.get("other_ms", 0)) / 1000.0 * val_env.episode_steps
        eta = ep_meter.eta(total - ep) + (per_val or 0.0) * n_val_left
        if ep % max(1, config.logging.log_interval) == 0 or ep == first + 1 or ep == total:
            _episode_line(ep, total, rep, agent, wall, eta)

        if val_env is not None and ep % eval_every == 0:
            t0 = time.perf_counter()
            ev = trainer.evaluate(val_env, val_projection, episodes=1)[0]
            dt = time.perf_counter() - t0
            val_meter.add(dt)
            better = ev.profit > trainer.best_val
            if better:
                trainer.best_val = ev.profit
                trainer.save(os.path.join(ckpt_dir, "best.pt"))
            info = val_env.episode_info
            print(f"  [val {args.eval_split}] profit ${ev.profit:,.0f}  service {ev.served / max(ev.loaded, 1):.1%} "
                  f"({ev.served:,}/{ev.loaded:,}) | feeder peak {info.feeder_peak_kw:,.0f} kW, violations "
                  f"{info.feeder_violation_steps}"
                  + (f" | MILP fallback {ev.projection.get('n_fallback', 0)}/{ev.projection.get('calls', 0)}" if ev.projection else "")
                  + f" | {fmt_duration(dt)} | "
                  f"{'new best -> best.pt' if better else 'best $%s' % format(trainer.best_val, ',.0f')}", flush=True)
            if writer is not None:
                writer.add_scalar("eval/profit", ev.profit, ep)
                writer.add_scalar("eval/service", ev.served / max(ev.loaded, 1), ep)
        if config.checkpoint.save_interval and ep % config.checkpoint.save_interval == 0:
            trainer.save(os.path.join(ckpt_dir, f"episode_{ep}.pt"))
    trainer.save(os.path.join(ckpt_dir, "final.pt"))
    if writer is not None:
        writer.close()
    best = f"best validation profit ${trainer.best_val:,.0f} (best.pt)" if trainer.best_val > -math.inf else "no validation"
    print(rule())
    print(f"[done] {total} episodes in {fmt_duration(time.perf_counter() - t_start)}; {best}; checkpoints in {ckpt_dir}; "
          f"history in {log_dir}")
    return trainer
