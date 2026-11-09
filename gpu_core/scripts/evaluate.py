#!/usr/bin/env python3
"""
Evaluate a vehicle-level SAC / PD-RSAC checkpoint on one data split (workstream F3).

Usage:
    python evaluate.py --checkpoint checkpoints/full/best.pt --split test     # one episode = whole test week
    python evaluate.py --checkpoint checkpoints/full/best.pt --split nye      # 31 Dec 2009 (OOD)

The configuration is the one embedded in the checkpoint, including the projection: a checkpoint
trained with the MILP is evaluated with it (``--milp`` / ``--no-milp`` override and are reported).
The policy is deterministic unless --no-deterministic. Results carry the economic totals, feeder
and station metrics, D11 projection statistics, per-epoch timings and a per-day table; the full
protocol over test / nye is scripts/run_protocol.py. MAPPO / MADDPG checkpoints use
evaluate_mappo.py / evaluate_maddpg.py until workstream G.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.data.factory import SIM_DEVICE, add_data_args, apply_data_args, build_run, config_for_evaluation, split_hours  # noqa: E402
from gpu_core.evaluation.protocol import evaluate_agent_on_split, print_result, write_rows_csv  # noqa: E402
from gpu_core.learning.evaluation import is_vehicle_checkpoint, load_vehicle_agent  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', default=None, help='YAML instead of the config embedded in the checkpoint')
    p.add_argument('--label', default=None, help='name in the outputs (default: checkpoint file stem)')
    p.add_argument('--episodes', type=int, default=1, help='episodes with seeds seed, seed + 1, ...')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=10_000)
    p.add_argument('--save-results', default=None, help='JSON with one result per episode')
    p.add_argument('--epochs-csv', default=None, help='per-epoch rows of the last episode (feeder, ports, economics)')
    p.add_argument('--days-csv', default=None, help='per-day table of the last episode')
    p.add_argument('--num-vehicles', type=int, default=None)
    p.add_argument('--episode-duration-hours', type=float, default=None, help='default: the whole split')
    p.add_argument('--max-steps', type=int, default=None, help='stop after N epochs (smoke tests / benchmarks)')
    add_data_args(p, default_split='test')
    p.add_argument('--milp', dest='milp', action='store_true', default=None,
                   help='force the MILP projection (default: as trained)')
    p.add_argument('--no-milp', dest='milp', action='store_false', help='force the raw executor')
    p.add_argument('--projection-csv', default=None, help='with the projection: one row per call (D11)')
    p.add_argument('--deterministic', action='store_true', default=True)
    p.add_argument('--no-deterministic', dest='deterministic', action='store_false')
    return p.parse_args(argv)


def create_config(args, checkpoint):
    config = config_for_evaluation(checkpoint, args.config)
    apply_data_args(config, args)
    if args.num_vehicles is not None:
        config.environment.num_vehicles = args.num_vehicles
    if args.episode_duration_hours is not None:
        config.episode.duration_hours = args.episode_duration_hours
    elif not args.synthetic:
        config.episode.duration_hours = split_hours(config, args.split)
    trained_milp = bool(config.milp.enabled)
    if args.milp is not None:
        config.milp.enabled = bool(args.milp)
    return config, trained_milp


def evaluate(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not is_vehicle_checkpoint(checkpoint):
        raise SystemExit("not a vehicle-level SAC / PD-RSAC checkpoint; use evaluate_mappo.py or evaluate_maddpg.py")
    config, trained_milp = create_config(args, checkpoint)
    run = build_run(config, SIM_DEVICE, split=args.split, synthetic=args.synthetic)
    agent = load_vehicle_agent(checkpoint, config, run.env, device, projected=run.projection is not None)
    label = args.label or Path(args.checkpoint).stem
    split = 'synthetic' if args.synthetic else args.split
    override = '' if bool(config.milp.enabled) == trained_milp else f" (trained {'with' if trained_milp else 'without'} the MILP)"
    from gpu_core.utils.console import rule
    print(rule(f"evaluate | {label} | {split}"))
    print(f"checkpoint {args.checkpoint} ({checkpoint.get('agent_algo', 'sac')}) | {run.env.num_vehicles} vehicles, "
          f"{run.env.num_hexes} cells | {run.env.episode_steps} epochs | {'MILP projection' if run.projection else 'raw executor'}"
          f"{override} | {'deterministic' if args.deterministic else 'stochastic'} policy")
    results = []
    rec = None
    for ep in range(args.episodes):
        res, rec = evaluate_agent_on_split(run, agent, split=split, label=label, seed=args.seed + ep,
                                           deterministic=args.deterministic, max_steps=args.max_steps, progress=True)
        res["episode"] = ep
        res["milp_as_trained"] = bool(config.milp.enabled) == trained_milp
        results.append(res)
        print_result(res)
    if args.save_results:
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Results saved to: {args.save_results}")
    if rec is not None and args.epochs_csv:
        write_rows_csv(args.epochs_csv, rec.rows)
    if rec is not None and args.days_csv:
        write_rows_csv(args.days_csv, results[-1]["days"])
    if run.projection is not None and args.projection_csv:
        run.projection.log.write_csv(args.projection_csv)
    return results


def main(argv=None):
    return evaluate(parse_args(argv))


if __name__ == '__main__':
    main()
