#!/usr/bin/env python3
"""
Greedy heuristic baseline of Tian et al. (2025), Sec. 5.2 (see gpu_core/baselines/greedy.py).

    python gpu_core/scripts/heuristic_matching.py --split test --output results/greedy_test.json
    python gpu_core/scripts/heuristic_matching.py --suite --out-dir results/greedy      # test, nye

Runs on the shared simulator and evaluation protocol (one deterministic episode per split, same
recorder and metrics as the learned methods). The configuration is the ``greedy`` preset
(base.yaml: environment, data, reward, physics, stations).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.baselines.greedy import GreedyPolicy  # noqa: E402
from gpu_core.config.presets import resolve_preset  # noqa: E402
from gpu_core.data.factory import add_data_args, apply_data_args, build_run, split_hours  # noqa: E402
from gpu_core.evaluation.protocol import (SUITE_SPLITS, evaluate_policy_on_split, print_result,  # noqa: E402
                                          run_suite_for_greedy, write_rows_csv)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p, default_split='test')
    p.add_argument('--suite', action='store_true', help='run the whole protocol (test, nye) into --out-dir')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--num-vehicles', type=int, default=None)
    p.add_argument('--num-hexes', type=int, default=None, help='synthetic runs only')
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--seed', type=int, default=10_000)
    p.add_argument('--low-soc', type=float, default=None, help='vehicles below this battery fraction charge (default 0.2)')
    p.add_argument('--charge-target', type=float, default=1.0, help='battery fraction a charging session ends at')
    p.add_argument('--output', default=None, help='JSON result of the single split')
    p.add_argument('--epochs-csv', default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = resolve_preset("greedy")
    apply_data_args(config, args)
    if args.num_vehicles is not None:
        config.environment.num_vehicles = args.num_vehicles
    if args.num_hexes is not None:
        config.environment.num_hexes = args.num_hexes
    kw = dict(low=args.low_soc, charge_target=args.charge_target)
    device = torch.device('cpu')
    if args.suite:
        if not args.out_dir:
            raise SystemExit("--suite needs --out-dir")
        return run_suite_for_greedy(config, device, out_dir=args.out_dir, splits=SUITE_SPLITS, seed=args.seed,
                                    max_steps=args.max_steps, **kw)
    if not args.synthetic:
        config.episode.duration_hours = split_hours(config, args.split)
    run = build_run(config, device, split=args.split, synthetic=args.synthetic)
    split = 'synthetic' if args.synthetic else args.split
    res, rec = evaluate_policy_on_split(run, GreedyPolicy(run.env, **kw), split=split, label="greedy",
                                        seed=args.seed, max_steps=args.max_steps, progress=True)
    print_result(res)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, default=float)
    if args.epochs_csv:
        write_rows_csv(args.epochs_csv, rec.rows)
    return res


if __name__ == '__main__':
    main()
