#!/usr/bin/env python3
"""
Test protocol (workstream F7): evaluate one checkpoint on the test week (25-31 Jan 2009, one
continuous 2016-epoch episode with a per-day table) and on New Year's Eve 2009 (``nye``, the OOD
day), each with the checkpoint's own pipeline. ``--sample-ratio`` evaluates on another trip sample
than the one the checkpoint was trained on. A second mode joins several runs into one comparison
table.

    python gpu_core/scripts/run_protocol.py --checkpoint checkpoint/full/best.pt --label full --out-dir results/full
    python gpu_core/scripts/run_protocol.py --preset greedy --out-dir results/greedy        # heuristic, no checkpoint
    python gpu_core/scripts/run_protocol.py --checkpoint checkpoint/full/best.pt --label full \
        --sample-ratio 0.15 --splits test --out-dir results/full_s15                        # another trip sample
    python gpu_core/scripts/run_protocol.py --compare results/full results/sac results/no_milp --out results/table.csv

Each run directory gets <split>.json, <split>_epochs.csv (feeder load, ports, economics per
epoch), <split>_days.csv, <split>_projection.csv (D11, with the MILP), summary.json, summary.csv
and eval_<label>.log, a copy of everything printed to the console.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.data.factory import apply_data_args, config_for_evaluation  # noqa: E402
from gpu_core.config.presets import get_preset, resolve_preset  # noqa: E402
from gpu_core.evaluation.protocol import (SUITE_SPLITS, compare_rows, run_suite_for_checkpoint,  # noqa: E402
                                          run_suite_for_greedy, write_rows_csv)
from gpu_core.learning.evaluation import is_vehicle_checkpoint  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--preset', default=None, help='greedy: run the heuristic instead of a checkpoint')
    p.add_argument('--num-vehicles', type=int, default=None, help='with --preset greedy')
    p.add_argument('--label', default=None, help='algorithm / preset name (default: the run record preset)')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--splits', nargs='+', default=list(SUITE_SPLITS))
    p.add_argument('--seed', type=int, default=10_000)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--config', default=None, help='YAML instead of the config embedded in the checkpoint')
    p.add_argument('--manifest', default=None)
    p.add_argument('--processed-root', default=None)
    p.add_argument('--sample-ratio', type=float, default=None,
                   help='trip sample of the evaluation splits (default: the checkpoint\'s, 0.2)')
    p.add_argument('--milp', dest='milp', action='store_true', default=None, help='force the projection')
    p.add_argument('--no-milp', dest='milp', action='store_false', help='force the raw executor')
    p.add_argument('--max-steps', type=int, default=None, help='stop each split after N epochs (smoke tests)')
    p.add_argument('--compare', nargs='+', default=None, help='run directories (with summary.json) to tabulate')
    p.add_argument('--out', default=None, help='with --compare: CSV of the comparison table')
    return p.parse_args(argv)


def compare(dirs, out):
    summaries = []
    for d in dirs:
        with open(Path(d) / "summary.json", encoding="utf-8") as f:
            summaries.append(json.load(f))
    rows = compare_rows(summaries)
    cols = ("label", "split", "profit", "revenue", "service_rate", "trips_dropped", "feeder_peak_kw",
            "feeder_violation_steps", "proj_effective_fallback")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols))
    if out:
        write_rows_csv(out, rows)
        print(f"Table written to {out}")
    return rows


class _Tee:
    """Writes to the console and to the run's log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def _logged(out_dir, label):
    """Console output also goes to <out_dir>/eval_<label>.log (as the logs kept for the trained runs)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / f"eval_{label}.log", "w", encoding="utf-8") as f:
        stdout = sys.stdout
        sys.stdout = _Tee(stdout, f)
        try:
            yield
        finally:
            sys.stdout = stdout


def main(argv=None):
    args = parse_args(argv)
    if args.compare:
        return compare(args.compare, args.out)
    if args.preset:
        if get_preset(args.preset).trainable:
            raise SystemExit(f"preset '{args.preset}' is trained: evaluate its checkpoint with --checkpoint")
        if not args.out_dir:
            raise SystemExit("--out-dir is required")
        config = resolve_preset(args.preset)
        apply_data_args(config, args)
        if args.num_vehicles is not None:
            config.environment.num_vehicles = args.num_vehicles
        label = args.label or args.preset
        with _logged(args.out_dir, label):
            return run_suite_for_greedy(config, torch.device('cpu'), label=label, out_dir=args.out_dir,
                                        splits=args.splits, seed=args.seed, max_steps=args.max_steps)
    if not args.checkpoint or not args.out_dir:
        raise SystemExit("--checkpoint and --out-dir are required (or use --preset greedy / --compare)")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not is_vehicle_checkpoint(checkpoint):
        raise SystemExit("not a vehicle-level checkpoint (SAC, PD-RSAC, MAPPO, MADDPG)")
    config = config_for_evaluation(checkpoint, args.config)
    apply_data_args(config, args)
    if args.milp is not None:
        config.milp.enabled = bool(args.milp)
    label = args.label or (checkpoint.get("manifest") or {}).get("preset") or Path(args.checkpoint).stem
    with _logged(args.out_dir, label):
        return run_suite_for_checkpoint(checkpoint, config, device, label=label, out_dir=args.out_dir,
                                        splits=args.splits, seed=args.seed, max_steps=args.max_steps)


if __name__ == '__main__':
    main()
