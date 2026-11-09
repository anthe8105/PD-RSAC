"""Evaluation protocol shared by every algorithm (workstream F): per-epoch recorder (grid,
station and economic metrics), per-day tables, the test suite and cross-algorithm comparison."""

from .recorder import StepRecorder
from .protocol import (SUITE_SPLITS, compare_rows, evaluate_agent_on_split, evaluate_policy_on_split,
                       run_suite, run_suite_for_checkpoint, run_suite_for_greedy, write_rows_csv)

__all__ = ["StepRecorder", "SUITE_SPLITS", "compare_rows", "evaluate_agent_on_split", "evaluate_policy_on_split",
           "run_suite", "run_suite_for_checkpoint", "run_suite_for_greedy", "write_rows_csv"]
