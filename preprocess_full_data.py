#!/usr/bin/env python3
"""Preprocess one raw NYC TLC 2009 yellow-taxi parquet file.

Thin wrapper around ``gpu_core.data.preprocess`` (streaming, per-day partitions, per-row
exclusion reasons, direct H3 mapping). Example::

    python preprocess_full_data.py --input ../yellow_tripdata_2009-01.parquet --output-root ../data/processed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_core.data.preprocess import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
