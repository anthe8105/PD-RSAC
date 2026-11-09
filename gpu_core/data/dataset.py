"""Trip dataset for one split, read from processed partitions through the spatial manifest.

Workstream B2/B3 (audit D02, D12, D16, D18, D20, D25-D27, D29; decisions Q14, Q22, Q23):

- days are loaded for a half-open [start, end) range; a missing day is an error (no silent
  synthetic fallback), an empty-but-present day is valid zero demand;
- trips are sampled per trip with a deterministic hash of (dataset, source_row, seed), so the
  same trip is in or out no matter which date range is loaded;
- endpoints are mapped by H3 id through the manifest; trips with an endpoint outside the
  domain are dropped and counted, never remapped;
- trips are kept sorted by pickup time; a decision epoch t sees arrivals with
  pickup time in (t - step, t], looked up with ``searchsorted``;
- episode starts are calendar times aligned to the step, drawn uniformly over the split.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from ..spatial.manifest import SpatialManifest, day_range, find_day_files

REPO_ROOT = Path(__file__).resolve().parents[2]
_COLUMNS = ["source_row", "pickup_ts", "duration_min", "distance_km", "fare", "pickup_cell", "dropoff_cell"]


def resolve_path(path: str) -> Path:
    """Relative config paths are resolved against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _dataset_key(name: str) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little"))


def sample_mask(dataset: str, source_rows: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Deterministic per-trip Bernoulli(ratio) via a splitmix64 hash of (dataset, row, seed)."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"sample ratio must be in (0, 1], got {ratio}")
    if ratio == 1.0:
        return np.ones(len(source_rows), dtype=bool)
    with np.errstate(over="ignore"):
        x = source_rows.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        x ^= _dataset_key(dataset) ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53) < ratio


class TripDataset:
    """All sampled, in-domain trips of one split, sorted by pickup time (kept on CPU)."""

    def __init__(
        self,
        processed_root: str,
        datasets: Sequence[str],
        manifest: SpatialManifest,
        start: str,
        end: str,
        sample_ratio: float,
        sample_seed: int,
        split_name: str = "",
        verbose: bool = True,
    ):
        self.manifest = manifest
        self.split_name = split_name
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.sample_ratio = sample_ratio
        self.sample_seed = sample_seed
        days = day_range(start, end)
        files = find_day_files(str(resolve_path(processed_root)), datasets, days)

        t, p, d, fare, dist, dur, t_out = [], [], [], [], [], [], []
        self.per_day: Dict[str, Dict[str, int]] = {}
        self.source_datasets = set()
        for day, f in zip(days, files):
            ds_dir = f.parent.parent
            meta = json.loads((ds_dir / "metadata.json").read_text(encoding="utf-8"))
            if meta["h3_resolution"] != manifest.resolution:
                raise ValueError(f"{ds_dir.name} is r{meta['h3_resolution']} but the manifest is r{manifest.resolution}")
            self.source_datasets.add(ds_dir.name)
            frame = pq.read_table(f, columns=_COLUMNS).to_pandas()
            keep = sample_mask(ds_dir.name, frame["source_row"].to_numpy(), sample_ratio, sample_seed)
            frame = frame[keep]
            pi = manifest.map_cells(frame["pickup_cell"].to_numpy(np.uint64))
            di = manifest.map_cells(frame["dropoff_cell"].to_numpy(np.uint64))
            inside = (pi >= 0) & (di >= 0)
            ts = frame["pickup_ts"].to_numpy("datetime64[s]").astype(np.int64)
            t.append(ts[inside]); p.append(pi[inside]); d.append(di[inside])
            fare.append(frame["fare"].to_numpy(np.float32)[inside])
            dist.append(frame["distance_km"].to_numpy(np.float32)[inside])
            dur.append(frame["duration_min"].to_numpy(np.float32)[inside])
            t_out.append(ts[~inside])
            self.per_day[day] = {"sampled": int(len(frame)), "in_domain": int(inside.sum()), "out_of_domain": int((~inside).sum())}

        self.pickup_s = np.concatenate(t)
        if np.any(np.diff(self.pickup_s) < 0):
            raise ValueError("processed partitions are not sorted by pickup time")
        self.pickup_idx = torch.from_numpy(np.concatenate(p))
        self.dropoff_idx = torch.from_numpy(np.concatenate(d))
        self.fare = torch.from_numpy(np.concatenate(fare))
        self.distance_km = torch.from_numpy(np.concatenate(dist))
        self.duration_min = torch.from_numpy(np.concatenate(dur))
        self._out_s = np.sort(np.concatenate(t_out))
        if verbose:
            n_out = len(self._out_s)
            print(f"[data] split '{split_name}' [{start}, {end}): {len(self.pickup_s):,} trips in domain, "
                  f"{n_out:,} out of domain ({n_out / max(1, n_out + len(self.pickup_s)):.3%}), sample {sample_ratio}")

    # ---------------------------------------------------------------- info
    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def total_trips(self) -> int:
        return int(len(self.pickup_s))

    @property
    def num_hexes(self) -> int:
        return self.manifest.num_cells

    @property
    def hex_ids(self) -> List[str]:
        return list(self.manifest.cells)

    @property
    def start_s(self) -> int:
        return int(self.start.value // 10**9)

    @property
    def end_s(self) -> int:
        return int(self.end.value // 10**9)

    @property
    def duration_hours(self) -> float:
        return (self.end_s - self.start_s) / 3600.0

    @property
    def fare_stats(self) -> Dict[str, float]:
        if self.total_trips == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0}
        return {"min": float(self.fare.min()), "max": float(self.fare.max()), "mean": float(self.fare.mean())}

    def mean_arrivals_per_step(self, step_minutes: float) -> float:
        steps = (self.end_s - self.start_s) / (step_minutes * 60)
        return self.total_trips / max(steps, 1.0)

    # ---------------------------------------------------------------- access
    def window(self, t_lo: int, t_hi: int) -> Tuple[int, int]:
        """Index range of trips with pickup time in (t_lo, t_hi] (epoch seconds)."""
        i0 = int(np.searchsorted(self.pickup_s, t_lo, side="right"))
        i1 = int(np.searchsorted(self.pickup_s, t_hi, side="right"))
        return i0, i1

    def trips_between(self, t_lo: int, t_hi: int, device) -> Dict[str, torch.Tensor]:
        i0, i1 = self.window(t_lo, t_hi)
        return {
            "pickup": self.pickup_idx[i0:i1].to(device),
            "dropoff": self.dropoff_idx[i0:i1].to(device),
            "fare": self.fare[i0:i1].to(device),
            "distance_km": self.distance_km[i0:i1].to(device),
        }

    def out_of_domain_between(self, t_lo: int, t_hi: int) -> int:
        return int(np.searchsorted(self._out_s, t_hi, side="right") - np.searchsorted(self._out_s, t_lo, side="right"))

    def episode_starts(self, episode_steps: int, step_minutes: float) -> np.ndarray:
        """Step-aligned calendar start times t0 with [t0, t0 + episode) inside the split."""
        step_s = int(round(step_minutes * 60))
        last = self.end_s - episode_steps * step_s
        if last < self.start_s:
            raise ValueError(f"split '{self.split_name}' ({self.duration_hours:.1f} h) is shorter than one episode "
                             f"({episode_steps} x {step_minutes} min)")
        return np.arange(self.start_s, last + 1, step_s, dtype=np.int64)
