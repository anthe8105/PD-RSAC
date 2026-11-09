"""Streaming preprocessing of raw NYC TLC 2009 yellow-taxi parquet files (workstream B2).

One raw monthly file becomes one *dataset* directory::

    <output_root>/<name>/
        trips/YYYY-MM-DD.parquet   kept trips, one file per pickup day, sorted by (pickup_ts, source_row)
        excluded.parquet           source_row + exclusion reason for every dropped row
        metadata.json              raw hash, schema, per-reason counts, per-day counts, flags, versions

Rules (audit D06, D11, D13, D14, D16, D17, D19, D30, D40; decision Q15):
- raw rows are read in batches with column projection; nothing loads the whole file;
- endpoints are mapped directly from lat/lon to the target H3 resolution (no parent/coarsening);
- intra-cell trips are kept unchanged;
- every raw row is either kept or excluded with exactly one reason, so counts reconcile;
- clear impossibilities are excluded, other oddities are only flagged;
- output is written to a staging directory and moved into place only when complete.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import h3
import h3.api.basic_int as h3i
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MILES_TO_KM = 1.60934
EARTH_RADIUS_KM = 6371.0

# Column names of the 2009 TLC yellow-taxi schema (identical in every 2009 monthly file).
RAW_COLUMNS = {
    "pickup": "Trip_Pickup_DateTime",
    "dropoff": "Trip_Dropoff_DateTime",
    "distance_mi": "Trip_Distance",
    "pickup_lon": "Start_Lon",
    "pickup_lat": "Start_Lat",
    "dropoff_lon": "End_Lon",
    "dropoff_lat": "End_Lat",
    "fare": "Fare_Amt",
    "passengers": "Passenger_Count",
}

# Coarse sanity box around NYC; the service domain itself is applied later by the spatial manifest.
BBOX = {"lon_min": -74.5, "lon_max": -73.5, "lat_min": 40.4, "lat_max": 41.0}
DURATION_MIN, DURATION_MAX = 1.0, 180.0
MAX_SPEED_KMH = 100.0

# Exclusion reasons, checked in this order; a row gets the first one that applies.
REASONS = (
    "timestamp_unparseable",
    "outside_file_month",
    "outside_bbox",
    "nonpositive_or_nonfinite_distance",
    "nonpositive_or_nonfinite_fare",
    "duration_out_of_range",
    "speed_above_100kmh",
    "h3_error",
)

# Quality flags on kept rows (bitmask): kept, but marked for audits.
FLAG_PASSENGERS_NONPOSITIVE = 1
FLAG_DISTANCE_BELOW_STRAIGHT_LINE = 2  # recorded distance + 0.1 km < great-circle distance of endpoints
FLAG_FARE_ABOVE_200 = 4
FLAGS = {
    "passengers_nonpositive": FLAG_PASSENGERS_NONPOSITIVE,
    "distance_below_straight_line": FLAG_DISTANCE_BELOW_STRAIGHT_LINE,
    "fare_above_200": FLAG_FARE_ABOVE_200,
}

TRIP_SCHEMA = pa.schema([
    ("source_row", pa.int64()),
    ("pickup_ts", pa.timestamp("s")),
    ("dropoff_ts", pa.timestamp("s")),
    ("duration_min", pa.float32()),
    ("distance_km", pa.float32()),
    ("fare", pa.float32()),
    ("passengers", pa.int16()),
    ("pickup_lat", pa.float64()),
    ("pickup_lon", pa.float64()),
    ("dropoff_lat", pa.float64()),
    ("dropoff_lon", pa.float64()),
    ("pickup_cell", pa.uint64()),
    ("dropoff_cell", pa.uint64()),
    ("quality_flags", pa.uint8()),
])


class RawSchemaError(ValueError):
    """The input is not a raw 2009 TLC file (e.g. an already processed file)."""


def validate_raw_schema(path: str) -> None:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = [c for c in RAW_COLUMNS.values() if c not in names]
    if missing:
        hint = ""
        if {"pickup_time", "pickup_hex"} & names:
            hint = " The file looks already processed; give the raw TLC monthly file instead."
        raise RawSchemaError(f"{path}: missing raw 2009 TLC columns {missing}.{hint}")


def file_month(path: str) -> Optional[pd.Period]:
    m = re.search(r"(\d{4})-(\d{2})", Path(path).stem)
    return pd.Period(f"{m.group(1)}-{m.group(2)}", freq="M") if m else None


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def process_batch(df: pd.DataFrame, first_row: int, resolution: int, month: Optional[pd.Period]):
    """Return (kept DataFrame matching TRIP_SCHEMA, excluded DataFrame[source_row, reason])."""
    n = len(df)
    source_row = np.arange(first_row, first_row + n, dtype=np.int64)
    reason = np.full(n, -1, dtype=np.int8)

    def mark(mask, code):
        sel = (reason < 0) & np.asarray(mask, dtype=bool)
        reason[sel] = code

    pick = pd.to_datetime(df[RAW_COLUMNS["pickup"]], errors="coerce")
    drop = pd.to_datetime(df[RAW_COLUMNS["dropoff"]], errors="coerce")
    mark(pick.isna() | drop.isna(), 0)
    if month is not None:
        mark(pick.dt.to_period("M") != month, 1)

    plon, plat = df[RAW_COLUMNS["pickup_lon"]].to_numpy(float), df[RAW_COLUMNS["pickup_lat"]].to_numpy(float)
    dlon, dlat = df[RAW_COLUMNS["dropoff_lon"]].to_numpy(float), df[RAW_COLUMNS["dropoff_lat"]].to_numpy(float)
    with np.errstate(invalid="ignore"):
        in_box = ((plon > BBOX["lon_min"]) & (plon < BBOX["lon_max"]) & (plat > BBOX["lat_min"]) & (plat < BBOX["lat_max"])
                  & (dlon > BBOX["lon_min"]) & (dlon < BBOX["lon_max"]) & (dlat > BBOX["lat_min"]) & (dlat < BBOX["lat_max"]))
    mark(~in_box, 2)

    dist_km = df[RAW_COLUMNS["distance_mi"]].to_numpy(float) * MILES_TO_KM
    fare = df[RAW_COLUMNS["fare"]].to_numpy(float)
    with np.errstate(invalid="ignore"):
        mark(~(np.isfinite(dist_km) & (dist_km > 0)), 3)
        mark(~(np.isfinite(fare) & (fare > 0)), 4)
    duration = ((drop - pick).dt.total_seconds() / 60.0).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mark(~((duration >= DURATION_MIN) & (duration <= DURATION_MAX)), 5)
        speed = dist_km / duration * 60.0
        mark(speed > MAX_SPEED_KMH, 6)

    keep_idx = np.nonzero(reason < 0)[0]
    pcell = np.zeros(len(keep_idx), dtype=np.uint64)
    dcell = np.zeros(len(keep_idx), dtype=np.uint64)
    for j, i in enumerate(keep_idx):
        try:
            pcell[j] = h3i.latlng_to_cell(plat[i], plon[i], resolution)
            dcell[j] = h3i.latlng_to_cell(dlat[i], dlon[i], resolution)
        except Exception:
            reason[i] = 7
    ok = reason[keep_idx] < 0
    keep_idx, pcell, dcell = keep_idx[ok], pcell[ok], dcell[ok]

    passengers = df[RAW_COLUMNS["passengers"]].to_numpy()[keep_idx]
    straight = haversine_km(plat[keep_idx], plon[keep_idx], dlat[keep_idx], dlon[keep_idx])
    flags = np.zeros(len(keep_idx), dtype=np.uint8)
    flags |= np.where(passengers <= 0, FLAG_PASSENGERS_NONPOSITIVE, 0).astype(np.uint8)
    flags |= np.where(dist_km[keep_idx] + 0.1 < straight, FLAG_DISTANCE_BELOW_STRAIGHT_LINE, 0).astype(np.uint8)
    flags |= np.where(fare[keep_idx] > 200.0, FLAG_FARE_ABOVE_200, 0).astype(np.uint8)

    kept = pd.DataFrame({
        "source_row": source_row[keep_idx],
        "pickup_ts": pick.to_numpy()[keep_idx].astype("datetime64[s]"),
        "dropoff_ts": drop.to_numpy()[keep_idx].astype("datetime64[s]"),
        "duration_min": duration[keep_idx].astype(np.float32),
        "distance_km": dist_km[keep_idx].astype(np.float32),
        "fare": fare[keep_idx].astype(np.float32),
        "passengers": passengers.astype(np.int16),
        "pickup_lat": plat[keep_idx], "pickup_lon": plon[keep_idx],
        "dropoff_lat": dlat[keep_idx], "dropoff_lon": dlon[keep_idx],
        "pickup_cell": pcell, "dropoff_cell": dcell,
        "quality_flags": flags,
    })
    ex_idx = np.nonzero(reason >= 0)[0]
    excluded = pd.DataFrame({"source_row": source_row[ex_idx], "reason": reason[ex_idx]})
    return kept, excluded


def _sha256(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def preprocess_month(
    raw_path: str,
    output_root: str,
    name: Optional[str] = None,
    resolution: int = 7,
    batch_size: int = 65536,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Preprocess one raw monthly file into ``<output_root>/<name>``. Returns the dataset directory."""
    if not 0 <= resolution <= 15:
        raise ValueError(f"H3 resolution must be in [0, 15], got {resolution}")
    validate_raw_schema(raw_path)
    name = name or Path(raw_path).stem
    root = Path(output_root)
    final = root / name
    if final.exists() and not overwrite:
        raise FileExistsError(f"{final} exists; pass overwrite=True (--overwrite) to rebuild it")
    staging = root / f".{name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    parts_dir = staging / "parts"
    parts_dir.mkdir(parents=True)

    month = file_month(raw_path)
    pf = pq.ParquetFile(raw_path, memory_map=True, pre_buffer=False, buffer_size=0)
    cols = list(RAW_COLUMNS.values())
    reason_counts = np.zeros(len(REASONS), dtype=np.int64)
    flag_counts = {k: 0 for k in FLAGS}
    excluded_frames = []
    n_raw = n_kept = 0
    t0 = time.time()
    for b, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=cols, use_threads=False)):
        df = batch.to_pandas()
        kept, excluded = process_batch(df, n_raw, resolution, month)
        n_raw += len(df)
        n_kept += len(kept)
        reason_counts += np.bincount(excluded["reason"].to_numpy(), minlength=len(REASONS))
        for k, bit in FLAGS.items():
            flag_counts[k] += int((kept["quality_flags"].to_numpy() & bit).astype(bool).sum())
        excluded_frames.append(excluded)
        if len(kept):
            day = kept["pickup_ts"].dt.strftime("%Y-%m-%d")
            for d, part in kept.groupby(day, sort=False):
                ddir = parts_dir / d
                ddir.mkdir(exist_ok=True)
                pq.write_table(pa.Table.from_pandas(part, schema=TRIP_SCHEMA, preserve_index=False),
                               ddir / f"part-{b:05d}.parquet")
        if verbose and b % 20 == 0:
            print(f"  batch {b}: {n_raw:,} raw rows, {n_kept:,} kept ({time.time() - t0:.0f}s)", flush=True)

    trips_dir = staging / "trips"
    trips_dir.mkdir()
    per_day = {}
    for ddir in sorted(parts_dir.iterdir()):
        table = pq.read_table(ddir)
        frame = table.to_pandas().sort_values(["pickup_ts", "source_row"], kind="stable")
        pq.write_table(pa.Table.from_pandas(frame, schema=TRIP_SCHEMA, preserve_index=False),
                       trips_dir / f"{ddir.name}.parquet")
        per_day[ddir.name] = len(frame)
    shutil.rmtree(parts_dir)

    excluded_all = pd.concat(excluded_frames, ignore_index=True) if excluded_frames else pd.DataFrame(
        {"source_row": pd.Series(dtype=np.int64), "reason": pd.Series(dtype=np.int8)})
    excluded_all["reason"] = pd.Categorical.from_codes(excluded_all["reason"].to_numpy(), categories=list(REASONS))
    excluded_all.to_parquet(staging / "excluded.parquet", index=False)

    if int(reason_counts.sum()) + n_kept != n_raw:
        raise AssertionError("row accounting failed: kept + excluded != raw")
    metadata = {
        "format_version": 1,
        "name": name,
        "raw": {"path": str(Path(raw_path).resolve()), "rows": n_raw, "sha256": _sha256(raw_path),
                "schema": "TLC yellow 2009", "file_month": str(month) if month is not None else None},
        "h3_resolution": resolution,
        "h3_version": h3.__version__,
        "timezone_policy": "naive local wall-clock time as recorded by TLC (America/New_York); no conversion",
        "predicates": {"bbox": BBOX, "duration_min": [DURATION_MIN, DURATION_MAX], "max_speed_kmh": MAX_SPEED_KMH,
                       "distance_unit": "raw miles converted to km"},
        "kept_rows": n_kept,
        "excluded_by_reason": {r: int(c) for r, c in zip(REASONS, reason_counts)},
        "flags_on_kept_rows": flag_counts,
        "kept_rows_per_day": per_day,
        "created_unix": int(time.time()),
    }
    with open(staging / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)
    if verbose:
        print(f"[preprocess] {name}: {n_raw:,} raw -> {n_kept:,} kept in {len(per_day)} day(s) ({time.time() - t0:.0f}s)")
    return final


def main(argv=None):
    p = argparse.ArgumentParser(description="Preprocess a raw 2009 NYC TLC yellow-taxi parquet file")
    p.add_argument("--input", required=True, help="Raw monthly parquet, e.g. yellow_tripdata_2009-01.parquet")
    p.add_argument("--output-root", default="data/processed", help="Directory that holds processed datasets")
    p.add_argument("--name", default=None, help="Dataset name (default: input file stem)")
    p.add_argument("--resolution", type=int, default=7, help="H3 resolution for trip endpoints")
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--overwrite", action="store_true", help="Replace an existing dataset of the same name")
    args = p.parse_args(argv)
    preprocess_month(args.input, args.output_root, args.name, args.resolution, args.batch_size, args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
