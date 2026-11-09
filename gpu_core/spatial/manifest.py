"""Fixed spatial + infrastructure manifest (workstream B1, B5, B6; decisions Q13, Q13b, Q15, Q23).

The manifest is built once from the *training* split and then reused unchanged by every
split, month and algorithm. It fixes:

- the ordered set of H3 cells (service domain), their centres, local IJ coordinates and
  1-ring adjacency inside the domain;
- the road-detour factor and the mean intra-cell travel distance used by the simulator;
- the charging-station layout (cells, ports) and its coverage metrics.

Domain rule (Q23), two variants for the seed set of cells:

- ``min_daily_endpoints`` (default 0.5, vong 4): every cell with on average at least that many
  trip endpoints (pickups + drop-offs) per day in the training trips, all of them (before the
  trip sample);
- ``coverage``: rank cells by endpoint count and keep the smallest prefix such that a target
  share of trips has *both* endpoints inside (the earlier 99.9 % rule).

Then keep the largest connected component, connect other components only when they carry a
meaningful share, and fill enclosed holes. Station rule (Q13b): demand-weighted set cover so that every
"active" cell lies within ``cover_rings`` of a station, then fill the remaining stations by
pickup demand; at most one station per cell.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FORMAT_VERSION = 2
EARTH_RADIUS_KM = 6371.0


class ManifestError(ValueError):
    """Invalid or inconsistent manifest."""


# --------------------------------------------------------------------------- helpers

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def day_range(start: str, end: str) -> List[str]:
    """Days in the half-open interval [start, end)."""
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1), freq="D")
    if len(days) == 0:
        raise ValueError(f"empty day range [{start}, {end})")
    return [d.strftime("%Y-%m-%d") for d in days]


def find_day_files(processed_root: str, datasets: Sequence[str], days: Sequence[str]) -> List[Path]:
    """Locate the partition of each day; every day must exist in exactly one dataset."""
    root = Path(processed_root)
    files = []
    for d in days:
        hits = [root / ds / "trips" / f"{d}.parquet" for ds in datasets if (root / ds / "trips" / f"{d}.parquet").exists()]
        if not hits:
            raise FileNotFoundError(f"no processed partition for {d} in datasets {list(datasets)} under {root}")
        if len(hits) > 1:
            raise ManifestError(f"day {d} present in several datasets: {hits}")
        files.append(hits[0])
    return files


def mean_intra_cell_distance_km(cell: str, n_pairs: int = 20000, seed: int = 0) -> float:
    """Mean great-circle distance between two uniform random points inside the cell (Monte Carlo)."""
    rng = np.random.default_rng(seed)
    boundary = np.array(h3.cell_to_boundary(cell))  # (lat, lon)
    lat0 = boundary[:, 0].mean()
    kx = 111.32 * math.cos(math.radians(lat0))
    poly = np.c_[(boundary[:, 1] - boundary[:, 1].mean()) * kx, (boundary[:, 0] - lat0) * 110.57]
    lo, hi = poly.min(0), poly.max(0)

    def inside(pts):
        res = np.ones(len(pts), dtype=bool)
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            cross = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
            res &= cross >= 0
        if not res.any():  # boundary in clockwise order
            res = np.ones(len(pts), dtype=bool)
            for k in range(len(poly)):
                a, b = poly[k], poly[(k + 1) % len(poly)]
                cross = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
                res &= cross <= 0
        return res

    pts = []
    while sum(len(p) for p in pts) < 2 * n_pairs:
        cand = rng.uniform(lo, hi, size=(4 * n_pairs, 2))
        pts.append(cand[inside(cand)])
    pts = np.concatenate(pts)[: 2 * n_pairs]
    return float(np.linalg.norm(pts[:n_pairs] - pts[n_pairs:], axis=1).mean())


# --------------------------------------------------------------------------- domain

def _map_with(cells: List[str], cell_ints: np.ndarray) -> np.ndarray:
    """Domain index of each uint64 H3 id (cells in manifest order), -1 outside."""
    ints = np.array([h3.str_to_int(c) for c in cells], dtype=np.uint64)
    order = np.argsort(ints)
    srt = ints[order]
    pos = np.minimum(np.searchsorted(srt, cell_ints), len(srt) - 1)
    return np.where(srt[pos] == cell_ints, order[pos], -1).astype(np.int64)


def _components(cells: set) -> List[set]:
    seen, comps = set(), []
    for c in cells:
        if c in seen:
            continue
        comp, queue = set(), deque([c])
        seen.add(c)
        while queue:
            x = queue.popleft()
            comp.add(x)
            for nb in h3.grid_ring(x, 1):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        comps.append(comp)
    return comps


def select_domain(
    pickup_cells: np.ndarray,
    dropoff_cells: np.ndarray,
    coverage: float = 0.999,
    min_component_share: float = 0.0005,
    hole_min_neighbors: int = 4,
    min_daily_endpoints: Optional[float] = None,
    num_days: Optional[int] = None,
) -> Tuple[List[str], Dict]:
    """Return (sorted cell ids, report) for the service domain (Q23). With
    ``min_daily_endpoints`` the seed cells are those with at least that many trip endpoints per
    day on average over ``num_days`` days; otherwise the ``coverage`` prefix rule is used."""
    n_trips = len(pickup_cells)
    if n_trips == 0:
        raise ManifestError("no training trips to build the domain from")
    uniq, counts = np.unique(np.concatenate([pickup_cells, dropoff_cells]), return_counts=True)
    order = np.lexsort((uniq, -counts))                       # by count desc, then id
    if min_daily_endpoints is not None:
        if not num_days or num_days <= 0:
            raise ValueError("min_daily_endpoints needs the number of training days")
        if min_daily_endpoints <= 0:
            raise ValueError("min_daily_endpoints must be positive")
        k = int((counts / num_days >= min_daily_endpoints).sum())
        rule = {"rule": "min_daily_endpoints", "min_daily_endpoints": min_daily_endpoints, "days": int(num_days)}
    else:
        if not 0 < coverage <= 1:
            raise ValueError("coverage must be in (0, 1]")
        rank = np.empty(len(uniq), dtype=np.int64)
        rank[order] = np.arange(len(uniq))
        need = np.maximum(rank[np.searchsorted(uniq, pickup_cells)], rank[np.searchsorted(uniq, dropoff_cells)])
        k = int(np.sort(need)[int(math.ceil(coverage * n_trips)) - 1]) + 1
        rule = {"rule": "coverage", "coverage_target": coverage}
    if k == 0:
        raise ManifestError("the domain rule keeps no cell")
    top = {h3.int_to_str(int(c)) for c in uniq[order[:k]]}
    weight = {h3.int_to_str(int(c)): int(n) for c, n in zip(uniq, counts)}
    total_w = float(counts.sum())

    # Sets of strings iterate in a per-process order (hash randomisation): every choice below
    # breaks ties by H3 id so the domain is identical across runs.
    comps = sorted(_components(top), key=lambda s: (-sum(weight.get(c, 0) for c in s), min(map(h3.str_to_int, s))))
    domain = set(comps[0])
    dropped, connectors = [], set()
    for comp in comps[1:]:
        share = sum(weight.get(c, 0) for c in comp) / total_w
        if share >= min_component_share:
            a, b = min(((x, y) for x in comp for y in domain),
                       key=lambda p: (h3.grid_distance(*p), h3.str_to_int(p[0]), h3.str_to_int(p[1])))
            path = set(h3.grid_path_cells(a, b))
            connectors |= path - domain - comp
            domain |= comp | path
        else:
            dropped.append({"cells": sorted(comp), "endpoint_share": share})

    holes = set()
    for _ in range(50):
        frontier = {nb for c in domain for nb in h3.grid_ring(c, 1)} - domain
        add = {c for c in frontier if sum(nb in domain for nb in h3.grid_ring(c, 1)) >= hole_min_neighbors}
        if not add:
            break
        holes |= add
        domain |= add

    cells = sorted(domain, key=h3.str_to_int)
    dset = set(h3.str_to_int(c) for c in cells)
    inside = np.isin(pickup_cells, list(dset)) & np.isin(dropoff_cells, list(dset))
    report = {
        **rule,
        "top_k_cells": k,
        "cells": len(cells),
        "trips_considered": int(n_trips),
        "trip_share_inside": float(inside.mean()),
        "dropped_components": dropped,
        "connector_cells_added": sorted(connectors),
        "hole_cells_added": sorted(holes),
    }
    return cells, report


# --------------------------------------------------------------------------- stations

def place_stations(
    cells: Sequence[str],
    daily_pickups: Dict[str, float],
    num_stations: int,
    cover_rings: int = 2,
    min_daily_pickups: float = 1.0,
) -> Tuple[List[str], Dict]:
    """Cover-then-fill placement (Q13b). Returns (station cells in placement order, report)."""
    cellset = set(cells)
    if num_stations > len(cells):
        raise ManifestError(f"{num_stations} stations requested for {len(cells)} cells")
    demand = {c: float(daily_pickups.get(c, 0.0)) for c in cells}
    active = {c for c in cells if demand[c] >= min_daily_pickups}
    disk = {c: set(h3.grid_disk(c, cover_rings)) & cellset for c in cells}

    chosen: List[str] = []
    uncovered = set(active)
    while uncovered and len(chosen) < num_stations:
        best = max((c for c in cells if c not in chosen),
                   key=lambda c: (sum(demand[x] for x in disk[c] & uncovered), demand[c], -h3.str_to_int(c)))
        if not disk[best] & uncovered:
            break
        chosen.append(best)
        uncovered -= disk[best]
    n_cover = len(chosen)
    for c in sorted(cells, key=lambda c: (-demand[c], h3.str_to_int(c))):
        if len(chosen) >= num_stations:
            break
        if c not in chosen:
            chosen.append(c)

    def gd(a, b):
        return h3.grid_distance(a, b)

    total = sum(demand.values()) or 1.0
    nearest_rings = {c: min(gd(c, s) for s in chosen) for c in cells}
    centres = {c: h3.cell_to_latlng(c) for c in cells}
    km_to_station = {c: min(float(haversine_km(*centres[c], *centres[s])) for s in chosen) for c in cells}
    catch = Counter()
    for c in cells:
        catch[min(chosen, key=lambda s: (gd(c, s), h3.str_to_int(s)))] += demand[c]
    shares = [catch[s] / total for s in chosen]
    report = {
        "rule": "cover_then_fill",
        "cover_rings": cover_rings,
        "min_daily_pickups": min_daily_pickups,
        "active_cells": len(active),
        "stations_for_cover": n_cover,
        "active_uncovered_after_cover": len(uncovered),
        "max_km_active_cell_to_station": max((km_to_station[c] for c in active), default=0.0),
        "max_rings_active_cell_to_station": max((nearest_rings[c] for c in active), default=0),
        "demand_share_within_rings": {str(r): sum(demand[c] for c in cells if nearest_rings[c] <= r) / total for r in (0, 1, 2)},
        "catchment_demand_share_max": max(shares),
        "catchment_demand_share_min": min(shares),
    }
    if uncovered:
        report["warning"] = f"{len(uncovered)} active cell(s) not within {cover_rings} rings of any station"
    return chosen, report


# --------------------------------------------------------------------------- manifest

@dataclass
class SpatialManifest:
    resolution: int
    cells: List[str]
    centers: List[Tuple[float, float]]
    local_ij: List[Tuple[int, int]]
    anchor: str
    adjacency: List[List[int]]
    accessible: List[bool]
    detour_factor: float
    intra_cell_km: float
    station_cells: List[str]
    ports_per_station: int
    station_report: Dict = field(default_factory=dict)
    # Training pickups per hour of day and cell [24][cells]; initial vehicle positions (C6).
    pickup_by_hour: List[List[int]] = field(default_factory=list)
    source: Dict = field(default_factory=dict)
    spatial_hash: str = ""
    infra_hash: str = ""

    # ---- derived lookups (not serialised)
    def __post_init__(self):
        self._index = {c: i for i, c in enumerate(self.cells)}
        ints = np.array([h3.str_to_int(c) for c in self.cells], dtype=np.uint64)
        self._order = np.argsort(ints)
        self._sorted_ints = ints[self._order]

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    def index_of(self, cell: str) -> int:
        return self._index[cell]

    @property
    def station_indices(self) -> List[int]:
        return [self._index[c] for c in self.station_cells]

    def map_cells(self, cell_ints: np.ndarray) -> np.ndarray:
        """Map uint64 H3 ints to domain indices; -1 for cells outside the domain."""
        cell_ints = np.asarray(cell_ints, dtype=np.uint64)
        pos = np.searchsorted(self._sorted_ints, cell_ints)
        pos_c = np.minimum(pos, len(self._sorted_ints) - 1)
        hit = self._sorted_ints[pos_c] == cell_ints
        return np.where(hit, self._order[pos_c], -1).astype(np.int64)

    # ---- hashing / io
    def compute_hashes(self) -> Tuple[str, str]:
        spatial = _canonical_hash({"resolution": self.resolution, "cells": self.cells})
        infra = _canonical_hash({"spatial": spatial, "stations": self.station_cells, "ports": self.ports_per_station,
                                 "detour": round(self.detour_factor, 9), "intra": round(self.intra_cell_km, 9),
                                 "pickup_by_hour": self.pickup_by_hour})
        return spatial, infra

    def to_dict(self) -> Dict:
        return {
            "format_version": FORMAT_VERSION,
            "resolution": self.resolution, "cells": self.cells, "centers": self.centers, "local_ij": self.local_ij,
            "anchor": self.anchor, "adjacency": self.adjacency, "accessible": self.accessible,
            "detour_factor": self.detour_factor, "intra_cell_km": self.intra_cell_km,
            "stations": {"cells": self.station_cells, "ports_per_station": self.ports_per_station, "report": self.station_report},
            "pickup_by_hour": self.pickup_by_hour,
            "source": self.source, "spatial_hash": self.spatial_hash, "infra_hash": self.infra_hash,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str) -> "SpatialManifest":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("format_version") != FORMAT_VERSION:
            raise ManifestError(f"{path}: unsupported manifest format {d.get('format_version')}")
        m = cls(
            resolution=d["resolution"], cells=d["cells"], centers=[tuple(x) for x in d["centers"]],
            local_ij=[tuple(x) for x in d["local_ij"]], anchor=d["anchor"], adjacency=d["adjacency"],
            accessible=d["accessible"], detour_factor=d["detour_factor"], intra_cell_km=d["intra_cell_km"],
            station_cells=d["stations"]["cells"], ports_per_station=d["stations"]["ports_per_station"],
            station_report=d["stations"].get("report", {}), pickup_by_hour=d["pickup_by_hour"], source=d.get("source", {}),
            spatial_hash=d["spatial_hash"], infra_hash=d["infra_hash"],
        )
        m.validate()
        return m

    def validate(self) -> None:
        n = len(self.cells)
        if n == 0:
            raise ManifestError("manifest has no cells")
        if len(set(self.cells)) != n:
            raise ManifestError("duplicate cells in manifest")
        for c in self.cells:
            if not h3.is_valid_cell(c) or h3.get_resolution(c) != self.resolution:
                raise ManifestError(f"cell {c!r} is not a valid resolution-{self.resolution} H3 cell")
        if self.cells != sorted(self.cells, key=h3.str_to_int):
            raise ManifestError("cells must be ordered by H3 integer id")
        for name, seq in (("centers", self.centers), ("local_ij", self.local_ij), ("adjacency", self.adjacency),
                          ("accessible", self.accessible)):
            if len(seq) != n:
                raise ManifestError(f"{name} has {len(seq)} entries for {n} cells")
        for i, nbrs in enumerate(self.adjacency):
            for j in nbrs:
                if not 0 <= j < n or i not in self.adjacency[j]:
                    raise ManifestError(f"adjacency is not symmetric/in range at cell {i}")
        if len(set(self.station_cells)) != len(self.station_cells) or not set(self.station_cells) <= set(self.cells):
            raise ManifestError("station cells must be unique cells of the domain")
        if len(self.pickup_by_hour) != 24 or any(len(row) != n for row in self.pickup_by_hour)                 or any(sum(row) <= 0 for row in self.pickup_by_hour):
            raise ManifestError("pickup_by_hour must be 24 rows of per-cell counts, each with some pickups")
        if (self.spatial_hash, self.infra_hash) != self.compute_hashes():
            raise ManifestError("manifest hash mismatch: file was edited or corrupted")


def build_manifest(
    processed_root: str,
    datasets: Sequence[str],
    train_start: str,
    train_end: str,
    resolution: int = 7,
    coverage: float = 0.999,
    num_stations: int = 35,
    ports_per_station: int = 5,
    cover_rings: int = 2,
    cover_min_daily_pickups: float = 1.0,
    sample_ratio: float = 0.12,
    charge_power_kw: float = 50.0,
    feeder_kw: float = 5500.0,
    detour_sample: int = 200_000,
    seed: int = 0,
    verbose: bool = True,
    min_daily_endpoints: Optional[float] = None,
) -> SpatialManifest:
    """Build the manifest from the processed training days [train_start, train_end)."""
    if num_stations * ports_per_station * charge_power_kw <= feeder_kw:
        raise ManifestError(
            f"{num_stations} stations x {ports_per_station} ports x {charge_power_kw} kW does not exceed the "
            f"{feeder_kw} kW feeder: the feeder would never bind (needs > {feeder_kw / (ports_per_station * charge_power_kw):.0f} stations)")
    days = day_range(train_start, train_end)
    files = find_day_files(processed_root, datasets, days)
    t0 = time.time()
    for ds in sorted({f.parent.parent for f in files}):
        meta = json.loads((ds / "metadata.json").read_text(encoding="utf-8"))
        if meta["h3_resolution"] != resolution:
            raise ManifestError(f"{ds.name} was preprocessed at r{meta['h3_resolution']}, manifest wants r{resolution}")
    # Domain: only the two cell columns of every training trip (16 bytes per trip).
    cell_tables = [pq.read_table(f, columns=["pickup_cell", "dropoff_cell"]) for f in files]
    pc = np.concatenate([t.column("pickup_cell").to_numpy() for t in cell_tables]).astype(np.uint64)
    dc = np.concatenate([t.column("dropoff_cell").to_numpy() for t in cell_tables]).astype(np.uint64)
    del cell_tables
    cells, domain_report = select_domain(pc, dc, coverage=coverage, min_daily_endpoints=min_daily_endpoints,
                                         num_days=len(days))
    cellset = set(cells)
    index = {c: i for i, c in enumerate(cells)}

    pick_uniq, pick_cnt = np.unique(pc, return_counts=True)
    anchor = h3.int_to_str(int(pick_uniq[np.argmax(pick_cnt)]))
    if anchor not in cellset:
        raise ManifestError("most frequent pickup cell is not in the domain")
    local_ij = [tuple(int(v) for v in h3.cell_to_local_ij(anchor, c)) for c in cells]
    centers = [tuple(h3.cell_to_latlng(c)) for c in cells]
    adjacency = [sorted(index[nb] for nb in h3.grid_ring(c, 1) if nb in cellset) for c in cells]

    # Road-detour factor on a per-file row sample: median(recorded / straight line),
    # straight line >= 0.5 km, rows not flagged as shorter than the straight line.
    rng = np.random.default_rng(seed)
    per_file = max(1, detour_sample // len(files))
    ratios = []
    detour_cols = ["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon", "distance_km", "quality_flags"]
    for f in files:
        t = pq.read_table(f, columns=detour_cols).to_pandas()
        if len(t) > per_file:
            t = t.iloc[np.sort(rng.choice(len(t), per_file, replace=False))]
        straight = haversine_km(t.pickup_lat.to_numpy(), t.pickup_lon.to_numpy(), t.dropoff_lat.to_numpy(), t.dropoff_lon.to_numpy())
        ok = (straight >= 0.5) & ((t.quality_flags.to_numpy() & 2) == 0)
        ratios.append(t.distance_km.to_numpy()[ok] / straight[ok])
    ratios = np.concatenate(ratios)
    detour = float(np.median(ratios))
    intra = mean_intra_cell_distance_km(anchor, seed=seed) * detour

    # Pickups by hour of day, in-domain cells only, full training split (initial positions, C6).
    by_hour = np.zeros((24, len(cells)), dtype=np.int64)
    for f in files:
        t = pq.read_table(f, columns=["pickup_ts", "pickup_cell"])
        hours = (t.column("pickup_ts").to_numpy().astype("datetime64[s]").astype(np.int64) % 86400) // 3600
        idx = _map_with(cells, t.column("pickup_cell").to_numpy().astype(np.uint64))
        ok = idx >= 0
        np.add.at(by_hour, (hours[ok], idx[ok]), 1)

    pick_counts = {h3.int_to_str(int(c)): int(n) for c, n in zip(pick_uniq, pick_cnt)}
    daily = {c: pick_counts.get(c, 0) * sample_ratio / len(days) for c in cells}
    stations, station_report = place_stations(cells, daily, num_stations, cover_rings, cover_min_daily_pickups)

    m = SpatialManifest(
        resolution=resolution, cells=cells, centers=centers, local_ij=local_ij, anchor=anchor,
        adjacency=adjacency, accessible=[True] * len(cells), detour_factor=detour, intra_cell_km=intra,
        station_cells=stations, ports_per_station=ports_per_station, station_report=station_report,
        pickup_by_hour=by_hour.tolist(),
        source={
            "datasets": [{"name": ds, "raw_sha256": json.loads((Path(processed_root) / ds / "metadata.json").read_text(encoding="utf-8"))["raw"]["sha256"]}
                         for ds in sorted({f.parent.parent.name for f in files})],
            "train_range": [train_start, train_end], "train_days": len(days),
            "sample_ratio_for_station_demand": sample_ratio, "domain": domain_report,
            "detour": {"trips_used": int(len(ratios)), "rule": "median(recorded_km / great_circle_km), great_circle >= 0.5 km, unflagged"},
            "intra_cell": {"rule": "Monte-Carlo mean distance of two uniform points in the anchor cell x detour factor"},
            "build_seconds": round(time.time() - t0, 1),
        },
    )
    m.spatial_hash, m.infra_hash = m.compute_hashes()
    m.validate()
    if verbose:
        dr = domain_report
        seed = (f">= {dr['min_daily_endpoints']:g} endpoints/day" if dr["rule"] == "min_daily_endpoints"
                else f"{dr['coverage_target']:.3%} coverage")
        print(f"[manifest] r{resolution}: {len(cells)} cells (seed {seed}: {dr['top_k_cells']}, +{len(dr['connector_cells_added'])} connectors, "
              f"+{len(dr['hole_cells_added'])} holes, {len(dr['dropped_components'])} dropped comps); "
              f"trips inside {dr['trip_share_inside']:.4%}; detour {detour:.3f}; intra-cell {intra:.2f} km")
        sr = station_report
        print(f"[manifest] stations: {len(stations)} ({sr['stations_for_cover']} for cover), max {sr['max_km_active_cell_to_station']:.1f} km "
              f"/ {sr['max_rings_active_cell_to_station']} rings from active cells; demand within 0/1/2 rings "
              f"{', '.join(f'{v:.1%}' for v in sr['demand_share_within_rings'].values())}")
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the fixed spatial + station manifest from the training split")
    p.add_argument("--processed-root", required=True)
    p.add_argument("--datasets", nargs="+", required=True, help="Processed dataset names (e.g. yellow_tripdata_2009-01)")
    p.add_argument("--train-start", required=True, help="First training day (inclusive), YYYY-MM-DD")
    p.add_argument("--train-end", required=True, help="Day after the last training day (exclusive), YYYY-MM-DD")
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, default=7)
    p.add_argument("--min-daily-endpoints", type=float, default=0.5,
                   help="domain seed: cells with at least this many trip endpoints per day (all training trips); "
                        "set 0 to use --coverage instead")
    p.add_argument("--coverage", type=float, default=0.999,
                   help="domain seed when --min-daily-endpoints is 0: smallest set of cells holding this share of trips")
    p.add_argument("--num-stations", type=int, default=35)
    p.add_argument("--ports-per-station", type=int, default=5)
    p.add_argument("--cover-rings", type=int, default=2)
    p.add_argument("--cover-min-daily-pickups", type=float, default=1.0)
    p.add_argument("--sample-ratio", type=float, default=0.12)
    p.add_argument("--charge-power-kw", type=float, default=50.0)
    p.add_argument("--feeder-kw", type=float, default=5500.0)
    a = p.parse_args(argv)
    m = build_manifest(a.processed_root, a.datasets, a.train_start, a.train_end, a.resolution, a.coverage,
                       a.num_stations, a.ports_per_station, a.cover_rings, a.cover_min_daily_pickups,
                       a.sample_ratio, a.charge_power_kw, a.feeder_kw,
                       min_daily_endpoints=a.min_daily_endpoints if a.min_daily_endpoints > 0 else None)
    m.save(a.out)
    print(f"[manifest] saved {a.out} (spatial {m.spatial_hash[:12]}, infra {m.infra_hash[:12]})")


if __name__ == "__main__":
    sys.exit(main())
