"""Single place that builds the spatial grid, the split dataset and the environment.

Every entrypoint (train, evaluate, power-grid evaluation, baselines, heuristic) goes through
``build_env`` so that cells, distances, stations and trips are identical across scripts
(audit D22, D23, F40, F41). The simulator runs on ``SIM_DEVICE`` (CPU) whatever device the
networks use: matching, bookings and the MILP are sequential / numpy and the fleet tensors are
small, so a GPU would only add transfers; the learners move what they read to their device. Real data always needs the spatial manifest; synthetic trips
are only used when explicitly requested (``synthetic=True``), never as a silent fallback.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import h3
import numpy as np
import torch

from ..config import Config
from ..simulator.environment import GPUEnvironment
from ..spatial.grid import HexGrid
from ..spatial.manifest import SpatialManifest, haversine_km
from .dataset import TripDataset, resolve_path

SYNTHETIC_CENTER = (40.758, -73.9855)  # Midtown Manhattan
SIM_DEVICE = torch.device("cpu")


def load_manifest(config: Config) -> SpatialManifest:
    path = resolve_path(config.data.manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"spatial manifest {path} not found; build it with "
            f"python -m gpu_core.spatial.manifest --processed-root ... --train-start ... --train-end ... --out {path}")
    manifest = SpatialManifest.load(str(path))
    if manifest.resolution != config.data.h3_resolution:
        raise ValueError(f"manifest is r{manifest.resolution} but data.h3_resolution is {config.data.h3_resolution}")
    return manifest


def split_range(config: Config, split: str) -> Tuple[str, str]:
    if split not in config.data.splits:
        raise KeyError(f"unknown split '{split}'; configured splits: {sorted(config.data.splits)}")
    start, end = config.data.splits[split]
    return start, end


def split_hours(config: Config, split: str) -> float:
    start, end = split_range(config, split)
    return (np.datetime64(end) - np.datetime64(start)) / np.timedelta64(1, "h")


def assert_disjoint_splits(config: Config, names=("train", "val", "test")) -> None:
    """Training, validation and test intervals must not overlap (audit D24)."""
    spans = sorted((np.datetime64(config.data.splits[n][0]), np.datetime64(config.data.splits[n][1]), n)
                   for n in names if n in config.data.splits)
    for (s0, e0, a), (s1, e1, b) in zip(spans, spans[1:]):
        if s1 < e0:
            raise ValueError(f"splits '{a}' and '{b}' overlap")


def load_split(config: Config, split: str, manifest: SpatialManifest, verbose: bool = True) -> TripDataset:
    start, end = split_range(config, split)
    return TripDataset(config.data.processed_root, config.data.datasets, manifest, start, end,
                       config.data.sample_ratio, config.data.sample_seed, split_name=split, verbose=verbose)


def grid_from_manifest(manifest: SpatialManifest, device) -> HexGrid:
    """Cells, centres, distances (great circle x detour, intra-cell on the diagonal) and 1-ring adjacency."""
    grid = HexGrid(device=str(device))
    lat = torch.tensor([c[0] for c in manifest.centers], dtype=torch.float64)
    lon = torch.tensor([c[1] for c in manifest.centers], dtype=torch.float64)
    _fill_grid(grid, list(manifest.cells), lat, lon, manifest.detour_factor, manifest.intra_cell_km, device)
    adj = torch.zeros(manifest.num_cells, manifest.num_cells)
    for i, nbrs in enumerate(manifest.adjacency):
        adj[i, nbrs] = 1.0
    grid.neighbors.compute_from_adjacency(adj.to(device), hex_ids=list(manifest.cells))
    return grid


SYNTHETIC_DETOUR = 1.302      # road / straight-line factor of the January training split (manifest)
SYNTHETIC_INTRA_KM = 1.52     # mean intra-cell distance at r7 x detour (manifest)


def synthetic_grid(num_hexes: int, device, resolution: int = 7) -> HexGrid:
    """Real H3 geometry for synthetic runs: the smallest disk around Midtown with >= num_hexes cells,
    with the same detour factor and intra-cell distance as the real manifest."""
    center = h3.latlng_to_cell(*SYNTHETIC_CENTER, resolution)
    rings = 0
    while 1 + 3 * rings * (rings + 1) < num_hexes:
        rings += 1
    cells = sorted(h3.grid_disk(center, rings), key=h3.str_to_int)
    lat = torch.tensor([h3.cell_to_latlng(c)[0] for c in cells], dtype=torch.float64)
    lon = torch.tensor([h3.cell_to_latlng(c)[1] for c in cells], dtype=torch.float64)
    grid = HexGrid(device=str(device))
    _fill_grid(grid, cells, lat, lon, SYNTHETIC_DETOUR, SYNTHETIC_INTRA_KM, device)
    grid.neighbors.compute(cells, k=1)
    return grid


def _fill_grid(grid: HexGrid, cells, lat, lon, detour: float, intra_km: float, device) -> None:
    grid._hex_ids = list(cells)
    grid._hex_to_idx = {c: i for i, c in enumerate(cells)}
    grid._latitudes = lat.float().to(device)
    grid._longitudes = lon.float().to(device)
    grid._initialized = True
    d = haversine_km(lat.numpy()[:, None], lon.numpy()[:, None], lat.numpy()[None, :], lon.numpy()[None, :]) * detour
    np.fill_diagonal(d, intra_km)
    grid.distance_matrix._distances = torch.tensor(d, dtype=torch.float32, device=device)
    grid.distance_matrix._num_hexes = len(cells)
    grid.distance_matrix._hex_ids = list(cells)


def add_data_args(parser, default_split: str = "test") -> None:
    """CLI options shared by every evaluation entrypoint."""
    parser.add_argument("--split", default=default_split,
                        help="Configured data split to run on (e.g. test, val, nye); one episode spans the split")
    parser.add_argument("--synthetic", action="store_true", help="Synthetic trips on a small H3 disk (smoke tests only)")
    parser.add_argument("--manifest", default=None, help="Override data.manifest_path")
    parser.add_argument("--processed-root", default=None, help="Override data.processed_root")


def apply_data_args(config: Config, args) -> None:
    if getattr(args, "manifest", None):
        config.data.manifest_path = args.manifest
    if getattr(args, "processed_root", None):
        config.data.processed_root = args.processed_root
    if getattr(args, "sample_ratio", None) is not None:
        config.data.sample_ratio = float(args.sample_ratio)


def config_for_evaluation(checkpoint: dict, yaml_path: Optional[str] = None) -> Config:
    """Run config for evaluating a checkpoint: an explicit YAML, else the config embedded in the
    checkpoint by the training run (A4). Old checkpoints without one need ``--config``."""
    from ..config import ConfigLoader
    if yaml_path:
        return ConfigLoader.from_yaml(yaml_path)
    embedded = checkpoint.get("config")
    if not isinstance(embedded, dict) or "environment" not in embedded:
        raise ValueError("checkpoint has no embedded run config; pass --config")
    config = Config()
    ConfigLoader._update_dataclass(config, embedded)
    # evaluation splits added after the checkpoint was trained (e.g. nye, round 11); the
    # training splits stay those of the checkpoint
    from ..config.base import _default_splits
    for name, span in _default_splits().items():
        config.data.splits.setdefault(name, span)
    # checkpoints written before milp.enabled existed recorded the switch only as a CLI argument
    if "enabled" not in (embedded.get("milp") or {}) and (checkpoint.get("run_args") or {}).get("milp"):
        config.milp.enabled = True
    return config


def build_env(
    config: Config,
    device,
    split: Optional[str] = "train",
    synthetic: bool = False,
    manifest: Optional[SpatialManifest] = None,
    verbose: bool = True,
) -> Tuple[GPUEnvironment, Optional[TripDataset], Optional[SpatialManifest]]:
    """Return (env, dataset, manifest). Sets ``config.environment.num_hexes`` from the grid."""
    if synthetic:
        grid = synthetic_grid(config.environment.num_hexes, device, config.data.h3_resolution)
        config.environment.num_hexes = grid.num_hexes
        return GPUEnvironment(config=config, hex_grid=grid, trip_loader=None, device=str(device)), None, None
    assert_disjoint_splits(config)
    manifest = manifest or load_manifest(config)
    dataset = load_split(config, split, manifest, verbose=verbose)
    grid = grid_from_manifest(manifest, device)
    config.environment.num_hexes = manifest.num_cells
    env = GPUEnvironment(config=config, hex_grid=grid, trip_loader=dataset, device=str(device), manifest=manifest)
    return env, dataset, manifest


@dataclass
class Run:
    """One split's pipeline: environment, dataset, manifest and (when ``milp.enabled``) projection."""
    env: GPUEnvironment
    dataset: Optional[TripDataset]
    manifest: Optional[SpatialManifest]
    projection: Optional[Any]


def build_run(
    config: Config,
    device,
    split: Optional[str] = "train",
    synthetic: bool = False,
    manifest: Optional[SpatialManifest] = None,
    verbose: bool = True,
) -> Run:
    """Environment of ``split`` plus its MILP projection when ``config.milp.enabled`` (F1).
    Training, checkpoint selection, evaluation, power-grid evaluation and the test protocol all
    assemble their pipeline here, so a preset is evaluated exactly as it was trained (F2, F39)."""
    env, dataset, manifest = build_env(config, device, split=split, synthetic=synthetic,
                                       manifest=manifest, verbose=verbose)
    projection = None
    if config.milp.enabled:
        from ..simulator.projection import make_projection
        projection = make_projection(env)
    return Run(env, dataset, manifest, projection)
