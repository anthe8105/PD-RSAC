"""Data pipeline: raw preprocessing, split datasets and the environment factory.

``gpu_core.data.factory`` (build_env) is imported explicitly by entrypoints to avoid an
import cycle with the simulator.
"""

from .dataset import TripDataset, resolve_path, sample_mask

__all__ = ["TripDataset", "resolve_path", "sample_mask"]
