"""
GPU Core - High-performance GPU-native EV Fleet RL System

Modules:
- config: Configuration management
- state: Tensor-based state representations
- spatial: Spatial operations (distance, neighbors, grid)
- simulator: Vectorized simulation engine
- features: Feature building for neural networks
- memory: GPU replay buffer and caching
"""

__version__ = "2.0.0"

import sys as _sys

# Log lines contain non-ASCII characters (arrows, Greek letters); on a cp1252 Windows console
# they used to crash a run mid-episode. Replace what the console cannot encode instead.
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure") and (getattr(_stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
        _stream.reconfigure(errors="replace")

from .config import (
    Config,
    EnvironmentConfig,
    TrainingConfig,
    RewardConfig,
    EpisodeConfig,
    DataConfig,
    ReplayBufferConfig,
    PhysicsConfig,
    VehicleConfig,
    StationConfig,
    DistributedConfig,
    CheckpointConfig,
    LoggingConfig,
    DebugConfig,
    ConfigLoader,
)

from .state import (
    TensorFleetState,
    TensorTripState,
    TensorStationState,
    VehicleStatus,
)

from .spatial import (
    DistanceMatrix,
    HexNeighbors,
    HexGrid,
)

__all__ = [
    # Config
    "Config",
    "EnvironmentConfig", 
    "TrainingConfig",
    "RewardConfig",
    "EpisodeConfig",
    "DataConfig",
    "ReplayBufferConfig",
    "PhysicsConfig",
    "VehicleConfig",
    "StationConfig",
    "DistributedConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "DebugConfig",
    "ConfigLoader",
    # State
    "TensorFleetState",
    "TensorTripState",
    "TensorStationState",
    "VehicleStatus",
    # Spatial
    "DistanceMatrix",
    "HexNeighbors",
    "HexGrid",
]