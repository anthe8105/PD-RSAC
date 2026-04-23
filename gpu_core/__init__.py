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

from .config import (
    Config,
    EnvironmentConfig,
    TrainingConfig,
    RewardConfig,
    EpisodeConfig,
    DataConfig,
    ReplayBufferConfig,
    CurriculumConfig,
    PhysicsConfig,
    VehicleConfig,
    StationConfig,
    DistributedConfig,
    CheckpointConfig,
    LoggingConfig,
    DebugConfig,
    NetworkConfig,
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
    "CurriculumConfig",
    "PhysicsConfig",
    "VehicleConfig",
    "StationConfig",
    "DistributedConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "DebugConfig",
    "NetworkConfig",
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