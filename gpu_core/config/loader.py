"""Configuration loader with YAML and CLI support."""

import argparse
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import fields, is_dataclass
from .base import (
    Config,
    EnvironmentConfig,
    EpisodeConfig,
    DataConfig,
    TrainingConfig,
    LearningRateConfig,
    ReplayBufferConfig,
    CurriculumConfig,
    PickupDistanceConfig,
    RewardConfig,
    PhysicsConfig,
    VehicleConfig,
    StationConfig,
    DistributedConfig,
    CheckpointConfig,
    LoggingConfig,
    DebugConfig,
    NetworkConfig,
)


class ConfigLoader:
    
    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    
    @staticmethod
    def _update_dataclass(obj: Any, data: Dict[str, Any]) -> None:
        if not is_dataclass(obj) or not data:
            return
        for f in fields(obj):
            if f.name in data:
                val = data[f.name]
                current = getattr(obj, f.name)
                if is_dataclass(current) and isinstance(val, dict):
                    ConfigLoader._update_dataclass(current, val)
                else:
                    setattr(obj, f.name, val)
    
    @staticmethod
    def from_yaml(path: str) -> Config:
        data = ConfigLoader.load_yaml(path)
        config = Config()
        ConfigLoader._update_dataclass(config, data)
        return config
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="GPU-native EV Fleet RL Training")
        parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
        
        # Environment
        parser.add_argument("--num-vehicles", type=int, default=None)
        parser.add_argument("--num-stations", type=int, default=None)
        parser.add_argument("--max-hexagons", type=int, default=None)
        
        # Episode
        parser.add_argument("--episode-duration", type=float, default=None)
        parser.add_argument("--step-duration", type=float, default=None)
        
        # Data
        parser.add_argument("--parquet-path", type=str, default=None)
        parser.add_argument("--cache-dir", type=str, default=None)
        parser.add_argument("--trip-percentage", type=float, default=None)
        parser.add_argument("--start-date", type=str, default=None)
        parser.add_argument("--target-h3-resolution", type=int, default=None,
                    help="Downsample H3 resolution for coarse grid")
        parser.add_argument("--max-hex-count", type=int, default=None,
                    help="Limit unique hex count after coarsening")
        
        # Training
        parser.add_argument("--total-episodes", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--episode-batch-size", type=int, default=None)
        parser.add_argument("--lr-actor", type=float, default=None)
        parser.add_argument("--lr-critic", type=float, default=None)
        parser.add_argument("--gamma", type=float, default=None)
        parser.add_argument("--warmup-steps", type=int, default=None)
        parser.add_argument("--no-mixed-precision", action="store_true")
        parser.add_argument("--no-compile", action="store_true")
        
        # Distributed
        parser.add_argument("--gpus", type=str, default=None)
        parser.add_argument("--distributed", action="store_true")
        
        # Checkpoint
        parser.add_argument("--checkpoint-dir", type=str, default=None)
        parser.add_argument("--resume", type=str, default=None)
        parser.add_argument("--save-every", type=int, default=None)
        
        # Logging
        parser.add_argument("--log-dir", type=str, default=None)
        parser.add_argument("--log-level", type=str, default=None)
        parser.add_argument("--wandb", action="store_true")
        
        # Debug
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--deterministic", action="store_true")
        parser.add_argument("--profile", action="store_true")
        
        return parser
    
    @staticmethod
    def from_args(args: Optional[argparse.Namespace] = None) -> Config:
        parser = ConfigLoader.create_parser()
        if args is None:
            args = parser.parse_args()
        
        if args.config:
            config = ConfigLoader.from_yaml(args.config)
        else:
            config = Config()
        
        # Apply CLI overrides
        if args.num_vehicles is not None:
            config.environment.num_vehicles = args.num_vehicles
        if args.num_stations is not None:
            config.environment.num_stations = args.num_stations
        if args.max_hexagons is not None:
            config.environment.max_hexagons = args.max_hexagons
            
        if args.episode_duration is not None:
            config.episode.duration_hours = args.episode_duration
        if args.step_duration is not None:
            config.episode.step_duration_minutes = args.step_duration
            
        if args.parquet_path is not None:
            config.data.parquet_path = args.parquet_path
        if args.cache_dir is not None:
            config.data.cache_dir = args.cache_dir
        if args.trip_percentage is not None:
            config.data.trip_percentage = args.trip_percentage
        if args.start_date is not None:
            config.data.start_date = args.start_date
        if args.target_h3_resolution is not None:
            config.data.target_h3_resolution = args.target_h3_resolution
        if args.max_hex_count is not None:
            config.data.max_hex_count = args.max_hex_count
            
        if args.total_episodes is not None:
            config.training.total_episodes = args.total_episodes
        if args.batch_size is not None:
            config.training.batch_size = args.batch_size
        if args.episode_batch_size is not None:
            config.training.episode_batch_size = args.episode_batch_size
        if args.lr_actor is not None:
            config.training.learning_rate.actor = args.lr_actor
        if args.lr_critic is not None:
            config.training.learning_rate.critic = args.lr_critic
        if args.gamma is not None:
            config.training.gamma = args.gamma
        if args.warmup_steps is not None:
            config.training.warmup_steps = args.warmup_steps
        if args.no_mixed_precision:
            config.training.mixed_precision = False
        if args.no_compile:
            config.training.compile_model = False
            
        if args.gpus is not None:
            config.distributed.gpus = args.gpus
        if args.distributed:
            config.distributed.enabled = True
            
        if args.checkpoint_dir is not None:
            config.checkpoint.checkpoint_dir = args.checkpoint_dir
        if args.resume is not None:
            config.checkpoint.resume_from = args.resume
        if args.save_every is not None:
            config.checkpoint.save_interval = args.save_every
            
        if args.log_dir is not None:
            config.logging.log_dir = args.log_dir
        if args.log_level is not None:
            config.logging.level = args.log_level
        if args.wandb:
            config.logging.use_wandb = True
            
        if args.seed is not None:
            config.debug.seed = args.seed
        if args.deterministic:
            config.debug.deterministic = True
        if args.profile:
            config.debug.profile_gpu = True
        
        return config
    
    @staticmethod
    def save_yaml(config: Config, path: str) -> None:
        def to_dict(obj: Any) -> Any:
            if is_dataclass(obj):
                return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
            return obj
        
        data = to_dict(config)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
