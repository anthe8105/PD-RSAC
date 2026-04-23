"""Configuration dataclasses for GPU-native EV Fleet RL System."""

from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path


@dataclass
class EnvironmentConfig:
    num_vehicles: int = 1000
    num_stations: int = 150
    num_hexes: int = 1300
    hex_resolution: int = 8
    max_hexagons: int = 1500
    compact_state: bool = False  # Aggregate vehicle features per hex for faster training
    
    max_trips_per_step: int = 2000  # Max trips to consider per step (Actor & Critic)
    max_serve_per_step: int = 500   # Max vehicle assignments per step
    max_charge_per_step: int = 200  # Max charging decisions per step


@dataclass
class EpisodeConfig:
    duration_hours: float = 10.0
    step_duration_minutes: float = 5.0
    
    @property
    def steps_per_episode(self) -> int:
        return int(self.duration_hours * 60 / self.step_duration_minutes)


@dataclass
class DataConfig:
    parquet_path: str = "data/yellow_tripdata_2009-01.parquet"
    cache_dir: str = "cache/trips"
    trip_percentage: float = 0.3
    prefetch_episodes: int = 5
    start_date: str = "2009-01-01"
    target_h3_resolution: Optional[int] = None
    max_hex_count: Optional[int] = None


@dataclass
class LearningRateConfig:
    actor: float = 3.0e-4
    critic: float = 3.0e-4
    alpha: float = 3.0e-4


@dataclass
class TrainingConfig:
    total_episodes: int = 5000
    batch_size: int = 256          # Standard for SAC
    episode_batch_size: int = 4
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)
    gamma: float = 0.99            # Standard for long-horizon tasks
    tau: float = 0.005             # Polyak averaging (SAC paper)
    warmup_steps: int = 10000      # Random steps before training
    update_every: int = 1          # Update after every step
    mixed_precision: bool = True
    compile_model: bool = False    
    max_grad_norm: float = 1.0     # Gradient clipping

    # PPO/MAPPO-specific fields (ignored by SAC trainers)
    algo: str = 'sac'              # Algorithm selector: 'sac' | 'ppo'
    gae_lambda: float = 0.95       # GAE λ for advantage estimation
    update_epochs: int = 4         # PPO epochs per rollout
    clip_eps: float = 0.2          # PPO actor clip range ε
    vf_coef: float = 0.5           # Critic loss coefficient (standard MAPPO)
    ent_coef: float = 0.01          # Entropy bonus coefficient
    vf_clip_eps: float = 10.0      # Critic clip range (loose)
    rollout_steps: int = 288       # Steps per PPO rollout (24h @ 5min/step)
    use_trip_head: bool = False    # MAPPO: disable policy trip-ID head by default
    use_execution_consistency_mask: bool = False  # MAPPO: optional strict executed/intended gating
    mappo_repos_khop: int = 4  
    mappo_max_k_neighbors: int = 0  # MAPPO: optional fixed max_k override (0=derive from adjacency)
    mappo_use_khop_candidates: bool = False  
    learn_charge_power: bool = False  # MAPPO: optional per-vehicle charging power control
    mappo_reward_mix_alpha: float = 1.0  # MAPPO: per-vehicle reward weight in mixed credit target
    mappo_use_execution_soft_weight: bool = False  # MAPPO: softly down-weight failed execution gradients
    mappo_failed_action_weight: float = 1.0  # MAPPO: actor-loss weight for generic failed actions
    mappo_failed_serve_weight: float = 1.0  # MAPPO: actor-loss weight for failed SERVE actions

    # MADDPG-specific fields (ignored by SAC and PPO trainers)
    maddpg_lr_actor:          float = 1e-4     # Actor optimizer LR
    maddpg_lr_critic:         float = 1e-3     # Critic optimizer LR
    maddpg_buffer_capacity:   int   = 5000     # Replay buffer cap (N×16 per step = memory-bound)
    maddpg_tau:               float = 0.005    # Polyak soft-update coefficient
    maddpg_gumbel_tau:        float = 0.5      # Gumbel-Softmax temperature (lower = sharper gradients)
    maddpg_action_embed_dim:  int   = 64       # Joint action projection dim in critic
    maddpg_eps_start:         float = 0.3      # ε-greedy exploration start
    maddpg_eps_end:           float = 0.05     # ε-greedy exploration floor
    maddpg_eps_decay_episodes: int  = 200      # Episodes to anneal ε
    maddpg_reward_mix_alpha: float = 0.4       # MADDPG: per-vehicle reward weight in mixed credit target
    maddpg_use_execution_soft_weight: bool = False  # MADDPG: reserved gate for future execution weighting
    maddpg_repos_khop: int = 4  
    maddpg_max_k_neighbors: int = 0  


@dataclass
class ReplayBufferConfig:
    capacity: int = 500000
    prioritized: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_annealing_steps: int = 100000


@dataclass
class PickupDistanceConfig:
    start: float = 5.0
    end: float = 5.0
    schedule: str = "cosine"


@dataclass
class CurriculumConfig:
    enabled: bool = False
    pickup_distance: PickupDistanceConfig = field(default_factory=PickupDistanceConfig)
    curriculum_end_fraction: float = 0.8


@dataclass
class RewardConfig:
    driving_cost_per_km: float = 0.30
    electricity_cost_per_kwh: float = 0.18
    wait_penalty_per_step: float = 0.02    
    drop_penalty_per_order: float = 0.5    
    scale_factor: float = 10.0
    max_wait_steps: int = 6                # 6 steps = 30 min max wait
    serve_bonus: float = 1.0               # Bonus per trip served to encourage serving
    # Action penalties (configurable via YAML)
    reposition_penalty: float = 0.0        # Penalty per REPOSITION action
    idle_penalty: float = 0.0              # Penalty per IDLE action when SOC > 30%
    serve_fail_penalty: float = 3.0        # Penalty per failed SERVE action
    high_soc_charge_penalty: float = 0.0   # Penalty for CHARGE when SOC > 60%
    very_high_soc_charge_penalty: float = 0.0  # Extra penalty for CHARGE when SOC > 80%
    # Demand-aware reposition rewards
    enable_demand_reposition_bonus: bool = True   # Enable reposition success bonus
    reposition_success_bonus: float = 0.5         # Bonus per unit demand at target hex
    reposition_nearby_decay: float = 0.5          # Weight for nearby-hex demand (0=exact only, 1=full)
    reposition_action_bonus: float = 0.0          # Flat bonus per vehicle choosing REPOSITION (action-level)
    reposition_dispatch_bonus: float = 0.0        # Demand-proportional bonus at dispatch time (immediate signal)
    reposition_dispatch_demand_cap: float = 25.0  # Reference demand for proportionality scaling (not a hard saturation)
    reposition_dispatch_per_vehicle_cap: float = 2.0  # Hard ceiling on bonus per repositioning vehicle


@dataclass
class PhysicsConfig:
    energy_per_km: float = 0.2             # kWh/km (typical EV)
    avg_speed_kmh: float = 25.0            # NYC average (realistic)
    max_soc: float = 100.0
    min_soc_reserve: float = 10.0
    charge_power_kw: float = 50.0          # Realistic fast charging (50 kW)


@dataclass
class VehicleConfig:
    initial_soc: float = 80.0
    soc_low_threshold: float = 20.0
    soc_med_threshold: float = 50.0
    soc_charge_interrupt_threshold: float = 90.0  # Changed: interrupt at 90%


@dataclass
class StationConfig:
    max_power: float = 50.0
    electricity_price: float = 0.15
    price_per_kwh: float = 0.15
    num_ports: int = 1
    placement_mode: str = "demand_spaced"
    radius_hops: int = 4
    fallback_mode: str = "fill_by_demand"


@dataclass
class DistributedConfig:
    enabled: bool = False
    backend: str = "nccl"
    gpus: str = "auto"
    
    def get_gpu_list(self) -> List[int]:
        if self.gpus == "auto" or self.gpus == "all":
            import torch
            return list(range(torch.cuda.device_count()))
        elif self.gpus:
            return [int(g.strip()) for g in self.gpus.split(",")]
        return [0]


@dataclass
class EntropyConfig:
    """SAC entropy settings for exploration/exploitation balance."""
    auto_alpha: bool = True           # Auto-tune entropy coefficient
    initial_alpha: float = 0.2        # Starting exploration level
    target_entropy_ratio: float = 0.98  # Multiply by -log(action_dim)
    min_alpha: float = 0.01           # Minimum alpha for stability
    max_alpha: float = 1.0            # Maximum alpha ceiling


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 20
    save_best: bool = True
    keep_last: int = 5
    resume_from: Optional[str] = None


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    log_interval: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "ev-fleet-rl"


@dataclass
class DebugConfig:
    profile_gpu: bool = False
    log_memory: bool = False
    benchmark_cudnn: bool = True
    deterministic: bool = False
    seed: int = 42


@dataclass
class FleetActorConfig:
    """Fleet-level actor settings for hex-based action allocation."""
    repos_khop: int = 4                    # K-hop neighborhood for reposition targets (~61 hexes)
    hex_vehicle_agg_dim: int = 8           # Aggregated vehicle summary dimension per hex
    hex_decision_hidden_dim: int = 256     # Hex decision encoder hidden dim
    entropy_target_ratio: float = 0.5      # Target entropy as ratio of max (log(action_dim))
    assignment_soc_priority: bool = True   # Low-SOC vehicles assigned to CHARGE first


@dataclass
class NetworkConfig:
    actor_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    hex_embedding_dim: int = 64
    gcn_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    gcn_output_dim: int = 64
    context_dim: int = 128
    hidden_dim: int = 256
    option_dim: int = 64
    gcn_hidden_dim: int = 128
    gcn_layers: int = 2
    dropout: float = 0.1


@dataclass
class Config:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    station: StationConfig = field(default_factory=StationConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    fleet_actor: FleetActorConfig = field(default_factory=FleetActorConfig)

    def get_device(self) -> str:
        import torch
        if self.distributed.enabled:
            gpus = self.distributed.get_gpu_list()
            return f"cuda:{gpus[0]}" if gpus else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
