"""Configuration dataclasses for GPU-native EV Fleet RL System."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union
from pathlib import Path


@dataclass
class EnvironmentConfig:
    num_vehicles: int = 1000
    num_stations: int = 35
    num_hexes: int = 1300          # overwritten from the spatial manifest when real data is used
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


def _default_splits() -> Dict[str, List[str]]:
    # Half-open [start, end) day ranges. Train skips the 1 Jan holiday (Q22).
    return {
        "train": ["2009-01-02", "2009-01-22"],
        "val": ["2009-01-22", "2009-01-25"],
        "test": ["2009-01-25", "2009-02-01"],
        # OOD test (round 11): New Year's Eve 2009 (Thursday). Volume of a January Thursday (0.98x,
        # ~54k orders at 12 %) with the pickup distribution shifted in all 17 active hours; in the
        # evening Times Square is closed and demand moves to the party districts.
        "nye": ["2009-12-31", "2010-01-01"],
    }


@dataclass
class DataConfig:
    # Relative paths are resolved against the repository root.
    processed_root: str = "../data/processed"
    datasets: List[str] = field(default_factory=lambda: ["yellow_tripdata_2009-01", "yellow_tripdata_2009-12"])
    manifest_path: str = "../data/manifests/nyc_r7_v3.json"
    h3_resolution: int = 7
    sample_ratio: float = 0.12     # per-trip deterministic sample, applied after the domain is fixed (round 9: 12 %;
                                   # paper and rounds 5-8: 20 %, where 1000 vehicles served only ~46 % of the orders)
    sample_seed: int = 2009
    splits: Dict[str, List[str]] = field(default_factory=_default_splits)


@dataclass
class LearningRateConfig:
    actor: float = 3.0e-4
    critic: float = 3.0e-4
    alpha: float = 3.0e-4


@dataclass
class TrainingConfig:
    """Shared by every learner of the vehicle-level stack (SAC / PD-RSAC, MAPPO, MADDPG)."""
    algo: str = 'sac'              # learner: 'sac' (SAC / PD-RSAC family), 'mappo', 'maddpg'
    total_episodes: int = 5000
    batch_size: int = 256          # per-vehicle transitions per minibatch
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)
    gamma: float = 0.999           # per step (semi-MDP) or per decision; one value for every family
    tau: float = 0.005             # Polyak averaging of target networks (Q11)
    warmup_steps: int = 10000      # decision epochs of uniform intentions before learning (off-policy)
    mixed_precision: bool = True
    max_grad_norm: float = 1.0     # gradient clipping
    # MAPPO (on-policy)
    gae_lambda: float = 0.95       # GAE lambda along each vehicle's decisions
    update_epochs: int = 4         # PPO passes over each episode
    clip_eps: float = 0.2          # PPO ratio clip
    vf_coef: float = 0.5           # value loss weight
    ent_coef: float = 0.01         # entropy bonus (rewards are / 100, like the SAC alpha)
    vf_clip_eps: float = 10.0      # value clip range (loose)


@dataclass
class ReplayBufferConfig:
    capacity: int = 500000
    prioritized: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_annealing_steps: int = 100000


@dataclass
class RewardConfig:
    """Paper Eq. (8) only (Q8: no shaping). Penalties use the code's values (Q10)."""
    driving_cost_per_km: float = 0.30      # every driven km: serve, reposition, travel to a station
    electricity_cost_per_kwh: float = 0.18 # per kWh drawn from the grid
    wait_penalty_per_step: float = 0.03    # per waiting order per decision epoch
    drop_penalty_per_order: float = 0.5    # per order dropped after max_wait_steps epochs
    scale_factor: float = 100.0            # constant reward scale
    max_wait_steps: int = 6                # 6 epochs = 30 min


@dataclass
class PhysicsConfig:
    """Vehicle physics shared by simulator, MILP, fallback, masks and the projection surrogate (C4)."""
    battery_kwh: float = 50.0              # E_max
    reserve_kwh: float = 7.5               # E_min (15 %); the drive to a station only needs E >= 0 (Q4)
    energy_per_km: float = 0.2             # eta_drv, kWh/km
    charge_efficiency: float = 0.95        # eta_c
    avg_speed_kmh: float = 25.0            # fixed speed for travel times (Q14b)


@dataclass
class VehicleConfig:
    initial_energy_range: List[float] = field(default_factory=lambda: [0.5, 0.9])  # x battery, uniform (C6)
    low_energy_fraction: float = 0.2       # "low battery" for the greedy baseline and SoC features


@dataclass
class StationConfig:
    num_ports: int = 5
    max_power: float = 50.0                # P_max^s per port, kW
    min_power: float = 20.0                # p_min when charging under the projection, kW
    feeder_kw: float = 5500.0              # P_feed: enforced by the MILP projection, measured on the raw path (Q9);
                                           # 5500 kW binds ~4-11 % of the time at the 12 % sample (5000 kW: 7-20 %, 7000 kW: < 1 %)
    # Placement is done once by the manifest builder (cover-then-fill, Q13b); these
    # values are recorded here so the run config states what the manifest must contain.
    cover_rings: int = 2
    cover_min_daily_pickups: float = 1.0


@dataclass
class MILPConfig:
    """Feasible-action projection (Eq. 12-16, Algorithm 1; workstream D). Shared by every variant
    that projects, so it lives in base.yaml. ``enabled`` is the variant switch; it is set by the
    preset (F4) or by ``--milp`` and recorded in the resolved config, never in a YAML."""
    enabled: bool = False           # project intentions with the MILP (else the raw executor)
    mu: float = 4.0                # intention-proximity weight; needs max repos km < 2 mu / c_drv (D7)
    tau_max: float = 0.5            # solver time limit per decision epoch, seconds (Q11)
    warm_start: bool = True         # start from Algorithm 1 (diagnostic switch, D11)
    threads: int = 0                # Gurobi threads (0 = automatic)
    strict: bool = True             # every projected action must execute (infeasible_executed == 0)


@dataclass
class AgentConfig:
    """Vehicle-level actor-critic (workstream E). Learning rates, gamma, Polyak tau, batch size,
    warm-up and gradient clipping live in ``training``."""
    hidden_dim: int = 256
    gcn_hidden: int = 128
    emb_dim: int = 64
    key_dim: int = 64
    lr_value: float = 1.0e-4
    updates_per_step: int = 1          # updates after every environment step (E11)
    replay_window: int = 20            # episodes kept in the replay (Q17)
    replay_recency: float = 0.8        # sampling weight per episode of age
    tau_start: float = 1.0             # Gumbel-softmax temperature, annealed linearly ...
    tau_end: float = 0.3
    tau_anneal_fraction: float = 0.6   # ... over this fraction of the episodes (Q16)
    init_alpha: float = 0.01           # initial value of the three entropy coefficients (rewards are / 100)
    entropy_ratio_mode: float = 0.5    # target H(mode) = ratio * log(#valid modes)
    entropy_ratio_target: float = 0.5  # target H(slot / station) = ratio * log(#valid candidates)
    entropy_target_power: float = -1.0
    semi_mdp: bool = True              # gamma^Delta bootstrap; False = fixed gamma (Q12)
    critic_population: bool = True     # False = independent critic without m_i (IAC ablation)
    explore_std: float = 0.3           # MADDPG: Gaussian exploration noise on the pre-squash power


@dataclass
class WDROConfig:
    enabled: bool = True
    rho: float = 0.1                 # budget in the robust target (21); = rho_target (round 7); 0.1 with the metric step (round 8)
    rho_target: float = 0.1
    beta: float = 0.3
    eta0: float = 0.5                # dual step eta0 / sqrt(t), t = dual updates (round 7)
    inner_steps: int = 5
    step_size: float = 0.5             # 'sigma' rule only: step in standard deviations of each coordinate
    step_rule: str = "metric"          # 'metric' (round 8): p = S^-1 grad f, length step_size_d in d_Q; 'sigma': rounds <= 7
    step_size_d: float = 0.05          # 'metric' rule: length of each inner step in d_Q units (= rho / 2)
    schur: bool = True                 # 2-hop restriction by the Schur complement (round 8); False = plain block Q_ii
    support_sigmas: float = 3.0        # box support B around xi-hat
    warmup_episodes: int = 20          # non-robust episodes before the robust target (Q7)
    metric: str = "graph"              # graph | diag | identity
    lambda_init: float = 0.0          # dual variable starts at 0 and rises only while the realised radius exceeds rho_target


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
class Config:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    station: StationConfig = field(default_factory=StationConfig)
    milp: MILPConfig = field(default_factory=MILPConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    wdro: WDROConfig = field(default_factory=WDROConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def get_device(self) -> str:
        import torch
        if self.distributed.enabled:
            gpus = self.distributed.get_gpu_list()
            return f"cuda:{gpus[0]}" if gpus else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
