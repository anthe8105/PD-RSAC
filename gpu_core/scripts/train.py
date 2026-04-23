"""
GPU-accelerated EV Fleet RL Training Script.

Usage:
    python train.py --config config.yaml
    python train.py --num-vehicles 1000 --num-hexes 1300 --gpus 0,1
    python train.py --real-data ./data/nyc_real/trips_processed.parquet
    python train.py --help
"""

import argparse
import copy
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque
import time
from collections import defaultdict
import csv

# Timing profiler
_timing_stats = defaultdict(list)

def _time_section(name):
    """Context manager for timing code sections."""
    class TimingContext:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start = time.time()
            return self
        def __exit__(self, *args):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - self.start
            _timing_stats[name].append(elapsed)
    return TimingContext(name)



import torch
import torch.distributed as dist

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import ConfigLoader, Config, EnvironmentConfig, TrainingConfig, DistributedConfig
from gpu_core.networks.sac import SACAgent, FleetSACAgent
from gpu_core.spatial.neighbors import HexNeighbors
from gpu_core.features.replay_buffer import GPUReplayBuffer
from gpu_core.features.builder import FeatureBuilder
from gpu_core.training.trainer import SACTrainer, FleetSACTrainer
from gpu_core.training.enhanced_trainer import (
    EnhancedSACTrainer,
    EnhancedTrainingConfig,
    DEFAULT_WDRO_VALUE_SOURCE_SWITCH_EPISODE,
    DEFAULT_WDRO_VALUE_TRAIN_STOP_EPISODE,
)
from gpu_core.training.enhanced_collector import EnhancedEpisodeCollector
from gpu_core.training.distributed import (
    DistributedTrainer,
    MixedPrecisionTrainer,
    setup_distributed,
    cleanup_distributed,
    spawn_distributed_training
)
from gpu_core.training.episode_collector import EpisodeCollector, BatchedEpisodeCollector
from gpu_core.simulator.environment import GPUEnvironment, GPUEnvironmentV2, BatchedGPUEnvironment
from gpu_core.spatial.grid import HexGrid
from gpu_core.spatial.distance import DistanceMatrix
from gpu_core.data.real_trip_loader import RealTripLoader

# MAPPO baseline
from gpu_core.networks.ppo_agent import PPOAgent, PPOOutput
from gpu_core.features.ppo_buffer import PPORolloutBuffer
from gpu_core.training.ppo_collector import PPOCollector
from gpu_core.training.ppo_trainer_enhanced import EnhancedPPOTrainer as MAPPOTrainer

# MADDPG baseline
from gpu_core.networks.maddpg_agent  import MADDPGAgent
from gpu_core.features.maddpg_buffer import MADDPGReplayBuffer
from gpu_core.training.maddpg_trainer    import MADDPGTrainer
from gpu_core.training.maddpg_collector  import MADDPGCollector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train EV Fleet RL Agent with GPU acceleration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    
    parser.add_argument('--num-vehicles', type=int, default=None,
                        help='Number of vehicles in fleet')
    parser.add_argument('--num-hexes', type=int, default=None,
                        help='Number of hexagons in grid')
    parser.add_argument('--compact-state', action='store_true',
                        help='Use compact state (aggregate vehicle features per hex) for faster training')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Total training episodes')
    parser.add_argument('--episode-duration-hours', type=float, default=None,
                        help='Duration of each episode in hours (e.g., 24.0 for full day, 10.0 for 10 hours)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size')
    
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs (e.g., "0,1,2")')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training (DDP - advanced)')
    parser.add_argument('--data-parallel', action='store_true',
                        help='Enable DataParallel for multi-GPU (simpler than DDP)')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training (AMP)')
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--start-episode', type=int, default=None,
                        help='Override start episode (use when checkpoint has wrong episode counter)')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Save checkpoint every N episodes')
    
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log metrics every N episodes')
    parser.add_argument('--eval-interval', type=int, default=50,
                        help='Evaluate agent every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Number of episodes for evaluation')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='Number of random steps to warmup replay buffer')
    parser.add_argument('--update-ratio', type=int, default=1,
                        help='Number of steps per training update (higher = faster but less learning)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    # Logging options
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging')
    parser.add_argument('--tensorboard-dir', type=str, default='runs',
                        help='TensorBoard log directory')
    parser.add_argument('--log-loss-interval', type=int, default=100,
                        help='Log loss every N training updates')
    
    # Real data options
    parser.add_argument('--real-data', type=str, default=None,
                        help='Path to real trip data parquet file (e.g., ./data/nyc_real/trips_processed.parquet)')
    parser.add_argument('--trip-sample', type=float, default=None,
                        help='Sample ratio for trip data (0.0-1.0). Use 0.1 for 10%%, 0.5 for 50%%, 1.0 for 100%%')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Filter trips from this date (format: YYYY-MM-DD, e.g., 2009-01-15)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Filter trips until this date inclusive (format: YYYY-MM-DD, e.g., 2009-01-20)')
    parser.add_argument('--eval-start-date', type=str, default=None,
                        help='Held-out evaluation trips start date during training (format: YYYY-MM-DD)')
    parser.add_argument('--eval-end-date', type=str, default=None,
                        help='Held-out evaluation trips end date during training inclusive (format: YYYY-MM-DD)')
    parser.add_argument('--target-h3-resolution', type=int, default=None,
                        help='Coarsen hex grid to target H3 resolution (e.g., 7, 8). Lower = larger hexes.')
    parser.add_argument('--max-hex-count', type=int, default=None,
                        help='Limit number of hexes by keeping most frequent (e.g., 5000)')
    
    # Paper-compliant training options
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use greedy (argmax) action selection instead of stochastic sampling')
    parser.add_argument('--milp', action='store_true', default=False,
                        help='Enable exact MILP assignment via Gurobi (IEEE_T_IV.pdf formulation)')
    parser.add_argument('--semi-mdp', action='store_true', default=True,  # CHANGED: Enable by default per paper
                        help='Enable Semi-MDP duration discounting (paper Eq. 12-13)')
    parser.add_argument('--no-semi-mdp', dest='semi_mdp', action='store_false',
                        help='Disable Semi-MDP discounting')
    parser.add_argument('--wdro', action='store_true', default=True,  # CHANGED: Enable by default per paper
                        help='Enable WDRO robust targets (paper Section 4)')
    parser.add_argument('--no-wdro', dest='wdro', action='store_false',
                        help='Disable WDRO robust targets')

    # ── MAPPO baseline ─────────────────────────────────────────────────
    parser.add_argument('--mappo', action='store_true', default=False,
                        help='Use MAPPO baseline (per-vehicle MLP PPO, bypasses SAC/WDRO/semi-MDP)')
    parser.add_argument('--ent-coef', type=float, default=None,
                        help='[PPO] Entropy coefficient (default: 0.05)')
    parser.add_argument('--vf-clip-eps', type=float, default=None,
                        help='[PPO] Critic clip range (default: 10.0)')
    parser.add_argument('--rollout-steps', type=int, default=None,
                        help='[PPO] Steps per rollout (default: 288 = 24h @ 5min/step)')
    parser.add_argument('--max-trips', type=int, default=None,
                        help='[PPO] Max trips tracked per step (default: env config)')

    # ── MADDPG baseline ────────────────────────────────────────────────
    parser.add_argument('--maddpg', action='store_true', default=False,
                        help='Use MADDPG baseline (centralized critic, decentralized actor, off-policy)')
    parser.add_argument('--maddpg-lr-actor', type=float, default=None,
                        help='[MADDPG] Actor learning rate (default: 1e-4)')
    parser.add_argument('--maddpg-lr-critic', type=float, default=None,
                        help='[MADDPG] Critic learning rate (default: 1e-3)')
    parser.add_argument('--maddpg-buffer-capacity', type=int, default=None,
                        help='[MADDPG] Replay buffer capacity (default: 5000)')
    parser.add_argument('--maddpg-eps-start', type=float, default=None,
                        help='[MADDPG] ε-greedy start value (default: 0.3)')
    parser.add_argument('--maddpg-eps-end', type=float, default=None,
                        help='[MADDPG] ε-greedy end value (default: 0.05)')

    parser.add_argument('--wdro-rho', type=float, default=0.3,
                        help='WDRO ambiguity radius (default: 0.3)')
    parser.add_argument('--wdro-metric', type=str, default='mag', choices=['mag', 'euclidean'],
                        help='WDRO ground metric: graph-aligned Mahalanobis (mag) or Euclidean')
    parser.add_argument('--wdro-value-source-switch-episode', type=int, default=DEFAULT_WDRO_VALUE_SOURCE_SWITCH_EPISODE,
                        help='First episode where WDRO uses V_phi instead of exact Eq. 27 value (default: 80)')
    parser.add_argument('--wdro-value-train-stop-episode', type=int, default=DEFAULT_WDRO_VALUE_TRAIN_STOP_EPISODE,
                        help='First episode where V_phi stops training; can equal or exceed the switch episode (default: 80)')
    parser.add_argument('--temperature-annealing', action='store_true', default=True,
                        help='Enable temperature annealing (paper Eq. 16)')
    parser.add_argument('--initial-temperature', type=float, default=1.0,
                        help='Initial softmax temperature')
    parser.add_argument('--final-temperature', type=float, default=0.1,
                        help='Final softmax temperature')
    parser.add_argument('--temperature-decay-episodes', type=int, default=200,
                        help='Episodes over which temperature anneals from initial to final (default: 200; was hardcoded 500)')
    parser.add_argument('--env-v2', action='store_true', default=False,
                        help='Use GPUEnvironmentV2 (refactored modular environment)')
    
    return parser.parse_args()


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(args) -> torch.device:
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
        return torch.device(f'cuda:{gpu_ids[0]}')
    elif torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def apply_enhanced_schedule_args(args, enhanced_config):
    """Apply resolved WDRO schedule args to the enhanced trainer config."""
    enhanced_config.wdro_value_source_switch_episode = args.wdro_value_source_switch_episode
    enhanced_config.wdro_value_train_stop_episode = args.wdro_value_train_stop_episode
    return enhanced_config


def create_config(args) -> Config:
    if args.config:
        config = ConfigLoader.from_yaml(args.config)
    else:
        config = Config()
    
    if args.num_vehicles is not None:
        config.environment.num_vehicles = args.num_vehicles
    if args.num_hexes is not None:
        config.environment.num_hexes = args.num_hexes
    if args.compact_state:
        config.environment.compact_state = True
    if args.episodes is not None:
        config.training.total_episodes = args.episodes
    if args.episode_duration_hours is not None:
        config.episode.duration_hours = args.episode_duration_hours
        print(f"[Episode] Duration set to {args.episode_duration_hours} hours ({config.episode.steps_per_episode} steps @ {config.episode.step_duration_minutes} min/step)")
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    if args.gpus:
        config.distributed.gpus = [int(x) for x in args.gpus.split(',')]
    if args.distributed:
        config.distributed.enabled = True
        config.distributed.world_size = len(config.distributed.gpus) if config.distributed.gpus else torch.cuda.device_count()
    if args.mixed_precision:
        config.training.mixed_precision = True
    
    config.checkpoint.checkpoint_dir = args.checkpoint_dir
    config.checkpoint.save_interval = args.save_interval
    config.logging.log_interval = args.log_interval
    
    if args.warmup_steps is not None:
        config.training.warmup_steps = args.warmup_steps

    # PPO / MAPPO flag propagation
    if getattr(args, 'mappo', False):
        config.training.algo = 'ppo'
        # Standard MAPPO max_grad_norm (0.5)
        config.training.max_grad_norm = 0.5
    if getattr(args, 'ent_coef', None) is not None:
        config.training.ent_coef = args.ent_coef
    if getattr(args, 'vf_clip_eps', None) is not None:
        config.training.vf_clip_eps = args.vf_clip_eps
    if getattr(args, 'rollout_steps', None) is not None:
        config.training.rollout_steps = args.rollout_steps
    if getattr(args, 'max_trips', None) is not None:
        config.environment.max_trips_per_step = args.max_trips

    # MADDPG flag propagation
    if getattr(args, 'maddpg', False):
        config.training.algo = 'maddpg'
    if getattr(args, 'maddpg_lr_actor', None) is not None:
        config.training.maddpg_lr_actor = args.maddpg_lr_actor
    if getattr(args, 'maddpg_lr_critic', None) is not None:
        config.training.maddpg_lr_critic = args.maddpg_lr_critic
    if getattr(args, 'maddpg_buffer_capacity', None) is not None:
        config.training.maddpg_buffer_capacity = args.maddpg_buffer_capacity
    if getattr(args, 'maddpg_eps_start', None) is not None:
        config.training.maddpg_eps_start = args.maddpg_eps_start
    if getattr(args, 'maddpg_eps_end', None) is not None:
        config.training.maddpg_eps_end = args.maddpg_eps_end

    return config


def create_agent(config: Config, device: torch.device, env=None):
    algo = getattr(config.training, 'algo', 'sac').lower()
    env_config = config.environment

    if algo == 'ppo':
        if not hasattr(config.training, 'use_trip_head') or config.training.use_trip_head is None:
            config.training.use_trip_head = False
    entropy_config = config.entropy
    vehicle_config = config.vehicle
    fleet_actor_config = config.fleet_actor

    # Get dimensions from environment if available, otherwise use defaults
    if env is not None:
        vehicle_feature_dim_full = env._vehicle_feature_dim   # 16 (full flat-state)
        vehicle_feature_dim_fleet = getattr(env, '_fleet_vehicle_feature_dim', env._vehicle_feature_dim)  # 13 (slim fleet)
        hex_feature_dim = env._hex_feature_dim
        context_dim = env._context_dim
    else:
        vehicle_feature_dim_full = 16
        vehicle_feature_dim_fleet = 13
        hex_feature_dim = 5
        context_dim = 9

    # ── MAPPO branch ──────────────────────────────────────────────────
    if algo == 'ppo':
        vehicle_feature_dim = vehicle_feature_dim_full
        state_dim = (
            env_config.num_vehicles * vehicle_feature_dim
            + env_config.num_hexes * hex_feature_dim
            + context_dim
        )
        print(f"[State Dim] Vehicles: {env_config.num_vehicles}×{vehicle_feature_dim}, "
              f"Hexes: {env_config.num_hexes}×{hex_feature_dim}, Context: {context_dim}, Total: {state_dim}")

        mappo_dropout = (
            config.network.dropout
            if hasattr(config, 'network') and hasattr(config.network, 'dropout')
            else 0.0
        )
        if mappo_dropout is None:
            mappo_dropout = 0.0

        mappo_khop = max(1, int(getattr(config.training, 'mappo_repos_khop', 4)))
        mappo_max_k = int(getattr(config.training, 'mappo_max_k_neighbors', 0) or 0)
        if mappo_max_k <= 0:
            mappo_max_k = 61
            if env is not None and hasattr(env, '_adjacency_matrix') and env._adjacency_matrix is not None:
                try:
                    khop_mask_hh = HexNeighbors.compute_khop_mask(env._adjacency_matrix, k=mappo_khop)
                    _, _, derived_max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh)
                    mappo_max_k = int(derived_max_k)
                except Exception as e:
                    print(f"[MAPPO] Could not derive max_k_neighbors, defaulting to {mappo_max_k}: {e}")

        agent = PPOAgent(
            state_dim           = state_dim,
            action_dim          = 3,
            num_hexes           = env_config.num_hexes,
            actor_hidden_dims   = [128, 128],
            critic_hidden_dims  = [256, 256],
            gamma               = config.training.gamma,
            gae_lambda          = config.training.gae_lambda,
            clip_eps            = config.training.clip_eps,
            vf_coef             = config.training.vf_coef,
            ent_coef            = config.training.ent_coef,
            lr_actor            = config.training.learning_rate.actor,
            lr_critic           = config.training.learning_rate.critic,
            dropout             = mappo_dropout,
            device              = str(device),
            num_vehicles        = env_config.num_vehicles,
            vehicle_feature_dim = vehicle_feature_dim,
            hex_feature_dim     = hex_feature_dim,
            context_dim         = context_dim,
            max_trips           = env_config.max_trips_per_step,
            use_trip_head       = config.training.use_trip_head,
            learn_charge_power  = getattr(config.training, 'learn_charge_power', False),
            max_k_neighbors     = mappo_max_k,
        )
        print(f"[MAPPO] PPOAgent created — parameters: {sum(p.numel() for p in agent.parameters()):,}")
        return agent

    # ── MADDPG branch ────────────────────────────────────────────────
    if algo == 'maddpg':
        vehicle_feature_dim = vehicle_feature_dim_full
        state_dim = (
            env_config.num_vehicles * vehicle_feature_dim
            + env_config.num_hexes * hex_feature_dim
            + context_dim
        )
        print(f"[MADDPG] Vehicle features: {vehicle_feature_dim}, "
              f"Context: {context_dim}, Vehicles: {env_config.num_vehicles}")

        maddpg_khop = max(1, int(getattr(config.training, 'maddpg_repos_khop', 4)))
        maddpg_max_k = int(getattr(config.training, 'maddpg_max_k_neighbors', 0) or 0)
        if maddpg_max_k <= 0:
            maddpg_max_k = 61
            if env is not None and hasattr(env, '_adjacency_matrix') and env._adjacency_matrix is not None:
                try:
                    khop_mask_hh = HexNeighbors.compute_khop_mask(env._adjacency_matrix, k=maddpg_khop)
                    _, _, derived_max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh)
                    maddpg_max_k = int(derived_max_k)
                except Exception as e:
                    print(f"[MADDPG] Could not derive max_k_neighbors, defaulting to {maddpg_max_k}: {e}")

        agent = MADDPGAgent(
            vehicle_feature_dim  = vehicle_feature_dim,
            context_dim          = context_dim,
            num_vehicles         = env_config.num_vehicles,
            num_hexes            = env_config.num_hexes,
            actor_hidden_dims    = [256, 256],
            critic_hidden_dims   = [256, 256],
            action_dim           = 3,
            max_trips            = env_config.max_trips_per_step,
            action_embed_dim     = config.training.maddpg_action_embed_dim,
            dropout              = config.network.dropout if hasattr(config, 'network') else 0.1,
            gamma                = config.training.gamma,
            tau                  = config.training.maddpg_tau,
            lr_actor             = config.training.maddpg_lr_actor,
            lr_critic            = config.training.maddpg_lr_critic,
            gumbel_tau           = config.training.maddpg_gumbel_tau,
            device               = str(device),
            state_dim            = state_dim,
            hex_feature_dim      = hex_feature_dim,
            max_k_neighbors      = maddpg_max_k,
        )
        print(f"[MADDPG] MADDPGAgent created — parameters: "
              f"{sum(p.numel() for p in agent.parameters()):,}")
        return agent.to(device)

    # ── SAC branch (existing, unchanged) ──────────────────────────────
    vehicle_feature_dim = vehicle_feature_dim_fleet
    
    # State dimension: vehicle features + hex features + context
    state_dim = (
        env_config.num_vehicles * vehicle_feature_dim +
        env_config.num_hexes * hex_feature_dim +
        context_dim
    )
    
    print(f"[State Dim] Vehicle: {env_config.num_vehicles}x{vehicle_feature_dim}, "
          f"Hex: {env_config.num_hexes}x{hex_feature_dim}, Context: {context_dim}, Total: {state_dim}")
    
    # Calculate mixed-action target entropy from config.
    # We use one scalar alpha across both the discrete allocation policy and
    # the continuous charge-power head, so the target entropy must cover both.
    action_dim = 3  # SERVE=0, CHARGE=1, REPOSITION=2
    target_entropy = -entropy_config.target_entropy_ratio * (
        torch.log(torch.tensor(float(action_dim))).item() + 1.0
    )
    
    # GCN-based actor (paper Section 5.1.2)
    # Use config network dimensions if available, otherwise paper defaults
    actor_hidden_dims = config.network.actor_hidden_dims if hasattr(config, 'network') and hasattr(config.network, 'actor_hidden_dims') else [128, 128]
    critic_hidden_dims = config.network.critic_hidden_dims if hasattr(config, 'network') and hasattr(config.network, 'critic_hidden_dims') else [256, 256]
    dropout_rate = config.network.dropout if hasattr(config, 'network') and hasattr(config.network, 'dropout') else 0.1
    
    # Determine max_k_neighbors based on adjacency matrix if available
    max_k_neighbors = 61
    if env is not None and hasattr(env, '_adjacency_matrix') and env._adjacency_matrix is not None:
        try:
            k_hops = getattr(config.fleet_actor, 'repos_khop', 4)
            khop_mask_hh = HexNeighbors.compute_khop_mask(env._adjacency_matrix, k=k_hops)
            _, _, max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh)
            max_k_neighbors = max_k
            print(f"[Agent Setup] Detected dynamic max_k_neighbors: {max_k_neighbors} for K={k_hops}")
        except Exception as e:
            print(f"[Agent Setup] Could not pre-compute max_k_neighbors, defaulting to 61: {e}")
            
    agent = FleetSACAgent(
        num_hexes=env_config.num_hexes,
        num_vehicles=env_config.num_vehicles,
        vehicle_feature_dim=vehicle_feature_dim,
        hex_feature_dim=hex_feature_dim,
        hex_vehicle_agg_dim=getattr(fleet_actor_config, 'hex_vehicle_agg_dim', 8),
        context_dim=context_dim,
        action_dim=action_dim,
        max_K_neighbors=max_k_neighbors,
        gcn_hidden_dim=actor_hidden_dims[0] if actor_hidden_dims else 128,
        gcn_output_dim=64,
        hex_decision_hidden_dim=getattr(fleet_actor_config, 'hex_decision_hidden_dim', 256),
        critic_hidden_dim=critic_hidden_dims[0] if critic_hidden_dims else 256,
        gamma=config.training.gamma,
        tau=config.training.tau,
        alpha=entropy_config.initial_alpha,
        auto_alpha=entropy_config.auto_alpha,
        target_entropy=target_entropy,
        lr_actor=config.training.learning_rate.actor,
        lr_critic=config.training.learning_rate.critic,
        lr_alpha=config.training.learning_rate.alpha,
        dropout=dropout_rate,
        device=str(device),
        min_alpha=getattr(entropy_config, 'min_alpha', 0.05),
        max_alpha=getattr(entropy_config, 'max_alpha', 1.0),
        repos_aux_weight=getattr(config.training, 'repos_aux_weight', 0.1),
        soc_low_threshold=getattr(vehicle_config, 'soc_low_threshold', 20.0),
        assignment_soc_priority=getattr(fleet_actor_config, 'assignment_soc_priority', True),
        use_semi_mdp=config.training.use_semi_mdp if hasattr(config.training, 'use_semi_mdp') else True,
    )
    
    return agent.to(device)


def create_replay_buffer(config: Config, device: torch.device, env=None) -> GPUReplayBuffer:
    env_config = config.environment
    buffer_config = config.replay_buffer
    
    # Get dimensions from environment if available, otherwise use defaults
    if env is not None:
        vehicle_feature_dim = getattr(env, '_fleet_vehicle_feature_dim', env._vehicle_feature_dim)
        hex_feature_dim = env._hex_feature_dim
        context_dim = env._context_dim
    else:
        # Default fleet dimensions
        vehicle_feature_dim = 13
        hex_feature_dim = 5
        context_dim = 9
    
    # Replay buffer lives on CPU (pinned memory) to avoid consuming GPU VRAM.
    # GPU VRAM is reserved entirely for the model, GCN, and environment tensors.
    # Batches are DMA-transferred to GPU non-blocking during sampling (~1-2 ms,
    # fully hidden behind the GCN forward pass which takes 20-100 ms).
    #
    # Memory estimate (500k cap, 1000 vehicles, 3985 hexes, vdim≈13, hdim=5, cdim=9):
    #   500k × (1000×16 + 3985×5 + 9) × 4B × 2 ≈ 143 GB on CPU RAM
    #   → use config capacity directly; reduce if system RAM is limited
    capacity = buffer_config.capacity
    bytes_per_transition = (env_config.num_vehicles * vehicle_feature_dim +
                            env_config.num_hexes * hex_feature_dim + context_dim +
                            env_config.num_vehicles * 2 + 1 + 1) * 4 * 2
    size_gb = capacity * bytes_per_transition / (1024 ** 3)
    print(f"Replay buffer: {capacity:,} transitions on CPU RAM (~{size_gb:.1f} GB pinned)")

    training_device = str(device) if device.type != "cpu" else None

    return GPUReplayBuffer(
        capacity=capacity,
        num_vehicles=env_config.num_vehicles,
        vehicle_feature_dim=vehicle_feature_dim,
        num_hexes=env_config.num_hexes,
        hex_feature_dim=hex_feature_dim,
        context_dim=context_dim,
        prioritized=buffer_config.prioritized,
        alpha=buffer_config.alpha,
        beta_start=buffer_config.beta_start,
        beta_end=buffer_config.beta_end,
        device="cpu",
        training_device=training_device,
    )


def create_trainer(
    agent: SACAgent,
    replay_buffer: GPUReplayBuffer,
    config: Config,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1
):
    training_config = config.training
    
    if world_size > 1:
        if config.training.mixed_precision:
            return MixedPrecisionTrainer(
                agent=agent,
                replay_buffer=replay_buffer,
                training_config=training_config,
                distributed_config=config.distributed,
                rank=rank,
                world_size=world_size,
                checkpoint_config=config.checkpoint,
                logging_config=config.logging,
                use_amp=True
                # device is set by DistributedTrainer based on rank
            )
        else:
            return DistributedTrainer(
                agent=agent,
                replay_buffer=replay_buffer,
                training_config=training_config,
                distributed_config=config.distributed,
                rank=rank,
                world_size=world_size,
                checkpoint_config=config.checkpoint,
                logging_config=config.logging
                # device is set by DistributedTrainer based on rank
            )
    else:
        if isinstance(agent, FleetSACAgent):
            return FleetSACTrainer(
                agent=agent,
                replay_buffer=replay_buffer,
                training_config=training_config,
                checkpoint_config=config.checkpoint,
                logging_config=config.logging,
                device=str(device),
                use_amp=config.training.mixed_precision,
            )
        return SACTrainer(
            agent=agent,
            replay_buffer=replay_buffer,
            training_config=training_config,
            checkpoint_config=config.checkpoint,
            logging_config=config.logging,
            device=str(device),
            use_amp=config.training.mixed_precision,
        )


def compute_od_matrix(trip_loader: RealTripLoader, device: str = "cuda") -> Optional[torch.Tensor]:
    """Compute Origin-Destination flow matrix from trip data for MAG metric (paper Eq. 14)."""
    if trip_loader is None or not trip_loader.is_loaded:
        return None
    num_hexes = trip_loader.num_hexes
    pickup = trip_loader._pickup_hexes   # [num_trips] long tensor
    dropoff = trip_loader._dropoff_hexes  # [num_trips] long tensor
    od = torch.zeros(num_hexes, num_hexes, device=device)
    od.index_put_((pickup, dropoff), torch.ones(pickup.shape[0], device=device), accumulate=True)
    print(f"[WDRO] Built OD matrix: {num_hexes}x{num_hexes}, {(od > 0).sum().item()} non-zero entries")
    return od


def create_environment(config: Config, device: torch.device, args, trip_loader: Optional[RealTripLoader] = None) -> GPUEnvironment:
    """Create GPU environment for training.
    
    Args:
        config: Configuration object
        device: Torch device
        args: Command line arguments
        trip_loader: Optional real trip data loader
        
    Returns:
        GPUEnvironment instance
    """
    num_hexes = config.environment.num_hexes
    device_str = str(device)
    hex_grid = HexGrid(device=device_str)
    
    if trip_loader and trip_loader.is_loaded:
        hex_ids = trip_loader.hex_ids
        lats, lons = trip_loader.get_hex_coordinates()
        num_hexes = len(hex_ids)
        config.environment.num_hexes = num_hexes
        hex_grid._hex_ids = hex_ids
        hex_grid._hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
        hex_grid._latitudes = lats.to(device)
        hex_grid._longitudes = lons.to(device)
        hex_grid._initialized = True
        hex_grid.distance_matrix.compute(hex_grid._latitudes, hex_grid._longitudes, hex_ids=hex_ids)
        try:
            hex_grid.neighbors.compute(hex_ids, k=1)
        except Exception as e:
            print(f"[Stations] Warning: failed to build 1-hop hex neighbors from H3 IDs: {e}")
    else:
        # Fallback: synthetic grid covering ~20km^2 of NYC
        grid_size = int(num_hexes ** 0.5) + 1
        hex_spacing_km = 20.0 / max(grid_size, 1)
        fake_hex_ids = [f"hex_{i}" for i in range(num_hexes)]
        hex_grid._hex_ids = fake_hex_ids
        hex_grid._hex_to_idx = {h: i for i, h in enumerate(fake_hex_ids)}
        base_lat, base_lon = 40.7128, -74.0060
        lat_per_km, lon_per_km = 0.009, 0.012
        indices = torch.arange(num_hexes, device=device)
        rows = indices // grid_size
        cols = indices % grid_size
        lats = base_lat + rows.float() * hex_spacing_km * lat_per_km
        lons = base_lon + cols.float() * hex_spacing_km * lon_per_km
        hex_grid._latitudes = lats
        hex_grid._longitudes = lons
        hex_grid._initialized = True
        lat_diff = lats.unsqueeze(1) - lats.unsqueeze(0)
        lon_diff = lons.unsqueeze(1) - lons.unsqueeze(0)
        lat_km = lat_diff / lat_per_km
        lon_km = lon_diff / lon_per_km
        distances = torch.sqrt(lat_km**2 + lon_km**2)
        distances = distances * (0.8 + 0.4 * torch.rand_like(distances))
        distances.fill_diagonal_(0)
        hex_grid.distance_matrix._distances = distances
        hex_grid.distance_matrix._num_hexes = num_hexes
        positive = distances[distances > 0]
        if positive.numel() > 0:
            neighbor_threshold = positive.min().item() * 1.5
            synthetic_adj = (distances <= neighbor_threshold).float()
            synthetic_adj.fill_diagonal_(0.0)
            hex_grid.neighbors.compute_from_adjacency(synthetic_adj, hex_ids=fake_hex_ids)
    
    # Choose environment version based on flag
    if args.env_v2:
        print("[Environment] Using GPUEnvironmentV2 (modular refactored version)")
        env = GPUEnvironmentV2(
            config=config,
            hex_grid=hex_grid,
            trip_loader=trip_loader,
            device=str(device)
        )
    else:
        env = GPUEnvironment(
            config=config,
            hex_grid=hex_grid,
            trip_loader=trip_loader,  # Pass real trip loader if available
            device=str(device)
        )
    
    return env


def warmup_replay_buffer(
    env: GPUEnvironment,
    replay_buffer: GPUReplayBuffer,
    num_steps: int,
    device: torch.device
):
    """Fill replay buffer with random transitions."""
    print(f"Warming up replay buffer with {num_steps} random transitions...")

    fleet_state_only = (
        hasattr(env, "_fleet_vehicle_feature_dim")
        and replay_buffer.vehicle_states.shape[2] == env._fleet_vehicle_feature_dim
    )

    state = env.reset(fleet_state_only=fleet_state_only)
    
    for step in range(num_steps):
        # SERVE=0, CHARGE=1, REPOSITION=2 (IDLE removed)
        action_type = torch.randint(0, 3, (env.num_vehicles,), device=device)
        reposition_target = torch.randint(0, env.num_hexes, (env.num_vehicles,), device=device)
        selected_trip = torch.randint(0, 500, (env.num_vehicles,), device=device)  # NEW: random trip selections

        if fleet_state_only:
            current_vehicle_hex_ids = env.fleet_state.positions.long().clone()
            available_mask = env.fleet_state.get_available_mask(env.current_step)

            next_state, reward, done, info = env.step(
                action_type,
                reposition_target,
                selected_trip,
                fleet_state_only=True,
            )

            hex_allocations = torch.zeros(env.num_hexes, 3, dtype=torch.float32, device=device)
            if available_mask.any():
                avail_hex_ids = current_vehicle_hex_ids[available_mask]
                avail_action_type = action_type[available_mask]
                hex_action_counts = torch.zeros(
                    env.num_hexes, 3, dtype=torch.float32, device=device
                )
                hex_action_counts.index_put_(
                    (avail_hex_ids, avail_action_type),
                    torch.ones_like(avail_action_type, dtype=torch.float32),
                    accumulate=True,
                )
                hex_totals = hex_action_counts.sum(dim=1, keepdim=True)
                active_hexes = hex_totals.squeeze(1) > 0
                hex_allocations[active_hexes] = (
                    hex_action_counts[active_hexes] / hex_totals[active_hexes]
                )

            hex_repos_targets = torch.arange(env.num_hexes, device=device, dtype=torch.long)
            if available_mask.any():
                repos_mask = available_mask & (action_type == 2)
                if repos_mask.any():
                    repos_hexes = current_vehicle_hex_ids[repos_mask]
                    repos_targets = reposition_target[repos_mask]
                    seen_hexes = set()
                    for hex_idx, target_idx in zip(repos_hexes.tolist(), repos_targets.tolist()):
                        if hex_idx not in seen_hexes:
                            hex_repos_targets[hex_idx] = target_idx
                            seen_hexes.add(hex_idx)

            hex_charge_power = torch.zeros(env.num_hexes, dtype=torch.float32, device=device)

            replay_buffer.push_fleet(
                state=state,
                hex_allocations=hex_allocations,
                hex_repos_targets=hex_repos_targets,
                hex_charge_power=hex_charge_power,
                vehicle_hex_ids=current_vehicle_hex_ids,
                reward=reward.item() if isinstance(reward, torch.Tensor) else reward,
                next_state=next_state,
                done=done.item() if isinstance(done, torch.Tensor) else done,
            )
        else:
            next_state, reward, done, info = env.step(action_type, reposition_target, selected_trip)

            # Convert flat state to dict format for replay buffer
            state_dict = env._build_state_dict(state)
            next_state_dict = env._build_state_dict(next_state)

            # Action: [num_vehicles, 2]
            action = torch.zeros(env.num_vehicles, 2, dtype=torch.long, device=device)
            action[:, 0] = action_type
            action[:, 1] = reposition_target

            replay_buffer.push(
                state=state_dict,
                action=action,
                reward=reward.item() if isinstance(reward, torch.Tensor) else reward,
                next_state=next_state_dict,
                done=done.item() if isinstance(done, torch.Tensor) else done
            )
        
        if done.item() if isinstance(done, torch.Tensor) else done:
            state = env.reset(fleet_state_only=fleet_state_only)
        else:
            state = next_state
        
        if (step + 1) % 1000 == 0:
            print(f"  Warmup: {step + 1}/{num_steps} transitions")
    
    print(f"Warmup complete. Buffer size: {len(replay_buffer)}")


def train_single_gpu(args):
    print("=" * 60)
    print("EV Fleet RL Training - GPU Accelerated")
    print("=" * 60)
    
    setup_seed(args.seed)
    device = get_device(args)
    print(f"Device: {device}")
    
    config = create_config(args)
    
    # Load real trip data if provided (CLI overrides YAML)
    trip_loader = None
    data_path = args.real_data or config.data.parquet_path
    sample_ratio = args.trip_sample if args.trip_sample is not None else config.data.trip_percentage
    target_res = args.target_h3_resolution if hasattr(args, 'target_h3_resolution') and args.target_h3_resolution is not None else config.data.target_h3_resolution
    max_hex_count = args.max_hex_count if hasattr(args, 'max_hex_count') and args.max_hex_count is not None else config.data.max_hex_count
    if data_path:
        resolved_path = Path(data_path)
        resolved_str = str(resolved_path)
        sample_ratio = sample_ratio if sample_ratio is not None else 1.0
        print(f"\n[Real Data] Loading from: {resolved_str}")
        if sample_ratio < 1.0:
            print(f"[Real Data] Using {sample_ratio*100:.0f}% sample of trips")
        if target_res is not None:
            print(f"[Real Data] Target H3 resolution: {target_res}")
        if max_hex_count is not None:
            print(f"[Real Data] Max hex count: {max_hex_count}")
        if not resolved_path.exists():
            print(f"[Real Data] File not found: {resolved_str}. Falling back to synthetic data.")
        else:
            try:
                trip_loader = RealTripLoader(
                    parquet_path=resolved_str,
                    device=str(device),
                    sample_ratio=sample_ratio,
                    target_h3_resolution=target_res,
                    max_hex_count=max_hex_count,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
                trip_loader.load()
            except Exception as exc:
                print(f"[Real Data] Failed to load due to: {exc}. Falling back to synthetic data.")
                trip_loader = None
        if trip_loader and trip_loader.is_loaded:
            config.environment.num_hexes = trip_loader.num_hexes
            print(f"[Real Data] Loaded {trip_loader.total_trips:,} trips | Hexes: {trip_loader.num_hexes:,}")
            print(f"[Real Data] Fare range: ${trip_loader.fare_stats['min']:.2f} - ${trip_loader.fare_stats['max']:.2f}")
        elif trip_loader is None:
            print("[Real Data] Warning: proceeding with synthetic trip generator")
    else:
        print("\n[Data] Using synthetic trips (set data.parquet_path or --real-data for NYC trips)")
    
    print(f"Vehicles: {config.environment.num_vehicles}")
    print(f"Hexes: {config.environment.num_hexes}")
    print(f"Episodes: {config.training.total_episodes}")
    print(f"Batch Size: {config.training.batch_size}")
    
    env = create_environment(config, device, args, trip_loader=trip_loader)
    print(f"\nEnvironment created with {env.num_vehicles} vehicles")
    
    # Create agent with GCN support
    agent = create_agent(config, device, env=env)
    algo = getattr(config.training, 'algo', 'sac').lower()

    # Set adjacency matrix and K-hop mask for Fleet actor (SAC only — MAPPO/MADDPG are pure MLP)
    if algo not in ('ppo', 'maddpg'):
        if hasattr(env, '_adjacency_matrix') and env._adjacency_matrix is not None:
            agent.set_adjacency(env._adjacency_matrix)
            print(f"[GCN] Adjacency matrix set for GCN actor")

            # Compute K-hop mask for FleetSACAgent
            try:
                k_hops = getattr(config.fleet_actor, 'repos_khop', 4)
                khop_mask_hh = HexNeighbors.compute_khop_mask(env._adjacency_matrix, k=k_hops)
                khop_indices, khop_counts, max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh)
                padded_khop_mask = khop_indices != -1
                agent.set_khop_data(khop_indices, padded_khop_mask)
                print(f"[GCN] K-hop mask computed (K={k_hops}, max_neighbors={max_k}) and passed to FleetSACAgent")
            except Exception as e:
                print(f"[GCN] Error computing K-hop mask: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[GCN] Warning: No adjacency matrix available, GCN may not work properly")
        print(f"[GCN] Using FleetGCN actor for hex-level spatial reasoning")
    else:
        print(f"[{algo.upper()}] Pure-MLP actor — skipping GCN/adjacency/K-hop setup")
    
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    gpu_ids = [int(x) for x in args.gpus.split(',')] if args.gpus else None

    # DataParallel and torch.compile only for SAC (MAPPO/MADDPG are pure MLP)
    if algo not in ('ppo', 'maddpg'):
        # Auto-enable data_parallel if multiple GPUs were passed without --data-parallel flag
        if gpu_ids and len(gpu_ids) > 1 and not (args.data_parallel or args.distributed):
            print(f"\n[Warning] Multiple GPUs ({gpu_ids}) specified but no parallel flag was used.")
            print(f"[Warning] Auto-enabling --data-parallel to utilize multiple GPUs!\n")
            args.data_parallel = True

        # DataParallel for multi-GPU training (simpler than DDP)
        if args.data_parallel and (torch.cuda.device_count() > 1 or (gpu_ids and len(gpu_ids) > 1)):
            num_gpus = len(gpu_ids) if gpu_ids else torch.cuda.device_count()
            print(f"[DataParallel] Using {num_gpus} GPUs (IDs: {gpu_ids or 'All visible'}) for training")
            agent.actor = torch.nn.DataParallel(agent.actor, device_ids=gpu_ids)
            agent.critic = torch.nn.DataParallel(agent.critic, device_ids=gpu_ids)
            agent.critic_target = torch.nn.DataParallel(agent.critic_target, device_ids=gpu_ids)
            print(f"[DataParallel] Models wrapped - effective split batch size: {args.batch_size}")
            print(f"[DataParallel Note] DataParallel keeps the Replay Buffer and Environment on GPU 0.")
            print(f"[DataParallel Note] VRAM usage will appear unbalanced (GPU 0 maxed, GPU 1 low usage).")
            print(f"[DataParallel Note] For balanced VRAM usage, use --distributed instead of --data-parallel.\n")

        # Compile models for faster execution (PyTorch 2.0+)
        # Note: torch.compile with DataParallel may have issues
        if hasattr(torch, 'compile') and not args.data_parallel:
            try:
                print("[Optimization] Compiling models with torch.compile...")
                agent.actor = torch.compile(agent.actor, mode='reduce-overhead', options={"triton.cudagraphs": False})
                agent.critic = torch.compile(agent.critic, mode='reduce-overhead', options={"triton.cudagraphs": False})
                agent.critic_target = torch.compile(agent.critic_target, mode='reduce-overhead', options={"triton.cudagraphs": False})
                print("[Optimization] Model compilation complete")
            except Exception as e:
                print(f"[Optimization] Warning: torch.compile failed: {e}")
                print("[Optimization] Continuing without compilation")
    
    # ── MAPPO on-policy path ────────────────────────────────────────────
    if algo == 'ppo':
        replay_buffer = PPORolloutBuffer()
        print("[MAPPO] PPORolloutBuffer created (on-policy, no warmup needed)")

        trainer = MAPPOTrainer(
            agent             = agent,
            replay_buffer     = replay_buffer,
            training_config   = config.training,
            checkpoint_config = config.checkpoint,
            logging_config    = config.logging,
            device            = str(device),
        )
        tc = config.training
        print(f"[MAPPO] Trainer: γ={tc.gamma}, λ_GAE={tc.gae_lambda}, "
              f"ε={tc.clip_eps}, vf_coef={tc.vf_coef}, "
              f"ent_coef={tc.ent_coef}, epochs={tc.update_epochs}")

        collector = PPOCollector(
            env=env,
            replay_buffer=replay_buffer,
            device=str(device),
            repos_khop=getattr(config.training, 'mappo_repos_khop', 4),
            use_khop_candidates=getattr(config.training, 'mappo_use_khop_candidates', False),
        )

        use_enhanced = False  # MAPPO has no WDRO/semi-MDP

    # ── MADDPG off-policy path ──────────────────────────────────────────
    elif algo == 'maddpg':
        tc = config.training
        maddpg_local_dim = env._vehicle_feature_dim
        maddpg_buffer = MADDPGReplayBuffer(
            capacity            = tc.maddpg_buffer_capacity,
            num_vehicles        = config.environment.num_vehicles,
            vehicle_feature_dim = maddpg_local_dim,
            context_dim         = env._context_dim,
            action_dim          = 3,
            device              = 'cpu',
        )
        bytes_per = (config.environment.num_vehicles * maddpg_local_dim * 2 +
                     env._context_dim * 2 + config.environment.num_vehicles * 2 + 2) * 4
        size_mb = tc.maddpg_buffer_capacity * bytes_per / (1024**2)
        print(f"[MADDPG] Replay buffer: {tc.maddpg_buffer_capacity:,} transitions "
              f"(~{size_mb:.0f} MB CPU)")

        trainer = MADDPGTrainer(
            agent             = agent,
            replay_buffer     = maddpg_buffer,
            training_config   = tc,
            checkpoint_config = config.checkpoint,
            logging_config    = config.logging,
            device            = str(device),
            use_amp           = tc.mixed_precision,
        )
        print(f"[MADDPG] Trainer: γ={tc.gamma}, τ={tc.maddpg_tau}, "
              f"lr_actor={tc.maddpg_lr_actor}, lr_critic={tc.maddpg_lr_critic}")

        collector = MADDPGCollector(
            env           = env,
            replay_buffer = maddpg_buffer,
            device        = str(device),
            repos_khop    = getattr(config.training, 'maddpg_repos_khop', 4),
        )
        collector._ensure_khop_neighbors()
        trainer._khop_neighbor_indices = collector._khop_neighbor_indices
        trainer._khop_neighbor_mask = collector._khop_neighbor_mask

        effective_warmup = (tc.batch_size * 4 if args.resume
                           else max(tc.batch_size * 4,
                                    getattr(tc, 'warmup_steps', 1000)))
        print(f"[MADDPG] Warming up buffer with {effective_warmup} random steps...")
        state = env.reset()
        for _w in range(effective_warmup):
            act_type = torch.randint(0, 3, (env.num_vehicles,), device=device)
            repos = torch.randint(0, env.num_hexes, (env.num_vehicles,), device=device)
            selected_trip = torch.randint(
                0,
                max(1, config.environment.max_trips_per_step),
                (env.num_vehicles,),
                device=device,
            )

            available_mask = env.get_available_actions()
            vehicle_hex_ids = env.fleet_state.positions.long() if hasattr(env, 'fleet_state') else None

            next_state, reward, done_tensor, info = env.step(
                act_type,
                repos,
                selected_trip,
            )
            done_val = done_tensor.item() if isinstance(done_tensor, torch.Tensor) else bool(done_tensor)

            vehicle_size = env.num_vehicles * env._vehicle_feature_dim
            hex_size = env.num_hexes * env._hex_feature_dim
            curr_vf = state[:vehicle_size].view(env.num_vehicles, env._vehicle_feature_dim)
            curr_context = state[vehicle_size + hex_size:]
            next_vf = next_state[:vehicle_size].view(env.num_vehicles, env._vehicle_feature_dim)
            next_context = next_state[vehicle_size + hex_size:]

            if hasattr(info, 'per_vehicle_reward') and info.per_vehicle_reward is not None:
                per_vehicle = info.per_vehicle_reward.clone().detach()
            else:
                scalar_r = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                per_vehicle = torch.full(
                    (env.num_vehicles,),
                    scalar_r / env.num_vehicles,
                    dtype=torch.float32,
                    device=device,
                )

            maddpg_buffer.push(
                vehicle_features=curr_vf,
                context_features=curr_context,
                next_vehicle_features=next_vf,
                next_context_features=next_context,
                actions_type=act_type,
                actions_repos=repos,
                per_vehicle_rewards=per_vehicle,
                done=done_val,
                action_mask=available_mask,
                reposition_mask=None,
                trip_mask=None,
                vehicle_hex_ids=vehicle_hex_ids,
                executed_actions_type=act_type,
                executed_actions_repos=repos,
            )
            state = env.reset() if done_val else next_state
        print(f"[MADDPG] Warmup complete. Buffer size: {len(maddpg_buffer)}")

        replay_buffer = maddpg_buffer
        use_enhanced  = False  # MADDPG has no WDRO/semi-MDP

    # ── SAC off-policy path (existing, unchanged) ─────────────────────
    else:
        replay_buffer = create_replay_buffer(config, device, env=env)
        # Capacity already printed by create_replay_buffer() with auto-adjustment info

        warmup_steps = config.training.warmup_steps
        if warmup_steps > 0:
            # On resume, do a shorter warmup to re-populate the buffer with
            # on-policy transitions from the restored policy before the first
            # critic update. Without this the buffer starts empty and the
            # first batch (after just 2 episodes) is highly correlated,
            # causing Q-value explosion.
            effective_warmup = config.training.batch_size * 4 if args.resume else warmup_steps
            warmup_replay_buffer(env, replay_buffer, effective_warmup, device)

        # Create trainer - use EnhancedSACTrainer if Semi-MDP or WDRO is enabled
        use_enhanced = args.semi_mdp or args.wdro
        if use_enhanced:
            enhanced_config = EnhancedTrainingConfig(
                use_semi_mdp=args.semi_mdp,
                use_wdro=args.wdro,
                wdro_rho=args.wdro_rho,
                wdro_metric=args.wdro_metric,
                use_temperature_annealing=args.temperature_annealing,
                initial_temperature=args.initial_temperature,
                final_temperature=args.final_temperature,
                temperature_decay_episodes=args.temperature_decay_episodes,
                use_milp=args.milp,
                use_amp=True,  # Enable mixed precision training
            )
            enhanced_config = apply_enhanced_schedule_args(args, enhanced_config)
            # Get adjacency matrix for WDRO graph-aligned metric (paper Eq. 17)
            adj_matrix = getattr(env, '_adjacency_matrix', None)
            # Compute OD matrix for MAG metric graph component (paper Eq. 14)
            od_matrix = compute_od_matrix(trip_loader, str(device)) if (args.wdro and args.wdro_metric == 'mag') else None
            if isinstance(agent, FleetSACAgent):
                from gpu_core.training.enhanced_trainer import FleetEnhancedSACTrainer
                TrainerClass = FleetEnhancedSACTrainer
            else:
                from gpu_core.training.enhanced_trainer import EnhancedSACTrainer
                TrainerClass = EnhancedSACTrainer

            trainer = TrainerClass(
                agent=agent,
                replay_buffer=replay_buffer,
                training_config=config.training,
                enhanced_config=enhanced_config,
                checkpoint_config=config.checkpoint,
                logging_config=config.logging,
                device=str(device),
                adjacency_matrix=adj_matrix,
                od_matrix=od_matrix
            )
            print(f"[Enhanced] Semi-MDP: {args.semi_mdp}, WDRO: {args.wdro}, MILP: {args.milp}")
            print(f"[Optimization] Mixed Precision Training (AMP): Enabled")
        else:
            trainer = create_trainer(agent, replay_buffer, config, device)
            if config.training.mixed_precision:
                print(f"[Optimization] Mixed Precision Training (AMP): Enabled")

        # Create collector - use EnhancedEpisodeCollector for duration tracking if Semi-MDP enabled
        if args.semi_mdp:
            collector = EnhancedEpisodeCollector(
                env=env,
                replay_buffer=replay_buffer,
                device=str(device),
                use_milp=args.milp,
                track_durations=True
            )
            print(f"[Enhanced] Using EnhancedEpisodeCollector with duration tracking")
        else:
            collector = EpisodeCollector(
                env=env,
                replay_buffer=replay_buffer,
                device=str(device),
                use_milp=args.milp
            )
            if args.milp:
                print(f"[Assignment] MILP EXACT ASSIGNMENT ENABLED (Gurobi)")
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print("=" * 60)
    print("Training started...")
    print("=" * 60)
    
    # Setup TensorBoard and CSV logging
    writer = None
    csv_file = None
    csv_writer = None
    
    # Setup log directory
    log_dir = os.path.join(config.logging.log_dir, f"run_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize CSV logging
    csv_path = os.path.join(log_dir, "training_history.csv")
    try:
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        # Write CSV header
        csv_writer.writerow([
            'Episode', 'Reward', 'Avg100', 'Profit', 'Revenue', 'Serve_Pct', 'Steps', 'Trips_Served', 
            'Trips_Loaded', 'Avg_SOC', 'Actor_Loss', 'Critic_Loss', 'Alpha', 'Value_Loss',
            'WDRO_Lambda', 'WDRO_Rho_Hat', 'WDRO_Target_Mean', 'WDRO_Target_Abs_Max',
            'WDRO_Distance_Mean', 'WDRO_Worst_Value_Mean', 'WDRO_Phase', 'WDRO_Value_Source_Is_Learned', 'WDRO_Value_Training_Active',
            'Actor_Serve_Pct', 'Actor_Charge_Pct', 'Actor_Repos_Pct', 'V_Label_Mean', 'V_Pred_Mean',
            'WDRO_Value_Source_Switch_Episode', 'WDRO_Value_Train_Stop_Episode',
            'Idle_Pct', 'Serve_Act_Pct', 'Charge_Pct', 'Repos_Pct', 'Forced_Charge_Pct', 'Serve_Failed_Pct', 'Elapsed_Sec'
        ])
        print(f"CSV logging to: {csv_path}")
    except Exception as e:
        print(f"Failed to initialize CSV logging: {e}")
        
    if args.tensorboard and HAS_TENSORBOARD:
        run_name = f"ev_fleet_{config.environment.num_vehicles}v_{config.environment.num_hexes}h_{int(time.time())}"
        writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir, run_name))
        print(f"TensorBoard logging to: {args.tensorboard_dir}/{run_name}")
    elif args.tensorboard and not HAS_TENSORBOARD:
        print("Warning: TensorBoard requested but not installed. Run: pip install tensorboard")
    
    start_time = time.time()
    total_steps = getattr(trainer, 'total_steps', 0)       # env-step counter (not gradient steps)
    total_updates = getattr(trainer, 'global_step', 0)
    best_eval_reward = getattr(trainer, 'best_reward', float('-inf'))
    best_train_reward = getattr(trainer, 'best_train_reward', float('-inf'))  # restored on resume
    
    # Metrics tracking
    recent_rewards = deque(getattr(trainer, 'recent_rewards_window', []), maxlen=100)
    recent_losses = {
        'actor_loss': deque(maxlen=100),
        'critic_loss': deque(maxlen=100),
        'alpha': deque(maxlen=100),
        'q_mean': deque(maxlen=100),
        'repos_aux_loss': deque(maxlen=100),
        'value_loss': deque(maxlen=100),
        'wdro_lambda': deque(maxlen=100),
        'wdro_rho_hat': deque(maxlen=100),
        'wdro_target_mean': deque(maxlen=100),
        'wdro_target_abs_max': deque(maxlen=100),
        'wdro_distance_mean': deque(maxlen=100),
        'wdro_worst_value_mean': deque(maxlen=100),
        'wdro_phase': deque(maxlen=100),
        'wdro_value_source_is_learned': deque(maxlen=100),
        'wdro_value_training_active': deque(maxlen=100),
        'actor_serve_frac': deque(maxlen=100),
        'actor_charge_frac': deque(maxlen=100),
        'actor_repos_frac': deque(maxlen=100),
        'v_label_mean': deque(maxlen=100),
        'v_pred_mean': deque(maxlen=100),
        'serve_consistency': deque(maxlen=100),
        'charge_consistency': deque(maxlen=100),
        'reposition_consistency': deque(maxlen=100),
        'failed_action_fraction': deque(maxlen=100),
    }
    
    start_episode = getattr(trainer, 'episode', 0)
    if args.start_episode is not None:
        start_episode = args.start_episode
        trainer.episode = start_episode
    if start_episode > 0:
        print(f"Resuming training loop from episode {start_episode}")
    
    # Track time per 20 episodes
    episode_batch_start_time = time.time()
    episode_batch_count = 0
        
    for episode in range(start_episode, config.training.total_episodes):
        # Track episode in trainer so checkpoints save the correct episode number
        trainer.episode = episode

        if algo == 'ppo':
            # ── MAPPO: collect one episode → PPO update → clear buffer ──
            with _time_section("episode_collection"):
                episode_stats = collector.collect_episode(
                    agent         = agent,
                    rollout_steps = config.training.rollout_steps,
                    deterministic = args.deterministic,
                    seed          = args.seed + episode,
                )
            total_steps += episode_stats.steps

            if len(replay_buffer) > 0:
                with _time_section("training_step"):
                    train_info = trainer.train_step(next_value=episode_stats.next_value)
                    # buffer is cleared inside train_step()
                total_updates += 1
                if train_info:
                    for key in ['actor_loss', 'critic_loss']:
                        if key in train_info:
                            recent_losses[key].append(train_info[key])
                    # MAPPO has no alpha; keep this column as entropy scale
                    # (ent_coef * entropy) for comparable magnitude diagnostics.
                    if 'ent_scale' in train_info:
                        recent_losses['alpha'].append(train_info['ent_scale'])
                    elif 'entropy' in train_info:
                        recent_losses['alpha'].append(train_info['entropy'])
                    if writer and total_updates % args.log_loss_interval == 0:
                        for key, value in train_info.items():
                            writer.add_scalar(f'Loss/{key}', value, total_updates)

        elif algo == 'maddpg':
            # ── MADDPG: off-policy collect → train_step (like SAC) ─────
            # Compute decayed epsilon for this episode
            eps_progress = min(1.0, episode / max(1, config.training.maddpg_eps_decay_episodes))
            exploration_eps = (config.training.maddpg_eps_start * (1.0 - eps_progress) +
                               config.training.maddpg_eps_end   *        eps_progress)

            with _time_section("episode_collection"):
                episode_stats = collector.collect_episode(
                    agent           = agent,
                    rollout_steps   = config.training.rollout_steps,
                    deterministic   = args.deterministic,
                    seed            = args.seed + episode,
                    exploration_eps = exploration_eps,
                )
            total_steps += episode_stats.steps

            if len(replay_buffer) >= config.training.batch_size:
                update_ratio = getattr(args, 'update_ratio',
                                       getattr(config.training, 'update_every', 1))
                num_updates = max(1, episode_stats.steps // update_ratio)

                for update_i in range(num_updates):
                    with _time_section("training_step"):
                        train_info = trainer.train_step()
                    total_updates += 1

                    if train_info:
                        for key in ['actor_loss', 'critic_loss', 'q_mean', 'repos_aux_loss', 'serve_consistency', 'charge_consistency', 'reposition_consistency', 'failed_action_fraction']:
                            if key in train_info:
                                recent_losses[key].append(train_info[key])
                        if 'alpha' in train_info:
                            recent_losses['alpha'].append(train_info['alpha'])

                        if writer and total_updates % args.log_loss_interval == 0:
                            for key, value in train_info.items():
                                writer.add_scalar(f'Loss/{key}', value, total_updates)

                    if args.debug and total_updates % (args.log_loss_interval * 10) == 0:
                        print(f"  [MADDPG Update {total_updates}] "
                              f"Actor: {train_info.get('actor_loss', 0):.4f} | "
                              f"Critic: {train_info.get('critic_loss', 0):.4f} | "
                              f"Q: {train_info.get('q_mean', 0):.2f} | "
                              f"ε: {exploration_eps:.3f}")

        else:
            # ── SAC: existing logic, unchanged ──────────────────────────
            # Lower epsilon-greedy noise since we now have per-vehicle stochastic sampling
            # The policy entropy (via alpha) provides main exploration
            # Fixed low noise - SAC entropy (alpha + temperature) handles exploration
            exploration_noise = getattr(config.training, 'exploration_noise', 0.01)

            # Get temperature for this episode (applies to both SAC and WDRO)
            if args.temperature_annealing:
                progress = min(1.0, episode / args.temperature_decay_episodes)
                temperature = args.initial_temperature * (1 - progress) + args.final_temperature * progress
            else:
                temperature = 1.0

            with _time_section("episode_collection"):
                episode_stats = collector.collect_episode_fleet(
                    agent=agent,
                    seed=args.seed + episode,
                    temperature=temperature,
                    deterministic=args.deterministic
                )

            total_steps += episode_stats.steps

            if len(replay_buffer) >= config.training.batch_size:
                # Use configurable update-to-data ratio
                # Default: 1 update per step for balance between learning and throughput
                update_ratio = getattr(args, 'update_ratio', getattr(config.training, 'update_every', 1))
                num_updates = max(1, episode_stats.steps // update_ratio)

                for update_i in range(num_updates):
                    # Use enhanced train_step if available
                    if use_enhanced:
                        with _time_section("training_step"):
                            train_info = trainer.train_step_enhanced()
                    else:
                            with _time_section("training_step"):
                                train_info = trainer.train_step()
                    total_updates += 1

                    # Track loss metrics
                    if train_info:
                        for key in ['actor_loss', 'critic_loss', 'alpha', 'q_mean', 'value_loss', 'wdro_lambda', 'wdro_rho_hat', 'wdro_target_mean', 'wdro_target_abs_max', 'wdro_distance_mean', 'wdro_worst_value_mean', 'wdro_phase', 'wdro_value_source_is_learned', 'wdro_value_training_active', 'actor_serve_frac', 'actor_charge_frac', 'actor_repos_frac', 'v_label_mean', 'v_pred_mean']:
                            if key in train_info:
                                recent_losses[key].append(train_info[key])

                        # Log to TensorBoard
                        if writer and total_updates % args.log_loss_interval == 0:
                            for key, value in train_info.items():
                                writer.add_scalar(f'Loss/{key}', value, total_updates)
                    
                    # Print detailed loss periodically
                    if args.debug and total_updates % (args.log_loss_interval * 10) == 0:
                        print(f"  [Update {total_updates}] "
                              f"Actor: {train_info.get('actor_loss', 0):.4f} | "
                              f"Critic: {train_info.get('critic_loss', 0):.4f} | "
                              f"Alpha: {train_info.get('alpha', 0):.4f} | "
                              f"Q: {train_info.get('q_mean', 0):.2f}")
        
        # Track episode reward
        recent_rewards.append(episode_stats.total_reward)
        episode_batch_count += 1
        # Keep trainer counters/window in sync so checkpoint saves have accurate resume state
        trainer.total_steps = total_steps
        trainer.recent_rewards_window = list(recent_rewards)
        
        # Log timing every 20 episodes
        if episode_batch_count == 20:
            batch_elapsed = time.time() - episode_batch_start_time
            print(f"[Timing] Last 20 episodes took {batch_elapsed:.2f}s ({batch_elapsed/20:.2f}s per episode)")
            episode_batch_start_time = time.time()
            episode_batch_count = 0
        
        if (episode + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed
            updates_per_sec = total_updates / elapsed if elapsed > 0 else 0
            
            # Calculate average losses
            avg_actor_loss = sum(recent_losses['actor_loss']) / len(recent_losses['actor_loss']) if recent_losses['actor_loss'] else 0
            avg_critic_loss = sum(recent_losses['critic_loss']) / len(recent_losses['critic_loss']) if recent_losses['critic_loss'] else 0
            avg_alpha = sum(recent_losses['alpha']) / len(recent_losses['alpha']) if recent_losses['alpha'] else 0
            avg_value_loss = sum(recent_losses['value_loss']) / len(recent_losses['value_loss']) if recent_losses['value_loss'] else 0
            avg_wdro_lambda = sum(recent_losses['wdro_lambda']) / len(recent_losses['wdro_lambda']) if recent_losses['wdro_lambda'] else 0
            avg_wdro_rho_hat = sum(recent_losses['wdro_rho_hat']) / len(recent_losses['wdro_rho_hat']) if recent_losses['wdro_rho_hat'] else 0
            avg_wdro_target_mean = sum(recent_losses['wdro_target_mean']) / len(recent_losses['wdro_target_mean']) if recent_losses['wdro_target_mean'] else 0
            avg_wdro_target_abs_max = sum(recent_losses['wdro_target_abs_max']) / len(recent_losses['wdro_target_abs_max']) if recent_losses['wdro_target_abs_max'] else 0
            avg_wdro_distance_mean = sum(recent_losses['wdro_distance_mean']) / len(recent_losses['wdro_distance_mean']) if recent_losses['wdro_distance_mean'] else 0
            avg_wdro_worst_value_mean = sum(recent_losses['wdro_worst_value_mean']) / len(recent_losses['wdro_worst_value_mean']) if recent_losses['wdro_worst_value_mean'] else 0
            avg_wdro_phase = round(sum(recent_losses['wdro_phase']) / len(recent_losses['wdro_phase'])) if recent_losses['wdro_phase'] else 0
            avg_wdro_value_source_is_learned = round(sum(recent_losses['wdro_value_source_is_learned']) / len(recent_losses['wdro_value_source_is_learned'])) if recent_losses['wdro_value_source_is_learned'] else 0
            avg_wdro_value_training_active = round(sum(recent_losses['wdro_value_training_active']) / len(recent_losses['wdro_value_training_active'])) if recent_losses['wdro_value_training_active'] else 0
            avg_actor_serve_pct = 100.0 * (sum(recent_losses['actor_serve_frac']) / len(recent_losses['actor_serve_frac'])) if recent_losses['actor_serve_frac'] else 0
            avg_actor_charge_pct = 100.0 * (sum(recent_losses['actor_charge_frac']) / len(recent_losses['actor_charge_frac'])) if recent_losses['actor_charge_frac'] else 0
            avg_actor_repos_pct = 100.0 * (sum(recent_losses['actor_repos_frac']) / len(recent_losses['actor_repos_frac'])) if recent_losses['actor_repos_frac'] else 0
            avg_v_label_mean = sum(recent_losses['v_label_mean']) / len(recent_losses['v_label_mean']) if recent_losses['v_label_mean'] else 0
            avg_v_pred_mean = sum(recent_losses['v_pred_mean']) / len(recent_losses['v_pred_mean']) if recent_losses['v_pred_mean'] else 0
            avg_serve_consistency = sum(recent_losses['serve_consistency']) / len(recent_losses['serve_consistency']) if recent_losses['serve_consistency'] else 0
            avg_charge_consistency = sum(recent_losses['charge_consistency']) / len(recent_losses['charge_consistency']) if recent_losses['charge_consistency'] else 0
            avg_reposition_consistency = sum(recent_losses['reposition_consistency']) / len(recent_losses['reposition_consistency']) if recent_losses['reposition_consistency'] else 0
            avg_failed_action_fraction = sum(recent_losses['failed_action_fraction']) / len(recent_losses['failed_action_fraction']) if recent_losses['failed_action_fraction'] else 0
            avg_reward_100 = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            
            # Get action distribution
            action_counts = getattr(episode_stats, 'action_counts', None)
            if action_counts:
                total_actions = sum(action_counts.values()) or 1
                action_pct = {k: v * 100.0 / total_actions for k, v in action_counts.items()}
                action_str = f"S:{action_pct['serve']:4.1f}% C:{action_pct['charge']:4.1f}% R:{action_pct['reposition']:4.1f}%"
            else:
                action_str = ""
            
            serve_pct_stat = (episode_stats.trips_served / max(1, episode_stats.trips_loaded)) * 100.0
            print(f"Episode {episode + 1:5d} | "
                f"Reward: {episode_stats.total_reward:8.2f} | "
                f"Avg100: {avg_reward_100:8.2f} | "
                f"Profit: ${episode_stats.profit:6.2f} | "
                f"Rev: ${episode_stats.revenue:6.2f} | "
                f"Serve%: {serve_pct_stat:4.1f}% | "
                f"SOC: {episode_stats.avg_soc:5.1f}%")
            print(f"         Actor: {avg_actor_loss:7.4f} | "
                f"Critic: {avg_critic_loss:7.4f} | "
                f"Alpha: {avg_alpha:5.3f} | "
                f"Actions: {action_str}")
            forced_charge_pct = (100.0 * getattr(episode_stats, 'forced_charge_count', 0) / max(1, getattr(episode_stats, 'forced_charge_total_idle', 0)))
            print(f"         ActorMix: S:{avg_actor_serve_pct:4.1f}% C:{avg_actor_charge_pct:4.1f}% R:{avg_actor_repos_pct:4.1f}% | "
                f"ForcedCharge: {forced_charge_pct:4.1f}% | "
                f"VLbl: {avg_v_label_mean:6.2f} | VPred: {avg_v_pred_mean:6.2f}")
            if algo == 'maddpg':
                print(f"         Consistency: Serve:{100.0*avg_serve_consistency:4.1f}% "
                    f"Charge:{100.0*avg_charge_consistency:4.1f}% "
                    f"Repos:{100.0*avg_reposition_consistency:4.1f}% | "
                    f"FailedActs:{100.0*avg_failed_action_fraction:4.1f}%")
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Episode/reward', episode_stats.total_reward, episode + 1)
                writer.add_scalar('Episode/avg_reward_100', avg_reward_100, episode + 1)
                writer.add_scalar('Episode/steps', episode_stats.steps, episode + 1)
                writer.add_scalar('Episode/trips_served', episode_stats.trips_served, episode + 1)
                writer.add_scalar('Episode/trips_loaded', episode_stats.trips_loaded, episode + 1)
                writer.add_scalar('Episode/avg_soc', episode_stats.avg_soc, episode + 1)
                writer.add_scalar('Speed/steps_per_sec', steps_per_sec, episode + 1)
                writer.add_scalar('Speed/updates_per_sec', updates_per_sec, episode + 1)
                writer.add_scalar('Loss/avg_actor_loss', avg_actor_loss, episode + 1)
                writer.add_scalar('Loss/avg_critic_loss', avg_critic_loss, episode + 1)
                writer.add_scalar('Loss/avg_alpha', avg_alpha, episode + 1)
                # Log action distribution
                if action_counts:
                    for action_name, count in action_counts.items():
                        writer.add_scalar(f'Actions/{action_name}', count, episode + 1)
                        writer.add_scalar(f'ActionPct/{action_name}', action_pct[action_name], episode + 1)
                if algo == 'maddpg':
                    writer.add_scalar('MADDPG/serve_consistency', avg_serve_consistency, episode + 1)
                    writer.add_scalar('MADDPG/charge_consistency', avg_charge_consistency, episode + 1)
                    writer.add_scalar('MADDPG/reposition_consistency', avg_reposition_consistency, episode + 1)
                    writer.add_scalar('MADDPG/failed_action_fraction', avg_failed_action_fraction, episode + 1)

            # Log to CSV
            if csv_writer and csv_file:
                idle_pct = action_pct.get('idle', 0) if action_counts else 0
                serve_act_pct = action_pct.get('serve', 0) if action_counts else 0
                charge_pct = action_pct.get('charge', 0) if action_counts else 0
                repos_pct = action_pct.get('reposition', 0) if action_counts else 0
                
                serve_pct_stat = (episode_stats.trips_served / max(1, episode_stats.trips_loaded)) * 100.0
                serve_attempted = int(getattr(episode_stats, 'num_serve_attempted', 0))
                serve_success = int(getattr(episode_stats, 'num_serve_success', episode_stats.trips_served))
                serve_failed_pct = (max(0, serve_attempted - serve_success) / max(1, serve_attempted)) * 100.0
                forced_charge_pct = (100.0 * getattr(episode_stats, 'forced_charge_count', 0) / max(1, getattr(episode_stats, 'forced_charge_total_idle', 0)))
                csv_writer.writerow([
                    episode + 1,
                    round(episode_stats.total_reward, 2),
                    round(avg_reward_100, 2),
                    round(episode_stats.profit, 2),
                    round(episode_stats.revenue, 2),
                    round(serve_pct_stat, 1),
                    episode_stats.steps,
                    episode_stats.trips_served,
                    episode_stats.trips_loaded,
                    round(episode_stats.avg_soc, 1),
                    round(avg_actor_loss, 4),
                    round(avg_critic_loss, 4),
                    round(avg_alpha, 4),
                    round(avg_value_loss, 4),
                    round(avg_wdro_lambda, 4),
                    round(avg_wdro_rho_hat, 4),
                    round(avg_wdro_target_mean, 4),
                    round(avg_wdro_target_abs_max, 4),
                    round(avg_wdro_distance_mean, 4),
                    round(avg_wdro_worst_value_mean, 4),
                    avg_wdro_phase,
                    avg_wdro_value_source_is_learned,
                    avg_wdro_value_training_active,
                    round(avg_actor_serve_pct, 1),
                    round(avg_actor_charge_pct, 1),
                    round(avg_actor_repos_pct, 1),
                    round(avg_v_label_mean, 4),
                    round(avg_v_pred_mean, 4),
                    args.wdro_value_source_switch_episode,
                    args.wdro_value_train_stop_episode,
                    round(idle_pct, 1),
                    round(serve_act_pct, 1),
                    round(charge_pct, 1),
                    round(repos_pct, 1),
                    round(forced_charge_pct, 1),
                    round(serve_failed_pct, 1),
                    round(elapsed, 1)
                ])
                csv_file.flush()  # Ensure data is written immediately
        
        if (episode + 1) % args.eval_interval == 0:
            eval_reward = evaluate_agent(
                agent, env, args.eval_episodes, device, config, args, trip_loader
            )
            print(f"  [Eval] Avg Reward: {eval_reward:.2f}")
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                trainer.best_reward = best_eval_reward   # keep trainer in sync for checkpoint
                trainer.save_checkpoint('best_eval.pt')
                print(f"  [Best Eval] New best eval model saved!")
        
        # Save best model based on training reward (Avg100)
        avg_reward_100 = sum(recent_rewards) / len(recent_rewards) if recent_rewards else float('-inf')
        if len(recent_rewards) >= 50 and avg_reward_100 > best_train_reward:
            best_train_reward = avg_reward_100
            trainer.best_train_reward = best_train_reward   # keep trainer in sync for checkpoint
            trainer.save_checkpoint('best.pt')
            print(f"  [Best Train] New best model saved! Avg100: {avg_reward_100:.2f}")
        
        if (episode + 1) % args.save_interval == 0:
            trainer.save_checkpoint(f'checkpoint_{episode + 1}.pt')
    
    elapsed = time.time() - start_time
    steps_per_sec = total_steps / elapsed
    
    print("=" * 60)
    print("Training complete!")
    print(f"Total episodes: {config.training.total_episodes}")
    print(f"Total steps: {total_steps:,}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Average steps/second: {steps_per_sec:.1f}")
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"Best train reward (Avg100): {best_train_reward:.2f}")
    print("=" * 60)
    
    # Print timing report
    print("\n" + "=" * 60)
    print("TIMING STATISTICS")
    print("=" * 60)
    for name, times in sorted(_timing_stats.items(), key=lambda x: sum(x[1]), reverse=True):
        total = sum(times)
        mean = total / len(times) if times else 0
        max_time = max(times) if times else 0
        count = len(times)
        print(f"{name:<30} Total: {total:>8.2f}s  Mean: {mean:>7.4f}s  Max: {max_time:>7.4f}s  Count: {count:>6}")
    print("=" * 60)
    
    trainer.save_checkpoint('final.pt')
    print(f"Final model saved to: {config.checkpoint.checkpoint_dir}/final.pt")
    
    # Close TensorBoard writer and CSV file
    if writer:
        writer.close()
        print(f"TensorBoard logs saved. View with: tensorboard --logdir={args.tensorboard_dir}")
        
    if csv_file:
        csv_file.close()
        print(f"CSV logs saved to: {csv_path}")


def evaluate_agent(
    agent,
    env,
    num_episodes: int,
    device: torch.device,
    config: Config,
    args,
    training_trip_loader: Optional[RealTripLoader] = None,
) -> float:
    """Evaluate agent performance."""
    total_reward = 0.0
    total_revenue = 0.0
    total_profit = 0.0
    total_served = 0
    total_loaded = 0

    eval_env = env
    eval_config = config

    # Optional held-out evaluation window using a separate environment.
    data_path = args.real_data or config.data.parquet_path
    if data_path and args.eval_start_date and args.eval_end_date:
        try:
            eval_config = copy.deepcopy(config)
            start_dt = datetime.strptime(args.eval_start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(args.eval_end_date, "%Y-%m-%d")
            if end_dt < start_dt:
                raise ValueError("eval_end_date must be on or after eval_start_date")
            eval_days = max(1, (end_dt - start_dt).days + 1)
            eval_config.episode.duration_hours = float(eval_days * 24)

            sample_ratio = args.trip_sample if args.trip_sample is not None else eval_config.data.trip_percentage
            target_res = args.target_h3_resolution if args.target_h3_resolution is not None else eval_config.data.target_h3_resolution
            max_hex_count = args.max_hex_count if args.max_hex_count is not None else eval_config.data.max_hex_count
            reference_hex_ids = None
            if training_trip_loader is not None and training_trip_loader.is_loaded:
                reference_hex_ids = list(training_trip_loader.hex_ids)

            eval_trip_loader = RealTripLoader(
                parquet_path=str(data_path),
                device=str(device),
                sample_ratio=sample_ratio if sample_ratio is not None else 1.0,
                target_h3_resolution=target_res,
                max_hex_count=max_hex_count,
                start_date=args.eval_start_date,
                end_date=args.eval_end_date,
                reference_hex_ids=reference_hex_ids,
            )
            eval_trip_loader.load()
            if eval_trip_loader.is_loaded:
                eval_config.environment.num_hexes = eval_trip_loader.num_hexes

            expected_hexes = int(getattr(agent, 'num_hexes', eval_config.environment.num_hexes))
            if eval_config.environment.num_hexes != expected_hexes:
                raise ValueError(
                    f"Held-out eval hex count mismatch (eval={eval_config.environment.num_hexes}, agent={expected_hexes}). "
                    "Use training reference hex IDs for held-out eval."
                )

            eval_env = create_environment(eval_config, device, args, trip_loader=eval_trip_loader)
            print(f"  [Eval Data] Held-out window: {args.eval_start_date} -> {args.eval_end_date} ({eval_days} day(s))")
        except Exception as exc:
            print(f"  [Eval Data] Held-out eval setup failed ({exc}); falling back to training env")
            eval_env = env
            eval_config = config

    prev_step_log_suppressed = getattr(eval_env, '_suppress_step_logs', False)
    reward_computer = getattr(eval_env, '_reward_computer', None)
    prev_reward_log_suppressed = getattr(reward_computer, 'suppress_debug_logs', False) if reward_computer is not None else False

    agent.eval()

    algo = getattr(config.training, 'algo', 'sac').lower()

    from gpu_core.training.episode_collector import EpisodeCollector

    if algo == 'ppo':
        # ── MAPPO eval path ─────────────────────────────────────────────
        dummy_buf = PPORolloutBuffer()
        eval_collector = PPOCollector(
            env=eval_env,
            replay_buffer=dummy_buf,
            device=str(device),
            repos_khop=getattr(config.training, 'mappo_repos_khop', 4),
            use_khop_candidates=getattr(config.training, 'mappo_use_khop_candidates', False),
        )
    elif algo == 'maddpg':
        # ── MADDPG eval path ────────────────────────────────────────────
        dummy_maddpg_buf = MADDPGReplayBuffer(
            capacity            = 1,
            num_vehicles        = eval_config.environment.num_vehicles,
            vehicle_feature_dim = eval_env._vehicle_feature_dim,
            context_dim         = eval_env._context_dim,
            action_dim          = 3,
            device              = str(device),
        )
        eval_collector = MADDPGCollector(
            env=eval_env,
            replay_buffer=dummy_maddpg_buf,
            device=str(device),
            repos_khop=getattr(config.training, 'maddpg_repos_khop', 4),
        )
    else:
        # ── SAC eval path ───────────────────────────────────────────────
        class DummyReplayBuffer:
            def push_fleet(self, *args, **kwargs): pass
            def push(self, *args, **kwargs): pass

        eval_collector = EpisodeCollector(
            env=eval_env, replay_buffer=DummyReplayBuffer(), device=str(device)
        )

    eval_env._suppress_step_logs = True
    if reward_computer is not None:
        reward_computer.suppress_debug_logs = True

    eval_episodes = 1 if (args.eval_start_date and args.eval_end_date) else num_episodes
    print(f"  [Eval Start] Running {eval_episodes} evaluation episodes")
    try:
        for ep in range(eval_episodes):
            print(f"    [Eval Episode {ep + 1}/{eval_episodes}]")
            if algo == 'ppo':
                stats = eval_collector.collect_episode(
                    agent=agent, rollout_steps=99999,
                    deterministic=True, seed=ep * 1000,
                )
            elif algo == 'maddpg':
                stats = eval_collector.collect_episode(
                    agent=agent, rollout_steps=99999,
                    deterministic=True, exploration_eps=0.0, seed=ep * 1000,
                )
            else:
                stats = eval_collector.collect_episode_fleet(
                    agent=agent, seed=ep * 1000, temperature=1.0, deterministic=True
                )
            total_reward += stats.total_reward
            total_revenue += stats.revenue
            total_profit += stats.profit
            total_served += stats.trips_served
            total_loaded += stats.trips_loaded
    finally:
        eval_env._suppress_step_logs = prev_step_log_suppressed
        if reward_computer is not None:
            reward_computer.suppress_debug_logs = prev_reward_log_suppressed
        agent.train()

    avg_reward = total_reward / eval_episodes
    avg_revenue = total_revenue / eval_episodes
    avg_profit = total_profit / eval_episodes
    serve_pct = (total_served / max(1, total_loaded)) * 100.0

    print(f"  [Eval Detail] Profit: ${avg_profit:.2f} | Revenue: ${avg_revenue:.2f} | Serve: {serve_pct:.1f}%")
    print("  [Eval End]")
    return avg_reward


def train_distributed(rank: int, world_size: int, args):
    # setup_distributed already called by _distributed_worker_wrapper
    try:
        device = torch.device(f'cuda:{rank}')
        setup_seed(args.seed + rank)
        
        if rank == 0:
            print("=" * 60)
            print("EV Fleet RL Training - Distributed (Multi-GPU)")
            print("=" * 60)
        config = create_config(args)
        
        # Load real trip data if provided
        trip_loader = None
        data_path = args.real_data or config.data.parquet_path
        sample_ratio = args.trip_sample if args.trip_sample is not None else config.data.trip_percentage
        target_res = args.target_h3_resolution if hasattr(args, 'target_h3_resolution') and args.target_h3_resolution is not None else config.data.target_h3_resolution
        max_hex_count = args.max_hex_count if hasattr(args, 'max_hex_count') and args.max_hex_count is not None else config.data.max_hex_count
        if data_path:
            resolved_path = Path(data_path)
            resolved_str = str(resolved_path)
            sample_ratio = sample_ratio if sample_ratio is not None else 1.0
            if rank == 0:
                print(f"\n[Real Data] Loading from: {resolved_str}")
                if sample_ratio < 1.0:
                    print(f"[Real Data] Using {sample_ratio*100:.0f}% sample of trips")
            if resolved_path.exists():
                try:
                    trip_loader = RealTripLoader(
                        parquet_path=resolved_str,
                        device=str(device),
                        sample_ratio=sample_ratio,
                        target_h3_resolution=target_res,
                        max_hex_count=max_hex_count,
                        start_date=args.start_date,
                        end_date=args.end_date,
                    )
                    trip_loader.load()
                except Exception as exc:
                    if rank == 0:
                        print(f"[Real Data] Failed to load: {exc}. Falling back to synthetic data.")
                    trip_loader = None
            if trip_loader and trip_loader.is_loaded:
                config.environment.num_hexes = trip_loader.num_hexes
                if rank == 0:
                    print(f"[Real Data] Loaded {trip_loader.total_trips:,} trips | Hexes: {trip_loader.num_hexes:,}")
        
        env = create_environment(config, device, args, trip_loader=trip_loader)
        if rank == 0:
            print(f"Vehicles: {config.environment.num_vehicles}")
            print(f"Hexes: {config.environment.num_hexes}")
            print(f"Episodes: {config.training.total_episodes}")
        
        agent = create_agent(config, device, env=env)
        
        # Set adjacency matrix and K-hop mask for Fleet actor
        if hasattr(env, '_adjacency_matrix') and env._adjacency_matrix is not None:
            agent.set_adjacency(env._adjacency_matrix)
            
            # Compute K-hop mask for FleetSACAgent
            try:
                k_hops = getattr(config.fleet_actor, 'repos_khop', 4)
                khop_mask_hh = HexNeighbors.compute_khop_mask(env._adjacency_matrix, k=k_hops)
                khop_indices, khop_counts, max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh)
                padded_khop_mask = khop_indices != -1
                agent.set_khop_data(khop_indices, padded_khop_mask)
                if rank == 0:
                    print(f"[GCN] K-hop mask computed (K={k_hops}, max_neighbors={max_k}) and passed to FleetSACAgent")
            except Exception as e:
                if rank == 0:
                    print(f"[GCN] Error computing K-hop mask: {e}")
        
        if rank == 0:
            print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")
        
        replay_buffer = create_replay_buffer(config, device, env=env)
        
        warmup_steps = config.training.warmup_steps
        if warmup_steps > 0 and not args.resume:
            if rank == 0:
                print(f"Warming up replay buffer with {warmup_steps} random steps...")
            warmup_replay_buffer(env, replay_buffer, warmup_steps, device)
        
        # Create trainer - use EnhancedSACTrainer if Semi-MDP or WDRO is enabled
        use_enhanced = args.semi_mdp or args.wdro
        if use_enhanced:
            enhanced_config = EnhancedTrainingConfig(
                use_semi_mdp=args.semi_mdp,
                use_wdro=args.wdro,
                wdro_rho=args.wdro_rho,
                wdro_metric=args.wdro_metric,
                use_temperature_annealing=args.temperature_annealing,
                initial_temperature=args.initial_temperature,
                final_temperature=args.final_temperature,
                temperature_decay_episodes=args.temperature_decay_episodes,
                use_amp=True,
            )
            enhanced_config = apply_enhanced_schedule_args(args, enhanced_config)
            adj_matrix = getattr(env, '_adjacency_matrix', None)
            od_matrix = compute_od_matrix(trip_loader, str(device)) if (args.wdro and args.wdro_metric == 'mag') else None
            if isinstance(agent, FleetSACAgent):
                from gpu_core.training.enhanced_trainer import FleetEnhancedSACTrainer
                TrainerClass = FleetEnhancedSACTrainer
            else:
                from gpu_core.training.enhanced_trainer import EnhancedSACTrainer
                TrainerClass = EnhancedSACTrainer
                
            trainer = TrainerClass(
                agent=agent,
                replay_buffer=replay_buffer,
                training_config=config.training,
                enhanced_config=enhanced_config,
                checkpoint_config=config.checkpoint,
                logging_config=config.logging,
                device=str(device),
                adjacency_matrix=adj_matrix,
                od_matrix=od_matrix
            )
            if rank == 0:
                print(f"[Enhanced] Semi-MDP: {args.semi_mdp}, WDRO: {args.wdro}")
                print(f"[Optimization] Mixed Precision Training (AMP): Enabled")
        else:
            trainer = create_trainer(
                agent, replay_buffer, config, device,
                rank=rank, world_size=world_size
            )
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
            if rank == 0:
                print(f"Resumed from checkpoint: {args.resume}")
        
        # Create collector
        if args.semi_mdp:
            from gpu_core.training.enhanced_collector import EnhancedEpisodeCollector
            collector = EnhancedEpisodeCollector(
                env=env, replay_buffer=replay_buffer, device=str(device),
                use_milp=args.milp, track_durations=True
            )
        else:
            from gpu_core.training.episode_collector import EpisodeCollector
            collector = EpisodeCollector(
                env=env, replay_buffer=replay_buffer, device=str(device),
                use_milp=args.milp
            )
        
        # === CSV Logging (rank 0 only) ===
        csv_file = None
        csv_writer = None
        log_dir = None
        if rank == 0:
            log_dir = os.path.join(config.logging.log_dir, f"run_{int(time.time())}")
            os.makedirs(log_dir, exist_ok=True)
            csv_path = os.path.join(log_dir, "training_history.csv")
            try:
                csv_file = open(csv_path, mode='w', newline='')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([
                    'Episode', 'Reward', 'Avg100', 'Profit', 'Revenue', 'Serve_Pct', 'Steps', 'Trips_Served',
                    'Trips_Loaded', 'Avg_SOC', 'Actor_Loss', 'Critic_Loss', 'Alpha', 'Value_Loss',
                    'WDRO_Lambda', 'WDRO_Rho_Hat', 'WDRO_Target_Mean', 'WDRO_Target_Abs_Max',
                    'WDRO_Distance_Mean', 'WDRO_Worst_Value_Mean', 'WDRO_Phase', 'WDRO_Value_Source_Is_Learned', 'WDRO_Value_Training_Active',
                    'Actor_Serve_Pct', 'Actor_Charge_Pct', 'Actor_Repos_Pct', 'V_Label_Mean', 'V_Pred_Mean',
            'WDRO_Value_Source_Switch_Episode', 'WDRO_Value_Train_Stop_Episode',
                    'Idle_Pct', 'Serve_Act_Pct', 'Charge_Pct', 'Repos_Pct', 'Forced_Charge_Pct', 'Serve_Failed_Pct', 'Elapsed_Sec'
                ])
                print(f"CSV logging to: {csv_path}")
            except Exception as e:
                print(f"Failed to initialize CSV logging: {e}")
        
        # === Tracking ===
        recent_rewards = deque(getattr(trainer, 'recent_rewards_window', []), maxlen=100)
        recent_losses = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'alpha': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'wdro_lambda': deque(maxlen=100),
            'wdro_rho_hat': deque(maxlen=100),
            'wdro_target_mean': deque(maxlen=100),
            'wdro_target_abs_max': deque(maxlen=100),
            'wdro_distance_mean': deque(maxlen=100),
            'wdro_worst_value_mean': deque(maxlen=100),
            'wdro_phase': deque(maxlen=100),
            'wdro_value_source_is_learned': deque(maxlen=100),
            'wdro_value_training_active': deque(maxlen=100),
            'repos_aux_loss': deque(maxlen=100),
            'actor_serve_frac': deque(maxlen=100),
            'actor_charge_frac': deque(maxlen=100),
            'actor_repos_frac': deque(maxlen=100),
            'v_label_mean': deque(maxlen=100),
            'v_pred_mean': deque(maxlen=100),
            'serve_consistency': deque(maxlen=100),
            'charge_consistency': deque(maxlen=100),
            'reposition_consistency': deque(maxlen=100),
            'failed_action_fraction': deque(maxlen=100),
        }
        best_eval_reward = getattr(trainer, 'best_reward', float('-inf'))
        best_train_reward = getattr(trainer, 'best_train_reward', float('-inf'))
        total_steps = getattr(trainer, 'total_steps', 0)
        total_updates = getattr(trainer, 'global_step', 0)
        start_time = time.time()
        
        if rank == 0:
            print("=" * 60)
            print(f"Distributed training with {world_size} GPUs started...")
            print("=" * 60)
        
        num_episodes = config.training.total_episodes
        start_episode = getattr(trainer, 'episode', 0)
        if args.start_episode is not None:
            start_episode = args.start_episode
            trainer.episode = start_episode
        if rank == 0 and start_episode > 0:
            print(f"Resuming training loop from episode {start_episode}")
        action_pct = {}
        
        # Track time per 20 episodes
        episode_batch_start_time = time.time()
        episode_batch_count = 0
        
        for episode in range(start_episode, num_episodes):
            # Track episode in trainer so checkpoints save the correct episode number
            trainer.episode = episode
            
            exploration_noise = getattr(config.training, 'exploration_noise', 0.01)
            
            # Get temperature for this episode (applies to both SAC and WDRO)
            if args.temperature_annealing:
                progress = min(1.0, episode / args.temperature_decay_episodes)
                temperature = args.initial_temperature * (1 - progress) + args.final_temperature * progress
            else:
                temperature = 1.0
            
            # Collect episode
            episode_stats = collector.collect_episode_fleet(
                agent=agent,
                seed=args.seed + episode,
                temperature=temperature,
                deterministic=args.deterministic
            )
            total_steps += episode_stats.steps
            trainer.total_steps = total_steps

            # Check if all ranks have enough data to train
            local_ready = 1 if len(replay_buffer) >= config.training.batch_size else 0
            if args.distributed:
                ready_tensor = torch.tensor([local_ready], dtype=torch.int32, device=device)
                dist.all_reduce(ready_tensor, op=dist.ReduceOp.MIN)
                global_ready = (ready_tensor.item() == 1)
            else:
                global_ready = (local_ready == 1)
                
            # Train on collected data
            if global_ready:
                update_ratio = getattr(config.training, 'update_every', 1)
                local_updates = max(1, episode_stats.steps // update_ratio)
                
                # Make sure all ranks perform the exact same number of updates to prevent DDP deadlocks
                if args.distributed:
                    updates_tensor = torch.tensor([local_updates], dtype=torch.float32, device=device)
                    dist.all_reduce(updates_tensor, op=dist.ReduceOp.MAX)
                    num_updates = int(updates_tensor.item())
                else:
                    num_updates = local_updates
                
                for _ in range(num_updates):
                    if use_enhanced:
                        train_info = trainer.train_step_enhanced()
                    else:
                        train_info = trainer.train_step()
                    total_updates += 1
                    
                    if train_info:
                        for key in ['actor_loss', 'critic_loss', 'alpha', 'value_loss', 'wdro_lambda', 'wdro_rho_hat', 'wdro_target_mean', 'wdro_target_abs_max', 'wdro_distance_mean', 'wdro_worst_value_mean', 'wdro_phase', 'wdro_value_source_is_learned', 'wdro_value_training_active', 'actor_serve_frac', 'actor_charge_frac', 'actor_repos_frac', 'v_label_mean', 'v_pred_mean', 'serve_consistency', 'charge_consistency', 'reposition_consistency', 'failed_action_fraction']:
                            if key in train_info:
                                recent_losses[key].append(train_info[key])
            
            # Track reward
            recent_rewards.append(episode_stats.total_reward)
            episode_batch_count += 1
            trainer.recent_rewards_window = list(recent_rewards)
            
            # Log timing every 20 episodes (rank 0 only)
            if rank == 0 and episode_batch_count == 20:
                batch_elapsed = time.time() - episode_batch_start_time
                print(f"[Timing] Last 20 episodes took {batch_elapsed:.2f}s ({batch_elapsed/20:.2f}s per episode)")
                episode_batch_start_time = time.time()
                episode_batch_count = 0
            
            # === Logging (rank 0 only) ===
            if rank == 0 and (episode + 1) % args.log_interval == 0:
                avg_actor_loss = sum(recent_losses['actor_loss']) / len(recent_losses['actor_loss']) if recent_losses['actor_loss'] else 0
                avg_critic_loss = sum(recent_losses['critic_loss']) / len(recent_losses['critic_loss']) if recent_losses['critic_loss'] else 0
                avg_alpha = sum(recent_losses['alpha']) / len(recent_losses['alpha']) if recent_losses['alpha'] else 0
                avg_value_loss = sum(recent_losses['value_loss']) / len(recent_losses['value_loss']) if recent_losses['value_loss'] else 0
                avg_wdro_lambda = sum(recent_losses['wdro_lambda']) / len(recent_losses['wdro_lambda']) if recent_losses['wdro_lambda'] else 0
                avg_wdro_rho_hat = sum(recent_losses['wdro_rho_hat']) / len(recent_losses['wdro_rho_hat']) if recent_losses['wdro_rho_hat'] else 0
                avg_wdro_target_mean = sum(recent_losses['wdro_target_mean']) / len(recent_losses['wdro_target_mean']) if recent_losses['wdro_target_mean'] else 0
                avg_wdro_target_abs_max = sum(recent_losses['wdro_target_abs_max']) / len(recent_losses['wdro_target_abs_max']) if recent_losses['wdro_target_abs_max'] else 0
                avg_wdro_distance_mean = sum(recent_losses['wdro_distance_mean']) / len(recent_losses['wdro_distance_mean']) if recent_losses['wdro_distance_mean'] else 0
                avg_wdro_worst_value_mean = sum(recent_losses['wdro_worst_value_mean']) / len(recent_losses['wdro_worst_value_mean']) if recent_losses['wdro_worst_value_mean'] else 0
                avg_wdro_phase = round(sum(recent_losses['wdro_phase']) / len(recent_losses['wdro_phase'])) if recent_losses['wdro_phase'] else 0
                avg_wdro_value_source_is_learned = round(sum(recent_losses['wdro_value_source_is_learned']) / len(recent_losses['wdro_value_source_is_learned'])) if recent_losses['wdro_value_source_is_learned'] else 0
                avg_wdro_value_training_active = round(sum(recent_losses['wdro_value_training_active']) / len(recent_losses['wdro_value_training_active'])) if recent_losses['wdro_value_training_active'] else 0
                avg_actor_serve_pct = 100.0 * (sum(recent_losses['actor_serve_frac']) / len(recent_losses['actor_serve_frac'])) if recent_losses['actor_serve_frac'] else 0
                avg_actor_charge_pct = 100.0 * (sum(recent_losses['actor_charge_frac']) / len(recent_losses['actor_charge_frac'])) if recent_losses['actor_charge_frac'] else 0
                avg_actor_repos_pct = 100.0 * (sum(recent_losses['actor_repos_frac']) / len(recent_losses['actor_repos_frac'])) if recent_losses['actor_repos_frac'] else 0
                avg_v_label_mean = sum(recent_losses['v_label_mean']) / len(recent_losses['v_label_mean']) if recent_losses['v_label_mean'] else 0
                avg_v_pred_mean = sum(recent_losses['v_pred_mean']) / len(recent_losses['v_pred_mean']) if recent_losses['v_pred_mean'] else 0
                avg_serve_consistency = sum(recent_losses['serve_consistency']) / len(recent_losses['serve_consistency']) if recent_losses['serve_consistency'] else 0
                avg_charge_consistency = sum(recent_losses['charge_consistency']) / len(recent_losses['charge_consistency']) if recent_losses['charge_consistency'] else 0
                avg_reposition_consistency = sum(recent_losses['reposition_consistency']) / len(recent_losses['reposition_consistency']) if recent_losses['reposition_consistency'] else 0
                avg_failed_action_fraction = sum(recent_losses['failed_action_fraction']) / len(recent_losses['failed_action_fraction']) if recent_losses['failed_action_fraction'] else 0
                avg_reward_100 = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                
                # Action distribution
                action_counts = getattr(episode_stats, 'action_counts', None)
                if action_counts:
                    total_actions = sum(action_counts.values()) or 1
                    action_pct = {k: v * 100.0 / total_actions for k, v in action_counts.items()}
                    action_str = f"S:{action_pct['serve']:4.1f}% C:{action_pct['charge']:4.1f}% R:{action_pct['reposition']:4.1f}%"
                else:
                    action_str = ""
                    action_pct = {}
                
                serve_pct_stat = (episode_stats.trips_served / max(1, episode_stats.trips_loaded)) * 100.0
                print(f"Episode {episode + 1:5d} | "
                    f"Reward: {episode_stats.total_reward:8.2f} | "
                    f"Avg100: {avg_reward_100:8.2f} | "
                    f"Profit: ${episode_stats.profit:6.2f} | "
                    f"Rev: ${episode_stats.revenue:6.2f} | "
                    f"Serve%: {serve_pct_stat:4.1f}% | "
                    f"SOC: {episode_stats.avg_soc:5.1f}%")
                print(f"         Actor: {avg_actor_loss:7.4f} | "
                    f"Critic: {avg_critic_loss:7.4f} | "
                    f"Alpha: {avg_alpha:5.3f} | "
                    f"Actions: {action_str}")
                forced_charge_pct = (100.0 * getattr(episode_stats, 'forced_charge_count', 0) / max(1, getattr(episode_stats, 'forced_charge_total_idle', 0)))
                print(f"         ActorMix: S:{avg_actor_serve_pct:4.1f}% C:{avg_actor_charge_pct:4.1f}% R:{avg_actor_repos_pct:4.1f}% | ForcedCharge: {forced_charge_pct:4.1f}% | VLbl: {avg_v_label_mean:6.2f} | VPred: {avg_v_pred_mean:6.2f}")
                if algo == 'maddpg':
                    print(f"         Consistency: Serve:{100.0*avg_serve_consistency:4.1f}% "
                        f"Charge:{100.0*avg_charge_consistency:4.1f}% "
                        f"Repos:{100.0*avg_reposition_consistency:4.1f}% | "
                        f"FailedActs:{100.0*avg_failed_action_fraction:4.1f}%")
                
                # CSV logging
                if csv_writer and csv_file:
                    idle_pct = action_pct.get('idle', 0)
                    serve_act_pct = action_pct.get('serve', 0)
                    charge_pct = action_pct.get('charge', 0)
                    repos_pct = action_pct.get('reposition', 0)
                    
                    dist_elapsed = time.time() - start_time
                    serve_attempted = int(getattr(episode_stats, 'num_serve_attempted', 0))
                    serve_success = int(getattr(episode_stats, 'num_serve_success', episode_stats.trips_served))
                    serve_failed_pct = (max(0, serve_attempted - serve_success) / max(1, serve_attempted)) * 100.0
                    forced_charge_pct = (100.0 * getattr(episode_stats, 'forced_charge_count', 0) / max(1, getattr(episode_stats, 'forced_charge_total_idle', 0)))
                    csv_writer.writerow([
                        episode + 1,
                        round(episode_stats.total_reward, 2),
                        round(avg_reward_100, 2),
                        round(episode_stats.profit, 2),
                        round(episode_stats.revenue, 2),
                        round(serve_pct_stat, 1),
                        episode_stats.steps,
                        episode_stats.trips_served,
                        episode_stats.trips_loaded,
                        round(episode_stats.avg_soc, 1),
                        round(avg_actor_loss, 4),
                        round(avg_critic_loss, 4),
                        round(avg_alpha, 4),
                        round(avg_value_loss, 4),
                        round(avg_wdro_lambda, 4),
                        round(avg_wdro_rho_hat, 4),
                        round(avg_wdro_target_mean, 4),
                        round(avg_wdro_target_abs_max, 4),
                        round(avg_wdro_distance_mean, 4),
                        round(avg_wdro_worst_value_mean, 4),
                        avg_wdro_phase,
                        avg_wdro_value_source_is_learned,
                        avg_wdro_value_training_active,
                        round(avg_actor_serve_pct, 1),
                        round(avg_actor_charge_pct, 1),
                        round(avg_actor_repos_pct, 1),
                        round(avg_v_label_mean, 4),
                        round(avg_v_pred_mean, 4),
                        args.wdro_value_source_switch_episode,
                        args.wdro_value_train_stop_episode,
                        round(idle_pct, 1),
                        round(serve_act_pct, 1),
                        round(charge_pct, 1),
                        round(repos_pct, 1),
                        round(forced_charge_pct, 1),
                        round(serve_failed_pct, 1),
                        round(dist_elapsed, 1)
                    ])
                    csv_file.flush()
            
            # === Evaluation ===
            if (episode + 1) % args.eval_interval == 0:
                # All ranks must evaluate to keep NCCL in sync
                eval_reward = evaluate_agent(agent, env, args.eval_episodes, device, config, args, trip_loader)
                
                # Make absolutely certain all ranks agree on the exact same eval score
                eval_tensor = torch.tensor([eval_reward], dtype=torch.float32, device=device)
                dist.all_reduce(eval_tensor, op=dist.ReduceOp.SUM)
                agreed_eval_reward = (eval_tensor / world_size).item()
                
                if rank == 0:
                    print(f"  [Eval] Avg Reward: {agreed_eval_reward:.2f}")
                
                # ALL ranks must determine if this is the best eval and save
                if agreed_eval_reward > best_eval_reward:
                    best_eval_reward = agreed_eval_reward
                    trainer.best_reward = best_eval_reward
                    if rank == 0:
                        print(f"  [Best Eval] New best eval model saved!")
                    trainer.save_checkpoint('best_eval.pt')
            
            # === Checkpoint saving ===
            # ALL ranks must compute the average reward to decide whether to save
            # This is critical because save_checkpoint contains a dist.barrier()
            avg_reward_100 = sum(recent_rewards) / len(recent_rewards) if recent_rewards else float('-inf')
            
            # Make sure all ranks agree exactly on the average
            if args.distributed:
                avg_tensor = torch.tensor([avg_reward_100], dtype=torch.float32, device=device)
                dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
                avg_reward_100 = (avg_tensor / world_size).item()
            
            save_best = False
            if len(recent_rewards) >= 50 and avg_reward_100 > best_train_reward:
                best_train_reward = avg_reward_100
                trainer.best_train_reward = best_train_reward
                save_best = True
                if rank == 0:
                    print(f"  [Best Train] New best model saved! Avg100: {avg_reward_100:.2f}")
            
            if save_best:
                trainer.save_checkpoint('best.pt')
            
            if (episode + 1) % args.save_interval == 0:
                trainer.save_checkpoint(f'checkpoint_{episode + 1}.pt')
        
        # === Training complete ===
        if rank == 0:
            elapsed = time.time() - start_time
            print("=" * 60)
            print("Training complete!")
            print(f"Total episodes: {num_episodes}")
            print(f"Total steps: {total_steps:,}")
            print(f"Total time: {elapsed:.1f}s")
            print(f"Best eval reward: {best_eval_reward:.2f}")
            print(f"Best train reward (Avg100): {best_train_reward:.2f}")
            print("=" * 60)
            
        trainer.save_checkpoint('final.pt')
        
        if rank == 0:
            print(f"Final model saved to: {getattr(config, 'checkpoint', type('obj', (object,), {'checkpoint_dir': 'checkpoints'})).checkpoint_dir}/final.pt")
            
            if csv_file:
                csv_file.close()
                print(f"CSV logs saved to: {csv_path}")
            
    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        pass  # cleanup_distributed called by wrapper


def main():
    args = parse_args()
    
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    
    if args.distributed:
        gpus = [int(x) for x in args.gpus.split(',')] if args.gpus else list(range(torch.cuda.device_count()))
        world_size = len(gpus)
        
        print(f"Launching distributed training on GPUs: {gpus}")
        spawn_distributed_training(
            train_distributed,
            world_size=world_size,
            args=(args,)
        )
    else:
        train_single_gpu(args)


if __name__ == '__main__':
    main()
