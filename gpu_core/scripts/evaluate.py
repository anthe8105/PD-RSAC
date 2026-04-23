#!/usr/bin/env python3
"""
Evaluate trained EV Fleet RL Agent on Real Data for a Full Time Span

Usage:
    python evaluate.py \
        --checkpoint checkpoints/best.pt \
        --config config_wdro_mod.yaml \
        --real-data data/nyc_full/trips_processed.parquet \
        --start-date 2009-01-15 \
        --end-date 2009-01-15 \
        --episode-duration-hours 24
"""

import argparse
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import ConfigLoader, Config
from gpu_core.networks.sac import FleetSACAgent
from gpu_core.spatial.grid import HexGrid
from gpu_core.spatial.neighbors import HexNeighbors
from gpu_core.simulator.environment import GPUEnvironmentV2
from gpu_core.data.real_trip_loader import RealTripLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate trained EV Fleet RL Agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides checkpoint config)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of evaluation passes')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Save results to JSON file')

    # Environment + Data args
    parser.add_argument('--num-vehicles', type=int, default=None,
                        help='Number of vehicles in fleet')
    parser.add_argument('--num-hexes', type=int, default=None,
                        help='Number of hexagons in grid')
    parser.add_argument('--episode-duration-hours', type=float, default=None,
                        help='Episode duration in hours (e.g., 720 for a 30-day month)')

    parser.add_argument('--real-data', type=str, default=None,
                        help='Path to real trip data parquet file')
    parser.add_argument('--trip-sample', type=float, default=1.0,
                        help='Sample ratio for trip data (0.0-1.0)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Filter trips from this date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Filter trips until this date (YYYY-MM-DD)')
    parser.add_argument('--target-h3-resolution', type=int, default=None,
                        help='Target H3 resolution')
    parser.add_argument('--max-hex-count', type=int, default=None,
                        help='Maximum number of hexes')
    parser.add_argument('--milp', action='store_true', default=False,
                        help='Enable MILP trip assignment (requires Gurobi)')

    # Eval settings
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use greedy (argmax) action selection (default: True)')
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false',
                        help='Use stochastic sampling instead of greedy argmax')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Softmax temperature for action selection')

    return parser.parse_args()


def create_config(args, checkpoint_path: str) -> Config:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = checkpoint.get('config', {})

    if args.config:
        config = ConfigLoader.from_yaml(args.config)
    else:
        config = Config()

    if args.num_vehicles is not None:
        config.environment.num_vehicles = args.num_vehicles
    elif 'environment' in ckpt_config and 'num_vehicles' in ckpt_config['environment']:
        config.environment.num_vehicles = ckpt_config['environment']['num_vehicles']

    if args.num_hexes is not None:
        config.environment.num_hexes = args.num_hexes
    elif 'environment' in ckpt_config and 'num_hexes' in ckpt_config['environment']:
        config.environment.num_hexes = ckpt_config['environment']['num_hexes']

    if args.start_date and args.end_date:
        try:
            from datetime import datetime
            start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            days = max(1, (end_dt - start_dt).days + 1)
            calculated_duration = float(days * 24)
            config.episode.duration_hours = calculated_duration
            print(f"[Auto-Detect] Set episode duration to {config.episode.duration_hours} hours ({days} days)")
            if args.episode_duration_hours is not None and args.episode_duration_hours != calculated_duration:
                print(f"[Warning] Overriding requested --episode-duration-hours {args.episode_duration_hours} with calculated duration {calculated_duration}")
        except Exception as e:
            print(f"[Auto-Detect] Failed to parse dates for duration: {e}")
            if args.episode_duration_hours is not None:
                config.episode.duration_hours = args.episode_duration_hours
            elif 'episode' in ckpt_config and 'duration_hours' in ckpt_config['episode']:
                config.episode.duration_hours = ckpt_config['episode']['duration_hours']
    elif args.episode_duration_hours is not None:
        config.episode.duration_hours = args.episode_duration_hours
    elif 'episode' in ckpt_config and 'duration_hours' in ckpt_config['episode']:
        config.episode.duration_hours = ckpt_config['episode']['duration_hours']

    return config


def detect_algo(checkpoint_path: str, config: Config) -> str:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    algo = str(getattr(config.training, 'algo', 'sac')).lower()

    agent_sd = checkpoint.get('agent', checkpoint.get('agent_state_dict', {}))
    if isinstance(agent_sd, dict):
        if any(k.startswith('actor.gcn.') or k.startswith('actor.module.gcn.') for k in agent_sd.keys()):
            return 'sac'
        if any(k.startswith('actor.backbone.') or k.startswith('critic.backbone.') for k in agent_sd.keys()):
            return 'ppo'

    if 'actor_state_dict' in checkpoint and 'critic_state_dict' in checkpoint:
        return 'maddpg'

    return algo


def create_environment(config: Config, device: torch.device, args, trip_loader: Optional[RealTripLoader] = None):
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
        grid_size = int(num_hexes ** 0.5) + 1
        fake_hex_ids = [f"hex_{i}" for i in range(num_hexes)]
        hex_grid._hex_ids = fake_hex_ids
        hex_grid._hex_to_idx = {h: i for i, h in enumerate(fake_hex_ids)}
        base_lat, base_lon = 40.7128, -74.0060
        lat_per_km, lon_per_km = 0.009, 0.012
        lats = torch.zeros(num_hexes, device=device)
        lons = torch.zeros(num_hexes, device=device)
        for i in range(num_hexes):
            row = i // grid_size
            col = i % grid_size
            lats[i] = base_lat + row * 0.5 * lat_per_km
            lons[i] = base_lon + col * 0.5 * lon_per_km
        hex_grid._latitudes = lats
        hex_grid._longitudes = lons
        hex_grid._initialized = True
        lat_diff = lats.unsqueeze(1) - lats.unsqueeze(0)
        lon_diff = lons.unsqueeze(1) - lons.unsqueeze(0)
        lat_km = lat_diff / lat_per_km
        lon_km = lon_diff / lon_per_km
        distances = torch.sqrt(lat_km**2 + lon_km**2)
        distances.fill_diagonal_(0)
        hex_grid.distance_matrix._distances = distances
        hex_grid.distance_matrix._num_hexes = num_hexes

    env = GPUEnvironmentV2(
        config=config,
        hex_grid=hex_grid,
        trip_loader=trip_loader,
        device=device_str
    )

    return env


def _init_temporal_buckets():
    action_counts = defaultdict(int)
    hourly_metrics = defaultdict(lambda: {'revenue': 0.0, 'driving_cost': 0.0, 'energy_cost': 0.0})
    daily_metrics = defaultdict(lambda: {'revenue': 0.0, 'driving_cost': 0.0, 'energy_cost': 0.0})
    return action_counts, hourly_metrics, daily_metrics


def _accumulate_temporal_step(
    step: int,
    step_duration_minutes: float,
    cur_revenue: float,
    cur_driving_cost: float,
    cur_energy_cost: float,
    prev_env_revenue: float,
    prev_env_driving_cost: float,
    prev_env_energy_cost: float,
    hourly_metrics,
    daily_metrics,
):
    step_revenue = cur_revenue - prev_env_revenue
    step_serve_cost = cur_driving_cost - prev_env_driving_cost
    step_charge_cost = cur_energy_cost - prev_env_energy_cost

    current_hour = int((step * step_duration_minutes) // 60) % 24
    hourly_metrics[current_hour]['revenue'] += step_revenue
    hourly_metrics[current_hour]['driving_cost'] += step_serve_cost
    hourly_metrics[current_hour]['energy_cost'] += step_charge_cost

    current_day = int((step * step_duration_minutes) // (60 * 24))
    daily_metrics[current_day]['revenue'] += step_revenue
    daily_metrics[current_day]['driving_cost'] += step_serve_cost
    daily_metrics[current_day]['energy_cost'] += step_charge_cost

    return step_revenue, step_serve_cost, step_charge_cost


def _format_temporal_metrics(metrics_bucket):
    out = {}
    for idx in sorted(metrics_bucket.keys()):
        data = metrics_bucket[idx]
        net_profit = data['revenue'] - data['driving_cost'] - data['energy_cost']
        out[int(idx)] = {
            'revenue': float(data['revenue']),
            'driving_cost': float(data['driving_cost']),
            'energy_cost': float(data['energy_cost']),
            'net_profit': float(net_profit),
        }
    return out


def _build_result_common(
    ep: int,
    metrics: dict,
    action_counts: dict,
    elapsed_time: float,
    steps_completed: int,
    hourly_metrics,
    daily_metrics,
    extra: Optional[dict] = None,
):
    trips_served = metrics.get('trips_served', 0)
    trips_loaded = metrics.get('trips_loaded', 0)
    service_rate = float(trips_served) / max(float(trips_loaded), 1.0)

    result = {
        'episode': ep,
        'total_trips_loaded': int(trips_loaded),
        'total_trips_served': int(trips_served),
        'total_trips_dropped': int(metrics.get('trips_dropped', 0)),
        'service_rate': float(service_rate),
        'total_revenue': float(metrics.get('revenue', 0.0)),
        'total_driving_cost': float(metrics.get('driving_cost', 0.0)),
        'total_charging_cost': float(metrics.get('energy_cost', 0.0)),
        'net_profit': float(metrics.get('revenue', 0.0) - metrics.get('driving_cost', 0.0) - metrics.get('energy_cost', 0.0)),
        'final_avg_soc': float(metrics.get('avg_soc', 0.0)),
        'action_counts': dict(action_counts),
        'simulation_time_seconds': float(elapsed_time),
        'steps_completed': int(steps_completed),
        'hourly_metrics': _format_temporal_metrics(hourly_metrics),
        'daily_metrics': _format_temporal_metrics(daily_metrics),
    }
    if extra:
        result.update(extra)
    return result


def _print_result_common(result: dict, config: Config):
    print("\n--- Episode Summary ---")
    if 'total_reward' in result:
        print(f" Reward: {result['total_reward']:.2f}")
    print(f" Revenue: ${result['total_revenue']:.2f}")
    print(f" Costs: ${result['total_driving_cost'] + result['total_charging_cost']:.2f} "
          f"(Drive: ${result['total_driving_cost']:.2f}, Charge: ${result['total_charging_cost']:.2f})")
    print(f" Net Profit: ${result['net_profit']:.2f}")
    print(f" Trips: Served {result['total_trips_served']} / Loaded {result['total_trips_loaded']} "
          f"(Rate: {result['service_rate']*100:.1f}%)")
    print(f" Action Counts: {result['action_counts']}")

    if 'daily_metrics' in result and result['daily_metrics'] and config.episode.duration_hours > 48.0:
        print("\nDaily metrics:")
        for day in sorted(result['daily_metrics'].keys()):
            day_data = result['daily_metrics'][day]
            print(f"  Day {day+1:2d}: Revenue=${day_data['revenue']:>10,.2f}, "
                  f"Net Profit=${day_data['net_profit']:>10,.2f}")


def _print_timing(total_elapsed: float, episodes: int):
    print(f"\n{'='*60}")
    print(f"[Timing] Total inference time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
    if episodes > 1:
        print(f"[Timing] Average per episode: {total_elapsed/episodes:.2f}s")
    print(f"{'='*60}")


def _save_results(results: list, save_path: Optional[str]):
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {save_path}")


def load_fleet_agent(
    checkpoint_path: str,
    config: Config,
    env,
    device: torch.device,
) -> FleetSACAgent:
    """Load FleetSACAgent from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    env_config = config.environment
    entropy_config = config.entropy
    fleet_actor_config = config.fleet_actor
    vehicle_config = config.vehicle

    # Feature dims (match train.py logic)
    vehicle_feature_dim = getattr(env, '_fleet_vehicle_feature_dim',
                                  getattr(env, '_vehicle_feature_dim', 13))
    hex_feature_dim = getattr(env, '_hex_feature_dim', 5)
    context_dim = getattr(env, '_context_dim', 9)

    # Extract max_K_neighbors from checkpoint (must match repos_target_head shape)
    raw_state_dict = checkpoint.get('agent', checkpoint.get('agent_state_dict', {}))
    max_k_neighbors = None
    for key in ('actor.repos_target_head.weight', 'actor.module.repos_target_head.weight'):
        if key in raw_state_dict:
            max_k_neighbors = raw_state_dict[key].shape[0]
            print(f"[Agent Setup] max_K_neighbors={max_k_neighbors} (from checkpoint)")
            break
    eval_adj_max_k = None
    if hasattr(env, '_adjacency_matrix') and env._adjacency_matrix is not None:
        try:
            k_hops_eval = getattr(fleet_actor_config, 'repos_khop', 4)
            khop_mask_hh_eval = HexNeighbors.compute_khop_mask(env._adjacency_matrix, k=k_hops_eval)
            _, _, eval_adj_max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh_eval)
            print(f"[Agent Setup] eval adjacency max_K_neighbors={eval_adj_max_k} (K={k_hops_eval}, num_hexes={env.num_hexes})")
        except Exception as e:
            print(f"[Agent Setup] Could not compute eval adjacency max_K_neighbors: {e}")

    if max_k_neighbors is None:
        # Fallback: compute from adjacency
        max_k_neighbors = 61
        if eval_adj_max_k is not None:
            max_k_neighbors = int(eval_adj_max_k)
            print(f"[Agent Setup] max_K_neighbors={max_k_neighbors} (from adjacency fallback)")
        else:
            print(f"[Agent Setup] max_K_neighbors={max_k_neighbors} (default fallback)")
    elif eval_adj_max_k is not None and int(eval_adj_max_k) != int(max_k_neighbors):
        print(f"[Agent Setup] checkpoint max_K={max_k_neighbors} differs from eval adjacency max_K={int(eval_adj_max_k)}")

    action_dim = 3
    target_entropy = -getattr(entropy_config, 'target_entropy_ratio', 0.5) * \
        torch.log(torch.tensor(float(action_dim))).item()

    actor_hidden_dims = config.network.actor_hidden_dims if hasattr(config.network, 'actor_hidden_dims') else [256, 256]
    critic_hidden_dims = config.network.critic_hidden_dims if hasattr(config.network, 'critic_hidden_dims') else [256, 256]
    dropout_rate = getattr(config.network, 'dropout', 0.1)

    # Infer gcn_hidden_dim from checkpoint weights to avoid shape mismatch when
    # the config used at eval time differs from the one used during training.
    gcn_hidden_dim = actor_hidden_dims[0] if actor_hidden_dims else 128
    for key in ('actor.gcn.layers.0.linear.weight', 'actor.module.gcn.layers.0.linear.weight'):
        if key in raw_state_dict:
            gcn_hidden_dim = raw_state_dict[key].shape[0]
            print(f"[Agent Setup] gcn_hidden_dim={gcn_hidden_dim} (inferred from checkpoint)")
            break

    agent = FleetSACAgent(
        num_hexes=env_config.num_hexes,
        num_vehicles=env_config.num_vehicles,
        vehicle_feature_dim=vehicle_feature_dim,
        hex_feature_dim=hex_feature_dim,
        hex_vehicle_agg_dim=getattr(fleet_actor_config, 'hex_vehicle_agg_dim', 8),
        context_dim=context_dim,
        action_dim=action_dim,
        max_K_neighbors=max_k_neighbors,
        gcn_hidden_dim=gcn_hidden_dim,
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
    )

    # Load state dict (handle key format variants)
    raw_state_dict = checkpoint.get('agent', checkpoint.get('agent_state_dict', None))
    if raw_state_dict is None:
        raise ValueError('SAC checkpoint missing agent state_dict')

    # Strip DDP 'module.' prefixes
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        clean_key = k.replace('.module.', '.') if '.module.' in k else k
        clean_state_dict[clean_key] = v

    model_state = agent.state_dict()
    shape_mismatches = [
        (k, tuple(v.shape), tuple(model_state[k].shape))
        for k, v in clean_state_dict.items()
        if k in model_state and model_state[k].shape != v.shape
    ]
    missing_keys = [k for k in model_state.keys() if k not in clean_state_dict]
    unexpected_keys = [k for k in clean_state_dict.keys() if k not in model_state]

    if shape_mismatches or missing_keys or unexpected_keys:
        details = []
        if shape_mismatches:
            details.append('[SAC] shape mismatches:')
            details.extend([f'  - {k}: ckpt={ck} model={md}' for k, ck, md in shape_mismatches[:20]])
        if missing_keys:
            details.append(f"[SAC] missing keys in checkpoint: {len(missing_keys)}")
            details.extend([f'  - {k}' for k in missing_keys[:20]])
        if unexpected_keys:
            details.append(f"[SAC] unexpected keys in checkpoint: {len(unexpected_keys)}")
            details.extend([f'  - {k}' for k in unexpected_keys[:20]])
        raise RuntimeError('Incompatible SAC checkpoint for strict evaluation load:\n' + '\n'.join(details))

    agent.load_state_dict(clean_state_dict, strict=True)
    print(f"[Checkpoint] Loaded {len(clean_state_dict)} keys from checkpoint (strict)")

    # Set adjacency and K-hop data
    if hasattr(env, '_adjacency_matrix') and env._adjacency_matrix is not None:
        agent.set_adjacency(env._adjacency_matrix)
        print(f"[GCN] Adjacency matrix loaded into agent")

        try:
            k_hops = getattr(fleet_actor_config, 'repos_khop', 4)
            khop_mask_hh = HexNeighbors.compute_khop_mask(env._adjacency_matrix, k=k_hops)
            khop_indices, khop_counts, max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh)
            # Pad or truncate to match checkpoint max_K (handles coordinate-driven differences)
            if max_k < max_k_neighbors:
                pad = torch.full((khop_indices.shape[0], max_k_neighbors - max_k), -1,
                                 dtype=torch.long, device=khop_indices.device)
                khop_indices = torch.cat([khop_indices, pad], dim=1)
                print(f"[GCN] K-hop padded {max_k} -> {max_k_neighbors}")
            elif max_k > max_k_neighbors:
                khop_indices = khop_indices[:, :max_k_neighbors]
                print(f"[GCN] K-hop truncated {max_k} -> {max_k_neighbors}")
            padded_khop_mask = khop_indices != -1
            agent.set_khop_data(khop_indices, padded_khop_mask)
            print(f"[GCN] K-hop mask computed (K={k_hops}, env_max_K={max_k}, agent_max_K={max_k_neighbors})")
        except Exception as e:
            print(f"[GCN] Error computing K-hop mask: {e}")
    else:
        # Compute adjacency from distance matrix
        print("[GCN] Computing missing adjacency matrix...")
        distance_matrix = env.hex_grid.distance_matrix._distances
        adjacency_threshold = 3.0
        adj = (distance_matrix < adjacency_threshold).float()
        adj = adj + torch.eye(env.num_hexes, device=device)
        adj = torch.clamp(adj, 0, 1)

        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)

        adj_matrix = D_inv_sqrt @ adj @ D_inv_sqrt
        agent.set_adjacency(adj_matrix)
        print(f"[GCN] Manual adjacency matrix loaded into agent")

        try:
            k_hops = getattr(fleet_actor_config, 'repos_khop', 4)
            khop_mask_hh = HexNeighbors.compute_khop_mask(adj_matrix, k=k_hops)
            khop_indices, khop_counts, max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh)
            if max_k < max_k_neighbors:
                pad = torch.full((khop_indices.shape[0], max_k_neighbors - max_k), -1,
                                 dtype=torch.long, device=khop_indices.device)
                khop_indices = torch.cat([khop_indices, pad], dim=1)
            elif max_k > max_k_neighbors:
                khop_indices = khop_indices[:, :max_k_neighbors]
            padded_khop_mask = khop_indices != -1
            agent.set_khop_data(khop_indices, padded_khop_mask)
            print(f"[GCN] K-hop mask computed (K={k_hops}, env_max_K={max_k}, agent_max_K={max_k_neighbors})")
        except Exception as e:
            print(f"[GCN] Error computing K-hop mask: {e}")

    agent.to(device)
    agent.eval()
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    return agent


def evaluate(args):
    print("=" * 60)
    print("EV Fleet RL Agent Evaluation (Fleet Actor)")
    print("=" * 60)

    total_start_time = time.time()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_config = checkpoint.get('config', {})
    train_episode_hours = None
    if isinstance(ckpt_config, dict):
        train_episode_hours = ckpt_config.get('episode', {}).get('duration_hours')

    config = create_config(args, args.checkpoint)
    algo = detect_algo(args.checkpoint, config)
    print(f"[Eval] Detected algorithm: {algo}")

    # Setup data
    trip_loader = None
    data_path = args.real_data or config.data.parquet_path
    if data_path:
        resolved_path = Path(data_path)
        print(f"\n[Real Data] Loading from: {resolved_path}")
        if not resolved_path.exists():
            print(f"[Real Data] File not found: {resolved_path}. Falling back to synthetic.")
        else:
            try:
                trip_loader = RealTripLoader(
                    parquet_path=str(resolved_path),
                    device=str(device),
                    sample_ratio=args.trip_sample,
                    target_h3_resolution=args.target_h3_resolution,
                    max_hex_count=args.max_hex_count,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
                trip_loader.load()
            except Exception as exc:
                print(f"[Real Data] Failed to load due to: {exc}. Falling back to synthetic data.")
                trip_loader = None

    # Create environment
    env = create_environment(config, device, args, trip_loader=trip_loader)

    # Keep long-run demand loading but normalize time features to the training
    # episode horizon so policy/context inputs stay in-distribution.
    if train_episode_hours is not None and train_episode_hours > 0:
        train_episode_steps = int(train_episode_hours * 60 / config.episode.step_duration_minutes)
        if train_episode_steps > 0 and train_episode_steps != env.episode_steps:
            env._feature_norm_steps = train_episode_steps
            print(f"[Eval] Using training-horizon normalization: {train_episode_hours}h ({train_episode_steps} steps)")

    print(f"\n[Environment] Created GPUEnvironmentV2 with {env.num_vehicles} vehicles and {env.num_hexes} hexes")
    print(f"[Environment] Time span: {config.episode.duration_hours} hours")

    print(f"Loading checkpoint: {args.checkpoint}")
    eval_temperature = float(args.temperature)
    print(f"[Temperature] {eval_temperature:.4f}")

    if algo != 'sac':
        raise ValueError(
            f"evaluate.py is SAC-only now (detected algo={algo}). "
            f"Use gpu_core/scripts/evaluate_mappo.py for MAPPO checkpoints."
        )

    agent = load_fleet_agent(args.checkpoint, config, env, device)

    # Eval override: treat long-idle as >= 1 hour (instead of the default 2h).
    # With 5-min steps this is 12 steps.
    idle_repos_steps = max(1, int(round(60.0 / config.episode.step_duration_minutes)))
    agent.assigner.idle_force_charge_steps = idle_repos_steps
    print(f"[Override] Long-idle reposition threshold: {idle_repos_steps} steps (~1 hour)")

    # Optionally set up MILP assigner
    milp_assigner = None
    if args.milp:
        try:
            from gpu_core.training.milp_assignment import MILPAssignment
            step_hours = config.episode.step_duration_minutes / 60.0
            milp_assigner = MILPAssignment(
                num_vehicles=env.num_vehicles,
                num_hexes=env.num_hexes,
                num_stations=env.station_state.num_stations if hasattr(env.station_state, 'num_stations') else config.environment.num_stations,
                device=str(device),
                delta_t=step_hours,
                mu=4.0,
                p_max_s=config.physics.charge_power_kw,
                p_min=20.0,
                p_max_feed=7000.0,
                charge_action_penalty=0.2,
                lambda_power=0.02,
                max_pickup_distance=getattr(env, 'max_pickup_distance', 5.0),
            )
            env.set_feeder_power_limit(7000.0)
            print(f"[MILP] Assigner initialised (delta_t={step_hours:.4f}h, p_max_feed=7000.0kW, p_min=20.0kW, charge_action_penalty=0.2, lambda_power=0.02)")
            print("[MILP] Runtime feeder cap in env: 7000.0kW")
        except ImportError as e:
            print(f"[MILP] Could not import MILPAssignment ({e}); running without MILP")

    print("=" * 60)

    all_results = []

    for ep in range(args.episodes):
        print(f"\nEvaluating Episode {ep + 1}/{args.episodes}...")

        action_counts, hourly_metrics, daily_metrics = _init_temporal_buckets()

        state = env.reset()
        max_steps = config.episode.steps_per_episode
        step_duration_minutes = config.episode.step_duration_minutes

        prev_revenue = 0.0
        prev_driving_cost = 0.0
        prev_energy_cost = 0.0

        prev_env_revenue = 0.0
        prev_env_driving_cost = 0.0
        prev_env_energy_cost = 0.0

        log_interval_steps = 20
        interval_revenue = 0.0
        interval_driving_cost = 0.0
        interval_energy_cost = 0.0

        # Cumulative failure counters (env.episode_info is overwritten each step)
        cumulative_charge_failed = 0
        cumulative_serve_failed = 0
        interval_charge_failed = 0
        interval_serve_failed = 0

        done = False
        step = 0
        start_time = time.time()

        while not done and step < max_steps:
            with torch.no_grad():
                # Get fleet policy inputs (matches training collector)
                policy_inputs = env.get_fleet_policy_inputs()

                # Fleet actor → hex allocations → HexVehicleAssigner → per-vehicle actions
                select_kwargs = dict(
                    hex_features=policy_inputs['hex_features'],
                    hex_vehicle_summary=policy_inputs['hex_vehicle_summary'],
                    context_features=policy_inputs['context_features'],
                    vehicle_hex_ids=policy_inputs['vehicle_hex_ids'],
                    vehicle_socs=policy_inputs['vehicle_socs'],
                    vehicle_status=policy_inputs['vehicle_status'],
                    temperature=eval_temperature,
                    deterministic=args.deterministic,
                )
                idle_steps = policy_inputs.get('idle_steps', None)
                if idle_steps is not None:
                    try:
                        fleet_out = agent.select_action_fleet(idle_steps=idle_steps, **select_kwargs)
                    except TypeError as exc:
                        if "idle_steps" not in str(exc):
                            raise
                        fleet_out = agent.select_action_fleet(**select_kwargs)
                else:
                    fleet_out = agent.select_action_fleet(**select_kwargs)

                # MILP projection: replace per-vehicle actions with MILP-optimal assignments
                if milp_assigner is not None:
                    unassigned = env.trip_state.get_unassigned_mask()
                    if unassigned.any():
                        trip_indices = unassigned.nonzero(as_tuple=True)[0]
                        trip_pickups = env.trip_state.pickup_hex[trip_indices]
                        trip_dropoffs = env.trip_state.dropoff_hex[trip_indices]
                        trip_fares = env.trip_state.fare[trip_indices]
                    else:
                        trip_indices = torch.empty(0, dtype=torch.long, device=device)
                        trip_pickups = torch.empty(0, dtype=torch.long, device=device)
                        trip_dropoffs = torch.empty(0, dtype=torch.long, device=device)
                        trip_fares = torch.empty(0, dtype=torch.float32, device=device)

                    available_mask = env.fleet_state.get_available_mask(env.current_step)
                    result = milp_assigner.assign_from_fleet(
                        vehicle_positions=env.fleet_state.positions.long(),
                        vehicle_socs=env.fleet_state.socs,
                        vehicle_status=env.fleet_state.status,
                        trip_pickups=trip_pickups,
                        trip_dropoffs=trip_dropoffs,
                        trip_fares=trip_fares,
                        distance_matrix=env.hex_grid.distance_matrix._distances,
                        allocation_probs=fleet_out.allocation_probs,
                        repos_sampled_targets=fleet_out.repos_sampled_targets,
                        charge_power=fleet_out.charge_power,
                        current_vehicle_charge_power=env.fleet_state.charge_power,
                        available_mask=available_mask,
                        current_step=env.current_step,
                        episode_steps=env.episode_steps,
                    )
                    milp_serve_trip_ids = torch.full(
                        (env.num_vehicles,), -1, dtype=torch.long, device=device
                    )
                    serve_mask = result.action_types == 0
                    if serve_mask.any() and unassigned.any():
                        serve_vehicle_indices = serve_mask.nonzero(as_tuple=True)[0]
                        milp_trip_pool_indices = result.serve_targets[serve_mask]
                        valid = (milp_trip_pool_indices >= 0) & (milp_trip_pool_indices < len(trip_indices))
                        if valid.any():
                            milp_serve_trip_ids[serve_vehicle_indices[valid]] = trip_indices[milp_trip_pool_indices[valid]]

                    from gpu_core.networks.sac import FleetSACOutput
                    fleet_out = FleetSACOutput(
                        action_type=result.action_types,
                        reposition_target=result.reposition_targets,
                        vehicle_charge_power=(
                            result.vehicle_charge_power
                            if result.vehicle_charge_power is not None
                            else fleet_out.vehicle_charge_power
                        ),
                        allocation_probs=fleet_out.allocation_probs,
                        allocation_log_probs=fleet_out.allocation_log_probs,
                        repos_sampled_targets=fleet_out.repos_sampled_targets,
                        charge_power=fleet_out.charge_power,
                        charge_power_log_prob=fleet_out.charge_power_log_prob,
                        allocation_entropy=fleet_out.allocation_entropy,
                        active_hex_mask=fleet_out.active_hex_mask,
                        hex_embeddings=fleet_out.hex_embeddings,
                        forced_charge_count=fleet_out.forced_charge_count,
                        forced_charge_total_idle=fleet_out.forced_charge_total_idle,
                        forced_reposition_count=getattr(fleet_out, 'forced_reposition_count', 0),
                        milp_serve_trip_ids=milp_serve_trip_ids,
                    )
                else:
                    milp_serve_trip_ids = None

                action_type = fleet_out.action_type          # [V]
                reposition_target = fleet_out.reposition_target  # [V]

            # Count actions for available vehicles only
            available_mask = env.fleet_state.get_available_mask(env.current_step)
            action_names = ['SERVE', 'CHARGE', 'REPOSITION']
            if available_mask is not None and available_mask.any():
                available_actions = action_type[available_mask].cpu().numpy()
            else:
                available_actions = action_type.cpu().numpy()
            for action_id in available_actions:
                if 0 <= action_id < 3:
                    action_counts[action_names[action_id]] += 1

            # Late-run diagnostics: check reachability bottlenecks without resetting state
            should_log_this_step = ((step + 1) % log_interval_steps == 0) or ((step + 1) == max_steps)
            diag_snapshot = None
            if should_log_this_step:
                unassigned_mask = env.trip_state.get_unassigned_mask()
                unassigned_count = int(unassigned_mask.sum().item())
                unique_pickup_hex_count = 0
                idle_available_count = int(((env.fleet_state.status == 0) & available_mask).sum().item())
                idle_with_reachable_trip = 0
                serve_attempted_count = int(((action_type == 0) & available_mask).sum().item())
                serve_attempt_no_reachable = 0
                serve_reachable_energy_infeasible = 0
                serve_reachable_energy_feasible = 0

                if unassigned_count > 0:
                    unassigned_trip_idx = unassigned_mask.nonzero(as_tuple=True)[0]
                    pickup_hexes = env.trip_state.pickup_hex[unassigned_trip_idx].long()
                    trip_distances = env.trip_state.distance_km[unassigned_trip_idx].float()
                    unique_pickup_hexes = pickup_hexes.unique()
                    unique_pickup_hex_count = int(unique_pickup_hexes.numel())
                    dist_mat = env.hex_grid.distance_matrix._distances
                    pickup_radius = float(getattr(env, 'max_pickup_distance', 5.0))

                    idle_available_idx = (((env.fleet_state.status == 0) & available_mask).nonzero(as_tuple=True)[0])
                    if idle_available_idx.numel() > 0:
                        idle_positions = env.fleet_state.positions[idle_available_idx].long()
                        idle_to_pickup = dist_mat[idle_positions.unsqueeze(1), unique_pickup_hexes.unsqueeze(0)]
                        idle_with_reachable_trip = int((idle_to_pickup <= pickup_radius).any(dim=1).sum().item())

                    serve_attempt_idx = (((action_type == 0) & available_mask).nonzero(as_tuple=True)[0])
                    if serve_attempt_idx.numel() > 0:
                        serve_positions = env.fleet_state.positions[serve_attempt_idx].long()
                        serve_socs = env.fleet_state.socs[serve_attempt_idx].float()

                        pickup_dist = dist_mat[serve_positions.unsqueeze(1), pickup_hexes.unsqueeze(0)]
                        reachable_mask = pickup_dist <= pickup_radius
                        serve_has_reachable = reachable_mask.any(dim=1)
                        serve_attempt_no_reachable = int((~serve_has_reachable).sum().item())

                        if serve_has_reachable.any():
                            total_dist = pickup_dist + trip_distances.unsqueeze(0)
                            energy_needed = env.energy_dynamics.compute_consumption(total_dist)
                            reserve = float(env.energy_dynamics.min_soc_reserve)
                            available_soc = (serve_socs - reserve).clamp(min=0.0).unsqueeze(1)
                            energy_ok = available_soc >= energy_needed
                            reachable_and_energy_ok = (reachable_mask & energy_ok).any(dim=1)

                            serve_reachable_energy_feasible = int((serve_has_reachable & reachable_and_energy_ok).sum().item())
                            serve_reachable_energy_infeasible = int((serve_has_reachable & ~reachable_and_energy_ok).sum().item())
                elif serve_attempted_count > 0:
                    serve_attempt_no_reachable = serve_attempted_count

                context_features = policy_inputs['context_features']
                progress = float(context_features[0].item())
                remaining = float(context_features[1].item())
                norm_trips = float(context_features[7].item())
                norm_fare = float(context_features[8].item())

                demand_feature = policy_inputs['hex_features'][:, 2]
                demand_saturated_ratio = float((demand_feature >= 0.999).float().mean().item())
                if fleet_out.active_hex_mask is not None and fleet_out.active_hex_mask.any():
                    active_demand = demand_feature[fleet_out.active_hex_mask]
                    active_demand_saturated_ratio = float((active_demand >= 0.999).float().mean().item())
                else:
                    active_demand_saturated_ratio = 0.0

                idle_mask_all = (env.fleet_state.status == 0) & available_mask
                idle_idx = idle_mask_all.nonzero(as_tuple=True)[0]
                idle_long_count = 0
                idle_long_repos = 0
                idle_long_serve = 0
                idle_long_charge = 0
                idle_p50 = 0.0
                idle_p90 = 0.0
                if idle_idx.numel() > 0:
                    idle_steps_all = env.fleet_state.idle_steps[idle_idx].float()
                    idle_p50 = float(torch.quantile(idle_steps_all, 0.5).item())
                    idle_p90 = float(torch.quantile(idle_steps_all, 0.9).item())
                    long_idle_mask = idle_steps_all >= float(agent.assigner.idle_force_charge_steps)
                    idle_long_count = int(long_idle_mask.sum().item())
                    if idle_long_count > 0:
                        long_idle_idx = idle_idx[long_idle_mask]
                        long_idle_actions = action_type[long_idle_idx]
                        idle_long_serve = int((long_idle_actions == 0).sum().item())
                        idle_long_charge = int((long_idle_actions == 1).sum().item())
                        idle_long_repos = int((long_idle_actions == 2).sum().item())

                diag_snapshot = {
                    'unassigned': unassigned_count,
                    'pickup_hexes': unique_pickup_hex_count,
                    'idle_available': idle_available_count,
                    'idle_reachable': idle_with_reachable_trip,
                    'serve_attempted': serve_attempted_count,
                    'serve_no_reachable': serve_attempt_no_reachable,
                    'serve_reachable_energy_infeasible': serve_reachable_energy_infeasible,
                    'serve_reachable_energy_feasible': serve_reachable_energy_feasible,
                    'progress': progress,
                    'remaining': remaining,
                    'norm_trips': norm_trips,
                    'norm_fare': norm_fare,
                    'demand_saturated_ratio': demand_saturated_ratio,
                    'active_demand_saturated_ratio': active_demand_saturated_ratio,
                    'idle_long_count': idle_long_count,
                    'idle_long_repos': idle_long_repos,
                    'idle_long_serve': idle_long_serve,
                    'idle_long_charge': idle_long_charge,
                    'idle_p50': idle_p50,
                    'idle_p90': idle_p90,
                }

            # Diagnostic: allocation distribution and entropy
            if step % log_interval_steps == 0:
                alloc_probs = fleet_out.allocation_probs  # [H, 3]
                active = fleet_out.active_hex_mask        # [H]
                if active is not None and active.any():
                    active_alloc = alloc_probs[active]    # [N_active, 3]
                    mean_alloc = active_alloc.mean(dim=0)
                    entropy = fleet_out.allocation_entropy[active].mean().item()
                    print(f"  [Policy] Active hexes: {active.sum().item()} | "
                          f"Mean alloc: SERVE={mean_alloc[0]:.3f} CHARGE={mean_alloc[1]:.3f} REPOS={mean_alloc[2]:.3f} | "
                          f"Entropy: {entropy:.3f}")

                    # Forced charge info
                    fc = getattr(fleet_out, 'forced_charge_count', 0)
                    fr = getattr(fleet_out, 'forced_reposition_count', 0)
                    fc_total = getattr(fleet_out, 'forced_charge_total_idle', 0)
                    if (fc + fr) > 0:
                        print(f"  [Override] Forced charge: {fc}/{fc_total} | Forced reposition: {fr}/{fc_total} idle vehicles")

            # Step the environment
            next_state, reward, done_tensor, info = env.step(
                action_type, reposition_target, None,
                vehicle_charge_power=fleet_out.vehicle_charge_power,
                milp_serve_trip_ids=milp_serve_trip_ids,
            )
            done = done_tensor.item()
            state = next_state

            # Track per-step failures from episode_info (overwritten each step)
            step_charge_attempted = getattr(info, 'num_charge_attempted', 0)
            step_charge_success = getattr(info, 'num_charge_success', 0)
            step_charge_fail = step_charge_attempted - step_charge_success
            step_serve_attempted = getattr(info, 'num_serve_attempted', 0)
            step_serve_success = getattr(info, 'num_serve_success', 0)
            step_serve_fail = step_serve_attempted - step_serve_success
            cumulative_charge_failed += step_charge_fail
            cumulative_serve_failed += step_serve_fail
            interval_charge_failed += step_charge_fail
            interval_serve_failed += step_serve_fail

            # Accumulate step metrics via delta tracking (V2 env)
            metrics = env.get_metrics()
            cur_rev = metrics.get('revenue', 0.0)
            cur_drive = metrics.get('driving_cost', 0.0)
            cur_energy = metrics.get('energy_cost', 0.0)

            step_revenue, step_serve_cost, step_charge_cost = _accumulate_temporal_step(
                step=step,
                step_duration_minutes=step_duration_minutes,
                cur_revenue=cur_rev,
                cur_driving_cost=cur_drive,
                cur_energy_cost=cur_energy,
                prev_env_revenue=prev_env_revenue,
                prev_env_driving_cost=prev_env_driving_cost,
                prev_env_energy_cost=prev_env_energy_cost,
                hourly_metrics=hourly_metrics,
                daily_metrics=daily_metrics,
            )

            prev_env_revenue = cur_rev
            prev_env_driving_cost = cur_drive
            prev_env_energy_cost = cur_energy

            prev_revenue += step_revenue
            prev_driving_cost += step_serve_cost
            prev_energy_cost += step_charge_cost

            interval_revenue += step_revenue
            interval_driving_cost += step_serve_cost
            interval_energy_cost += step_charge_cost

            step += 1
            if step % log_interval_steps == 0 or step == max_steps:
                # Fleet status with TO_CHARGE shown separately
                status_counts = env.fleet_state.get_status_counts()
                status_str = " ".join(f"{k}={v}" for k, v in status_counts.items() if v > 0)

                # Station utilization
                ports_occupied = env.station_state.occupied.sum().item()
                ports_total = env.station_state.ports.sum().item()
                port_util_pct = 100.0 * ports_occupied / max(ports_total, 1)

                print(f"  Step {step}/{max_steps} | Last {log_interval_steps} steps: "
                      f"Revenue=${interval_revenue:.2f}, Driving=${interval_driving_cost:.2f}, Charge=${interval_energy_cost:.2f} | "
                      f"Interval Failures: Charge={interval_charge_failed}, Serve={interval_serve_failed}")
                print(f"  Cumulative: Revenue=${prev_revenue:.2f}, Driving=${prev_driving_cost:.2f}, Charge=${prev_energy_cost:.2f} | "
                      f"Total Failures: Charge={cumulative_charge_failed}, Serve={cumulative_serve_failed}")
                print(f"  Fleet: {status_str} | Mean SOC: {env.fleet_state.mean_soc:.1f}% | "
                      f"Ports: {ports_occupied}/{ports_total} ({port_util_pct:.0f}%)")
                if diag_snapshot is not None:
                    print(
                        "  Reachability: "
                        f"Unassigned={diag_snapshot['unassigned']}, "
                        f"PickupHexes={diag_snapshot['pickup_hexes']}, "
                        f"IdleAvail={diag_snapshot['idle_available']}, "
                        f"IdleWithReachable={diag_snapshot['idle_reachable']}, "
                        f"ServeAttempted={diag_snapshot['serve_attempted']}, "
                        f"ServeNoReachable={diag_snapshot['serve_no_reachable']}, "
                        f"ServeReachableButEnergyInfeasible={diag_snapshot['serve_reachable_energy_infeasible']}, "
                        f"ServeReachableAndEnergyFeasible={diag_snapshot['serve_reachable_energy_feasible']}"
                    )
                    print(
                        "  Context/Demand: "
                        f"Progress={diag_snapshot['progress']:.3f}, "
                        f"Remaining={diag_snapshot['remaining']:.3f}, "
                        f"NormTrips={diag_snapshot['norm_trips']:.3f}, "
                        f"NormFare={diag_snapshot['norm_fare']:.3f}, "
                        f"DemandSatAll={diag_snapshot['demand_saturated_ratio']:.3f}, "
                        f"DemandSatActive={diag_snapshot['active_demand_saturated_ratio']:.3f}"
                    )
                    print(
                        "  IdleAge: "
                        f"P50={diag_snapshot['idle_p50']:.1f}, "
                        f"P90={diag_snapshot['idle_p90']:.1f}, "
                        f"LongIdle={diag_snapshot['idle_long_count']}, "
                        f"LongIdleServe={diag_snapshot['idle_long_serve']}, "
                        f"LongIdleCharge={diag_snapshot['idle_long_charge']}, "
                        f"LongIdleReposition={diag_snapshot['idle_long_repos']}"
                    )

                interval_revenue = 0.0
                interval_driving_cost = 0.0
                interval_energy_cost = 0.0
                interval_charge_failed = 0
                interval_serve_failed = 0

        elapsed_time = time.time() - start_time

        # Final metrics from env (authoritative)
        metrics = env.get_metrics()
        metrics_payload = {
            'trips_served': metrics.get('trips_served', 0),
            'trips_loaded': metrics.get('trips_loaded', 0),
            'trips_dropped': metrics.get('trips_dropped', 0),
            'avg_soc': metrics.get('avg_soc', env.fleet_state.mean_soc),
            'revenue': metrics.get('revenue', 0.0),
            'driving_cost': metrics.get('driving_cost', 0.0),
            'energy_cost': metrics.get('energy_cost', 0.0),
        }

        result = _build_result_common(
            ep=ep,
            metrics=metrics_payload,
            action_counts=action_counts,
            elapsed_time=elapsed_time,
            steps_completed=step,
            hourly_metrics=hourly_metrics,
            daily_metrics=daily_metrics,
        )

        _print_result_common(result, config)

        all_results.append(result)

    total_elapsed = time.time() - total_start_time
    _print_timing(total_elapsed, args.episodes)
    _save_results(all_results, args.save_results)


def main():
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
