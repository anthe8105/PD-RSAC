#!/usr/bin/env python3
"""
Evaluate MADDPG checkpoints on the EV fleet environment.

Usage:
    python gpu_core/scripts/evaluate_maddpg.py \
        --checkpoint gpu_core/scripts/checkpoint/maddpg/best_eval.pt \
        --config gpu_core/scripts/config_maddpg.yaml \
        --real-data data/nyc_full/trips_processed.parquet \
        --start-date 2009-01-15 --end-date 2009-01-15
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import Config, ConfigLoader
from gpu_core.data.real_trip_loader import RealTripLoader
from gpu_core.features.maddpg_buffer import MADDPGReplayBuffer
from gpu_core.networks.maddpg_agent import MADDPGAgent
from gpu_core.spatial.grid import HexGrid
from gpu_core.simulator.environment import GPUEnvironmentV2
from gpu_core.training.maddpg_collector import MADDPGCollector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate MADDPG checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to MADDPG checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--save-results', '--save-result', type=str, default=None, help='Save results JSON path')

    parser.add_argument('--num-vehicles', type=int, default=None, help='Override number of vehicles')
    parser.add_argument('--num-hexes', type=int, default=None, help='Override number of hexes')
    parser.add_argument('--episode-duration-hours', type=float, default=None, help='Override episode duration')

    parser.add_argument('--real-data', type=str, default=None, help='Path to trip parquet')
    parser.add_argument('--trip-sample', type=float, default=None, help='Trip sample ratio [0,1]')
    parser.add_argument('--start-date', type=str, default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--target-h3-resolution', type=int, default=None, help='Target H3 resolution')
    parser.add_argument('--max-hex-count', type=int, default=None, help='Maximum active hexes')

    parser.add_argument('--deterministic', action='store_true', default=True, help='Greedy action selection')
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false', help='Stochastic action selection')
    parser.add_argument('--log-interval-steps', type=int, default=20, help='Progress log interval in steps (<=0 to disable)')

    return parser.parse_args()


def create_config(args) -> Config:
    config = ConfigLoader.from_yaml(args.config) if args.config else Config()

    if args.num_vehicles is not None:
        config.environment.num_vehicles = args.num_vehicles
    if args.num_hexes is not None:
        config.environment.num_hexes = args.num_hexes

    if args.start_date and args.end_date:
        try:
            from datetime import datetime
            start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            days = max(1, (end_dt - start_dt).days + 1)
            config.episode.duration_hours = float(days * 24)
            print(f"[Auto-Detect] Set episode duration to {config.episode.duration_hours} hours ({days} days)")
            if args.episode_duration_hours is not None and args.episode_duration_hours != float(days * 24):
                print(f"[Warning] Overriding requested --episode-duration-hours {args.episode_duration_hours} with calculated duration {float(days * 24)}")
        except Exception as e:
            print(f"[Auto-Detect] Failed to parse dates for duration: {e}")
            if args.episode_duration_hours is not None:
                config.episode.duration_hours = args.episode_duration_hours
    elif args.episode_duration_hours is not None:
        config.episode.duration_hours = args.episode_duration_hours

    if args.trip_sample is not None:
        config.data.trip_percentage = args.trip_sample
    if args.target_h3_resolution is not None:
        config.data.target_h3_resolution = args.target_h3_resolution
    if args.max_hex_count is not None:
        config.data.max_hex_count = args.max_hex_count

    config.training.algo = 'maddpg'
    return config


def create_environment(config: Config, device: torch.device, trip_loader: Optional[RealTripLoader] = None):
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
            print(f"[Stations] warning: failed to build neighbors: {e}")
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
        distances = torch.sqrt(lat_km ** 2 + lon_km ** 2)
        distances.fill_diagonal_(0)
        hex_grid.distance_matrix._distances = distances
        hex_grid.distance_matrix._num_hexes = num_hexes

    return GPUEnvironmentV2(config=config, hex_grid=hex_grid, trip_loader=trip_loader, device=device_str)


def _strict_load(module: torch.nn.Module, incoming: dict, tag: str):
    model_state = module.state_dict()
    shape_mismatches = [
        (k, tuple(v.shape), tuple(model_state[k].shape))
        for k, v in incoming.items()
        if k in model_state and model_state[k].shape != v.shape
    ]
    missing = [k for k in model_state.keys() if k not in incoming]
    unexpected = [k for k in incoming.keys() if k not in model_state]

    if shape_mismatches or missing or unexpected:
        details = []
        if shape_mismatches:
            details.append(f'[{tag}] shape mismatches:')
            details.extend([f'  - {k}: ckpt={ck} model={md}' for k, ck, md in shape_mismatches[:20]])
        if missing:
            details.append(f'[{tag}] missing keys in checkpoint: {len(missing)}')
            details.extend([f'  - {k}' for k in missing[:20]])
        if unexpected:
            details.append(f'[{tag}] unexpected keys in checkpoint: {len(unexpected)}')
            details.extend([f'  - {k}' for k in unexpected[:20]])
        raise RuntimeError('Incompatible checkpoint for strict MADDPG load:\n' + '\n'.join(details))

    module.load_state_dict(incoming, strict=True)


def _init_temporal_buckets():
    hourly_metrics = defaultdict(lambda: {'revenue': 0.0, 'driving_cost': 0.0, 'energy_cost': 0.0})
    daily_metrics = defaultdict(lambda: {'revenue': 0.0, 'driving_cost': 0.0, 'energy_cost': 0.0})
    return hourly_metrics, daily_metrics


def _accumulate_temporal_step(step: int, step_duration_minutes: float, delta_revenue: float, delta_driving: float, delta_energy: float, hourly_metrics, daily_metrics):
    current_hour = int(((step - 1) * step_duration_minutes) // 60) % 24
    current_day = int(((step - 1) * step_duration_minutes) // (60 * 24))
    hourly_metrics[current_hour]['revenue'] += delta_revenue
    hourly_metrics[current_hour]['driving_cost'] += delta_driving
    hourly_metrics[current_hour]['energy_cost'] += delta_energy
    daily_metrics[current_day]['revenue'] += delta_revenue
    daily_metrics[current_day]['driving_cost'] += delta_driving
    daily_metrics[current_day]['energy_cost'] += delta_energy


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


def _print_maddpg_progress(step: int, max_steps: int, info, cumulative_reward: float) -> None:
    status_parts = []
    if info is not None:
        status_parts.append(f"Trips {int(getattr(info, 'trips_served', 0))}/{int(getattr(info, 'trips_loaded', 0))}")
        status_parts.append(f"ServeFail {int(getattr(info, 'num_serve_attempted', 0)) - int(getattr(info, 'num_serve_success', 0))}")
        status_parts.append(f"ChargeFail {int(getattr(info, 'num_charge_attempted', 0)) - int(getattr(info, 'num_charge_success', 0))}")
    status = " | ".join(status_parts) if status_parts else ""
    print(f"  Step {step}/{max_steps} | Reward={cumulative_reward:.2f}" + (f" | {status}" if status else ""))


def load_maddpg_agent(checkpoint_path: str, config: Config, env, device: torch.device) -> MADDPGAgent:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'actor_state_dict' not in ckpt or 'critic_state_dict' not in ckpt:
        raise ValueError('Checkpoint is not MADDPG format: missing actor_state_dict / critic_state_dict')

    actor_sd = ckpt['actor_state_dict']
    max_k_neighbors = int(getattr(config.training, 'maddpg_max_k_neighbors', 0) or 0)
    if 'reposition_head.weight' in actor_sd:
        max_k_neighbors = int(actor_sd['reposition_head.weight'].shape[0])
    elif max_k_neighbors <= 0:
        max_k_neighbors = 61

    agent = MADDPGAgent(
        vehicle_feature_dim=getattr(env, '_vehicle_feature_dim', 16),
        context_dim=getattr(env, '_context_dim', 9),
        num_vehicles=config.environment.num_vehicles,
        num_hexes=config.environment.num_hexes,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        action_dim=3,
        max_trips=config.environment.max_trips_per_step,
        action_embed_dim=config.training.maddpg_action_embed_dim,
        dropout=getattr(config.network, 'dropout', 0.1) if hasattr(config, 'network') else 0.1,
        gamma=config.training.gamma,
        tau=config.training.maddpg_tau,
        lr_actor=config.training.maddpg_lr_actor,
        lr_critic=config.training.maddpg_lr_critic,
        gumbel_tau=config.training.maddpg_gumbel_tau,
        device=str(device),
        state_dim=(
            config.environment.num_vehicles * getattr(env, '_vehicle_feature_dim', 16)
            + config.environment.num_hexes * getattr(env, '_hex_feature_dim', 5)
            + getattr(env, '_context_dim', 9)
        ),
        hex_feature_dim=getattr(env, '_hex_feature_dim', 5),
        max_k_neighbors=max_k_neighbors,
    )

    _strict_load(agent.actor, ckpt['actor_state_dict'], 'MADDPG actor')
    _strict_load(agent.critic, ckpt['critic_state_dict'], 'MADDPG critic')

    if 'actor_target_state_dict' in ckpt:
        _strict_load(agent.actor_target, ckpt['actor_target_state_dict'], 'MADDPG actor_target')
    else:
        agent.actor_target.load_state_dict(agent.actor.state_dict())

    if 'critic_target_state_dict' in ckpt:
        _strict_load(agent.critic_target, ckpt['critic_target_state_dict'], 'MADDPG critic_target')
    else:
        agent.critic_target.load_state_dict(agent.critic.state_dict())

    agent.to(device)
    agent.eval()

    print(f"[MADDPG] Loaded checkpoint: {checkpoint_path}")
    if 'episode' in ckpt or 'global_step' in ckpt:
        print(
            f"[MADDPG] ckpt metadata: episode={ckpt.get('episode', 'n/a')}, "
            f"global_step={ckpt.get('global_step', 'n/a')}"
        )

    return agent


def evaluate(args):
    print('=' * 60)
    print('MADDPG Evaluation')
    print('=' * 60)

    total_start = time.time()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = create_config(args)

    trip_loader = None
    data_path = args.real_data or config.data.parquet_path
    if data_path:
        resolved = Path(data_path)
        print(f"[Real Data] Loading from: {resolved}")
        if resolved.exists():
            try:
                trip_loader = RealTripLoader(
                    parquet_path=str(resolved),
                    device=str(device),
                    sample_ratio=(args.trip_sample if args.trip_sample is not None else config.data.trip_percentage),
                    target_h3_resolution=(args.target_h3_resolution if args.target_h3_resolution is not None else config.data.target_h3_resolution),
                    max_hex_count=(args.max_hex_count if args.max_hex_count is not None else config.data.max_hex_count),
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
                trip_loader.load()
            except Exception as exc:
                print(f"[Real Data] Failed to load ({exc}); falling back to synthetic.")
                trip_loader = None
        else:
            print(f"[Real Data] File not found: {resolved}. Falling back to synthetic.")

    env = create_environment(config, device, trip_loader=trip_loader)
    print(f"[Environment] vehicles={env.num_vehicles}, hexes={env.num_hexes}, duration={config.episode.duration_hours}h")

    agent = load_maddpg_agent(args.checkpoint, config, env, device)

    dummy_buf = MADDPGReplayBuffer(
        capacity=1,
        num_vehicles=config.environment.num_vehicles,
        vehicle_feature_dim=getattr(env, '_vehicle_feature_dim', 16),
        context_dim=getattr(env, '_context_dim', 9),
        action_dim=3,
        prioritized=False,
        device='cpu',
    )
    collector = MADDPGCollector(
        env=env,
        replay_buffer=dummy_buf,
        device=str(device),
        repos_khop=getattr(config.training, 'maddpg_repos_khop', 4),
    )

    all_results = []
    print("\nEvaluating Episode 1/1...")

    hourly_metrics, daily_metrics = _init_temporal_buckets()
    step_duration_minutes = float(config.episode.step_duration_minutes)
    prev_env_revenue = 0.0
    prev_env_driving_cost = 0.0
    prev_env_energy_cost = 0.0

    def _on_step(step: int, _max_steps: int, _info, _reward):
        nonlocal prev_env_revenue, prev_env_driving_cost, prev_env_energy_cost
        metrics_now = env.get_metrics()
        cur_revenue = float(metrics_now.get('revenue', 0.0))
        cur_driving = float(metrics_now.get('driving_cost', 0.0))
        cur_energy = float(metrics_now.get('energy_cost', 0.0))
        delta_revenue = cur_revenue - prev_env_revenue
        delta_driving = cur_driving - prev_env_driving_cost
        delta_energy = cur_energy - prev_env_energy_cost
        _accumulate_temporal_step(
            step=step,
            step_duration_minutes=step_duration_minutes,
            delta_revenue=delta_revenue,
            delta_driving=delta_driving,
            delta_energy=delta_energy,
            hourly_metrics=hourly_metrics,
            daily_metrics=daily_metrics,
        )
        prev_env_revenue = cur_revenue
        prev_env_driving_cost = cur_driving
        prev_env_energy_cost = cur_energy

    stats = collector.collect_episode(
        agent=agent,
        rollout_steps=int(env.episode_steps),
        seed=0,
        deterministic=args.deterministic,
        exploration_eps=0.0,
        log_interval_steps=int(args.log_interval_steps),
        progress_callback=_print_maddpg_progress,
        step_callback=_on_step,
    )

    metrics = env.get_metrics()
    trips_served = int(stats.trips_served)
    trips_loaded = int(stats.trips_loaded)
    service_rate = float(trips_served) / max(float(trips_loaded), 1.0)

    action_counts = {
        'CHARGE': int(stats.action_counts.get('charge', 0)),
        'REPOSITION': int(stats.action_counts.get('reposition', 0)),
        'SERVE': int(stats.action_counts.get('serve', 0)),
    }

    result = {
        'episode': 0,
        'total_trips_loaded': trips_loaded,
        'total_trips_served': trips_served,
        'total_trips_dropped': int(stats.trips_dropped),
        'service_rate': float(service_rate),
        'total_revenue': float(metrics.get('revenue', stats.revenue)),
        'total_driving_cost': float(metrics.get('driving_cost', 0.0)),
        'total_charging_cost': float(metrics.get('energy_cost', stats.energy_cost)),
        'net_profit': float(metrics.get('revenue', stats.revenue) - metrics.get('driving_cost', 0.0) - metrics.get('energy_cost', stats.energy_cost)),
        'final_avg_soc': float(stats.avg_soc),
        'action_counts': action_counts,
        'simulation_time_seconds': 0.0,
        'steps_completed': int(stats.steps),
        'hourly_metrics': _format_temporal_metrics(hourly_metrics),
        'daily_metrics': _format_temporal_metrics(daily_metrics),
    }
    all_results.append(result)

    print(
        f"  Profit=${result['net_profit']:.2f} | "
        f"Serve={result['service_rate']*100:.1f}% | SOC={result['final_avg_soc']:.1f}%"
    )

    total_elapsed = time.time() - total_start
    all_results[0]['simulation_time_seconds'] = float(total_elapsed)
    print(f"\n{'=' * 60}")
    print(f"[Timing] total={total_elapsed:.2f}s")
    print(f"{'=' * 60}")

    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {args.save_results}")


def main():
    evaluate(parse_args())


if __name__ == '__main__':
    main()
