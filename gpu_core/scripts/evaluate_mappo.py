#!/usr/bin/env python3
"""
Evaluate MAPPO checkpoints on real-data environment.

This evaluator is dedicated to MAPPO policy checkpoints and supports
non-khop-candidate training/eval mode via config.training.mappo_use_khop_candidates.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import ConfigLoader, Config
from gpu_core.networks.ppo_agent import PPOAgent
from gpu_core.features.ppo_buffer import PPORolloutBuffer
from gpu_core.training.ppo_collector import PPOCollector
from gpu_core.spatial.grid import HexGrid
from gpu_core.spatial.neighbors import HexNeighbors
from gpu_core.simulator.environment import GPUEnvironmentV2
from gpu_core.data.real_trip_loader import RealTripLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate MAPPO checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to MAPPO checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (optional)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of evaluation episodes')

    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--save-results', '--save-result', type=str, default=None, help='Save results JSON path')

    parser.add_argument('--num-vehicles', type=int, default=None, help='Override number of vehicles')
    parser.add_argument('--num-hexes', type=int, default=None, help='Override number of hexes')
    parser.add_argument('--episode-duration-hours', type=float, default=None, help='Episode duration in hours')

    parser.add_argument('--real-data', type=str, default=None, help='Trip parquet path')
    parser.add_argument('--trip-sample', type=float, default=None, help='Trip sample ratio [0,1]')
    parser.add_argument('--start-date', type=str, default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--target-h3-resolution', type=int, default=None, help='Target H3 resolution')
    parser.add_argument('--max-hex-count', type=int, default=None, help='Max active hex count')
    parser.add_argument('--reference-start-date', type=str, default=None,
                        help='Reference hex universe start date (defaults to unbounded)')
    parser.add_argument('--reference-end-date', type=str, default=None,
                        help='Reference hex universe end date (default: day before --start-date)')

    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use greedy action selection')
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false',
                        help='Use stochastic sampling')

    return parser.parse_args()


def create_config(args, checkpoint_path: str) -> Config:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    ckpt_config = checkpoint.get('config', {})

    config = ConfigLoader.from_yaml(args.config) if args.config else Config()

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
            config.episode.duration_hours = float(days * 24)
            print(f"[Auto-Detect] episode duration={config.episode.duration_hours}h ({days} days)")
        except Exception as e:
            print(f"[Auto-Detect] date parse failed: {e}")
            if args.episode_duration_hours is not None:
                config.episode.duration_hours = args.episode_duration_hours
            elif 'episode' in ckpt_config and 'duration_hours' in ckpt_config['episode']:
                config.episode.duration_hours = ckpt_config['episode']['duration_hours']
    elif args.episode_duration_hours is not None:
        config.episode.duration_hours = args.episode_duration_hours
    elif 'episode' in ckpt_config and 'duration_hours' in ckpt_config['episode']:
        config.episode.duration_hours = ckpt_config['episode']['duration_hours']

    if args.trip_sample is not None:
        config.data.trip_percentage = args.trip_sample
    if args.target_h3_resolution is not None:
        config.data.target_h3_resolution = args.target_h3_resolution
    if args.max_hex_count is not None:
        config.data.max_hex_count = args.max_hex_count

    if not hasattr(config.training, 'mappo_use_khop_candidates'):
        config.training.mappo_use_khop_candidates = False

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


def _resolve_reference_window(args, config):
    from datetime import datetime, timedelta

    ref_start = args.reference_start_date or getattr(config.data, 'start_date', None)
    ref_end = args.reference_end_date

    if ref_end is None and args.start_date:
        try:
            ref_end = (datetime.strptime(args.start_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            ref_end = None

    if not ref_start or not ref_end:
        return None, None

    try:
        start_dt = datetime.strptime(ref_start, "%Y-%m-%d")
        end_dt = datetime.strptime(ref_end, "%Y-%m-%d")
        if end_dt < start_dt:
            return None, None
    except Exception:
        return None, None

    return ref_start, ref_end


def _init_temporal_buckets():
    action_counts = defaultdict(int)
    hourly_metrics = defaultdict(lambda: {'revenue': 0.0, 'driving_cost': 0.0, 'energy_cost': 0.0})
    daily_metrics = defaultdict(lambda: {'revenue': 0.0, 'driving_cost': 0.0, 'energy_cost': 0.0})
    return action_counts, hourly_metrics, daily_metrics


def _accumulate_temporal_step(step: int, step_duration_minutes: float,
                              cur_revenue: float, cur_driving_cost: float, cur_energy_cost: float,
                              prev_env_revenue: float, prev_env_driving_cost: float, prev_env_energy_cost: float,
                              hourly_metrics, daily_metrics):
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


def _build_result(ep: int, metrics: dict, action_counts: dict, elapsed_time: float, steps_completed: int,
                  hourly_metrics, daily_metrics, total_reward: float,
                  total_serve_attempted: int, total_serve_success: int):
    trips_served = metrics.get('trips_served', 0)
    trips_loaded = metrics.get('trips_loaded', 0)
    service_rate = float(trips_served) / max(float(trips_loaded), 1.0)

    total_actions = sum(action_counts.values()) or 1
    action_mix = {
        'serve_pct': 100.0 * action_counts.get('SERVE', 0) / total_actions,
        'charge_pct': 100.0 * action_counts.get('CHARGE', 0) / total_actions,
        'reposition_pct': 100.0 * action_counts.get('REPOSITION', 0) / total_actions,
    }

    return {
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
        'total_reward': float(total_reward),
        'serve_pct': float(service_rate * 100.0),
        'action_mix': action_mix,
        'num_serve_attempted': int(total_serve_attempted),
        'num_serve_success': int(total_serve_success),
    }


def _print_result(result: dict, duration_hours: float):
    print("\n--- Episode Summary ---")
    print(f" Reward: {result['total_reward']:.2f}")
    print(f" Revenue: ${result['total_revenue']:.2f}")
    print(f" Costs: ${result['total_driving_cost'] + result['total_charging_cost']:.2f}")
    print(f" Net Profit: ${result['net_profit']:.2f}")
    print(f" Trips: Served {result['total_trips_served']} / Loaded {result['total_trips_loaded']} "
          f"(Rate: {result['service_rate'] * 100:.1f}%)")
    print(f" Action Counts: {result['action_counts']}")

    if result['daily_metrics'] and duration_hours > 48.0:
        print("\nDaily metrics:")
        for day in sorted(result['daily_metrics'].keys()):
            day_data = result['daily_metrics'][day]
            print(f"  Day {day + 1:2d}: Revenue=${day_data['revenue']:>10,.2f}, Net Profit=${day_data['net_profit']:>10,.2f}")


def load_mappo_agent(checkpoint_path: str, config: Config, env, device: torch.device) -> PPOAgent:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state_dict = checkpoint.get('agent', checkpoint.get('agent_state_dict', None))
    if raw_state_dict is None:
        raise ValueError('MAPPO checkpoint missing agent state_dict')

    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        clean_key = k.replace('.module.', '.') if '.module.' in k else k
        clean_state_dict[clean_key] = v

    env_config = config.environment
    vehicle_feature_dim = getattr(env, '_vehicle_feature_dim', 16)
    hex_feature_dim = getattr(env, '_hex_feature_dim', 5)
    context_dim = getattr(env, '_context_dim', 9)

    state_dim = (
        env_config.num_vehicles * vehicle_feature_dim
        + env_config.num_hexes * hex_feature_dim
        + context_dim
    )

    mappo_dropout = config.network.dropout if hasattr(config, 'network') and hasattr(config.network, 'dropout') else 0.0
    if mappo_dropout is None:
        mappo_dropout = 0.0

    mappo_khop = max(1, int(getattr(config.training, 'mappo_repos_khop', 4)))
    eval_adj_max_k = None
    if hasattr(env, '_adjacency_matrix') and env._adjacency_matrix is not None:
        try:
            khop_mask_hh_eval = HexNeighbors.compute_khop_mask(env._adjacency_matrix, k=mappo_khop)
            _, _, eval_adj_max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh_eval)
            print(f"[MAPPO] eval adjacency max_k_neighbors={int(eval_adj_max_k)} (K={mappo_khop}, num_hexes={env.num_hexes})")
        except Exception as e:
            print(f"[MAPPO] Could not compute eval adjacency max_k_neighbors: {e}")

    mappo_max_k = 0
    if 'actor.reposition_head.weight' in clean_state_dict:
        mappo_max_k = int(clean_state_dict['actor.reposition_head.weight'].shape[0])
        print(f"[MAPPO] checkpoint max_k_neighbors={mappo_max_k}")

    use_trip_head = 'actor.trip_head.weight' in clean_state_dict
    learn_charge_power = 'actor.charge_power_head.weight' in clean_state_dict

    if mappo_max_k <= 0:
        mappo_max_k = int(getattr(config.training, 'mappo_max_k_neighbors', 0) or 0)
    if mappo_max_k <= 0:
        mappo_max_k = 61
        if eval_adj_max_k is not None:
            mappo_max_k = int(eval_adj_max_k)

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=3,
        num_hexes=env_config.num_hexes,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[256, 256],
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_eps=config.training.clip_eps,
        vf_coef=config.training.vf_coef,
        ent_coef=config.training.ent_coef,
        lr_actor=config.training.learning_rate.actor,
        lr_critic=config.training.learning_rate.critic,
        dropout=mappo_dropout,
        device=str(device),
        num_vehicles=env_config.num_vehicles,
        vehicle_feature_dim=vehicle_feature_dim,
        hex_feature_dim=hex_feature_dim,
        context_dim=context_dim,
        max_trips=env_config.max_trips_per_step,
        use_trip_head=use_trip_head,
        learn_charge_power=learn_charge_power,
        max_k_neighbors=mappo_max_k,
    )

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
            details.append('[MAPPO] shape mismatches:')
            details.extend([f'  - {k}: ckpt={ck} model={md}' for k, ck, md in shape_mismatches[:20]])
        if missing_keys:
            details.append(f"[MAPPO] missing keys in checkpoint: {len(missing_keys)}")
            details.extend([f'  - {k}' for k in missing_keys[:20]])
        if unexpected_keys:
            details.append(f"[MAPPO] unexpected keys in checkpoint: {len(unexpected_keys)}")
            details.extend([f'  - {k}' for k in unexpected_keys[:20]])
        raise RuntimeError('Incompatible MAPPO checkpoint for strict evaluation load:\n' + '\n'.join(details))

    agent.load_state_dict(clean_state_dict, strict=True)

    if 'value_norm_mean' in checkpoint and hasattr(agent, 'value_norm_mean'):
        agent.value_norm_mean.copy_(checkpoint['value_norm_mean'].to(agent.value_norm_mean.device))
    if 'value_norm_var' in checkpoint and hasattr(agent, 'value_norm_var'):
        agent.value_norm_var.copy_(checkpoint['value_norm_var'].to(agent.value_norm_var.device))
    if 'value_norm_count' in checkpoint and hasattr(agent, 'value_norm_count'):
        agent.value_norm_count.copy_(checkpoint['value_norm_count'].to(agent.value_norm_count.device))

    agent.to(device)
    agent.eval()
    print(f"[MAPPO] Loaded {len(clean_state_dict)} keys from checkpoint (strict)")
    return agent


def evaluate(args):
    print("=" * 60)
    print("MAPPO Evaluation")
    print("=" * 60)

    total_start_time = time.time()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_config = checkpoint.get('config', {})
    train_episode_hours = ckpt_config.get('episode', {}).get('duration_hours') if isinstance(ckpt_config, dict) else None

    config = create_config(args, args.checkpoint)

    trip_loader = None
    data_path = args.real_data or config.data.parquet_path
    sample_ratio = args.trip_sample if args.trip_sample is not None else getattr(config.data, 'trip_percentage', 1.0)
    target_res = args.target_h3_resolution if args.target_h3_resolution is not None else getattr(config.data, 'target_h3_resolution', None)
    max_hex_count = args.max_hex_count if args.max_hex_count is not None else getattr(config.data, 'max_hex_count', None)

    if data_path:
        resolved_path = Path(data_path)
        print(f"[Real Data] Loading from: {resolved_path}")

        reference_hex_ids = None
        ref_start, ref_end = _resolve_reference_window(args, config)
        if ref_start and ref_end:
            try:
                reference_loader = RealTripLoader(
                    parquet_path=str(resolved_path),
                    device=str(device),
                    sample_ratio=sample_ratio if sample_ratio is not None else 1.0,
                    target_h3_resolution=target_res,
                    max_hex_count=max_hex_count,
                    start_date=ref_start,
                    end_date=ref_end,
                )
                reference_loader.load()
                if reference_loader.is_loaded:
                    reference_hex_ids = list(reference_loader.hex_ids)
                    print(f"[Reference Hex] Loaded training hex universe {ref_start} -> {ref_end}: {len(reference_hex_ids)} hexes")
            except Exception as exc:
                print(f"[Reference Hex] Failed to build reference hex universe ({exc}); continuing without it")

        if resolved_path.exists():
            try:
                trip_loader = RealTripLoader(
                    parquet_path=str(resolved_path),
                    device=str(device),
                    sample_ratio=sample_ratio if sample_ratio is not None else 1.0,
                    target_h3_resolution=target_res,
                    max_hex_count=max_hex_count,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    reference_hex_ids=reference_hex_ids,
                )
                trip_loader.load()
            except Exception as exc:
                print(f"[Real Data] Failed to load ({exc}); falling back to synthetic.")
                trip_loader = None
        else:
            print(f"[Real Data] File not found: {resolved_path}. Falling back to synthetic.")

    env = create_environment(config, device, trip_loader=trip_loader)

    if train_episode_hours is not None and train_episode_hours > 0:
        train_episode_steps = int(train_episode_hours * 60 / config.episode.step_duration_minutes)
        if train_episode_steps > 0 and train_episode_steps != env.episode_steps:
            env._feature_norm_steps = train_episode_steps
            print(f"[Eval] Using training-horizon normalization: {train_episode_hours}h ({train_episode_steps} steps)")

    print(f"[Environment] vehicles={env.num_vehicles}, hexes={env.num_hexes}, duration={config.episode.duration_hours}h")
    agent = load_mappo_agent(args.checkpoint, config, env, device)

    use_khop_candidates = bool(getattr(config.training, 'mappo_use_khop_candidates', False))
    print(f"[MAPPO] use_khop_candidates={use_khop_candidates}")

    all_results = []

    for ep in range(args.episodes):
        print(f"\nEvaluating Episode {ep + 1}/{args.episodes}...")

        ppo_buffer = PPORolloutBuffer()
        collector = PPOCollector(
            env=env,
            replay_buffer=ppo_buffer,
            device=str(device),
            repos_khop=getattr(config.training, 'mappo_repos_khop', 4),
            use_khop_candidates=use_khop_candidates,
        )
        if use_khop_candidates:
            collector._ensure_khop_neighbors()

        state = env.reset(seed=ep)
        max_steps = int(env.episode_steps)
        step_duration_minutes = config.episode.step_duration_minutes

        action_counts, hourly_metrics, daily_metrics = _init_temporal_buckets()
        prev_env_revenue = 0.0
        prev_env_driving_cost = 0.0
        prev_env_energy_cost = 0.0
        total_reward = 0.0
        total_serve_attempted = 0
        total_serve_success = 0

        done = False
        step = 0
        start_time = time.time()

        while (not done) and step < max_steps:
            available_mask = env.get_available_actions()
            vehicle_hex_ids = env.fleet_state.positions.long()
            reposition_mask = collector._build_reposition_mask(vehicle_hex_ids)

            trip_mask = None
            if hasattr(env, 'trip_state'):
                trip_mask = collector._build_per_vehicle_trip_mask(vehicle_hex_ids, agent._max_trips)
                if trip_mask is None:
                    available_mask[:, 0] = False
                else:
                    available_mask[:, 0] = available_mask[:, 0] & trip_mask.any(dim=1)

            with torch.no_grad():
                ppo_out = agent.select_action(
                    state=state,
                    action_mask=available_mask,
                    reposition_mask=reposition_mask,
                    trip_mask=trip_mask,
                    deterministic=args.deterministic,
                    khop_neighbor_indices=collector._khop_neighbor_indices if use_khop_candidates else None,
                    khop_neighbor_mask=collector._khop_neighbor_mask if use_khop_candidates else None,
                    vehicle_hex_ids=vehicle_hex_ids,
                )

            action_type = ppo_out.action_type
            reposition_target = ppo_out.reposition_target
            selected_trip = ppo_out.selected_trip

            if action_type.dim() > 1:
                action_type = action_type.squeeze(0)
                reposition_target = reposition_target.squeeze(0)
                if selected_trip is not None:
                    selected_trip = selected_trip.squeeze(0)

            selected_trip_for_env = selected_trip if getattr(agent, 'use_trip_head', True) else None

            next_state, reward, done_tensor, info = env.step(
                action_type,
                reposition_target,
                selected_trip_for_env,
                vehicle_charge_power=None,
            )

            done = done_tensor.item()
            state = next_state

            total_reward += reward.item() if isinstance(reward, torch.Tensor) else float(reward)
            total_serve_attempted += int(getattr(info, 'num_serve_attempted', 0))
            total_serve_success += int(getattr(info, 'num_serve_success', 0))

            available_1d = available_mask.any(dim=1)
            valid_actions = action_type[(action_type >= 0) & (action_type < 3) & available_1d]
            if valid_actions.numel() > 0:
                counts = torch.bincount(valid_actions, minlength=3)
                action_counts['SERVE'] += int(counts[0].item())
                action_counts['CHARGE'] += int(counts[1].item())
                action_counts['REPOSITION'] += int(counts[2].item())

            metrics_step = env.get_metrics()
            cur_rev = metrics_step.get('revenue', 0.0)
            cur_drive = metrics_step.get('driving_cost', 0.0)
            cur_energy = metrics_step.get('energy_cost', 0.0)

            _accumulate_temporal_step(
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
            step += 1

        elapsed_time = time.time() - start_time
        metrics = env.get_metrics()

        result = _build_result(
            ep=ep,
            metrics={
                'trips_served': metrics.get('trips_served', 0),
                'trips_loaded': metrics.get('trips_loaded', 0),
                'trips_dropped': metrics.get('trips_dropped', 0),
                'avg_soc': metrics.get('avg_soc', env.fleet_state.mean_soc),
                'revenue': metrics.get('revenue', 0.0),
                'driving_cost': metrics.get('driving_cost', 0.0),
                'energy_cost': metrics.get('energy_cost', 0.0),
            },
            action_counts=action_counts,
            elapsed_time=elapsed_time,
            steps_completed=step,
            hourly_metrics=hourly_metrics,
            daily_metrics=daily_metrics,
            total_reward=total_reward,
            total_serve_attempted=total_serve_attempted,
            total_serve_success=total_serve_success,
        )

        _print_result(result, config.episode.duration_hours)
        all_results.append(result)

    total_elapsed = time.time() - total_start_time
    print(f"\n{'=' * 60}")
    print(f"[Timing] total={total_elapsed:.2f}s")
    if args.episodes > 1:
        print(f"[Timing] avg/episode={total_elapsed / args.episodes:.2f}s")
    print(f"{'=' * 60}")

    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {args.save_results}")


def main():
    evaluate(parse_args())


if __name__ == '__main__':
    main()
