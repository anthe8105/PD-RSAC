#!/usr/bin/env python3
"""
Evaluate MAPPO checkpoint charging power for exactly one day.

Examples:
    python gpu_core/scripts/powergrid_evaluate_mappo.py \
        --checkpoint gpu_core/scripts/checkpoint/mappo/best_eval.pt \
        --day 2009-01-25 \
        --real-data data/nyc_full/trips_processed.parquet \
        --config gpu_core/scripts/config_mappo.yaml \
        --trip-sample 0.2 \
        --out-csv powergrid_mappo.csv \
        --out-json powergrid_mappo.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import Config, ConfigLoader
from gpu_core.data.real_trip_loader import RealTripLoader
from gpu_core.networks.ppo_agent import PPOAgent
from gpu_core.simulator.environment import GPUEnvironmentV2
from gpu_core.spatial.grid import HexGrid
from gpu_core.spatial.neighbors import HexNeighbors
from gpu_core.state import VehicleStatus
from gpu_core.training.ppo_collector import PPOCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MAPPO charging power for a selected day",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint")
    parser.add_argument("--day", type=str, required=True, help="Calendar day to evaluate (YYYY-MM-DD)")
    parser.add_argument("--real-data", type=str, required=True, help="Path to real trip parquet file")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override")
    parser.add_argument("--label", type=str, default=None, help="Optional label for outputs")
    parser.add_argument("--trip-sample", type=float, default=1.0, help="Trip sampling ratio")
    parser.add_argument("--target-h3-resolution", type=int, default=None, help="Optional H3 coarsening resolution")
    parser.add_argument("--max-hex-count", type=int, default=None, help="Optional cap on number of hexes")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step limit for debugging")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional CSV output path")
    parser.add_argument("--out-json", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--reference-start-date", type=str, default=None,
                        help="Reference hex universe start date (default: config.data.start_date)")
    parser.add_argument("--reference-end-date", type=str, default=None,
                        help="Reference hex universe end date (default: day before --day)")
    return parser.parse_args()


def clean_state_dict(raw_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean = {}
    for key, value in raw_state_dict.items():
        clean_key = key.replace('.module.', '.') if '.module.' in key else key
        clean[clean_key] = value
    return clean


def build_config(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> Config:
    config = ConfigLoader.from_yaml(args.config) if args.config else Config()
    ckpt_config = checkpoint.get('config', {})
    if isinstance(ckpt_config, dict) and ckpt_config:
        ConfigLoader._update_dataclass(config, ckpt_config)

    config.training.algo = 'ppo'
    config.episode.duration_hours = 24.0
    config.data.parquet_path = args.real_data
    config.data.start_date = args.day

    if args.trip_sample is not None:
        config.data.trip_percentage = args.trip_sample
    if args.target_h3_resolution is not None:
        config.data.target_h3_resolution = args.target_h3_resolution
    if args.max_hex_count is not None:
        config.data.max_hex_count = args.max_hex_count

    return config


def create_environment(config: Config, device: torch.device, trip_loader: Optional[RealTripLoader] = None) -> GPUEnvironmentV2:
    num_hexes = config.environment.num_hexes
    device_str = str(device)
    hex_grid = HexGrid(device=device_str)

    if trip_loader and trip_loader.is_loaded:
        hex_ids = trip_loader.hex_ids
        lats, lons = trip_loader.get_hex_coordinates()
        num_hexes = len(hex_ids)
        config.environment.num_hexes = num_hexes
        hex_grid._hex_ids = hex_ids
        hex_grid._hex_to_idx = {hex_id: idx for idx, hex_id in enumerate(hex_ids)}
        hex_grid._latitudes = lats.to(device)
        hex_grid._longitudes = lons.to(device)
        hex_grid._initialized = True
        hex_grid.distance_matrix.compute(hex_grid._latitudes, hex_grid._longitudes, hex_ids=hex_ids)
        try:
            hex_grid.neighbors.compute(hex_ids, k=1)
        except Exception as exc:
            print(f"[Stations] Warning: failed to build 1-hop neighbors from H3 IDs: {exc}")
    else:
        grid_size = int(num_hexes ** 0.5) + 1
        fake_hex_ids = [f"hex_{i}" for i in range(num_hexes)]
        hex_grid._hex_ids = fake_hex_ids
        hex_grid._hex_to_idx = {hex_id: idx for idx, hex_id in enumerate(fake_hex_ids)}
        base_lat, base_lon = 40.7128, -74.0060
        lat_per_km, lon_per_km = 0.009, 0.012
        lats = torch.zeros(num_hexes, device=device)
        lons = torch.zeros(num_hexes, device=device)
        for idx in range(num_hexes):
            row = idx // grid_size
            col = idx % grid_size
            lats[idx] = base_lat + row * 0.5 * lat_per_km
            lons[idx] = base_lon + col * 0.5 * lon_per_km
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

    return GPUEnvironmentV2(
        config=config,
        hex_grid=hex_grid,
        trip_loader=trip_loader,
        device=device_str,
    )


def apply_training_horizon_normalization(env: GPUEnvironmentV2, checkpoint: Dict[str, Any], config: Config) -> None:
    ckpt_config = checkpoint.get('config', {})
    if not isinstance(ckpt_config, dict):
        return
    train_episode_hours = ckpt_config.get('episode', {}).get('duration_hours')
    if train_episode_hours is None or train_episode_hours <= 0:
        return
    train_episode_steps = int(train_episode_hours * 60 / config.episode.step_duration_minutes)
    if train_episode_steps > 0 and train_episode_steps != env.episode_steps:
        env._feature_norm_steps = train_episode_steps
        print(f"[Eval] Using training-horizon normalization: {train_episode_hours}h ({train_episode_steps} steps)")


def _resolve_reference_window(args: argparse.Namespace, config: Config) -> (Optional[str], Optional[str]):
    from datetime import datetime, timedelta

    ref_start = args.reference_start_date or getattr(config.data, 'start_date', None)
    ref_end = args.reference_end_date

    if ref_end is None and args.day:
        try:
            ref_end = (datetime.strptime(args.day, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
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


def create_trip_loader(args: argparse.Namespace, config: Config, device: torch.device) -> RealTripLoader:
    reference_hex_ids = None
    ref_start, ref_end = _resolve_reference_window(args, config)
    if ref_start and ref_end:
        try:
            ref_loader = RealTripLoader(
                parquet_path=args.real_data,
                device=str(device),
                sample_ratio=args.trip_sample,
                target_h3_resolution=args.target_h3_resolution,
                max_hex_count=args.max_hex_count,
                start_date=ref_start,
                end_date=ref_end,
            )
            ref_loader.load()
            if ref_loader.is_loaded:
                reference_hex_ids = list(ref_loader.hex_ids)
                print(f"[Reference Hex] Loaded training hex universe {ref_start} -> {ref_end}: {len(reference_hex_ids)} hexes")
        except Exception as exc:
            print(f"[Reference Hex] Failed to build reference hex universe ({exc}); continuing without it")

    trip_loader = RealTripLoader(
        parquet_path=args.real_data,
        device=str(device),
        sample_ratio=args.trip_sample,
        target_h3_resolution=args.target_h3_resolution,
        max_hex_count=args.max_hex_count,
        start_date=args.day,
        end_date=args.day,
        reference_hex_ids=reference_hex_ids,
    )
    trip_loader.load()
    return trip_loader


def load_mappo_agent(
    checkpoint: Dict[str, Any],
    config: Config,
    env: GPUEnvironmentV2,
    device: torch.device,
) -> PPOAgent:
    raw_state_dict = checkpoint.get('agent', checkpoint.get('agent_state_dict', {}))
    raw_state_dict = clean_state_dict(raw_state_dict)

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
        except Exception as exc:
            print(f"[MAPPO] Could not compute eval adjacency max_k_neighbors: {exc}")

    mappo_max_k = 0
    if 'actor.reposition_head.weight' in raw_state_dict:
        mappo_max_k = int(raw_state_dict['actor.reposition_head.weight'].shape[0])
        print(f"[MAPPO] checkpoint max_k_neighbors={mappo_max_k}")

    use_trip_head = 'actor.trip_head.weight' in raw_state_dict
    learn_charge_power = 'actor.charge_power_head.weight' in raw_state_dict

    if mappo_max_k <= 0:
        mappo_max_k = int(getattr(config.training, 'mappo_max_k_neighbors', 0) or 0)
    if mappo_max_k <= 0:
        mappo_max_k = int(eval_adj_max_k) if eval_adj_max_k is not None else 61

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
        for k, v in raw_state_dict.items()
        if k in model_state and model_state[k].shape != v.shape
    ]
    missing_keys = [k for k in model_state.keys() if k not in raw_state_dict]
    unexpected_keys = [k for k in raw_state_dict.keys() if k not in model_state]

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

    agent.load_state_dict(raw_state_dict, strict=True)
    if 'value_norm_mean' in checkpoint and hasattr(agent, 'value_norm_mean'):
        agent.value_norm_mean.copy_(checkpoint['value_norm_mean'].to(agent.value_norm_mean.device))
    if 'value_norm_var' in checkpoint and hasattr(agent, 'value_norm_var'):
        agent.value_norm_var.copy_(checkpoint['value_norm_var'].to(agent.value_norm_var.device))
    if 'value_norm_count' in checkpoint and hasattr(agent, 'value_norm_count'):
        agent.value_norm_count.copy_(checkpoint['value_norm_count'].to(agent.value_norm_count.device))

    agent.to(device)
    agent.eval()
    print(f"[MAPPO] Loaded {len(raw_state_dict)} keys from checkpoint (strict)")
    return agent


def format_minute_of_day(minute: int) -> str:
    minute = int(minute) % (24 * 60)
    hh = minute // 60
    mm = minute % 60
    return f"{hh:02d}:{mm:02d}"


def collect_ppo_timeseries(
    env: GPUEnvironmentV2,
    agent: PPOAgent,
    max_steps: int,
    label: str,
    checkpoint_path: str,
    day: str,
    step_duration_minutes: float,
    use_khop_candidates: bool,
    repos_khop: int,
) -> List[Dict[str, Any]]:
    collector = PPOCollector(
        env=env,
        replay_buffer=None,
        device=str(env.device),
        repos_khop=repos_khop,
        use_khop_candidates=use_khop_candidates,
    )
    if use_khop_candidates:
        collector._ensure_khop_neighbors()

    state = env.reset(seed=0)
    rows: List[Dict[str, Any]] = []

    for step in range(max_steps):
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
                deterministic=True,
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
            if selected_trip is not None and selected_trip.dim() > 1:
                selected_trip = selected_trip.squeeze(0)

        selected_trip_for_env = selected_trip if getattr(agent, 'use_trip_head', True) else None
        next_state, _, done_tensor, _ = env.step(action_type, reposition_target, selected_trip_for_env, vehicle_charge_power=None)
        state = next_state

        charging_mask = env.fleet_state.status == VehicleStatus.CHARGING.value
        total_charging_power_kw = float(env.fleet_state.charge_power[charging_mask].sum().item()) if charging_mask.any() else 0.0
        num_vehicles_charging = int(charging_mask.sum().item())
        mean_power = total_charging_power_kw / num_vehicles_charging if num_vehicles_charging > 0 else 0.0
        minute_of_day = int(round(step * step_duration_minutes))

        rows.append(
            {
                'label': label,
                'checkpoint': checkpoint_path,
                'day': day,
                'step': step,
                'minute_of_day': minute_of_day,
                'timestamp_hhmm': format_minute_of_day(minute_of_day),
                'total_charging_power_kw': total_charging_power_kw,
                'num_vehicles_charging': num_vehicles_charging,
                'mean_power_per_vehicle_kw': mean_power,
            }
        )

        if bool(done_tensor.item()):
            break

    return rows


def build_summary(
    rows: List[Dict[str, Any]],
    checkpoint_path: str,
    label: str,
    day: str,
    step_duration_minutes: float,
) -> Dict[str, Any]:
    step_hours = step_duration_minutes / 60.0
    total_energy_kwh = sum(row['total_charging_power_kw'] * step_hours for row in rows)
    peak_row = max(rows, key=lambda row: row['total_charging_power_kw']) if rows else None
    average_power = sum(row['total_charging_power_kw'] for row in rows) / len(rows) if rows else 0.0

    return {
        'label': label,
        'checkpoint': checkpoint_path,
        'day': day,
        'algo': 'ppo',
        'num_steps': len(rows),
        'step_duration_minutes': step_duration_minutes,
        'average_charging_power_kw': average_power,
        'peak_charging_power_kw': peak_row['total_charging_power_kw'] if peak_row else 0.0,
        'peak_step': peak_row['step'] if peak_row else None,
        'peak_time': peak_row['timestamp_hhmm'] if peak_row else None,
        'total_charged_energy_kwh': total_energy_kwh,
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'label',
        'checkpoint',
        'day',
        'step',
        'minute_of_day',
        'timestamp_hhmm',
        'total_charging_power_kw',
        'num_vehicles_charging',
        'mean_power_per_vehicle_kw',
    ]
    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Output] Wrote CSV to {output_path}")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as handle:
        json.dump(payload, handle, indent=2)
    print(f"[Output] Wrote JSON to {output_path}")


def main() -> None:
    args = parse_args()
    checkpoint_path = str(Path(args.checkpoint).resolve())
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label = args.label or Path(checkpoint_path).stem

    print('=' * 60)
    print('Power Grid Evaluation (MAPPO)')
    print('=' * 60)
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Label: {label}')
    print(f'Day: {args.day}')
    print(f'Device: {device}')

    config = build_config(args, checkpoint)
    trip_loader = create_trip_loader(args, config, device)
    env = create_environment(config, device, trip_loader=trip_loader)
    apply_training_horizon_normalization(env, checkpoint, config)

    print(f"[Environment] {env.num_vehicles} vehicles | {env.num_hexes} hexes | {config.episode.duration_hours:.1f} hours")

    agent = load_mappo_agent(checkpoint, config, env, device)
    use_khop_candidates = bool(getattr(config.training, 'mappo_use_khop_candidates', False))
    print(f"[MAPPO] use_khop_candidates={use_khop_candidates}")

    max_steps = min(env.episode_steps, args.max_steps) if args.max_steps else env.episode_steps
    rows = collect_ppo_timeseries(
        env=env,
        agent=agent,
        max_steps=max_steps,
        label=label,
        checkpoint_path=checkpoint_path,
        day=args.day,
        step_duration_minutes=config.episode.step_duration_minutes,
        use_khop_candidates=use_khop_candidates,
        repos_khop=getattr(config.training, 'mappo_repos_khop', 4),
    )

    summary = build_summary(
        rows=rows,
        checkpoint_path=checkpoint_path,
        label=label,
        day=args.day,
        step_duration_minutes=config.episode.step_duration_minutes,
    )

    print('[Summary]')
    print(f"  Steps: {summary['num_steps']}")
    print(f"  Average charging power: {summary['average_charging_power_kw']:.3f} kW")
    print(f"  Peak charging power: {summary['peak_charging_power_kw']:.3f} kW at step {summary['peak_step']} ({summary['peak_time']})")
    print(f"  Total charged energy: {summary['total_charged_energy_kwh']:.3f} kWh")

    if args.out_csv:
        write_csv(args.out_csv, rows)
    if args.out_json:
        write_json(args.out_json, {'summary': summary, 'timeseries': rows})


if __name__ == '__main__':
    main()
