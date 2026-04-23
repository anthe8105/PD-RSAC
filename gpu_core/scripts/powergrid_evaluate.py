#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on a selected calendar day and record aggregate
charging power over time.

Examples:
    python gpu_core/scripts/powergrid_evaluate.py \
        --checkpoint checkpoints/best.pt \
        --day 2009-01-15 \
        --real-data data/nyc_full/trips_processed.parquet \
        --out-csv powergrid.csv \
        --out-json powergrid.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import Config, ConfigLoader
from gpu_core.data.real_trip_loader import RealTripLoader
from gpu_core.networks.ppo_agent import PPOAgent
from gpu_core.networks.sac import FleetSACAgent
from gpu_core.simulator.environment import GPUEnvironmentV2
from gpu_core.spatial.grid import HexGrid
from gpu_core.spatial.neighbors import HexNeighbors
from gpu_core.state.fleet import VehicleStatus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate aggregate charging power for a single checkpoint on a selected day",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--day", type=str, required=True, help="Calendar day to evaluate (YYYY-MM-DD)")
    parser.add_argument("--real-data", type=str, required=True, help="Path to real trip parquet file")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override")
    parser.add_argument("--label", type=str, default=None, help="Optional label for outputs")
    parser.add_argument("--milp", action="store_true", default=False, help="Enable MILP projection for fleet SAC checkpoints")
    parser.add_argument("--milp-p-max-feed", type=float, default=7000.0, help="MILP feeder power cap in kW")
    parser.add_argument("--trip-sample", type=float, default=1.0, help="Trip sampling ratio")
    parser.add_argument("--target-h3-resolution", type=int, default=None, help="Optional H3 coarsening resolution")
    parser.add_argument("--max-hex-count", type=int, default=None, help="Optional cap on number of hexes")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step limit for debugging")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional CSV output path")
    parser.add_argument("--out-json", type=str, default=None, help="Optional JSON output path")
    return parser.parse_args()


def clean_state_dict(raw_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean = {}
    for key, value in raw_state_dict.items():
        clean_key = key.replace(".module.", ".") if ".module." in key else key
        clean[clean_key] = value
    return clean


def infer_algo(ckpt_config: Dict[str, Any], raw_state_dict: Dict[str, torch.Tensor]) -> str:
    training_cfg = ckpt_config.get("training", {}) if isinstance(ckpt_config, dict) else {}
    algo = str(training_cfg.get("algo", "")).lower().strip()
    if algo in {"sac", "ppo"}:
        return algo

    keys = set(raw_state_dict.keys())
    if any(key.startswith("actor.gcn.") or key.startswith("critic_target.") for key in keys):
        return "sac"
    if any(key.startswith("actor.backbone.") for key in keys) and any(key.startswith("critic.backbone.") for key in keys):
        return "ppo"
    return "sac"


def build_config(args: argparse.Namespace, checkpoint: Dict[str, Any], algo: str) -> Config:
    config = Config()
    ckpt_config = checkpoint.get("config", {})
    if isinstance(ckpt_config, dict) and ckpt_config:
        ConfigLoader._update_dataclass(config, ckpt_config)

    if args.config:
        yaml_data = ConfigLoader.load_yaml(args.config)
        ConfigLoader._update_dataclass(config, yaml_data)

    metadata = checkpoint.get("checkpoint_metadata", {})
    if isinstance(metadata, dict):
        num_vehicles = metadata.get("num_vehicles")
        num_hexes = metadata.get("num_hexes")
        if isinstance(num_vehicles, int) and num_vehicles > 0:
            config.environment.num_vehicles = num_vehicles
        if isinstance(num_hexes, int) and num_hexes > 0:
            config.environment.num_hexes = num_hexes

    config.training.algo = algo
    config.episode.duration_hours = 24.0
    config.data.parquet_path = args.real_data
    config.data.start_date = args.day

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


def create_trip_loader(args: argparse.Namespace, device: torch.device) -> RealTripLoader:
    trip_loader = RealTripLoader(
        parquet_path=args.real_data,
        device=str(device),
        sample_ratio=args.trip_sample,
        target_h3_resolution=args.target_h3_resolution,
        max_hex_count=args.max_hex_count,
        start_date=args.day,
        end_date=args.day,
    )
    trip_loader.load()
    return trip_loader


def apply_training_horizon_normalization(env: GPUEnvironmentV2, checkpoint: Dict[str, Any], config: Config) -> None:
    ckpt_config = checkpoint.get("config", {})
    if not isinstance(ckpt_config, dict):
        return
    train_episode_hours = ckpt_config.get("episode", {}).get("duration_hours")
    if train_episode_hours is None or train_episode_hours <= 0:
        return
    train_episode_steps = int(train_episode_hours * 60 / config.episode.step_duration_minutes)
    if train_episode_steps > 0 and train_episode_steps != env.episode_steps:
        env._feature_norm_steps = train_episode_steps
        print(f"[Eval] Using training-horizon normalization: {train_episode_hours}h ({train_episode_steps} steps)")


def load_fleet_agent(
    checkpoint: Dict[str, Any],
    config: Config,
    env: GPUEnvironmentV2,
    device: torch.device,
) -> FleetSACAgent:
    env_config = config.environment
    entropy_config = config.entropy
    fleet_actor_config = config.fleet_actor
    vehicle_config = config.vehicle

    vehicle_feature_dim = getattr(env, "_fleet_vehicle_feature_dim", getattr(env, "_vehicle_feature_dim", 13))
    hex_feature_dim = getattr(env, "_hex_feature_dim", 5)
    context_dim = getattr(env, "_context_dim", 9)

    raw_state_dict = checkpoint.get("agent", checkpoint.get("agent_state_dict", {}))
    raw_state_dict = clean_state_dict(raw_state_dict)

    max_k_neighbors = None
    for key in ("actor.repos_target_head.weight",):
        if key in raw_state_dict:
            max_k_neighbors = raw_state_dict[key].shape[0]
            print(f"[Agent Setup] max_K_neighbors={max_k_neighbors} (from checkpoint)")
            break
    if max_k_neighbors is None:
        max_k_neighbors = 61
        if hasattr(env, "_adjacency_matrix") and env._adjacency_matrix is not None:
            try:
                k_hops = getattr(fleet_actor_config, "repos_khop", 4)
                khop_mask_hh = HexNeighbors.compute_khop_mask(env._adjacency_matrix, k=k_hops)
                _, _, max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh)
                max_k_neighbors = max_k
            except Exception:
                pass

    actor_hidden_dims = config.network.actor_hidden_dims if hasattr(config.network, "actor_hidden_dims") else [256, 256]
    critic_hidden_dims = config.network.critic_hidden_dims if hasattr(config.network, "critic_hidden_dims") else [256, 256]
    dropout_rate = getattr(config.network, "dropout", 0.1)

    gcn_hidden_dim = actor_hidden_dims[0] if actor_hidden_dims else 128
    for key in ("actor.gcn.layers.0.linear.weight",):
        if key in raw_state_dict:
            gcn_hidden_dim = raw_state_dict[key].shape[0]
            print(f"[Agent Setup] gcn_hidden_dim={gcn_hidden_dim} (inferred from checkpoint)")
            break

    action_dim = 3
    target_entropy = -getattr(entropy_config, "target_entropy_ratio", 0.5) * torch.log(torch.tensor(float(action_dim))).item()

    agent = FleetSACAgent(
        num_hexes=env_config.num_hexes,
        num_vehicles=env_config.num_vehicles,
        vehicle_feature_dim=vehicle_feature_dim,
        hex_feature_dim=hex_feature_dim,
        hex_vehicle_agg_dim=getattr(fleet_actor_config, "hex_vehicle_agg_dim", 8),
        context_dim=context_dim,
        action_dim=action_dim,
        max_K_neighbors=max_k_neighbors,
        gcn_hidden_dim=gcn_hidden_dim,
        gcn_output_dim=64,
        hex_decision_hidden_dim=getattr(fleet_actor_config, "hex_decision_hidden_dim", 256),
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
        min_alpha=getattr(entropy_config, "min_alpha", 0.05),
        max_alpha=getattr(entropy_config, "max_alpha", 1.0),
        repos_aux_weight=getattr(config.training, "repos_aux_weight", 0.1),
        soc_low_threshold=getattr(vehicle_config, "soc_low_threshold", 20.0),
        assignment_soc_priority=getattr(fleet_actor_config, "assignment_soc_priority", True),
        use_semi_mdp=getattr(config.training, "use_semi_mdp", True),
    )

    if raw_state_dict:
        agent.load_state_dict(raw_state_dict, strict=False)
        print(f"[Checkpoint] Loaded {len(raw_state_dict)} keys into FleetSACAgent")

    if hasattr(env, "_adjacency_matrix") and env._adjacency_matrix is not None:
        agent.set_adjacency(env._adjacency_matrix)
        try:
            k_hops = getattr(fleet_actor_config, "repos_khop", 4)
            khop_mask_hh = HexNeighbors.compute_khop_mask(env._adjacency_matrix, k=k_hops)
            khop_indices, _, max_k = HexNeighbors.khop_to_padded_indices(khop_mask_hh)
            if max_k < max_k_neighbors:
                pad = torch.full(
                    (khop_indices.shape[0], max_k_neighbors - max_k),
                    -1,
                    dtype=torch.long,
                    device=khop_indices.device,
                )
                khop_indices = torch.cat([khop_indices, pad], dim=1)
            elif max_k > max_k_neighbors:
                khop_indices = khop_indices[:, :max_k_neighbors]
            padded_khop_mask = khop_indices != -1
            agent.set_khop_data(khop_indices, padded_khop_mask)
        except Exception as exc:
            print(f"[GCN] Error computing K-hop mask: {exc}")

    agent.to(device)
    agent.eval()
    return agent


def load_ppo_agent(
    checkpoint: Dict[str, Any],
    config: Config,
    env: GPUEnvironmentV2,
    device: torch.device,
) -> PPOAgent:
    raw_state_dict = checkpoint.get("agent", checkpoint.get("agent_state_dict", {}))
    raw_state_dict = clean_state_dict(raw_state_dict)

    num_hexes = env.num_hexes
    if "actor.reposition_head.weight" in raw_state_dict:
        num_hexes = int(raw_state_dict["actor.reposition_head.weight"].shape[0])
        config.environment.num_hexes = num_hexes

    vehicle_feature_dim = getattr(env, "_vehicle_feature_dim", 16)
    context_dim = getattr(env, "_context_dim", 9)
    state_dim = config.environment.num_vehicles * vehicle_feature_dim + config.environment.num_hexes * getattr(env, "_hex_feature_dim", 5) + context_dim

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=3,
        num_hexes=config.environment.num_hexes,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[256, 256],
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_eps=config.training.clip_eps,
        vf_coef=config.training.vf_coef,
        ent_coef=config.training.ent_coef,
        lr_actor=config.training.learning_rate.actor,
        lr_critic=config.training.learning_rate.critic,
        dropout=config.network.dropout if hasattr(config, "network") else 0.1,
        device=str(device),
        num_vehicles=config.environment.num_vehicles,
        vehicle_feature_dim=vehicle_feature_dim,
        hex_feature_dim=getattr(env, "_hex_feature_dim", 5),
        context_dim=context_dim,
        max_trips=config.environment.max_trips_per_step,
    )

    if raw_state_dict:
        agent.load_state_dict(raw_state_dict, strict=False)
        print(f"[Checkpoint] Loaded {len(raw_state_dict)} keys into PPOAgent")

    agent.to(device)
    agent.eval()
    return agent


def create_milp_assigner(args: argparse.Namespace, config: Config, env: GPUEnvironmentV2, device: torch.device):
    if not args.milp:
        return None
    try:
        from gpu_core.training.milp_assignment import MILPAssignment
    except ImportError as exc:
        print(f"[MILP] Could not import MILPAssignment ({exc}); continuing without MILP")
        return None

    step_hours = config.episode.step_duration_minutes / 60.0
    station_positions = None
    if hasattr(env, "station_state") and hasattr(env.station_state, "hex_ids"):
        station_positions = env.station_state.hex_ids.detach().cpu().numpy()

    milp_assigner = MILPAssignment(
        num_vehicles=env.num_vehicles,
        num_hexes=env.num_hexes,
        num_stations=env.station_state.num_stations if hasattr(env.station_state, "num_stations") else config.environment.num_stations,
        device=str(device),
        delta_t=step_hours,
        mu=4.0,
        p_max_s=config.physics.charge_power_kw,
        p_min=20.0,
        p_max_feed=args.milp_p_max_feed,
        charge_action_penalty=0.2,
        lambda_power=0.02,
        max_pickup_distance=getattr(env, "max_pickup_distance", 5.0),
        station_positions=station_positions,
    )
    print(f"[MILP] Assigner initialised (delta_t={step_hours:.4f}h, p_max_feed={args.milp_p_max_feed:.1f}kW, p_min=20.0kW, charge_action_penalty=0.2, lambda_power=0.02)")
    return milp_assigner


def build_trip_mask(env: GPUEnvironmentV2, max_trips: int) -> Optional[torch.Tensor]:
    if not hasattr(env, "trip_state"):
        return None
    unassigned_mask = env.trip_state.get_unassigned_mask()
    if not unassigned_mask.any():
        return None
    available_indices = unassigned_mask.nonzero(as_tuple=True)[0]
    num_to_copy = min(len(available_indices), max_trips)
    trip_mask = torch.zeros(max_trips, dtype=torch.bool, device=env.device)
    trip_mask[:num_to_copy] = True
    return trip_mask


def format_minute_of_day(minute_of_day: int) -> str:
    hours = minute_of_day // 60
    minutes = minute_of_day % 60
    return f"{hours:02d}:{minutes:02d}"


def collect_sac_timeseries(
    env: GPUEnvironmentV2,
    agent: FleetSACAgent,
    milp_assigner,
    max_steps: int,
    label: str,
    checkpoint_path: str,
    day: str,
    step_duration_minutes: float,
) -> List[Dict[str, Any]]:
    env.reset()
    rows: List[Dict[str, Any]] = []

    for step in range(max_steps):
        with torch.no_grad():
            policy_inputs = env.get_fleet_policy_inputs()
            select_kwargs = dict(
                hex_features=policy_inputs["hex_features"],
                hex_vehicle_summary=policy_inputs["hex_vehicle_summary"],
                context_features=policy_inputs["context_features"],
                vehicle_hex_ids=policy_inputs["vehicle_hex_ids"],
                vehicle_socs=policy_inputs["vehicle_socs"],
                vehicle_status=policy_inputs["vehicle_status"],
                temperature=1.0,
                deterministic=True,
            )
            idle_steps = policy_inputs.get("idle_steps")
            if idle_steps is not None:
                try:
                    fleet_out = agent.select_action_fleet(idle_steps=idle_steps, **select_kwargs)
                except TypeError as exc:
                    if "idle_steps" not in str(exc):
                        raise
                    fleet_out = agent.select_action_fleet(**select_kwargs)
            else:
                fleet_out = agent.select_action_fleet(**select_kwargs)

            action_type = fleet_out.action_type
            reposition_target = fleet_out.reposition_target
            vehicle_charge_power = fleet_out.vehicle_charge_power
            milp_serve_trip_ids = None

            if milp_assigner is not None:
                unassigned = env.trip_state.get_unassigned_mask()
                if unassigned.any():
                    trip_indices = unassigned.nonzero(as_tuple=True)[0]
                    trip_pickups = env.trip_state.pickup_hex[trip_indices]
                    trip_dropoffs = env.trip_state.dropoff_hex[trip_indices]
                    trip_fares = env.trip_state.fare[trip_indices]
                else:
                    trip_indices = torch.empty(0, dtype=torch.long, device=env.device)
                    trip_pickups = torch.empty(0, dtype=torch.long, device=env.device)
                    trip_dropoffs = torch.empty(0, dtype=torch.long, device=env.device)
                    trip_fares = torch.empty(0, dtype=torch.float32, device=env.device)

                available_mask = env.fleet_state.get_available_mask(env.current_step)
                milp_result = milp_assigner.assign_from_fleet(
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
                action_type = milp_result.action_types
                reposition_target = milp_result.reposition_targets
                vehicle_charge_power = milp_result.vehicle_charge_power

                milp_serve_trip_ids = torch.full(
                    (env.num_vehicles,), -1, dtype=torch.long, device=env.device
                )
                serve_mask = action_type == 0
                if serve_mask.any() and unassigned.any():
                    serve_vehicle_indices = serve_mask.nonzero(as_tuple=True)[0]
                    milp_trip_pool_indices = milp_result.serve_targets[serve_mask]
                    valid = (milp_trip_pool_indices >= 0) & (milp_trip_pool_indices < len(trip_indices))
                    if valid.any():
                        milp_serve_trip_ids[serve_vehicle_indices[valid]] = trip_indices[milp_trip_pool_indices[valid]]

        _, _, done_tensor, _ = env.step(
            action_type,
            reposition_target,
            None,
            vehicle_charge_power=vehicle_charge_power,
            milp_serve_trip_ids=milp_serve_trip_ids,
        )

        charging_mask = env.fleet_state.status == VehicleStatus.CHARGING.value
        feeder_power_kw = float(env.fleet_state.charge_power[charging_mask].sum().item()) if charging_mask.any() else 0.0
        print(f"  [Feeder] Step {step + 1}/{max_steps}: {feeder_power_kw:.1f} kW")
        total_charging_power_kw = float(env.fleet_state.charge_power[charging_mask].sum().item()) if charging_mask.any() else 0.0
        num_vehicles_charging = int(charging_mask.sum().item())
        mean_power = total_charging_power_kw / num_vehicles_charging if num_vehicles_charging > 0 else 0.0
        minute_of_day = int(round(step * step_duration_minutes))

        rows.append(
            {
                "label": label,
                "checkpoint": checkpoint_path,
                "day": day,
                "step": step,
                "minute_of_day": minute_of_day,
                "timestamp_hhmm": format_minute_of_day(minute_of_day),
                "total_charging_power_kw": total_charging_power_kw,
                "num_vehicles_charging": num_vehicles_charging,
                "mean_power_per_vehicle_kw": mean_power,
            }
        )

        if bool(done_tensor.item()):
            break

    return rows


def collect_ppo_timeseries(
    env: GPUEnvironmentV2,
    agent: PPOAgent,
    max_steps: int,
    label: str,
    checkpoint_path: str,
    day: str,
    step_duration_minutes: float,
) -> List[Dict[str, Any]]:
    state = env.reset()
    rows: List[Dict[str, Any]] = []

    for step in range(max_steps):
        available_mask = env.get_available_actions()
        trip_mask = build_trip_mask(env, agent._max_trips)

        with torch.no_grad():
            ppo_out = agent.select_action(
                state=state,
                action_mask=available_mask,
                reposition_mask=None,
                trip_mask=trip_mask,
                deterministic=True,
            )

        action_type = ppo_out.action_type
        reposition_target = ppo_out.reposition_target
        selected_trip = ppo_out.selected_trip

        if action_type.dim() > 1:
            action_type = action_type.squeeze(0)
            reposition_target = reposition_target.squeeze(0)
            if selected_trip is not None and selected_trip.dim() > 1:
                selected_trip = selected_trip.squeeze(0)

        next_state, _, done_tensor, _ = env.step(action_type, reposition_target, selected_trip)
        state = next_state

        charging_mask = env.fleet_state.status == VehicleStatus.CHARGING.value
        total_charging_power_kw = float(env.fleet_state.charge_power[charging_mask].sum().item()) if charging_mask.any() else 0.0
        num_vehicles_charging = int(charging_mask.sum().item())
        mean_power = total_charging_power_kw / num_vehicles_charging if num_vehicles_charging > 0 else 0.0
        minute_of_day = int(round(step * step_duration_minutes))

        rows.append(
            {
                "label": label,
                "checkpoint": checkpoint_path,
                "day": day,
                "step": step,
                "minute_of_day": minute_of_day,
                "timestamp_hhmm": format_minute_of_day(minute_of_day),
                "total_charging_power_kw": total_charging_power_kw,
                "num_vehicles_charging": num_vehicles_charging,
                "mean_power_per_vehicle_kw": mean_power,
            }
        )

        if bool(done_tensor.item()):
            break

    return rows


def build_summary(rows: List[Dict[str, Any]], checkpoint_path: str, label: str, day: str, algo: str, step_duration_minutes: float, milp_enabled: bool) -> Dict[str, Any]:
    step_hours = step_duration_minutes / 60.0
    total_energy_kwh = sum(row["total_charging_power_kw"] * step_hours for row in rows)
    peak_row = max(rows, key=lambda row: row["total_charging_power_kw"]) if rows else None
    average_power = sum(row["total_charging_power_kw"] for row in rows) / len(rows) if rows else 0.0

    return {
        "label": label,
        "checkpoint": checkpoint_path,
        "day": day,
        "algo": algo,
        "milp_enabled": milp_enabled,
        "num_steps": len(rows),
        "step_duration_minutes": step_duration_minutes,
        "average_charging_power_kw": average_power,
        "peak_charging_power_kw": peak_row["total_charging_power_kw"] if peak_row else 0.0,
        "peak_step": peak_row["step"] if peak_row else None,
        "peak_time": peak_row["timestamp_hhmm"] if peak_row else None,
        "total_charged_energy_kwh": total_energy_kwh,
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "checkpoint",
        "day",
        "step",
        "minute_of_day",
        "timestamp_hhmm",
        "total_charging_power_kw",
        "num_vehicles_charging",
        "mean_power_per_vehicle_kw",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Output] Wrote CSV to {output_path}")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[Output] Wrote JSON to {output_path}")


def main() -> None:
    args = parse_args()
    checkpoint_path = str(Path(args.checkpoint).resolve())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_state_dict = checkpoint.get("agent", checkpoint.get("agent_state_dict", {}))
    raw_state_dict = clean_state_dict(raw_state_dict)
    algo = infer_algo(checkpoint.get("config", {}), raw_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = args.label or Path(checkpoint_path).stem

    print("=" * 60)
    print("Power Grid Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Label: {label}")
    print(f"Day: {args.day}")
    print(f"Algorithm: {algo}")
    print(f"Device: {device}")

    config = build_config(args, checkpoint, algo)
    trip_loader = create_trip_loader(args, device)
    env = create_environment(config, device, trip_loader=trip_loader)
    apply_training_horizon_normalization(env, checkpoint, config)
    if args.milp:
        env.set_feeder_power_limit(args.milp_p_max_feed)
        print(f"[MILP] Runtime feeder cap in env: {args.milp_p_max_feed:.1f}kW")

    print(f"[Environment] {env.num_vehicles} vehicles | {env.num_hexes} hexes | {config.episode.duration_hours:.1f} hours")

    if algo == "ppo":
        if args.milp:
            print("[MILP] Ignoring --milp for PPO/MAPPO checkpoints")
        agent = load_ppo_agent(checkpoint, config, env, device)
        rows = collect_ppo_timeseries(
            env=env,
            agent=agent,
            max_steps=min(env.episode_steps, args.max_steps) if args.max_steps else env.episode_steps,
            label=label,
            checkpoint_path=checkpoint_path,
            day=args.day,
            step_duration_minutes=config.episode.step_duration_minutes,
        )
    else:
        agent = load_fleet_agent(checkpoint, config, env, device)

        # Match evaluate.py mismatch #2 bandaid:
        # treat long-idle as >= 1 hour to force reposition earlier.
        idle_repos_steps = max(1, int(round(60.0 / config.episode.step_duration_minutes)))
        agent.assigner.idle_force_charge_steps = idle_repos_steps
        print(f"[Override] Long-idle reposition threshold: {idle_repos_steps} steps (~1 hour)")

        milp_assigner = create_milp_assigner(args, config, env, device)
        rows = collect_sac_timeseries(
            env=env,
            agent=agent,
            milp_assigner=milp_assigner,
            max_steps=min(env.episode_steps, args.max_steps) if args.max_steps else env.episode_steps,
            label=label,
            checkpoint_path=checkpoint_path,
            day=args.day,
            step_duration_minutes=config.episode.step_duration_minutes,
        )

    summary = build_summary(
        rows=rows,
        checkpoint_path=checkpoint_path,
        label=label,
        day=args.day,
        algo=algo,
        step_duration_minutes=config.episode.step_duration_minutes,
        milp_enabled=bool(args.milp and algo != "ppo"),
    )

    print("[Summary]")
    print(f"  Steps: {summary['num_steps']}")
    print(f"  Average charging power: {summary['average_charging_power_kw']:.3f} kW")
    print(f"  Peak charging power: {summary['peak_charging_power_kw']:.3f} kW at step {summary['peak_step']} ({summary['peak_time']})")
    print(f"  Total charged energy: {summary['total_charged_energy_kwh']:.3f} kWh")

    if args.out_csv:
        write_csv(args.out_csv, rows)
    if args.out_json:
        payload = {
            "summary": summary,
            "timeseries": rows,
        }
        write_json(args.out_json, payload)


if __name__ == "__main__":
    main()
