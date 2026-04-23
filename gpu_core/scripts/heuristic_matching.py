#!/usr/bin/env python3
"""
Heuristic Matching Baseline for EV Fleet Management

This script runs a heuristic matching algorithm (SOC-aware + profit-aware) to serve
as a baseline for comparing with the actor-critic model.

Heuristic Algorithm:
1. SOC-aware charging: Charge if SOC < critical_threshold (default: 20%)
2. Profit-aware serving: Serve trips with highest profit (fare - costs)
3. Idle: If no profitable actions available

Usage:
    python3 heuristic_matching.py \
        --config gpu_core/scripts/config.yaml \
        --real-data data/nyc_full/trips_processed.parquet \
        --start-date 2009-01-15 \
        --end-date 2009-01-15 \
        --trip-sample 0.2 \
        --num-vehicles 1000 \
        --num-hexes 1000 \
        --critical-soc-threshold 20.0 \
        --output results/heuristic_baseline.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import ConfigLoader, Config
from gpu_core.spatial.grid import HexGrid
from gpu_core.simulator.environment import GPUEnvironment
from gpu_core.simulator.environment import GPUEnvironmentV2
from gpu_core.data.real_trip_loader import RealTripLoader
from gpu_core.spatial.assignment import UltraFastGreedyAssignment


class HeuristicMatcher:
    """
    Heuristic matching algorithm: SOC-aware charging + Profit-aware serving.
    """
    
    def __init__(
        self,
        env,
        critical_soc_threshold: float = 20.0,
        target_soc_threshold: float = 90.0,
        max_pickup_distance: float = 150.0,
        min_profit_threshold: float = 0.0,  # Minimum profit to serve (can be negative)
    ):
        self.env = env
        self.critical_soc_threshold = critical_soc_threshold
        self.target_soc_threshold = target_soc_threshold
        self.max_pickup_distance = max_pickup_distance
        self.min_profit_threshold = min_profit_threshold
        
        # Get cost parameters from config
        self.driving_cost_per_km = env.config.reward.driving_cost_per_km
        self.electricity_cost_per_kwh = env.config.reward.electricity_cost_per_kwh
        self.energy_per_km = env.config.physics.energy_per_km
        
        # Legacy solver retained for compatibility with older experiments
        self.assignment_solver = UltraFastGreedyAssignment(
            device=torch.device(env.device),
            max_cost=1e6
        )

    
    def compute_profit_matrix(
        self,
        vehicle_indices: torch.Tensor,
        vehicle_positions: torch.Tensor,
        vehicle_socs: torch.Tensor,
        trip_indices: torch.Tensor,
        trip_pickup_hexes: torch.Tensor,
        trip_fares: torch.Tensor,
        trip_distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute profit matrix [num_vehicles, num_trips].
        
        Profit = fare - pickup_cost - trip_cost - energy_cost
        
        Returns negative profit (as cost) for minimization.
        """
        num_vehicles = len(vehicle_indices)
        num_trips = len(trip_indices)
        
        if num_trips == 0:
            return torch.zeros(num_vehicles, 0, device=self.env.device)
        
        # Get distance matrix
        distance_matrix = self.env.hex_grid.distance_matrix._distances
        
        # Compute pickup distances [num_vehicles, num_trips]
        vehicle_pos_expanded = vehicle_positions.unsqueeze(1).expand(num_vehicles, num_trips)
        trip_pickup_expanded = trip_pickup_hexes.unsqueeze(0).expand(num_vehicles, num_trips)
        
        pickup_distances = distance_matrix[
            vehicle_pos_expanded.reshape(-1),
            trip_pickup_expanded.reshape(-1)
        ].reshape(num_vehicles, num_trips)
        
        # Mask out trips too far
        valid_mask = pickup_distances <= self.max_pickup_distance
        
        # Trip distances [num_trips]
        trip_distances_expanded = trip_distances.unsqueeze(0).expand(num_vehicles, num_trips)
        
        # Total distance = pickup + trip
        total_distances = pickup_distances + trip_distances_expanded
        
        # Energy needed [num_vehicles, num_trips]
        energy_needed = total_distances * self.energy_per_km
        
        # Check energy feasibility: vehicle SOC must be sufficient
        energy_available = vehicle_socs.unsqueeze(1)  # kWh
        feasible_mask = energy_available >= (energy_needed + 10.0)  # 10 kWh reserve
        valid_mask = valid_mask & feasible_mask
        
        # Costs
        pickup_cost = pickup_distances * self.driving_cost_per_km
        trip_cost = trip_distances_expanded * self.driving_cost_per_km
        energy_cost = energy_needed * self.electricity_cost_per_kwh
        
        # Profit = fare - all costs
        fares_expanded = trip_fares.unsqueeze(0).expand(num_vehicles, num_trips)
        profit = fares_expanded - pickup_cost - trip_cost - energy_cost
        
        # Mask invalid/unprofitable trips
        profit[~valid_mask] = -1e6  # Very negative for invalid
        profit[profit < self.min_profit_threshold] = -1e6  # Filter unprofitable
        
        # Return negative profit (as cost) for minimization assignment
        return -profit
    
    def select_actions(
        self,
        state
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Select actions using nearest-feasible serving, station-aware charging,
        and relocation to largest supply-demand gap for simulator environments.

        Returns:
            (action_type, reposition_target)
            - action_type: [num_vehicles] with env IDs (0=SERVE, 1=CHARGE, 2=REPOSITION)
            - reposition_target: [num_vehicles] target hex for reposition actions
        """
        device = self.env.device
        num_vehicles = self.env.num_vehicles
        action_type = torch.zeros(num_vehicles, dtype=torch.long, device=device)  # default SERVE
        reposition_target = self.env.fleet_state.positions.clone().long()

        available_mask = self.env.fleet_state.get_available_mask(self.env.current_step)
        available_indices = available_mask.nonzero(as_tuple=True)[0]
        if len(available_indices) == 0:
            return action_type, reposition_target

        distance_matrix = self.env.hex_grid.distance_matrix._distances
        assigned_vehicles = torch.zeros(num_vehicles, dtype=torch.bool, device=device)

        # Step 1: nearest feasible vehicle per unassigned trip (oldest trips first)
        unassigned_trip_mask = self.env.trip_state.get_unassigned_mask()
        if unassigned_trip_mask.any():
            trip_indices = unassigned_trip_mask.nonzero(as_tuple=True)[0]
            trip_wait = self.env.trip_state.wait_steps[trip_indices].float()
            trip_fares = self.env.trip_state.fare[trip_indices]
            trip_priority = trip_wait * 1e6 + trip_fares
            sorted_order = torch.argsort(trip_priority, descending=True)
            ordered_trips = trip_indices[sorted_order]

            for trip_idx in ordered_trips.tolist():
                free_candidates = available_indices[~assigned_vehicles[available_indices]]
                if len(free_candidates) == 0:
                    break

                pickup_hex = int(self.env.trip_state.pickup_hex[trip_idx].item())
                trip_distance = self.env.trip_state.distance_km[trip_idx]

                candidate_positions = self.env.fleet_state.positions[free_candidates]
                candidate_socs = self.env.fleet_state.socs[free_candidates]

                pickup_distances = distance_matrix[candidate_positions, pickup_hex]
                total_distance = pickup_distances + trip_distance
                energy_needed = total_distance * self.energy_per_km

                feasible_mask = (
                    (pickup_distances <= self.max_pickup_distance)
                    & (candidate_socs >= (energy_needed + 10.0))
                )

                if not feasible_mask.any():
                    continue

                feasible_candidates = free_candidates[feasible_mask]
                feasible_pickup_distances = pickup_distances[feasible_mask]
                chosen_local = torch.argmin(feasible_pickup_distances)
                chosen_vehicle = int(feasible_candidates[chosen_local].item())

                action_type[chosen_vehicle] = 0  # SERVE
                assigned_vehicles[chosen_vehicle] = True

        # Step 2: assign low-SoC available vehicles to CHARGE
        unassigned_available = available_indices[~assigned_vehicles[available_indices]]
        if len(unassigned_available) > 0:
            low_soc_mask = self.env.fleet_state.socs[unassigned_available] < self.critical_soc_threshold
            charge_candidates = unassigned_available[low_soc_mask]
            if len(charge_candidates) > 0:
                action_type[charge_candidates] = 1  # CHARGE
                assigned_vehicles[charge_candidates] = True

        # Step 3: relocate all remaining available vehicles (not matched to serve/charge)
        unassigned_available = available_indices[~assigned_vehicles[available_indices]]
        if len(unassigned_available) > 0:
            num_hexes = self.env.num_hexes
            demand_counts = torch.zeros(num_hexes, dtype=torch.float32, device=device)
            supply_counts = torch.zeros(num_hexes, dtype=torch.float32, device=device)

            remaining_trip_mask = self.env.trip_state.get_unassigned_mask()
            if remaining_trip_mask.any():
                remaining_pickups = self.env.trip_state.pickup_hex[remaining_trip_mask].long()
                demand_counts.scatter_add_(
                    0,
                    remaining_pickups,
                    torch.ones_like(remaining_pickups, dtype=torch.float32),
                )

            current_positions = self.env.fleet_state.positions[unassigned_available].long()
            supply_counts.scatter_add_(
                0,
                current_positions,
                torch.ones_like(current_positions, dtype=torch.float32),
            )

            gap = demand_counts - supply_counts
            _, target_hex = torch.max(gap, dim=0)
            target_hex = int(target_hex.item())

            action_type[unassigned_available] = 2  # REPOSITION
            reposition_target[unassigned_available] = target_hex

        return action_type, reposition_target


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run heuristic matching baseline simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    
    # Environment
    parser.add_argument('--num-vehicles', type=int, default=None,
                        help='Number of vehicles in fleet')
    parser.add_argument('--num-hexes', type=int, default=None,
                        help='Number of hexagons in grid')
    parser.add_argument('--episode-duration-hours', type=float, default=None,
                        help='Episode duration in hours (e.g., 24.0 for full day)')
    parser.add_argument('--env-v2', action='store_true', default=False,
                        help='Use GPUEnvironmentV2')
    
    # Data
    parser.add_argument('--real-data', type=str, default=None,
                        help='Path to real trip data parquet file')
    parser.add_argument('--trip-sample', type=float, default=1.0,
                        help='Sample ratio for trip data (0.0-1.0). Default: 1.0 (all trips)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Filter trips from this date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Filter trips until this date (YYYY-MM-DD)')
    parser.add_argument('--target-h3-resolution', type=int, default=None,
                        help='Target H3 resolution')
    parser.add_argument('--max-hex-count', type=int, default=None,
                        help='Maximum number of hexes')
    
    # Heuristic parameters
    parser.add_argument('--critical-soc-threshold', type=float, default=20.0,
                        help='SOC threshold for forced charging (percentage)')
    parser.add_argument('--max-pickup-distance', type=float, default=5.0,
                        help='Maximum pickup distance in km (default: 5.0)')
    parser.add_argument('--min-profit-threshold', type=float, default=0.0,
                        help='Minimum profit to serve a trip')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def create_config(args) -> Config:
    """Create config from YAML and CLI arguments."""
    if args.config:
        config = ConfigLoader.from_yaml(args.config)
    else:
        config = Config()
    
    # Override with CLI args
    if args.num_vehicles is not None:
        config.environment.num_vehicles = args.num_vehicles
    if args.num_hexes is not None:
        config.environment.num_hexes = args.num_hexes
    if args.episode_duration_hours is not None:
        config.episode.duration_hours = args.episode_duration_hours
        print(f"[Episode] Duration set to {args.episode_duration_hours} hours "
              f"({config.episode.steps_per_episode} steps @ {config.episode.step_duration_minutes} min/step)")
    
    return config


def create_environment(config: Config, device: torch.device, args, trip_loader: Optional[RealTripLoader] = None):
    """Create GPU environment."""
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
    else:
        # Fallback: synthetic grid
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
    
    if args.env_v2:
        env = GPUEnvironmentV2(
            config=config,
            hex_grid=hex_grid,
            trip_loader=trip_loader,
            device=device_str
        )
    else:
        env = GPUEnvironment(
            config=config,
            hex_grid=hex_grid,
            trip_loader=trip_loader,
            device=device_str
        )
    
    return env


def run_heuristic_simulation(
    env,
    heuristic: HeuristicMatcher,
    max_steps: Optional[int] = None
) -> Dict:
    """
    Run heuristic simulation and collect metrics.
    
    Returns:
        Dictionary with simulation results
    """
    # Initialize metrics
    total_revenue = 0.0
    total_charging_cost = 0.0
    total_driving_cost = 0.0
    total_trips_served = 0
    total_trips_loaded = 0
    total_trips_dropped = 0
    
    # Track action counts
    action_counts = defaultdict(int)
    
    # Reset environment
    state = env.reset()
    max_steps = max_steps or env.episode_steps
    
    print(f"\nRunning heuristic simulation for {max_steps} steps...")
    start_time = time.time()
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Select actions using heuristic
        action_type, reposition_target = heuristic.select_actions(state)
        
        # Count actions (only for available vehicles)
        available_mask = env.fleet_state.get_available_mask(env.current_step)
        available_actions = action_type[available_mask]
        for action_id in available_actions.cpu().numpy():
            if action_id == 0:
                action_counts['SERVE'] += 1
            elif action_id == 1:
                action_counts['CHARGE'] += 1
            elif action_id == 2:
                action_counts['REPOSITION'] += 1
        
        # Step environment
        next_state, reward, done_tensor, info = env.step(action_type, reposition_target)
        done = done_tensor.item() if isinstance(done_tensor, torch.Tensor) else done_tensor
        
        # Note: revenue, energy_cost, trips_served, trips_loaded, trips_dropped 
        # in info are already CUMULATIVE (accumulated by environment),
        # so we just take the final values, don't accumulate again
        total_revenue = info.revenue
        total_charging_cost = info.energy_cost
        total_driving_cost = info.driving_cost
        total_trips_served = info.trips_served
        total_trips_loaded = info.trips_loaded
        total_trips_dropped = info.trips_dropped
        
        state = next_state
        step += 1
        
        # Progress update
        if step % 20 == 0:
            print(f"  Step {step}/{max_steps}: Revenue=${total_revenue:.2f}, "
                  f"Trips={total_trips_served}/{total_trips_loaded}, "
                  f"ServiceRate={total_trips_served/max(total_trips_loaded,1)*100:.1f}%")
    
    elapsed_time = time.time() - start_time
    
    net_profit = total_revenue - total_driving_cost - total_charging_cost
    
    # Service rate
    service_rate = total_trips_served / max(total_trips_loaded, 1)
    
    # Get final state info
    final_avg_soc = info.avg_soc if hasattr(info, 'avg_soc') else 0.0
    
    results = {
        'total_trips_loaded': int(total_trips_loaded),
        'total_trips_served': int(total_trips_served),
        'total_trips_dropped': int(total_trips_dropped),
        'service_rate': float(service_rate),
        'total_revenue': float(total_revenue),
        'total_driving_cost': float(total_driving_cost),
        'total_charging_cost': float(total_charging_cost),
        'net_profit': float(net_profit),
        'final_avg_soc': float(final_avg_soc),
        'action_counts': dict(action_counts),
        'simulation_time_seconds': float(elapsed_time),
        'steps_completed': int(step)
    }
    
    return results


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load config
    config = create_config(args)
    
    # Load trip data
    trip_loader = None
    if args.real_data:
        data_path = Path(args.real_data)
        if data_path.exists():
            print(f"\nLoading trip data from: {data_path}")
            if args.trip_sample is not None and args.trip_sample < 1.0:
                print(f"  Using {args.trip_sample*100:.1f}% sample of trips")
            try:
                trip_loader = RealTripLoader(
                    parquet_path=str(data_path),
                    device=str(device),
                    sample_ratio=args.trip_sample if args.trip_sample is not None else 1.0,
                    target_h3_resolution=args.target_h3_resolution,
                    max_hex_count=args.max_hex_count,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
                trip_loader.load()
                config.environment.num_hexes = trip_loader.num_hexes
                print(f"  Loaded {trip_loader.total_trips:,} trips after sampling")
            except Exception as e:
                print(f"Failed to load trip data: {e}")
                sys.exit(1)
        else:
            print(f"Trip data file not found: {data_path}")
            sys.exit(1)
    
    print(f"\nCreating heuristic environment...")
    print(f"  Vehicles: {config.environment.num_vehicles}")

    env = create_environment(config=config, device=device, args=args, trip_loader=trip_loader)

    # Sync hex count with runtime environment data
    if hasattr(env, 'num_hexes'):
        config.environment.num_hexes = int(env.num_hexes)

    print(f"  Hexes: {config.environment.num_hexes}")

    heuristic = HeuristicMatcher(
        env=env,
        critical_soc_threshold=args.critical_soc_threshold,
        max_pickup_distance=args.max_pickup_distance,
        min_profit_threshold=args.min_profit_threshold
    )
    
    print(f"\nInitializing heuristic matcher...")
    print(f"  Critical SOC threshold: {args.critical_soc_threshold}%")
    print(f"  Max pickup distance: {args.max_pickup_distance} km")
    print(f"  Min profit threshold: ${args.min_profit_threshold:.2f}")
    print(f"  Episode duration: {config.episode.duration_hours} hours ({config.episode.steps_per_episode} steps)")
    
    # Run simulation (true environment only)
    results = run_heuristic_simulation(env, heuristic)
    
    # Add metadata
    results['config'] = {
        'num_vehicles': config.environment.num_vehicles,
        'num_hexes': config.environment.num_hexes,
        'episode_duration_hours': config.episode.duration_hours,
        'steps_per_episode': config.episode.steps_per_episode,
        'trip_sample_ratio': args.trip_sample if args.trip_sample is not None else 1.0,
        'critical_soc_threshold': args.critical_soc_threshold,
        'max_pickup_distance': args.max_pickup_distance,
        'min_profit_threshold': args.min_profit_threshold,
    }
    
    # Add driving cost to results if available
    if 'total_driving_cost' in results:
        results['config']['driving_cost_tracked'] = True
    else:
        results['total_driving_cost'] = 0.0
    
    if args.start_date:
        results['date'] = args.start_date
    
    # Print summary
    print("\n" + "="*60)
    print("HEURISTIC MATCHING RESULTS")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Vehicles: {results['config']['num_vehicles']} (số xe trong fleet)")
    print(f"  Steps: {results['steps_completed']} (episode duration: {results['steps_completed']*5/60:.1f} hours @ 5 min/step)")
    print(f"  Trip sample: {results['config']['trip_sample_ratio']*100:.1f}%")
    print(f"\nTrips:")
    print(f"  Loaded: {results['total_trips_loaded']:,} (tổng trips xuất hiện trong episode)")
    print(f"  Served: {results['total_trips_served']:,} (số trips được serve)")
    print(f"  Dropped: {results['total_trips_dropped']:,} (số trips bị drop do quá thời gian chờ)")
    print(f"  Service rate: {results['service_rate']*100:.2f}% (served/loaded)")
    print(f"\nFinancial:")
    print(f"  Revenue: ${results['total_revenue']:,.2f}")
    print(f"  Driving cost: ${results.get('total_driving_cost', 0.0):,.2f}")
    print(f"  Charging cost: ${results['total_charging_cost']:,.2f}")
    print(f"  Net profit: ${results['net_profit']:,.2f} (Revenue - Driving Cost - Charging Cost)")
    
    # Print hourly or daily metrics depending on duration
    # If duration > 48 hours, print daily metrics to avoid flooding the console
    if 'daily_metrics' in results and results['daily_metrics'] and results['config']['episode_duration_hours'] > 48.0:
        print(f"\nDaily metrics:")
        for day in sorted(results['daily_metrics'].keys()):
            day_data = results['daily_metrics'][day]
            print(f"  Day {day+1:2d}: Revenue=${day_data['revenue']:>10,.2f}, "
                  f"Net Profit=${day_data['net_profit']:>10,.2f}")
    elif 'hourly_metrics' in results and results['hourly_metrics']:
        print(f"\nHourly metrics:")
        for hour in sorted(results['hourly_metrics'].keys()):
            hour_data = results['hourly_metrics'][hour]
            print(f"  Hour {hour:2d}: Revenue=${hour_data['revenue']:>10,.2f}, "
                  f"Net Profit=${hour_data['net_profit']:>10,.2f}")
    
    print(f"\nOther metrics:")
    print(f"  Final avg SOC: {results['final_avg_soc']:.1f}% (battery level trung bình)")
    print(f"  Action distribution: {results['action_counts']}")
    print(f"  Simulation time: {results['simulation_time_seconds']:.2f}s")
    print("="*60)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    else:
        print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

