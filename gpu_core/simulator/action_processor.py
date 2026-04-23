"""
Action Processing for EV Fleet Environment.

Handles SERVE, CHARGE, REPOSITION action execution.
Separated from main environment for modularity.
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..state import TensorFleetState, TensorTripState, TensorStationState
    from ..spatial import HexGrid
    from ..spatial.assignment import TripAssigner, StationAssigner
    from .dynamics import EnergyDynamics, TimeDynamics
from ..state import VehicleStatus


class ActionProcessor:
    """
    Processes vehicle actions: SERVE, CHARGE, REPOSITION.
    
    Features:
    - Vectorized batch processing for GPU efficiency
    - Optional TripAssigner/StationAssigner for optimal matching
    - Energy consumption tracking
    """
    
    def __init__(
        self,
        config: 'Config',
        hex_grid: 'HexGrid',
        fleet_state: 'TensorFleetState',
        trip_state: 'TensorTripState',
        station_state: 'TensorStationState',
        energy_dynamics: 'EnergyDynamics',
        time_dynamics: 'TimeDynamics',
        device: torch.device,
        trip_assigner: Optional['TripAssigner'] = None,
        station_assigner: Optional['StationAssigner'] = None,
        max_pickup_distance: float = 5.0,
        feeder_power_limit_kw: Optional[float] = None,
    ):
        self.config = config
        self.hex_grid = hex_grid
        self.fleet_state = fleet_state
        self.trip_state = trip_state
        self.station_state = station_state
        self.energy_dynamics = energy_dynamics
        self.time_dynamics = time_dynamics
        self.device = device
        self.trip_assigner = trip_assigner
        self.station_assigner = station_assigner
        
        self.num_stations = config.environment.num_stations
        self.max_soc = config.physics.max_soc
        self.max_pickup_distance = max_pickup_distance  # For curriculum learning
        self.feeder_power_limit_kw = float(feeder_power_limit_kw) if feeder_power_limit_kw is not None else None

    def _has_energy_budget(
        self,
        vehicle_indices: torch.Tensor,
        energy_needed: torch.Tensor,
    ) -> torch.Tensor:
        """Check whether vehicles can execute an action while keeping reserve SOC."""
        current_soc = self.fleet_state.socs[vehicle_indices]
        available_energy = current_soc - self.energy_dynamics.min_soc_reserve
        return available_energy >= energy_needed

    def _apply_feeder_cap_mask(self, candidate_powers_kw: torch.Tensor) -> torch.Tensor:
        """Return mask of candidates that can start charging under feeder cap."""
        if self.feeder_power_limit_kw is None or len(candidate_powers_kw) == 0:
            return torch.ones(len(candidate_powers_kw), dtype=torch.bool, device=self.device)

        current_total_kw = float(self.fleet_state.charge_power[self.fleet_state.status == VehicleStatus.CHARGING].sum().item())
        remaining_kw = max(0.0, self.feeder_power_limit_kw - current_total_kw)
        if remaining_kw <= 0.0:
            return torch.zeros(len(candidate_powers_kw), dtype=torch.bool, device=self.device)

        order = torch.argsort(candidate_powers_kw)  # pack smaller powers first
        sorted_powers = candidate_powers_kw[order]
        keep_sorted = torch.cumsum(sorted_powers, dim=0) <= (remaining_kw + 1e-6)

        keep_mask = torch.zeros(len(candidate_powers_kw), dtype=torch.bool, device=self.device)
        if keep_sorted.any():
            keep_mask[order[keep_sorted]] = True
        return keep_mask
    
    def process_serve_actions(
        self,
        serve_mask: torch.Tensor,
        current_step: int,
        selected_trip: Optional[torch.Tensor] = None,  # GCN trip selections [num_vehicles]
        milp_serve_trip_ids: Optional[torch.Tensor] = None,  # MILP pre-resolved trip IDs [num_vehicles]
    ) -> Tuple[int, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Process serve actions using GCN trip selections with Hungarian conflict resolution.
        When milp_serve_trip_ids is provided, uses MILP's pre-resolved vehicle-trip
        assignments directly (no re-matching needed).
        """
        serve_indices = serve_mask.nonzero(as_tuple=True)[0]
        num_attempted = len(serve_indices)

        empty_tensor = torch.tensor([], dtype=torch.long, device=self.device)
        zero = torch.tensor(0.0, device=self.device)

        if num_attempted == 0:
            return 0, zero, zero, 0, empty_tensor, empty_tensor

        unassigned_mask = self.trip_state.get_unassigned_mask()
        if not unassigned_mask.any():
            return 0, zero, zero, num_attempted, empty_tensor, empty_tensor

        trip_indices = unassigned_mask.nonzero(as_tuple=True)[0]

        if milp_serve_trip_ids is not None:
            matched_vehicles, matched_trips = self._match_with_milp_assignments(
                serve_indices, milp_serve_trip_ids, trip_indices
            )
        elif selected_trip is not None:
            matched_vehicles, matched_trips = self._match_with_gcn_and_hungarian(
                serve_indices, selected_trip, trip_indices
            )
        else:
            # Fleet path (SAC/WDRO): use global net-revenue matching instead of greedy.
            # Falls back to greedy automatically when V*T > HUNGARIAN_SIZE_LIMIT.
            matched_vehicles, matched_trips = self._hungarian_net_revenue_matching(
                serve_indices, trip_indices
            )

        trips_served = len(matched_trips)
        if trips_served == 0:
            return 0, zero, zero, num_attempted, empty_tensor, empty_tensor

        fares = self.trip_state.fare[matched_trips]
        trip_distances = self.trip_state.distance_km[matched_trips]
        dropoff_hexes = self.trip_state.dropoff_hex[matched_trips]
        pickup_hexes = self.trip_state.pickup_hex[matched_trips]
        vehicle_positions = self.fleet_state.positions[matched_vehicles]
        pickup_distances = self.hex_grid.distance_matrix._distances[
            vehicle_positions, pickup_hexes
        ]
        total_distances = pickup_distances + trip_distances

        avg_speed_kmh = self.config.physics.avg_speed_kmh
        step_minutes = self.config.episode.step_duration_minutes
        duration_hours = total_distances / avg_speed_kmh
        duration_minutes = duration_hours * 60.0
        durations = (duration_minutes / step_minutes).clamp(min=1).long()

        energy_costs = self.energy_dynamics.compute_consumption(total_distances)
        feasible_mask = self._has_energy_budget(matched_vehicles, energy_costs)
        if not feasible_mask.any():
            return 0, zero, zero, num_attempted, empty_tensor, empty_tensor

        matched_vehicles = matched_vehicles[feasible_mask]
        matched_trips = matched_trips[feasible_mask]
        fares = fares[feasible_mask]
        dropoff_hexes = dropoff_hexes[feasible_mask]
        total_distances = total_distances[feasible_mask]
        durations = durations[feasible_mask]
        energy_costs = energy_costs[feasible_mask]

        trips_served = len(matched_trips)
        if trips_served == 0:
            return 0, zero, zero, num_attempted, empty_tensor, empty_tensor

        self.trip_state.assigned[matched_trips] = True
        self.trip_state.assigned_vehicle[matched_trips] = matched_vehicles

        busy_until = (current_step + durations).long()
        revenues = fares - energy_costs * self.config.reward.electricity_cost_per_kwh
        revenue_per_step = revenues / durations.float()

        self._release_charging_ports(matched_vehicles)
        self.fleet_state.set_serving(
            vehicle_indices=matched_vehicles,
            trip_ids=matched_trips,
            target_hexes=dropoff_hexes,
            busy_until=busy_until,
            revenue_per_step=revenue_per_step,
        )

        self.fleet_state.socs[matched_vehicles] -= energy_costs
        self.fleet_state.socs = torch.clamp(self.fleet_state.socs, 0.0, self.max_soc)

        driving_cost = (total_distances * self.config.reward.driving_cost_per_km).sum()
        num_failed = num_attempted - trips_served
        return trips_served, zero, driving_cost, num_failed, matched_vehicles, matched_trips

    def process_serve_actions_with_preferences(
        self,
        serve_mask: torch.Tensor,
        current_step: int,
        serve_scores: Optional[torch.Tensor] = None,
        preference_weight: float = 0.5
    ) -> Tuple[int, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Process serve actions using TripAssigner with actor preferences.
        """
        serve_indices = serve_mask.nonzero(as_tuple=True)[0]
        num_attempted = len(serve_indices)

        empty_tensor = torch.tensor([], dtype=torch.long, device=self.device)
        zero = torch.tensor(0.0, device=self.device)

        if num_attempted == 0:
            return 0, zero, zero, 0, empty_tensor, empty_tensor

        unassigned_mask = self.trip_state.get_unassigned_mask()
        if not unassigned_mask.any():
            return 0, zero, zero, num_attempted, empty_tensor, empty_tensor

        trip_indices = unassigned_mask.nonzero(as_tuple=True)[0]

        if self.trip_assigner is not None:
            positions = self.fleet_state.positions[serve_indices]
            trip_hexes = self.trip_state.pickup_hex[trip_indices]
            vehicle_prefs = serve_scores[serve_indices][:, :len(trip_indices)] if serve_scores is not None else None
            result = self.trip_assigner.assign(
                vehicle_indices=serve_indices,
                vehicle_positions=positions,
                vehicle_preferences=vehicle_prefs,
                trip_indices=trip_indices,
                trip_pickup_hexes=trip_hexes,
                preference_weight=preference_weight
            )
            matched_vehicles = result.vehicle_indices
            matched_trips = result.target_indices
        else:
            return self.process_serve_actions(serve_mask, current_step)

        if len(matched_vehicles) == 0:
            return 0, zero, zero, num_attempted, empty_tensor, empty_tensor

        fares = self.trip_state.fare[matched_trips]
        trip_distances = self.trip_state.distance_km[matched_trips]
        dropoff_hexes = self.trip_state.dropoff_hex[matched_trips]
        pickup_hexes = self.trip_state.pickup_hex[matched_trips]
        vehicle_positions = self.fleet_state.positions[matched_vehicles]
        pickup_distances = self.hex_grid.distance_matrix._distances[
            vehicle_positions, pickup_hexes
        ]
        total_distances = pickup_distances + trip_distances

        avg_speed_kmh = self.config.physics.avg_speed_kmh
        step_minutes = self.config.episode.step_duration_minutes
        duration_hours = total_distances / avg_speed_kmh
        duration_minutes = duration_hours * 60.0
        durations = (duration_minutes / step_minutes).clamp(min=1).long()

        energy_costs = self.energy_dynamics.compute_consumption(total_distances)
        feasible_mask = self._has_energy_budget(matched_vehicles, energy_costs)
        if not feasible_mask.any():
            return 0, zero, zero, num_attempted, empty_tensor, empty_tensor

        matched_vehicles = matched_vehicles[feasible_mask]
        matched_trips = matched_trips[feasible_mask]
        fares = fares[feasible_mask]
        dropoff_hexes = dropoff_hexes[feasible_mask]
        total_distances = total_distances[feasible_mask]
        durations = durations[feasible_mask]
        energy_costs = energy_costs[feasible_mask]

        trips_served = len(matched_trips)
        if trips_served == 0:
            return 0, zero, zero, num_attempted, empty_tensor, empty_tensor

        self.trip_state.assigned[matched_trips] = True
        self.trip_state.assigned_vehicle[matched_trips] = matched_vehicles

        busy_until = (current_step + durations).long()
        revenues = fares - energy_costs * self.config.reward.electricity_cost_per_kwh
        revenue_per_step = revenues / durations.float()

        self._release_charging_ports(matched_vehicles)
        self.fleet_state.set_serving(
            vehicle_indices=matched_vehicles,
            trip_ids=matched_trips,
            target_hexes=dropoff_hexes,
            busy_until=busy_until,
            revenue_per_step=revenue_per_step
        )

        self.fleet_state.socs[matched_vehicles] -= energy_costs
        self.fleet_state.socs = torch.clamp(self.fleet_state.socs, 0.0, self.max_soc)

        driving_cost = (total_distances * self.config.reward.driving_cost_per_km).sum()
        num_failed = num_attempted - trips_served
        return trips_served, zero, driving_cost, num_failed, matched_vehicles, matched_trips

    def process_charge_actions(
        self,
        charge_mask: torch.Tensor,
        current_step: int,
        vehicle_charge_power: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Process charging actions - vectorized with distance-based station selection."""
        charge_indices = charge_mask.nonzero(as_tuple=True)[0]
        num_attempted = len(charge_indices)

        empty_tensor = torch.tensor([], dtype=torch.long, device=self.device)
        zero = torch.tensor(0.0, device=self.device)

        if num_attempted == 0:
            return zero, 0, empty_tensor, empty_tensor
        if self.num_stations == 0:
            return zero, num_attempted, empty_tensor, empty_tensor

        if self.station_assigner is not None:
            positions = self.fleet_state.positions[charge_indices]
            available_ports = self.station_state.get_available_ports()
            result = self.station_assigner.assign(
                vehicle_indices=charge_indices,
                vehicle_positions=positions,
                vehicle_preferences=None,
                available_ports=available_ports,
                preference_weight=0.0
            )
            if len(result.vehicle_indices) == 0:
                return zero, num_attempted, empty_tensor, empty_tensor
            valid_charge_indices = result.vehicle_indices
            valid_station_indices = result.target_indices
        else:
            positions = self.fleet_state.positions[charge_indices]
            valid_charge_indices = charge_indices
            valid_station_indices = positions % self.num_stations

        station_hex_positions = self.station_state.hex_ids[valid_station_indices].long()
        vehicle_positions = self.fleet_state.positions[valid_charge_indices].long()
        distances = self.hex_grid.distance_matrix.get_distances_batch(vehicle_positions, station_hex_positions)
        travel_energy_costs = self.energy_dynamics.compute_consumption(distances)

        # For charge-bound travel, do NOT enforce min_soc_reserve.
        # The reserve exists so vehicles can reach a station after serving.
        # A vehicle heading TO a charger doesn't need that reserve — it's
        # about to charge.  Without this, vehicles with SOC just above the
        # reserve (e.g. 12 kWh with 10 kWh reserve → 2 kWh budget → 10 km)
        # become stranded despite having enough raw energy to reach a station.
        current_socs = self.fleet_state.socs[valid_charge_indices]
        feasible_mask = current_socs >= travel_energy_costs
        if not feasible_mask.any():
            return zero, num_attempted, empty_tensor, empty_tensor

        valid_charge_indices = valid_charge_indices[feasible_mask]
        valid_station_indices = valid_station_indices[feasible_mask]
        station_hex_positions = station_hex_positions[feasible_mask]
        distances = distances[feasible_mask]
        travel_energy_costs = travel_energy_costs[feasible_mask]

        self.fleet_state.socs[valid_charge_indices] -= travel_energy_costs
        self.fleet_state.socs = torch.clamp(self.fleet_state.socs, 0.0, self.max_soc)

        travel_steps = self.time_dynamics.distance_to_steps(distances)
        charge_power_kw = self.config.physics.charge_power_kw
        if vehicle_charge_power is not None:
            charge_power = vehicle_charge_power[valid_charge_indices] * charge_power_kw
        else:
            charge_power = torch.full((len(valid_charge_indices),), charge_power_kw, device=self.device)

        at_station = travel_steps == 0
        if at_station.any():
            ready_indices = valid_charge_indices[at_station]
            ready_stations = valid_station_indices[at_station]
            ready_powers = charge_power[at_station]
            ready_station_hexes = station_hex_positions[at_station]

            success_mask = self.station_state.batch_occupy(ready_stations)
            if success_mask.any():
                succ_idx = success_mask.nonzero(as_tuple=True)[0]
                feeder_ok = self._apply_feeder_cap_mask(ready_powers[succ_idx])
                if feeder_ok.any():
                    success_mask[succ_idx[~feeder_ok]] = False
                else:
                    success_mask &= False
            if success_mask.any():
                successful_indices = ready_indices[success_mask]
                successful_stations = ready_stations[success_mask]
                successful_powers = ready_powers[success_mask]
                self._release_charging_ports(successful_indices)
                self.fleet_state.set_charging(
                    vehicle_indices=successful_indices,
                    station_ids=successful_stations,
                    charge_power=successful_powers,
                )
                self.fleet_state.positions[successful_indices] = ready_station_hexes[success_mask]

            waiting_mask = ~success_mask
            if waiting_mask.any():
                waiting_indices = ready_indices[waiting_mask]
                waiting_stations = ready_stations[waiting_mask]
                waiting_powers = ready_powers[waiting_mask]
                waiting_station_hexes = ready_station_hexes[waiting_mask]
                self._release_charging_ports(waiting_indices)
                self.fleet_state.set_waiting_for_charge(
                    vehicle_indices=waiting_indices,
                    station_ids=waiting_stations,
                    station_hexes=waiting_station_hexes,
                    charge_power=waiting_powers,
                    current_step=current_step,
                )
                self.fleet_state.positions[waiting_indices] = waiting_station_hexes

        if (~at_station).any():
            traveling_indices = valid_charge_indices[~at_station]
            traveling_stations = valid_station_indices[~at_station]
            traveling_targets = station_hex_positions[~at_station]
            traveling_steps = travel_steps[~at_station]
            traveling_powers = charge_power[~at_station]
            busy_until = (current_step + traveling_steps).long()
            self._release_charging_ports(traveling_indices)
            self.fleet_state.set_traveling_to_charge(
                vehicle_indices=traveling_indices,
                station_ids=traveling_stations,
                target_hexes=traveling_targets,
                busy_until=busy_until,
                charge_power=traveling_powers,
            )

        total_cost = (distances * self.config.reward.driving_cost_per_km).sum()
        num_success = len(valid_charge_indices)
        num_failed = num_attempted - num_success
        return total_cost, num_failed, valid_charge_indices, valid_station_indices

    def process_charge_actions_with_preferences(
        self,
        charge_mask: torch.Tensor,
        current_step: int,
        charge_scores: Optional[torch.Tensor] = None,
        preference_weight: float = 0.3
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Process charge actions using StationAssigner with actor preferences."""
        charge_indices = charge_mask.nonzero(as_tuple=True)[0]
        num_attempted = len(charge_indices)

        empty_tensor = torch.tensor([], dtype=torch.long, device=self.device)
        zero = torch.tensor(0.0, device=self.device)

        if num_attempted == 0:
            return zero, 0, empty_tensor, empty_tensor
        if self.num_stations == 0 or self.station_assigner is None:
            return self.process_charge_actions(charge_mask, current_step)

        positions = self.fleet_state.positions[charge_indices]
        available_ports = self.station_state.get_available_ports()
        vehicle_prefs = charge_scores[charge_indices] if charge_scores is not None else None
        result = self.station_assigner.assign(
            vehicle_indices=charge_indices,
            vehicle_positions=positions,
            vehicle_preferences=vehicle_prefs,
            available_ports=available_ports,
            preference_weight=preference_weight,
        )

        if len(result.vehicle_indices) == 0:
            return zero, num_attempted, empty_tensor, empty_tensor

        valid_charge_indices = result.vehicle_indices
        valid_station_indices = result.target_indices
        station_hex_positions = self.station_state.hex_ids[valid_station_indices].long()
        vehicle_positions = self.fleet_state.positions[valid_charge_indices].long()
        distances = self.hex_grid.distance_matrix.get_distances_batch(vehicle_positions, station_hex_positions)
        travel_energy_costs = self.energy_dynamics.compute_consumption(distances)

        # No min_soc_reserve for charge-bound travel (same rationale as
        # process_charge_actions — vehicle is heading to a charger).
        current_socs = self.fleet_state.socs[valid_charge_indices]
        feasible_mask = current_socs >= travel_energy_costs
        if not feasible_mask.any():
            return zero, num_attempted, empty_tensor, empty_tensor

        valid_charge_indices = valid_charge_indices[feasible_mask]
        valid_station_indices = valid_station_indices[feasible_mask]
        station_hex_positions = station_hex_positions[feasible_mask]
        distances = distances[feasible_mask]
        travel_energy_costs = travel_energy_costs[feasible_mask]

        self.fleet_state.socs[valid_charge_indices] -= travel_energy_costs
        self.fleet_state.socs = torch.clamp(self.fleet_state.socs, 0.0, self.max_soc)

        travel_steps = self.time_dynamics.distance_to_steps(distances)
        charge_power_kw = self.config.physics.charge_power_kw
        charge_power = torch.full((len(valid_charge_indices),), charge_power_kw, device=self.device)

        at_station = travel_steps == 0
        if at_station.any():
            ready_indices = valid_charge_indices[at_station]
            ready_stations = valid_station_indices[at_station]
            ready_powers = charge_power[at_station]
            ready_station_hexes = station_hex_positions[at_station]

            success_mask = self.station_state.batch_occupy(ready_stations)
            if success_mask.any():
                succ_idx = success_mask.nonzero(as_tuple=True)[0]
                feeder_ok = self._apply_feeder_cap_mask(ready_powers[succ_idx])
                if feeder_ok.any():
                    success_mask[succ_idx[~feeder_ok]] = False
                else:
                    success_mask &= False
            if success_mask.any():
                successful_indices = ready_indices[success_mask]
                successful_stations = ready_stations[success_mask]
                successful_powers = ready_powers[success_mask]
                self._release_charging_ports(successful_indices)
                self.fleet_state.set_charging(
                    vehicle_indices=successful_indices,
                    station_ids=successful_stations,
                    charge_power=successful_powers,
                )
                self.fleet_state.positions[successful_indices] = ready_station_hexes[success_mask]

            waiting_mask = ~success_mask
            if waiting_mask.any():
                waiting_indices = ready_indices[waiting_mask]
                waiting_stations = ready_stations[waiting_mask]
                waiting_powers = ready_powers[waiting_mask]
                waiting_station_hexes = ready_station_hexes[waiting_mask]
                self._release_charging_ports(waiting_indices)
                self.fleet_state.set_waiting_for_charge(
                    vehicle_indices=waiting_indices,
                    station_ids=waiting_stations,
                    station_hexes=waiting_station_hexes,
                    charge_power=waiting_powers,
                    current_step=current_step,
                )
                self.fleet_state.positions[waiting_indices] = waiting_station_hexes

        if (~at_station).any():
            traveling_indices = valid_charge_indices[~at_station]
            traveling_stations = valid_station_indices[~at_station]
            traveling_targets = station_hex_positions[~at_station]
            traveling_steps = travel_steps[~at_station]
            traveling_powers = charge_power[~at_station]
            busy_until = (current_step + traveling_steps).long()
            self._release_charging_ports(traveling_indices)
            self.fleet_state.set_traveling_to_charge(
                vehicle_indices=traveling_indices,
                station_ids=traveling_stations,
                target_hexes=traveling_targets,
                busy_until=busy_until,
                charge_power=traveling_powers,
            )

        total_cost = (distances * self.config.reward.driving_cost_per_km).sum()
        num_success = len(valid_charge_indices)
        num_failed = num_attempted - num_success
        return total_cost, num_failed, valid_charge_indices, valid_station_indices

    def process_reposition_actions(
        self,
        reposition_mask: torch.Tensor,
        targets: torch.Tensor,
        current_step: int,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Process repositioning actions - vectorized with SOC feasibility checks.

        Returns:
            total_cost, num_failed, failed_indices (vehicle indices that failed repos)
        """
        reposition_indices = reposition_mask.nonzero(as_tuple=True)[0]
        all_attempted = reposition_indices.clone()

        if len(reposition_indices) == 0:
            return torch.tensor(0.0, device=self.device), 0, torch.tensor([], dtype=torch.long, device=self.device)

        current_positions = self.fleet_state.positions[reposition_indices]
        target_positions = targets[reposition_indices] if targets.dim() > 0 else targets.expand(len(reposition_indices))

        # Skip vehicles with invalid targets (-1 from edge hexes with no K-hop neighbors)
        valid_target = target_positions >= 0
        if not valid_target.all():
            reposition_indices = reposition_indices[valid_target]
            current_positions = current_positions[valid_target]
            target_positions = target_positions[valid_target]
            if len(reposition_indices) == 0:
                return torch.tensor(0.0, device=self.device), len(all_attempted), all_attempted

        distances = self.hex_grid.distance_matrix.get_distances_batch(current_positions, target_positions)
        energy_costs = self.energy_dynamics.compute_consumption(distances)
        feasible_mask = self._has_energy_budget(reposition_indices, energy_costs)

        if not feasible_mask.any():
            return torch.tensor(0.0, device=self.device), len(reposition_indices), all_attempted

        reposition_indices = reposition_indices[feasible_mask]
        target_positions = target_positions[feasible_mask]
        distances = distances[feasible_mask]
        energy_costs = energy_costs[feasible_mask]

        durations = self.time_dynamics.distance_to_steps(distances).to(torch.long)
        busy_until = (current_step + durations).long()

        self._release_charging_ports(reposition_indices)
        self.fleet_state.set_repositioning(
            vehicle_indices=reposition_indices,
            target_hexes=target_positions.long(),
            busy_until=busy_until,
        )

        self.fleet_state.socs[reposition_indices] -= energy_costs
        self.fleet_state.socs = torch.clamp(self.fleet_state.socs, 0.0, self.max_soc)

        total_cost = (distances * self.config.reward.driving_cost_per_km).sum()
        num_failed = len(all_attempted) - len(reposition_indices)
        # Compute failed indices: vehicles in all_attempted but not in reposition_indices
        if num_failed > 0:
            success_set = set(reposition_indices.tolist())
            failed_mask = torch.tensor([idx.item() not in success_set for idx in all_attempted], device=self.device)
            failed_indices = all_attempted[failed_mask]
        else:
            failed_indices = torch.tensor([], dtype=torch.long, device=self.device)
        return total_cost, num_failed, failed_indices

    def update_ongoing_actions(
        self,
        current_step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update vehicles with ongoing actions."""
        step_charge_cost = torch.tensor(0.0, device=self.device)
        step_serve_revenue = torch.tensor(0.0, device=self.device)

        serving_mask = self.fleet_state.get_serving_mask()
        if serving_mask.any():
            step_serve_revenue = self.fleet_state.ongoing_revenue[serving_mask].sum()

        waiting_mask = self.fleet_state.status == VehicleStatus.TO_CHARGE
        self.station_state.queue_length.fill_(0)
        if waiting_mask.any():
            waiting_indices = waiting_mask.nonzero(as_tuple=True)[0]
            waiting_stations = self.fleet_state.charging_station[waiting_indices]
            valid_waiting = waiting_stations >= 0
            if valid_waiting.any():
                valid_indices = waiting_indices[valid_waiting]
                valid_stations = waiting_stations[valid_waiting]
                success_mask = self.station_state.batch_occupy(valid_stations)
                if success_mask.any():
                    candidate_powers = self.fleet_state.charge_power[valid_indices[success_mask]]
                    feeder_ok = self._apply_feeder_cap_mask(candidate_powers)
                    if feeder_ok.any():
                        succ_idx = success_mask.nonzero(as_tuple=True)[0]
                        success_mask[succ_idx[~feeder_ok]] = False
                    else:
                        success_mask &= False
                if success_mask.any():
                    charging_indices = valid_indices[success_mask]
                    charging_stations = valid_stations[success_mask]
                    charging_powers = self.fleet_state.charge_power[charging_indices]
                    self.fleet_state.set_charging(
                        vehicle_indices=charging_indices,
                        station_ids=charging_stations,
                        charge_power=charging_powers,
                    )
                remaining_wait = valid_stations[~success_mask]
                if remaining_wait.numel() > 0:
                    unique_stations, counts = torch.unique(remaining_wait, return_counts=True)
                    self.station_state.queue_length[unique_stations.long()] = counts.to(torch.int32)

            # Timeout: vehicles waiting too long for a port become IDLE.
            # 12 steps (60 min) gives enough time for port turnover before
            # giving up, preventing the rapid retry thrashing that causes
            # charge pile-up in long evaluations.
            CHARGE_WAIT_TIMEOUT = 12
            wait_starts = self.fleet_state.charge_wait_start[waiting_indices]
            timed_out = (wait_starts >= 0) & ((current_step - wait_starts) > CHARGE_WAIT_TIMEOUT)
            if timed_out.any():
                timed_out_indices = waiting_indices[timed_out]
                self.fleet_state.set_idle(timed_out_indices)

        charging_mask = self.fleet_state.get_charging_mask()
        if charging_mask.any():
            step_duration_hours = self.config.episode.step_duration_minutes / 60.0
            charge_power_per_vehicle = self.fleet_state.charge_power[charging_mask]
            requested_energy_added = charge_power_per_vehicle * step_duration_hours
            prev_socs = self.fleet_state.socs[charging_mask]
            next_socs = torch.clamp(prev_socs + requested_energy_added, 0.0, self.max_soc)
            actual_energy_added = torch.clamp(next_socs - prev_socs, min=0.0)

            self.fleet_state.socs[charging_mask] = next_socs
            step_charge_cost = (actual_energy_added * self.config.station.price_per_kwh).sum()

            charging_indices = charging_mask.nonzero(as_tuple=True)[0]
            fully_charged = self.fleet_state.socs[charging_mask] >= self.max_soc * 0.99

            # Early release: vehicles at >= 80% SOC vacate the port when
            # other vehicles are queued waiting, preventing port hogging.
            has_queue = self.station_state.queue_length.sum() > 0
            if has_queue:
                early_release_threshold = self.max_soc * 0.80
                early_release = (~fully_charged) & (self.fleet_state.socs[charging_mask] >= early_release_threshold)
                fully_charged = fully_charged | early_release

            if fully_charged.any():
                finished = charging_indices[fully_charged]
                station_indices = self.fleet_state.charging_station[finished]
                valid_stations = station_indices[station_indices >= 0]
                if len(valid_stations) > 0:
                    self.station_state.release_ports_batch(valid_stations)
                self.fleet_state.set_idle(finished)

        return step_charge_cost, step_serve_revenue

    def _release_charging_ports(self, vehicle_indices: torch.Tensor):
        if getattr(self, 'station_state', None) is None:
            return
        has_port = self.fleet_state.status[vehicle_indices] == VehicleStatus.CHARGING.value
        if has_port.any():
            port_indices = vehicle_indices[has_port]
            station_ids = self.fleet_state.charging_station[port_indices]
            valid_stations = station_ids[station_ids >= 0]
            if len(valid_stations) > 0:
                self.station_state.release_ports_batch(valid_stations)
            self.fleet_state.charging_station[port_indices] = -1

    # Maximum re-matching rounds for iterative conflict resolution.
    # Keeps computation bounded while recovering most serve failures.
    MAX_REMATCH_ROUNDS = 5
    HUNGARIAN_SIZE_LIMIT = 500_000  # V*T threshold; fall back to greedy above this

    def _match_with_milp_assignments(
        self,
        serve_indices: torch.Tensor,
        milp_serve_trip_ids: torch.Tensor,
        trip_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use MILP's pre-resolved vehicle-trip assignments directly.

        The MILP already solved a conflict-free, energy-feasible matching.
        We trust those assignments and only fall back to greedy for vehicles
        whose MILP trip is no longer available (e.g. new trips loaded mid-step).
        """
        device = serve_indices.device

        if len(serve_indices) == 0:
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        # Build set of currently unassigned trip IDs for validation
        unassigned_set = set(trip_indices.tolist())

        milp_trips_for_serve = milp_serve_trip_ids[serve_indices]  # [N_serve]

        # Split into vehicles with valid MILP assignments vs those needing fallback
        has_assignment = milp_trips_for_serve >= 0
        valid_vehicles = []
        valid_trips = []
        fallback_vehicles = []

        for i in range(len(serve_indices)):
            veh_idx = serve_indices[i]
            trip_id = milp_trips_for_serve[i].item()
            if has_assignment[i] and trip_id in unassigned_set:
                valid_vehicles.append(veh_idx)
                valid_trips.append(trip_id)
                # Remove from pool so no duplicate assignment
                unassigned_set.discard(trip_id)
            else:
                fallback_vehicles.append(veh_idx)

        # Greedy-match fallback vehicles to remaining trips
        if fallback_vehicles:
            remaining_trips = torch.tensor(
                sorted(unassigned_set), dtype=torch.long, device=device
            )
            fb_veh = torch.tensor(fallback_vehicles, dtype=torch.long, device=device)
            if len(remaining_trips) > 0:
                fb_matched_veh, fb_matched_trips = self._greedy_distance_matching(
                    fb_veh, remaining_trips
                )
                if len(fb_matched_veh) > 0:
                    valid_vehicles.extend(fb_matched_veh.tolist())
                    valid_trips.extend(fb_matched_trips.tolist())

        if not valid_vehicles:
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        return torch.tensor(valid_vehicles, dtype=torch.long, device=device), \
               torch.tensor(valid_trips, dtype=torch.long, device=device)

    def _match_with_gcn_and_hungarian(
        self,
        serve_indices: torch.Tensor,
        selected_trip: torch.Tensor,
        trip_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match vehicles to trips using GCN selections with iterative conflict resolution.

        The upstream per-vehicle trip mask (episode_collector._build_per_vehicle_trip_mask)
        already guarantees that each vehicle's GCN choice is within max_pickup_distance.
        Steps:
          1. Modulo wrap for index safety (max_trips > num_available edge-case).
          2. Greedy distance fallback for vehicles whose mask fell back to all-trips.
          3. Conflict resolution: nearest vehicle wins each trip.
          4. **Iterative re-matching**: losing vehicles are re-matched to remaining
             unassigned trips via greedy distance matching (up to MAX_REMATCH_ROUNDS).
        """
        device = serve_indices.device
        N_serve = len(serve_indices)

        if N_serve == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

        gcn_choices  = selected_trip[serve_indices]  # [N_serve] indices into [0, max_trips)
        num_available = len(trip_indices)

        if num_available == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

        # Modulo wrap: safety guard in case mask fallback produced an out-of-range index
        valid_gcn_choices = gcn_choices % num_available
        chosen_trips      = trip_indices[valid_gcn_choices]  # map to actual trip IDs

        # Check for the "mask fallback" case: vehicles with no nearby trips were given
        # all available trips in the mask. If their chosen trip is still out of range,
        # fall back to greedy nearest-trip matching for those vehicles.
        vehicle_positions  = self.fleet_state.positions[serve_indices]
        trip_pickup_hexes  = self.trip_state.pickup_hex[chosen_trips]
        distances          = self.hex_grid.distance_matrix._distances[vehicle_positions, trip_pickup_hexes]
        within_limit       = distances <= self.max_pickup_distance

        if not within_limit.all():
            # Only a few vehicles should have out-of-range choices (mask-fallback vehicles).
            # Handle them with greedy matching; keep in-range choices as-is.
            if not within_limit.any():
                return self._greedy_distance_matching(serve_indices, trip_indices)
            # Mixed: keep within-range vehicles, greedy-match the rest
            ok_vehicles    = serve_indices[within_limit]
            ok_trips       = chosen_trips[within_limit]
            far_vehicles   = serve_indices[~within_limit]
            g_veh, g_trip  = self._greedy_distance_matching(far_vehicles, trip_indices)
            if len(g_veh) > 0:
                serve_indices = torch.cat([ok_vehicles, g_veh])
                chosen_trips  = torch.cat([ok_trips,    g_trip])
            else:
                serve_indices = ok_vehicles
                chosen_trips  = ok_trips

        # ---- Conflict resolution with iterative re-matching ----
        # Round 1: resolve GCN conflicts (nearest vehicle wins per trip).
        # Subsequent rounds: losers get re-matched via greedy distance to
        # remaining trips, then conflicts in the new matches are resolved again.

        all_matched_vehicles = []
        all_matched_trips = []

        # Track which trips from the full pool are still available
        remaining_trip_set = set(trip_indices.tolist())

        current_vehicles = serve_indices
        current_trips = chosen_trips

        for _round in range(self.MAX_REMATCH_ROUNDS):
            if len(current_vehicles) == 0 or len(remaining_trip_set) == 0:
                break

            unique_trips, inverse_indices = current_trips.unique(return_inverse=True)

            if len(unique_trips) == len(current_trips):
                # No conflicts — all vehicles matched successfully
                all_matched_vehicles.append(current_vehicles)
                all_matched_trips.append(current_trips)
                # Remove these trips from the available pool
                remaining_trip_set -= set(current_trips.tolist())
                break

            # Resolve conflicts: nearest vehicle wins each contested trip
            veh_positions = self.fleet_state.positions[current_vehicles]
            trip_hexes = self.trip_state.pickup_hex[unique_trips]
            dist_matrix = self.hex_grid.distance_matrix._distances[
                veh_positions[:, None], trip_hexes[None, :]
            ]

            competition_mask = inverse_indices[None, :] == torch.arange(
                len(unique_trips), device=device
            )[:, None]
            masked_distances = dist_matrix.t().clone()  # [unique_trips, vehicles]
            masked_distances[~competition_mask] = float('inf')

            winner_indices = masked_distances.argmin(dim=1)  # [unique_trips]
            valid_winners = masked_distances[
                torch.arange(len(unique_trips), device=device), winner_indices
            ] < float('inf')

            if not valid_winners.any():
                break

            # Collect winners
            round_vehicles = current_vehicles[winner_indices[valid_winners]]
            round_trips = unique_trips[valid_winners]
            all_matched_vehicles.append(round_vehicles)
            all_matched_trips.append(round_trips)

            # Remove won trips from available pool
            remaining_trip_set -= set(round_trips.tolist())

            # Identify losers: vehicles that didn't win
            winner_set = set(winner_indices[valid_winners].tolist())
            loser_mask = torch.ones(len(current_vehicles), dtype=torch.bool, device=device)
            for w_idx in winner_set:
                loser_mask[w_idx] = False
            loser_vehicles = current_vehicles[loser_mask]

            if len(loser_vehicles) == 0 or len(remaining_trip_set) == 0:
                break

            # Re-match losers to remaining trips via greedy distance
            remaining_trips = torch.tensor(
                sorted(remaining_trip_set), dtype=torch.long, device=device
            )
            g_veh, g_trip = self._greedy_distance_matching(loser_vehicles, remaining_trips)

            if len(g_veh) == 0:
                break

            # Next round: resolve any new conflicts from greedy matching
            current_vehicles = g_veh
            current_trips = g_trip

        # Combine all rounds
        if len(all_matched_vehicles) == 0:
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        return torch.cat(all_matched_vehicles), torch.cat(all_matched_trips)

    def _hungarian_net_revenue_matching(
        self,
        serve_indices: torch.Tensor,
        trip_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Global bipartite matching maximising per-assignment net revenue.

        Cost = -(fare - c_drv*total_dist - c_elec*energy + 0.1*wait_steps)
        Infeasible pairs (energy or distance) get a large sentinel cost so
        the Hungarian solver never picks them.

        Falls back to greedy when V*T exceeds HUNGARIAN_SIZE_LIMIT.
        """
        device = serve_indices.device
        V = len(serve_indices)
        T = len(trip_indices)

        if V == 0 or T == 0:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device))

        if V * T > self.HUNGARIAN_SIZE_LIMIT:
            return self._greedy_distance_matching(serve_indices, trip_indices)

        # --- Vehicle data [V] ---
        veh_positions = self.fleet_state.positions[serve_indices].long()  # [V]
        veh_socs = self.fleet_state.socs[serve_indices]                   # [V]

        # --- Trip data [T] ---
        t_pickup = self.trip_state.pickup_hex[trip_indices].long()        # [T]
        t_dropoff = self.trip_state.dropoff_hex[trip_indices].long()      # [T]
        t_fare = self.trip_state.fare[trip_indices]                       # [T]
        t_dist = self.trip_state.distance_km[trip_indices]                # [T]
        t_wait = self.trip_state.wait_steps[trip_indices].float()         # [T]

        dist_mat = self.hex_grid.distance_matrix._distances               # [H, H]

        # Pickup distances [V, T]
        pickup_dist = dist_mat[veh_positions.unsqueeze(1), t_pickup.unsqueeze(0)]
        total_dist = pickup_dist + t_dist.unsqueeze(0)                    # [V, T]

        # Energy [V, T]
        energy_needed = self.energy_dynamics.compute_consumption(total_dist)  # [V, T]

        # Feasibility [V, T]
        reserve = self.energy_dynamics.min_soc_reserve
        available_soc = (veh_socs - reserve).clamp(min=0.0).unsqueeze(1)  # [V, 1]
        energy_ok = available_soc >= energy_needed
        dist_ok = pickup_dist <= self.max_pickup_distance
        feasible = energy_ok & dist_ok                                     # [V, T]

        # Cost matrix [V, T]  (Hungarian minimises)
        c_drv = self.config.reward.driving_cost_per_km
        c_elec = self.config.reward.electricity_cost_per_kwh
        net_revenue = (t_fare.unsqueeze(0)
                       - c_drv * total_dist
                       - c_elec * energy_needed
                       + 0.1 * t_wait.unsqueeze(0))                       # [V, T]
        cost = -net_revenue
        SENTINEL = 1e9
        cost[~feasible] = SENTINEL

        # Solve on CPU
        cost_np = cost.cpu().float().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # Keep only feasible assignments
        valid = cost_np[row_ind, col_ind] < SENTINEL * 0.5
        row_ind = row_ind[valid]
        col_ind = col_ind[valid]

        if len(row_ind) == 0:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device))

        matched_vehicles = serve_indices[torch.from_numpy(row_ind).to(device)]
        matched_trips = trip_indices[torch.from_numpy(col_ind).to(device)]
        return matched_vehicles, matched_trips

    def _greedy_distance_matching(
        self,
        serve_indices: torch.Tensor,
        trip_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy distance-based matching with iterative re-matching.

        Losers (vehicles whose nearest trip was claimed by another vehicle)
        are re-matched to remaining unclaimed trips, up to MAX_REMATCH_ROUNDS.
        """
        device = serve_indices.device

        if len(serve_indices) == 0 or len(trip_indices) == 0:
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        all_matched_vehicles = []
        all_matched_trips = []

        remaining_vehicles = serve_indices
        remaining_trips = trip_indices

        for _round in range(self.MAX_REMATCH_ROUNDS):
            if len(remaining_vehicles) == 0 or len(remaining_trips) == 0:
                break

            winners, won_trips, losers = self._greedy_distance_single_pass(
                remaining_vehicles, remaining_trips
            )

            if len(winners) == 0:
                break

            all_matched_vehicles.append(winners)
            all_matched_trips.append(won_trips)

            # Remove won trips from pool for next round
            won_set = set(won_trips.tolist())
            keep_mask = torch.tensor(
                [t.item() not in won_set for t in remaining_trips],
                dtype=torch.bool, device=device
            )
            remaining_trips = remaining_trips[keep_mask]
            remaining_vehicles = losers

        if len(all_matched_vehicles) == 0:
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        return torch.cat(all_matched_vehicles), torch.cat(all_matched_trips)

    def _greedy_distance_single_pass(
        self,
        serve_indices: torch.Tensor,
        trip_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-pass greedy nearest-trip matching (no re-matching).

        Returns:
            (matched_vehicles, matched_trips, unmatched_vehicles)
        """
        device = serve_indices.device
        num_vehicles = len(serve_indices)
        num_trips = len(trip_indices)

        empty = torch.empty(0, dtype=torch.long, device=device)

        if num_vehicles == 0 or num_trips == 0:
            return empty, empty, serve_indices

        positions = self.fleet_state.positions[serve_indices]
        trip_hexes = self.trip_state.pickup_hex[trip_indices]

        veh_positions_expanded = positions.unsqueeze(1).expand(num_vehicles, num_trips)
        trip_hexes_expanded = trip_hexes.unsqueeze(0).expand(num_vehicles, num_trips)

        distances = self.hex_grid.distance_matrix._distances[
            veh_positions_expanded.reshape(-1),
            trip_hexes_expanded.reshape(-1)
        ].reshape(num_vehicles, num_trips)

        valid_mask = distances <= self.max_pickup_distance

        if not valid_mask.any():
            return empty, empty, serve_indices

        masked_distances = distances.clone()
        masked_distances[~valid_mask] = float('inf')

        nearest_trip_local = masked_distances.argmin(dim=1)
        min_distances = masked_distances[torch.arange(num_vehicles, device=device), nearest_trip_local]

        valid_vehicles = min_distances < float('inf')

        if not valid_vehicles.any():
            return empty, empty, serve_indices

        valid_veh_local = valid_vehicles.nonzero(as_tuple=True)[0]
        matched_vehicles = serve_indices[valid_veh_local]
        matched_trip_local = nearest_trip_local[valid_veh_local]

        # For each trip picked by multiple vehicles, keep the nearest one
        sorted_order = matched_trip_local.argsort()
        sorted_trips = matched_trip_local[sorted_order]
        sorted_vehicles = matched_vehicles[sorted_order]
        sorted_distances = min_distances[valid_veh_local][sorted_order]

        unique_trips_local, inverse, counts = sorted_trips.unique_consecutive(
            return_inverse=True, return_counts=True
        )

        # For each unique trip, pick the vehicle with minimum distance
        first_indices = torch.zeros(len(unique_trips_local), dtype=torch.long, device=device)
        first_indices[1:] = counts[:-1].cumsum(0)

        # Within each group, find the argmin distance vehicle
        best_indices = []
        for i, (start, count) in enumerate(zip(first_indices, counts)):
            group_distances = sorted_distances[start:start + count]
            best_in_group = start + group_distances.argmin()
            best_indices.append(best_in_group.item())

        best_indices_t = torch.tensor(best_indices, dtype=torch.long, device=device)
        winners = sorted_vehicles[best_indices_t]
        won_trips = trip_indices[unique_trips_local]

        # Identify losers: vehicles that were valid but didn't win
        winner_set = set(winners.tolist())
        # Include vehicles that had no valid trip at all
        no_valid = serve_indices[~valid_vehicles] if (~valid_vehicles).any() else empty
        lost_valid = matched_vehicles[
            torch.tensor([v.item() not in winner_set for v in matched_vehicles],
                        dtype=torch.bool, device=device)
        ] if len(matched_vehicles) > 0 else empty

        losers = torch.cat([lost_valid, no_valid]) if (len(lost_valid) + len(no_valid)) > 0 else empty

        return winners, won_trips, losers
