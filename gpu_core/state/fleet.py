"""Tensor-based fleet state representation."""

import torch
from typing import Optional, Tuple
from enum import IntEnum


class VehicleStatus(IntEnum):
    IDLE = 0
    SERVING = 1
    REPOSITIONING = 2
    CHARGING = 3
    PICKUP = 4
    TO_CHARGE = 5


class TensorFleetState:
    """
    GPU-native fleet state using pure tensors.
    
    All vehicle attributes are stored as contiguous GPU tensors
    for maximum performance with vectorized operations.
    """
    
    def __init__(
        self,
        num_vehicles: int,
        device: str = "cuda",
        initial_soc: float = 80.0,
        max_soc: float = 100.0,
    ):
        self.num_vehicles = num_vehicles
        self.device = torch.device(device)
        self.max_soc = max_soc
        
        # Vehicle positions (hex indices)
        self.positions = torch.zeros(num_vehicles, dtype=torch.long, device=self.device)
        
        # Battery state of charge (kWh)
        self.socs = torch.empty(num_vehicles, dtype=torch.float32, device=self.device).uniform_(20.0, 80.0)
        
        # Vehicle status (VehicleStatus enum values)
        self.status = torch.zeros(num_vehicles, dtype=torch.int8, device=self.device)
        
        # Step when vehicle becomes available (0 = available now)
        self.busy_until = torch.zeros(num_vehicles, dtype=torch.int64, device=self.device)
        
        # Target hex for current action (-1 = no target)
        self.target_hex = torch.full((num_vehicles,), -1, dtype=torch.long, device=self.device)
        
        # Current trip ID being served (-1 = no trip)
        self.current_trip = torch.full((num_vehicles,), -1, dtype=torch.long, device=self.device)
        
        # Charging station ID (-1 = not charging)
        self.charging_station = torch.full((num_vehicles,), -1, dtype=torch.long, device=self.device)
        
        # Revenue being earned per step (for SERVING vehicles)
        self.ongoing_revenue = torch.zeros(num_vehicles, dtype=torch.float32, device=self.device)
        
        # Charge power being used (kW)
        self.charge_power = torch.zeros(num_vehicles, dtype=torch.float32, device=self.device)

        # Step when vehicle started waiting for a charging port (-1 = not waiting)
        self.charge_wait_start = torch.full((num_vehicles,), -1, dtype=torch.int64, device=self.device)

        # Consecutive steps a vehicle has been idle (reset when it gets a successful action)
        self.idle_steps = torch.zeros(num_vehicles, dtype=torch.int64, device=self.device)
    
    def clone(self) -> "TensorFleetState":
        new_state = TensorFleetState.__new__(TensorFleetState)
        new_state.num_vehicles = self.num_vehicles
        new_state.device = self.device
        new_state.max_soc = self.max_soc
        new_state.positions = self.positions.clone()
        new_state.socs = self.socs.clone()
        new_state.status = self.status.clone()
        new_state.busy_until = self.busy_until.clone()
        new_state.target_hex = self.target_hex.clone()
        new_state.current_trip = self.current_trip.clone()
        new_state.charging_station = self.charging_station.clone()
        new_state.ongoing_revenue = self.ongoing_revenue.clone()
        new_state.charge_power = self.charge_power.clone()
        new_state.charge_wait_start = self.charge_wait_start.clone()
        new_state.idle_steps = self.idle_steps.clone()
        return new_state
    
    def get_available_mask(self, current_step: int) -> torch.Tensor:
        """Get mask of vehicles available for new actions.
        
        A vehicle is available if:
        - It's IDLE and not busy, OR
        - It's CHARGING and SOC >= 90% (can be interrupted), OR  
        - It's REPOSITIONING and busy_until <= current_step (just completed)
        
        Vehicles SERVING are never interruptible until they complete.
        """
        idle_available = (self.status == VehicleStatus.IDLE) & (self.busy_until <= current_step)
        
        # CHARGING vehicles can be interrupted ONLY if SOC >= 90%
        charging_interruptible = (self.status == VehicleStatus.CHARGING) & (self.socs >= 90.0)
        
        # REPOSITIONING vehicles become available after busy_until
        # (complete_actions will have set them to IDLE, so they're caught by idle_available)
        
        return idle_available | charging_interruptible
    
    def get_busy_mask(self, current_step: int) -> torch.Tensor:
        return self.busy_until > current_step
    
    def get_status_counts(self) -> dict:
        counts = {}
        for s in VehicleStatus:
            counts[s.name] = (self.status == s.value).sum().item()
        return counts
    
    def get_charging_mask(self) -> torch.Tensor:
        return self.status == VehicleStatus.CHARGING
    
    def get_serving_mask(self) -> torch.Tensor:
        return self.status == VehicleStatus.SERVING
    
    def get_low_soc_mask(self, threshold: float) -> torch.Tensor:
        return self.socs < threshold
    
    def update_positions(self, vehicle_mask: torch.Tensor, new_positions: torch.Tensor) -> None:
        self.positions[vehicle_mask] = new_positions
    
    def update_socs(self, delta_soc: torch.Tensor) -> None:
        self.socs = torch.clamp(self.socs + delta_soc, 0.0, self.max_soc)
    
    def set_serving(
        self,
        vehicle_indices: torch.Tensor,
        trip_ids: torch.Tensor,
        target_hexes: torch.Tensor,
        busy_until: torch.Tensor,
        revenue_per_step: torch.Tensor,
    ) -> None:
        self.status[vehicle_indices] = VehicleStatus.SERVING
        self.current_trip[vehicle_indices] = trip_ids
        self.target_hex[vehicle_indices] = target_hexes
        self.busy_until[vehicle_indices] = busy_until
        self.ongoing_revenue[vehicle_indices] = revenue_per_step
        self.idle_steps[vehicle_indices] = 0
    
    def set_repositioning(
        self,
        vehicle_indices: torch.Tensor,
        target_hexes: torch.Tensor,
        busy_until: torch.Tensor,
    ) -> None:
        self.status[vehicle_indices] = VehicleStatus.REPOSITIONING
        self.target_hex[vehicle_indices] = target_hexes
        self.busy_until[vehicle_indices] = busy_until
        self.current_trip[vehicle_indices] = -1
        self.ongoing_revenue[vehicle_indices] = 0.0
        self.idle_steps[vehicle_indices] = 0
    
    def set_charging(
        self,
        vehicle_indices: torch.Tensor,
        station_ids: torch.Tensor,
        charge_power: torch.Tensor,
    ) -> None:
        self.status[vehicle_indices] = VehicleStatus.CHARGING
        self.charging_station[vehicle_indices] = station_ids
        self.charge_power[vehicle_indices] = charge_power
        self.busy_until[vehicle_indices] = 0  # Charging can be interrupted
        self.current_trip[vehicle_indices] = -1
        self.ongoing_revenue[vehicle_indices] = 0.0
        self.charge_wait_start[vehicle_indices] = -1
        self.idle_steps[vehicle_indices] = 0

    def set_traveling_to_charge(
        self,
        vehicle_indices: torch.Tensor,
        station_ids: torch.Tensor,
        target_hexes: torch.Tensor,
        busy_until: torch.Tensor,
        charge_power: torch.Tensor,
    ) -> None:
        self.status[vehicle_indices] = VehicleStatus.TO_CHARGE
        self.charging_station[vehicle_indices] = station_ids
        self.target_hex[vehicle_indices] = target_hexes
        self.busy_until[vehicle_indices] = busy_until
        self.charge_power[vehicle_indices] = charge_power
        self.current_trip[vehicle_indices] = -1
        self.ongoing_revenue[vehicle_indices] = 0.0
        self.charge_wait_start[vehicle_indices] = -1  # not waiting yet, still traveling
        self.idle_steps[vehicle_indices] = 0

    def set_waiting_for_charge(
        self,
        vehicle_indices: torch.Tensor,
        station_ids: torch.Tensor,
        station_hexes: torch.Tensor,
        charge_power: torch.Tensor,
        current_step: int = 0,
    ) -> None:
        self.status[vehicle_indices] = VehicleStatus.TO_CHARGE
        self.charging_station[vehicle_indices] = station_ids
        self.target_hex[vehicle_indices] = station_hexes
        self.busy_until[vehicle_indices] = 0
        self.charge_power[vehicle_indices] = charge_power
        self.current_trip[vehicle_indices] = -1
        self.ongoing_revenue[vehicle_indices] = 0.0
        self.charge_wait_start[vehicle_indices] = current_step
    
    def set_idle(self, vehicle_indices: torch.Tensor) -> None:
        self.status[vehicle_indices] = VehicleStatus.IDLE
        self.busy_until[vehicle_indices] = 0
        self.target_hex[vehicle_indices] = -1
        self.current_trip[vehicle_indices] = -1
        self.charging_station[vehicle_indices] = -1
        self.ongoing_revenue[vehicle_indices] = 0.0
        self.charge_power[vehicle_indices] = 0.0
        self.charge_wait_start[vehicle_indices] = -1
    
    def complete_actions(self, current_step: int) -> torch.Tensor:
        """Complete actions for vehicles that have finished. Returns mask of completed vehicles."""
        completed = (self.busy_until <= current_step) & (self.busy_until > 0)
        
        # Update positions for completed SERVING/REPOSITIONING/TO_CHARGE
        serving_completed = completed & (self.status == VehicleStatus.SERVING)
        repos_completed = completed & (self.status == VehicleStatus.REPOSITIONING)
        to_charge_completed = completed & (self.status == VehicleStatus.TO_CHARGE)
        
        # Move to target hex
        move_completed = serving_completed | repos_completed | to_charge_completed
        if move_completed.any():
            valid_targets = self.target_hex[move_completed] >= 0
            update_mask = move_completed.clone()
            update_mask[move_completed] = valid_targets
            if update_mask.any():
                self.positions[update_mask] = self.target_hex[update_mask]
        
        # Transition completed vehicles
        if completed.any():
            # TO_CHARGE transitions to an arrived waiting state; ActionProcessor
            # attempts to claim a charging port during update_ongoing_actions.
            if to_charge_completed.any():
                indices = to_charge_completed.nonzero(as_tuple=True)[0]
                self.status[indices] = VehicleStatus.TO_CHARGE
                self.busy_until[indices] = 0
                self.target_hex[indices] = -1
                self.charge_wait_start[indices] = current_step
            
            # Others transition to IDLE
            others_completed = completed & (~to_charge_completed)
            if others_completed.any():
                indices = others_completed.nonzero(as_tuple=True)[0]
                self.status[indices] = VehicleStatus.IDLE
                self.busy_until[indices] = 0
                self.target_hex[indices] = -1
                self.current_trip[indices] = -1
                self.ongoing_revenue[indices] = 0.0
        
        return completed
    
    def to_feature_tensor(self) -> torch.Tensor:
        """Convert fleet state to feature tensor [N, feature_dim]."""
        # Clamp status so TO_CHARGE (=5) is treated as CHARGING (=3)
        # This keeps the range [0, 1] without changing the feature dimension.
        _STATUS_MAX = float(VehicleStatus.PICKUP)  # = 4
        clamped_status = self.status.float().clamp(max=float(VehicleStatus.CHARGING))
        features = torch.stack([
            self.positions.float() / 1000.0,  # Normalized position
            self.socs / self.max_soc,  # Normalized SoC
            clamped_status / _STATUS_MAX,  # Normalized status (0..1)
            (self.busy_until > 0).float(),  # Is busy
            (self.charging_station >= 0).float(),  # Is at station
            self.ongoing_revenue / 10.0,  # Normalized revenue
        ], dim=1)
        return features
    
    @property
    def mean_soc(self) -> float:
        return self.socs.mean().item()
    
    @property
    def num_available(self) -> int:
        return (self.status == VehicleStatus.IDLE).sum().item()
