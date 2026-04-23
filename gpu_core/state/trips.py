"""Tensor-based trip state representation."""

import torch
from typing import Optional, Tuple


class TensorTripState:
    """
    GPU-native trip orders state using pure tensors.
    
    Stores trip information as contiguous GPU tensors for
    efficient batch processing and filtering.
    """
    
    def __init__(
        self,
        max_trips: int = 500,
        device: str = "cuda",
    ):
        self.max_trips = max_trips
        self.device = torch.device(device)
        
        # Number of active trips
        self.num_trips = 0
        
        # Trip attributes [max_trips]
        self.trip_ids = torch.zeros(max_trips, dtype=torch.long, device=self.device)
        self.pickup_hex = torch.zeros(max_trips, dtype=torch.long, device=self.device)
        self.dropoff_hex = torch.zeros(max_trips, dtype=torch.long, device=self.device)
        self.fare = torch.zeros(max_trips, dtype=torch.float32, device=self.device)
        self.distance_km = torch.zeros(max_trips, dtype=torch.float32, device=self.device)
        self.wait_steps = torch.zeros(max_trips, dtype=torch.int32, device=self.device)
        self.assigned_vehicle = torch.full((max_trips,), -1, dtype=torch.long, device=self.device)
        self.assigned = torch.zeros(max_trips, dtype=torch.bool, device=self.device)  # Boolean assigned flag
        
        # Valid mask (True for active trips)
        self.valid_mask = torch.zeros(max_trips, dtype=torch.bool, device=self.device)
    
    def reset(self) -> None:
        self.num_trips = 0
        self.valid_mask.fill_(False)
        self.assigned_vehicle.fill_(-1)
        self.assigned.fill_(False)
        self.wait_steps.fill_(0)
    
    def load_trips(
        self,
        trip_ids: torch.Tensor,
        pickup_hexes: torch.Tensor,
        dropoff_hexes: torch.Tensor,
        fares: torch.Tensor,
        distances: torch.Tensor,
    ) -> None:
        """Load new trips for current step."""
        n = min(len(trip_ids), self.max_trips)
        self.num_trips = n
        
        self.trip_ids[:n] = trip_ids[:n]
        self.pickup_hex[:n] = pickup_hexes[:n]
        self.dropoff_hex[:n] = dropoff_hexes[:n]
        self.fare[:n] = fares[:n]
        self.distance_km[:n] = distances[:n]
        self.wait_steps[:n] = 0
        self.assigned_vehicle[:n] = -1
        self.assigned[:n] = False
        
        self.valid_mask.fill_(False)
        self.valid_mask[:n] = True
    
    def add_trips(
        self,
        trip_ids: torch.Tensor,
        pickup_hexes: torch.Tensor,
        dropoff_hexes: torch.Tensor,
        fares: torch.Tensor,
        distances: torch.Tensor,
    ) -> int:
        """Add new trips to existing pool. Returns number added.

        Reuses invalid (served/dropped) slots first, then appends.
        This avoids _compact() which would invalidate buffer indices
        stored in fleet_state.current_trip.
        """
        n_new = len(trip_ids)
        if n_new == 0:
            return 0

        # Find invalid slots within current buffer to reuse
        invalid_slots = (~self.valid_mask[:self.num_trips]).nonzero(as_tuple=True)[0]
        n_reuse = min(len(invalid_slots), n_new)

        # Remaining trips go to fresh slots at the end
        n_append = min(n_new - n_reuse, self.max_trips - self.num_trips)
        n_add = n_reuse + n_append

        if n_add == 0:
            return 0

        # Build target slot indices
        slots_list = []
        if n_reuse > 0:
            slots_list.append(invalid_slots[:n_reuse])
        if n_append > 0:
            slots_list.append(torch.arange(
                self.num_trips, self.num_trips + n_append,
                dtype=torch.long, device=self.device
            ))
        slots = torch.cat(slots_list)

        self.trip_ids[slots] = trip_ids[:n_add]
        self.pickup_hex[slots] = pickup_hexes[:n_add]
        self.dropoff_hex[slots] = dropoff_hexes[:n_add]
        self.fare[slots] = fares[:n_add]
        self.distance_km[slots] = distances[:n_add]
        self.wait_steps[slots] = 0
        self.assigned_vehicle[slots] = -1
        self.assigned[slots] = False
        self.valid_mask[slots] = True

        self.num_trips += n_append
        return n_add
    
    def increment_wait(self) -> None:
        """Increment wait time for all unassigned trips."""
        unassigned = self.valid_mask & (self.assigned_vehicle < 0)
        self.wait_steps[unassigned] += 1
    
    def get_expired_mask(self, max_wait: int) -> torch.Tensor:
        """Get mask of trips that have exceeded max wait time."""
        return self.valid_mask & (self.wait_steps >= max_wait) & (self.assigned_vehicle < 0)
    
    def drop_expired(self, max_wait: int) -> int:
        """Drop expired trips and return count."""
        expired = self.get_expired_mask(max_wait)
        n_dropped = expired.sum().item()
        self.valid_mask[expired] = False
        # Don't compact — it reorders buffer indices, corrupting
        # fleet_state.current_trip references to in-progress trips.
        # add_trips() reuses invalid slots directly instead.
        return n_dropped
    
    def _compact(self) -> None:
        """Compact buffer by moving valid trips to the front, reusing dropped slots."""
        if self.num_trips == 0:
            return
            
        # Find valid trips
        valid_indices = self.valid_mask[:self.num_trips].nonzero(as_tuple=True)[0]
        n_valid = len(valid_indices)
        
        if n_valid == self.num_trips:
            # No gaps, nothing to compact
            return
        
        if n_valid == 0:
            # All trips gone, reset
            self.num_trips = 0
            return
        
        # Move valid trips to front
        self.trip_ids[:n_valid] = self.trip_ids[valid_indices]
        self.pickup_hex[:n_valid] = self.pickup_hex[valid_indices]
        self.dropoff_hex[:n_valid] = self.dropoff_hex[valid_indices]
        self.fare[:n_valid] = self.fare[valid_indices]
        self.distance_km[:n_valid] = self.distance_km[valid_indices]
        self.wait_steps[:n_valid] = self.wait_steps[valid_indices]
        self.assigned_vehicle[:n_valid] = self.assigned_vehicle[valid_indices]
        self.assigned[:n_valid] = self.assigned[valid_indices]
        
        # Update valid mask
        self.valid_mask.fill_(False)
        self.valid_mask[:n_valid] = True
        
        self.num_trips = n_valid
    
    def assign_trip(self, trip_idx: int, vehicle_idx: int) -> None:
        """Assign a trip to a vehicle."""
        self.assigned_vehicle[trip_idx] = vehicle_idx
    
    def assign_trips_batch(self, trip_indices: torch.Tensor, vehicle_indices: torch.Tensor) -> None:
        """
        Assign multiple trips to vehicles.
        OPTIMIZED: vectorized batch assignment.
        """
        if len(trip_indices) == 0:
            return
        self.assigned_vehicle[trip_indices] = vehicle_indices
    
    def complete_trip(self, trip_idx: int) -> float:
        """Complete a trip and return fare. Marks trip as invalid."""
        fare = self.fare[trip_idx].item()
        self.valid_mask[trip_idx] = False
        return fare
    
    def get_unassigned_mask(self) -> torch.Tensor:
        """Get mask of valid unassigned trips."""
        return self.valid_mask & (self.assigned_vehicle < 0)
    
    def get_assigned_mask(self) -> torch.Tensor:
        """Get mask of valid assigned (in-progress) trips."""
        return self.valid_mask & (self.assigned_vehicle >= 0)
    
    def get_active_trips(self) -> Tuple[torch.Tensor, ...]:
        """Get tensors for all active (valid and unassigned) trips."""
        mask = self.get_unassigned_mask()
        return (
            self.trip_ids[mask],
            self.pickup_hex[mask],
            self.dropoff_hex[mask],
            self.fare[mask],
            self.distance_km[mask],
        )
    
    def get_trips_in_hex(self, hex_idx: int) -> torch.Tensor:
        """Get indices of unassigned trips with pickup in given hex."""
        mask = self.get_unassigned_mask() & (self.pickup_hex == hex_idx)
        return torch.where(mask)[0]
    
    def get_trips_near_hex(
        self,
        hex_idx: int,
        hex_distances: torch.Tensor,
        max_distance: float,
    ) -> torch.Tensor:
        """Get indices of unassigned trips within max_distance of hex."""
        unassigned = self.get_unassigned_mask()
        pickup_hexes = self.pickup_hex[unassigned]
        distances = hex_distances[hex_idx, pickup_hexes]
        within_range = distances <= max_distance
        
        unassigned_indices = torch.where(unassigned)[0]
        return unassigned_indices[within_range]
    
    def to_feature_tensor(self) -> torch.Tensor:
        """Convert active trips to feature tensor [num_active, feature_dim]."""
        mask = self.get_unassigned_mask()
        n = mask.sum().item()
        
        if n == 0:
            return torch.zeros((0, 5), dtype=torch.float32, device=self.device)
        
        features = torch.stack([
            self.pickup_hex[mask].float() / 1000.0,
            self.dropoff_hex[mask].float() / 1000.0,
            self.fare[mask] / 100.0,
            self.distance_km[mask] / 50.0,
            self.wait_steps[mask].float() / 10.0,
        ], dim=1)
        return features
    
    @property
    def num_active(self) -> int:
        return self.get_unassigned_mask().sum().item()
    
    @property
    def total_fare(self) -> float:
        mask = self.get_unassigned_mask()
        return self.fare[mask].sum().item()
