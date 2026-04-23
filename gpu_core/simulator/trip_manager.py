"""
Trip Management for EV Fleet Environment.

Handles trip loading, generation, and lifecycle management.
Separated from main environment for modularity.
"""

import torch
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..state import TensorTripState
    from ..spatial import HexGrid
    from ..data.real_trip_loader import RealTripLoader


class TripManager:
    """
    Manages trip lifecycle: loading, generation, and state updates.
    
    Supports both real NYC taxi data and synthetic trip generation.
    """
    
    def __init__(
        self,
        config: 'Config',
        hex_grid: 'HexGrid',
        trip_state: 'TensorTripState',
        trip_loader: Optional['RealTripLoader'],
        device: torch.device,
        num_hexes: int,
        num_vehicles: int,
        episode_steps: int,
    ):
        self.config = config
        self.hex_grid = hex_grid
        self.trip_state = trip_state
        self.trip_loader = trip_loader
        self.device = device
        self.num_hexes = num_hexes
        self.num_vehicles = num_vehicles
        self.episode_steps = episode_steps
        
        # Trip generation state
        self._next_trip_id = 0
        self._trip_generation_rate = max(5, num_vehicles // 20)
        self._real_trip_step = 0
        self._episode_start_idx = 0
        self._use_time_based_loading = True
        self._initial_step_loaded = False

        # Track trips loaded per episode
        self.trips_loaded = 0
    
    def reset(self, episode_idx: int = 0, episode_start_idx: int = None):
        """Reset trip manager for new episode.
        
        Args:
            episode_idx: Episode number (used for deterministic start if no random)
            episode_start_idx: Optional explicit start index (from get_episode_start_indices)
        """
        self._next_trip_id = 0
        self._real_trip_step = 0
        self._initial_step_loaded = False
        self.trips_loaded = 0
        
        # Set episode start index for time-based loading
        if episode_start_idx is not None:
            # Use explicit start index (passed from environment)
            self._episode_start_idx = episode_start_idx
        elif self.trip_loader is not None and self.trip_loader.is_loaded and self._use_time_based_loading:
            # Choose random episode start from valid time-aligned indices (like V1)
            valid_starts = self.trip_loader.get_episode_start_indices(
                episode_duration_hours=self.config.episode.duration_hours
            )
            if len(valid_starts) > 0:
                random_idx = torch.randint(0, len(valid_starts), (1,)).item()
                self._episode_start_idx = valid_starts[random_idx].item()
            else:
                self._episode_start_idx = 0
        else:
            self._episode_start_idx = 0
    
    def calculate_max_trips(self) -> int:
        """
        Calculate max_trips buffer size based on episode configuration and data.
        
        Uses real data statistics if available, otherwise estimates based on:
        - Episode duration and step size
        - Number of vehicles (more vehicles = can serve more trips = less backlog)
        - Safety margin for peak hours
        """
        if self.trip_loader is not None and self.trip_loader.is_loaded:
            # Use real data: calculate average trips per step
            total_trips = self.trip_loader.total_trips
            # Data covers ~31 days, estimate trips per 5-min step
            # 31 days * 24 hours * 12 steps/hour = 8928 steps total
            total_steps_in_data = 31 * 24 * 12
            avg_trips_per_step = total_trips / total_steps_in_data
            
            # Peak hours have ~1.6x average (from data analysis)
            # Add backlog factor for trips waiting to be served
            peak_multiplier = 1.6
            backlog_factor = 1.5
            
            max_trips = int(avg_trips_per_step * self.episode_steps * peak_multiplier * backlog_factor)
        else:
            # Synthetic: estimate based on vehicle count
            # Assume ~10% of vehicles get new trips each step
            trips_per_step = max(10, self.num_vehicles // 10)
            max_trips = int(trips_per_step * self.episode_steps * 2)  # 2x safety margin
        
        # Clamp to reasonable range (up to 500k for full data)
        max_trips = max(1000, min(max_trips, 500000))
        return max_trips
    
    def load_initial_trips(self):
        """Load initial trips - use real data if available, otherwise synthetic."""
        if self.trip_loader is not None and self.trip_loader.is_loaded:
            self._load_real_initial_trips()
        else:
            self._generate_synthetic_initial_trips()
    
    def _load_real_initial_trips(self):
        """Load initial batch of real trips from NYC taxi data using TIME-BASED loading."""
        num_initial_trips = min(100, self.num_vehicles * 2)
        
        if self._use_time_based_loading:
            # Use time-based loading: get trips for step 0 of this episode
            pickup_hexes, dropoff_hexes, fares, distances = self.trip_loader.get_trips_for_episode_step(
                episode_start_idx=self._episode_start_idx,
                step=0,
                step_duration_minutes=self.config.episode.step_duration_minutes,
                episode_duration_hours=self.config.episode.duration_hours,
            )
        else:
            # Fallback to sequential loading
            pickup_hexes, dropoff_hexes, fares, distances = self.trip_loader.get_trips_for_step(
                step=0, trips_per_step=num_initial_trips
            )
        
        actual_trips = len(pickup_hexes)
        if actual_trips == 0:
            # Fallback to synthetic if no data
            self._generate_synthetic_initial_trips()
            return
        
        # Remap hex indices to our grid size
        pickup_hexes = pickup_hexes % self.num_hexes
        dropoff_hexes = dropoff_hexes % self.num_hexes
        
        # Ensure pickup != dropoff
        same_mask = pickup_hexes == dropoff_hexes
        dropoff_hexes[same_mask] = (dropoff_hexes[same_mask] + 1) % self.num_hexes
        
        trip_ids = torch.arange(actual_trips, device=self.device)
        
        self.trip_state.load_trips(
            trip_ids=trip_ids,
            pickup_hexes=pickup_hexes,
            dropoff_hexes=dropoff_hexes,
            fares=fares,
            distances=distances
        )
        
        self._next_trip_id = actual_trips
        self.trips_loaded += actual_trips
        self._real_trip_step = 1  # Next step to load (already loaded step 0)
        self._initial_step_loaded = True
    
    def _generate_synthetic_initial_trips(self):
        """Generate synthetic trips for training when no real data available."""
        num_initial_trips = min(100, self.num_vehicles * 2)
        
        trip_ids = torch.arange(num_initial_trips, device=self.device)
        pickup_hexes = torch.randint(0, self.num_hexes, (num_initial_trips,), device=self.device)
        dropoff_hexes = torch.randint(0, self.num_hexes, (num_initial_trips,), device=self.device)
        
        # Ensure pickup != dropoff
        same_mask = pickup_hexes == dropoff_hexes
        dropoff_hexes[same_mask] = (dropoff_hexes[same_mask] + 1) % self.num_hexes
        
        # Compute distances using distance matrix
        distances = self.hex_grid.distance_matrix.get_distances_batch(pickup_hexes, dropoff_hexes)
        distances = torch.clamp(distances, min=1.0, max=20.0)  # 1-20 km
        
        # Compute fares: base fare + per-km rate
        base_fare = 3.0
        per_km_rate = 2.5
        fares = base_fare + distances * per_km_rate
        
        self.trip_state.load_trips(
            trip_ids=trip_ids,
            pickup_hexes=pickup_hexes,
            dropoff_hexes=dropoff_hexes,
            fares=fares,
            distances=distances
        )
        
        self._next_trip_id = num_initial_trips
        self.trips_loaded += num_initial_trips
    
    def load_step_trips(self, current_step: int):
        """Load new trips for this step - use real data if available."""
        if self.trip_loader is not None and self.trip_loader.is_loaded:
            self._load_real_step_trips(current_step)
        else:
            self._generate_synthetic_step_trips()
    
    def _load_real_step_trips(self, current_step: int):
        """Load real trips for this step from NYC taxi data using TIME-BASED loading."""
        if self._use_time_based_loading:
            # Use time-based loading: get trips for current step of this episode
            pickup_hexes, dropoff_hexes, fares, distances = self.trip_loader.get_trips_for_episode_step(
                episode_start_idx=self._episode_start_idx,
                step=current_step,
                step_duration_minutes=self.config.episode.step_duration_minutes,
                episode_duration_hours=self.config.episode.duration_hours,
            )
        else:
            # Fallback to sequential loading
            trips_to_load = max(5, self._trip_generation_rate)
            step_index = self._real_trip_step
            
            pickup_hexes, dropoff_hexes, fares, distances = self.trip_loader.get_trips_for_step(
                step=step_index, trips_per_step=trips_to_load
            )
            self._real_trip_step = step_index + trips_to_load
        
        actual_trips = len(pickup_hexes)
        if actual_trips == 0:
            return

        # Avoid loading step-0 trips twice (already loaded during reset)
        if self._use_time_based_loading and current_step == 0 and self._initial_step_loaded:
            return

        # Remap hex indices to our grid size
        pickup_hexes = pickup_hexes % self.num_hexes
        dropoff_hexes = dropoff_hexes % self.num_hexes
        
        same_mask = pickup_hexes == dropoff_hexes
        dropoff_hexes[same_mask] = (dropoff_hexes[same_mask] + 1) % self.num_hexes
        
        trip_ids = torch.arange(self._next_trip_id, self._next_trip_id + actual_trips, device=self.device)
        self._next_trip_id += actual_trips
        
        self.trip_state.add_trips(
            trip_ids=trip_ids,
            pickup_hexes=pickup_hexes,
            dropoff_hexes=dropoff_hexes,
            fares=fares,
            distances=distances
        )
        self.trips_loaded += actual_trips
    
    def _generate_synthetic_step_trips(self):
        """Generate synthetic trips for this step."""
        # Probabilistic trip generation
        num_new_trips = torch.poisson(torch.tensor(float(self._trip_generation_rate))).int().item()
        num_new_trips = min(num_new_trips, 50)  # Cap at 50 per step
        
        if num_new_trips == 0:
            return
        
        trip_ids = torch.arange(self._next_trip_id, self._next_trip_id + num_new_trips, device=self.device)
        self._next_trip_id += num_new_trips
        
        pickup_hexes = torch.randint(0, self.num_hexes, (num_new_trips,), device=self.device)
        dropoff_hexes = torch.randint(0, self.num_hexes, (num_new_trips,), device=self.device)
        
        same_mask = pickup_hexes == dropoff_hexes
        dropoff_hexes[same_mask] = (dropoff_hexes[same_mask] + 1) % self.num_hexes
        
        distances = self.hex_grid.distance_matrix.get_distances_batch(pickup_hexes, dropoff_hexes)
        distances = torch.clamp(distances, min=1.0, max=20.0)
        
        base_fare = 3.0
        per_km_rate = 2.5
        fares = base_fare + distances * per_km_rate
        
        self.trip_state.add_trips(
            trip_ids=trip_ids,
            pickup_hexes=pickup_hexes,
            dropoff_hexes=dropoff_hexes,
            fares=fares,
            distances=distances
        )
        self.trips_loaded += num_new_trips
