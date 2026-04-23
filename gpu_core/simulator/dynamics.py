"""Energy and time dynamics."""

import torch
from typing import Tuple
from ..state import TensorFleetState, VehicleStatus


class EnergyDynamics:
    """Handles energy consumption and charging dynamics."""
    
    def __init__(
        self,
        energy_per_km: float = 0.2,
        charge_power_kw: float = 50.0,
        max_soc: float = 100.0,
        min_soc_reserve: float = 10.0,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.energy_per_km = energy_per_km
        self.charge_power_kw = charge_power_kw
        self.max_soc = max_soc
        self.min_soc_reserve = min_soc_reserve
    
    def compute_consumption(
        self,
        distance_km: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy consumption for given distances."""
        return distance_km * self.energy_per_km
    
    def compute_charge_gain(
        self,
        duration_hours: float,
        charge_power: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy gained from charging."""
        return charge_power * duration_hours
    
    def get_range_km(
        self,
        soc: torch.Tensor,
    ) -> torch.Tensor:
        """Get remaining range in km for given SoC."""
        usable_soc = torch.clamp(soc - self.min_soc_reserve, min=0.0)
        return usable_soc / self.energy_per_km
    
    def get_time_to_full(
        self,
        current_soc: torch.Tensor,
        charge_power: torch.Tensor,
    ) -> torch.Tensor:
        """Get time in hours to fully charge."""
        needed = self.max_soc - current_soc
        time = needed / torch.clamp(charge_power, min=1e-6)
        return time
    
    def needs_charging(
        self,
        soc: torch.Tensor,
        threshold: float = 20.0,
    ) -> torch.Tensor:
        """Check which vehicles need charging."""
        return soc < threshold
    
    def can_complete_trip(
        self,
        current_soc: torch.Tensor,
        trip_distance_km: torch.Tensor,
        return_distance_km: torch.Tensor,
    ) -> torch.Tensor:
        """Check if vehicle can complete trip and return with reserve."""
        total_distance = trip_distance_km + return_distance_km
        energy_needed = self.compute_consumption(total_distance)
        available = current_soc - self.min_soc_reserve
        return available >= energy_needed


class TimeDynamics:
    """Handles time-related computations."""
    
    def __init__(
        self,
        step_duration_minutes: float = 5.0,
        avg_speed_kmh: float = 30.0,
        episode_duration_hours: float = 10.0,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.step_duration_minutes = step_duration_minutes
        self.step_duration_hours = step_duration_minutes / 60.0
        self.avg_speed_kmh = avg_speed_kmh
        self.episode_duration_hours = episode_duration_hours
        
        self.steps_per_episode = int(episode_duration_hours * 60 / step_duration_minutes)
        self.km_per_step = avg_speed_kmh * self.step_duration_hours
    
    def distance_to_steps(
        self,
        distance_km: torch.Tensor,
    ) -> torch.Tensor:
        """Convert distance to travel time in steps."""
        time_hours = distance_km / self.avg_speed_kmh
        time_steps = torch.ceil(time_hours / self.step_duration_hours)
        return time_steps.to(torch.int32)
    
    def steps_to_hours(
        self,
        steps: torch.Tensor,
    ) -> torch.Tensor:
        """Convert steps to hours."""
        return steps.float() * self.step_duration_hours
    
    def hours_to_steps(
        self,
        hours: torch.Tensor,
    ) -> torch.Tensor:
        """Convert hours to steps."""
        return torch.ceil(hours / self.step_duration_hours).to(torch.int32)
    
    def get_step_from_time(
        self,
        hour: float,
        minute: float = 0.0,
    ) -> int:
        """Convert time of day to step number."""
        total_minutes = hour * 60 + minute
        return int(total_minutes / self.step_duration_minutes)
    
    def get_time_from_step(
        self,
        step: int,
    ) -> Tuple[int, int]:
        """Convert step to time of day (hour, minute)."""
        total_minutes = step * self.step_duration_minutes
        hour = int(total_minutes // 60)
        minute = int(total_minutes % 60)
        return hour, minute
    
    def is_peak_hour(
        self,
        step: int,
        morning_start: int = 7,
        morning_end: int = 10,
        evening_start: int = 17,
        evening_end: int = 20,
    ) -> bool:
        """Check if step is during peak hours."""
        hour, _ = self.get_time_from_step(step)
        return (morning_start <= hour < morning_end) or (evening_start <= hour < evening_end)
    
    def get_remaining_steps(
        self,
        current_step: int,
    ) -> int:
        """Get remaining steps in episode."""
        return max(0, self.steps_per_episode - current_step)
    
    def get_episode_progress(
        self,
        current_step: int,
    ) -> float:
        """Get episode progress as fraction [0, 1]."""
        return min(1.0, current_step / self.steps_per_episode)
