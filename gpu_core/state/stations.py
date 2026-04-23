"""Tensor-based charging station state representation."""

import torch
from typing import Optional


class TensorStationState:
    """
    GPU-native charging station state using pure tensors.
    
    Tracks station occupancy, queue lengths, and charging sessions.
    """
    
    def __init__(
        self,
        num_stations: int,
        device: str = "cuda",
        num_ports: int = 1,
        max_power: float = 50.0,
        electricity_price: float = 0.15,
    ):
        self.num_stations = num_stations
        self.device = torch.device(device)
        self.num_ports = num_ports
        self.max_power = max_power
        self.electricity_price = electricity_price
        
        # Station hex locations
        self.hex_ids = torch.zeros(num_stations, dtype=torch.long, device=self.device)
        
        # Number of ports per station
        self.ports = torch.full((num_stations,), num_ports, dtype=torch.int32, device=self.device)
        
        # Currently occupied ports
        self.occupied = torch.zeros(num_stations, dtype=torch.int32, device=self.device)
        
        # Queue length at each station
        self.queue_length = torch.zeros(num_stations, dtype=torch.int32, device=self.device)
        
        # Power capacity per station (kW)
        self.power_capacity = torch.full((num_stations,), max_power, dtype=torch.float32, device=self.device)
        
        # Current power usage per station (kW)
        self.power_usage = torch.zeros(num_stations, dtype=torch.float32, device=self.device)
    
    def set_locations(self, hex_ids: torch.Tensor) -> None:
        n = min(len(hex_ids), self.num_stations)
        self.hex_ids[:n] = hex_ids[:n]
    
    def get_available_ports(self) -> torch.Tensor:
        """Get number of available ports at each station."""
        return torch.clamp(self.ports - self.occupied, min=0)
    
    def get_available_mask(self) -> torch.Tensor:
        """Get mask of stations with available ports."""
        return self.get_available_ports() > 0
    
    def get_utilization(self) -> torch.Tensor:
        """Get utilization ratio for each station."""
        return self.occupied.float() / self.ports.float().clamp(min=1)
    
    def occupy_port(self, station_idx: int) -> bool:
        """Try to occupy a port. Returns True if successful."""
        if self.occupied[station_idx] < self.ports[station_idx]:
            self.occupied[station_idx] += 1
            return True
        return False
    
    def occupy_ports_batch(self, station_indices: torch.Tensor) -> None:
        """
        Occupy ports for multiple stations (assumes already validated).
        OPTIMIZED: vectorized port occupation.
        """
        if len(station_indices) == 0:
            return
        
        # Count requests per station
        unique_stations, counts = torch.unique(station_indices, return_counts=True)
        # Match dtype with self.occupied (Int32)
        self.occupied.index_add_(0, unique_stations.long(), counts.int())
    
    def release_port(self, station_idx: int) -> None:
        """Release a port at station."""
        if self.occupied[station_idx] > 0:
            self.occupied[station_idx] -= 1
    
    def release_ports_batch(self, station_indices: torch.Tensor) -> None:
        """
        Release ports for multiple stations.
        OPTIMIZED: vectorized port release.
        """
        if len(station_indices) == 0:
            return
        
        # Count releases per station
        unique_stations, counts = torch.unique(station_indices, return_counts=True)
        # Atomic subtract (clamp to avoid negative), match dtype
        self.occupied[unique_stations.long()] = (
            self.occupied[unique_stations.long()] - counts.int()
        ).clamp(min=0)
    
    def batch_occupy(self, station_indices: torch.Tensor) -> torch.Tensor:
        """
        Try to occupy ports at multiple stations. Returns success mask.
        
        IMPORTANT: When multiple vehicles request the same station, we need to
        limit how many can actually occupy based on available ports.
        """
        if len(station_indices) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        
        success_mask = torch.zeros(len(station_indices), dtype=torch.bool, device=self.device)
        
        # Process each unique station to respect port limits
        unique_stations = torch.unique(station_indices)
        
        for station in unique_stations:
            station_id = station.item()
            # Find all vehicles wanting this station
            wanting_mask = station_indices == station_id
            wanting_indices = wanting_mask.nonzero(as_tuple=True)[0]
            
            # How many ports available at this station?
            available = self.ports[station_id] - self.occupied[station_id]
            available = max(0, available.item())
            
            # Only allow up to 'available' vehicles
            num_can_occupy = min(len(wanting_indices), available)
            
            if num_can_occupy > 0:
                # Mark first num_can_occupy as successful
                success_indices = wanting_indices[:num_can_occupy]
                success_mask[success_indices] = True
                # Update occupied count
                self.occupied[station_id] += num_can_occupy
        
        return success_mask
    
    def batch_release(self, station_indices: torch.Tensor) -> None:
        """Release ports at multiple stations."""
        unique_stations, counts = torch.unique(station_indices, return_counts=True)
        self.occupied[unique_stations] = torch.clamp(
            self.occupied[unique_stations] - counts.to(torch.int32), min=0
        )
    
    def get_station_at_hex(self, hex_idx: int) -> int:
        """Get station ID at hex, or -1 if none."""
        matches = torch.where(self.hex_ids == hex_idx)[0]
        return matches[0].item() if len(matches) > 0 else -1
    
    def get_stations_at_hexes(self, hex_indices: torch.Tensor) -> torch.Tensor:
        """Get station IDs at hexes, -1 if none."""
        result = torch.full((len(hex_indices),), -1, dtype=torch.long, device=self.device)
        for i, h in enumerate(hex_indices):
            station = self.get_station_at_hex(h.item())
            if station >= 0:
                result[i] = station
        return result
    
    def get_nearest_stations(
        self,
        hex_idx: int,
        hex_distances: torch.Tensor,
        k: int = 3,
    ) -> torch.Tensor:
        """Get k nearest stations to hex, sorted by distance."""
        distances = hex_distances[hex_idx, self.hex_ids]
        _, sorted_indices = torch.sort(distances)
        return sorted_indices[:k]
    
    def get_nearest_available_station(
        self,
        hex_idx: int,
        hex_distances: torch.Tensor,
    ) -> int:
        """Get nearest station with available port, or -1 if none."""
        available_mask = self.get_available_mask()
        if not available_mask.any():
            return -1
        
        distances = hex_distances[hex_idx, self.hex_ids]
        distances[~available_mask] = float("inf")
        return torch.argmin(distances).item()
    
    def reset(self) -> None:
        """Reset all stations to empty state."""
        self.occupied.fill_(0)
        self.queue_length.fill_(0)
        self.power_usage.fill_(0)
    
    def to_feature_tensor(self) -> torch.Tensor:
        """Convert station state to feature tensor [num_stations, feature_dim]."""
        features = torch.stack([
            self.hex_ids.float() / 1000.0,
            self.get_utilization(),
            self.get_available_ports().float() / self.num_ports,
            self.queue_length.float() / 10.0,
            self.power_capacity / 100.0,
        ], dim=1)
        return features
