"""GPU-accelerated distance matrix computation."""

import torch
import math
from typing import Optional, Tuple, List


class DistanceMatrix:
    """
    Pre-computed distance matrix between all hex pairs.
    
    Uses GPU for fast haversine computation and caches results
    for O(1) lookup during simulation.
    """
    
    EARTH_RADIUS_KM = 6371.0
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self._distances: Optional[torch.Tensor] = None
        self._hex_ids: Optional[List[str]] = None
        self._num_hexes: int = 0
    
    @property
    def num_hexes(self) -> int:
        return self._num_hexes
    
    @property
    def distances(self) -> torch.Tensor:
        if self._distances is None:
            raise RuntimeError("Distance matrix not computed. Call compute() first.")
        return self._distances
    
    def compute(
        self,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        hex_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute pairwise haversine distances between all hex centers.
        
        Args:
            latitudes: [num_hexes] latitudes in degrees
            longitudes: [num_hexes] longitudes in degrees
            hex_ids: Optional list of hex ID strings
            
        Returns:
            Distance matrix [num_hexes, num_hexes] in km
        """
        n = len(latitudes)
        self._num_hexes = n
        self._hex_ids = hex_ids
        
        latitudes = latitudes.to(self.device)
        longitudes = longitudes.to(self.device)
        
        # Convert to radians
        lat_rad = torch.deg2rad(latitudes)
        lon_rad = torch.deg2rad(longitudes)
        
        # Expand for pairwise computation
        lat1 = lat_rad.unsqueeze(1)  # [n, 1]
        lat2 = lat_rad.unsqueeze(0)  # [1, n]
        lon1 = lon_rad.unsqueeze(1)  # [n, 1]
        lon2 = lon_rad.unsqueeze(0)  # [1, n]
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat / 2) ** 2 + \
            torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0, 1)))
        
        self._distances = self.EARTH_RADIUS_KM * c
        return self._distances
    
    def compute_from_hex_ids(self, hex_ids: List[str]) -> torch.Tensor:
        """Compute distance matrix from H3 hex IDs."""
        try:
            import h3
        except ImportError:
            raise ImportError("h3 library required for hex ID distance computation")
        
        n = len(hex_ids)
        lats = torch.zeros(n, dtype=torch.float32)
        lons = torch.zeros(n, dtype=torch.float32)
        
        for i, hex_id in enumerate(hex_ids):
            lat, lon = h3.cell_to_latlng(hex_id)
            lats[i] = lat
            lons[i] = lon
        
        return self.compute(lats, lons, hex_ids)
    
    def get_distance(self, hex_i: int, hex_j: int) -> float:
        """Get distance between two hexes by index."""
        return self.distances[hex_i, hex_j].item()
    
    def get_distances_from(self, hex_idx: int) -> torch.Tensor:
        """Get distances from one hex to all others."""
        return self.distances[hex_idx]
    
    def get_distances_batch(
        self,
        from_indices: torch.Tensor,
        to_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Get distances for batch of (from, to) pairs."""
        return self.distances[from_indices, to_indices]
    
    def get_hexes_within_distance(
        self,
        hex_idx: int,
        max_distance: float,
    ) -> torch.Tensor:
        """Get indices of hexes within max_distance of given hex."""
        distances = self.distances[hex_idx]
        return torch.where(distances <= max_distance)[0]
    
    def get_nearest_hexes(
        self,
        hex_idx: int,
        k: int,
        exclude_self: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get k nearest hexes to given hex.
        
        Returns:
            (indices, distances) of k nearest hexes
        """
        distances = self.distances[hex_idx].clone()
        if exclude_self:
            distances[hex_idx] = float("inf")
        
        sorted_distances, sorted_indices = torch.sort(distances)
        return sorted_indices[:k], sorted_distances[:k]
    
    def save(self, path: str) -> None:
        """Save distance matrix to file."""
        data = {
            "distances": self._distances.cpu(),
            "num_hexes": self._num_hexes,
            "hex_ids": self._hex_ids,
        }
        torch.save(data, path)
    
    def load(self, path: str) -> None:
        """Load distance matrix from file."""
        data = torch.load(path)
        self._distances = data["distances"].to(self.device)
        self._num_hexes = data["num_hexes"]
        self._hex_ids = data.get("hex_ids")
    
    @staticmethod
    def haversine_single(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Compute haversine distance between two points (CPU, single)."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat / 2) ** 2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        
        return DistanceMatrix.EARTH_RADIUS_KM * c
