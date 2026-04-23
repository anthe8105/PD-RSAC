"""Hex grid utilities."""

import torch
from typing import Optional, List, Dict, Tuple
from .distance import DistanceMatrix
from .neighbors import HexNeighbors


class HexGrid:
    """
    Combined hex grid manager with distance and neighbor lookups.
    
    Provides unified interface for all spatial operations
    with lazy computation and caching.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        
        self._hex_ids: Optional[List[str]] = None
        self._hex_to_idx: Optional[Dict[str, int]] = None
        self._latitudes: Optional[torch.Tensor] = None
        self._longitudes: Optional[torch.Tensor] = None
        
        self.distance_matrix = DistanceMatrix(device)
        self.neighbors = HexNeighbors(device)
        
        self._initialized = False
    
    @property
    def num_hexes(self) -> int:
        return len(self._hex_ids) if self._hex_ids else 0
    
    @property
    def hex_ids(self) -> List[str]:
        return self._hex_ids or []
    
    @property
    def hex_to_idx(self) -> Dict[str, int]:
        return self._hex_to_idx or {}
    
    def initialize(
        self,
        hex_ids: List[str],
        neighbor_rings: int = 1,
    ) -> None:
        """
        Initialize grid with hex IDs.
        
        Computes distance matrix and neighbor lookup tables.
        """
        try:
            import h3
        except ImportError:
            raise ImportError("h3 library required for grid initialization")
        
        self._hex_ids = hex_ids
        self._hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
        
        # Get hex centers
        lats = []
        lons = []
        for hex_id in hex_ids:
            lat, lon = h3.cell_to_latlng(hex_id)
            lats.append(lat)
            lons.append(lon)
        
        self._latitudes = torch.tensor(lats, dtype=torch.float32, device=self.device)
        self._longitudes = torch.tensor(lons, dtype=torch.float32, device=self.device)
        
        # Compute distance matrix
        self.distance_matrix.compute(self._latitudes, self._longitudes, hex_ids)
        
        # Compute neighbors
        self.neighbors.compute(hex_ids, k=neighbor_rings)
        
        self._initialized = True
    
    def initialize_from_cache(self, cache_path: str) -> bool:
        """Try to load from cache. Returns True if successful."""
        try:
            import os
            distance_path = os.path.join(cache_path, "distance_matrix.pt")
            neighbor_path = os.path.join(cache_path, "neighbors.pt")
            grid_path = os.path.join(cache_path, "grid_info.pt")
            
            if all(os.path.exists(p) for p in [distance_path, neighbor_path, grid_path]):
                self.distance_matrix.load(distance_path)
                self.neighbors.load(neighbor_path)
                
                grid_info = torch.load(grid_path)
                self._hex_ids = grid_info["hex_ids"]
                self._hex_to_idx = {h: i for i, h in enumerate(self._hex_ids)}
                self._latitudes = grid_info["latitudes"].to(self.device)
                self._longitudes = grid_info["longitudes"].to(self.device)
                
                self._initialized = True
                return True
        except Exception:
            pass
        return False
    
    def save_cache(self, cache_path: str) -> None:
        """Save grid data to cache."""
        import os
        os.makedirs(cache_path, exist_ok=True)
        
        self.distance_matrix.save(os.path.join(cache_path, "distance_matrix.pt"))
        self.neighbors.save(os.path.join(cache_path, "neighbors.pt"))
        
        grid_info = {
            "hex_ids": self._hex_ids,
            "latitudes": self._latitudes.cpu(),
            "longitudes": self._longitudes.cpu(),
        }
        torch.save(grid_info, os.path.join(cache_path, "grid_info.pt"))
    
    def get_distance(self, hex_i: int, hex_j: int) -> float:
        """Get distance between two hexes."""
        return self.distance_matrix.get_distance(hex_i, hex_j)
    
    def get_neighbors(self, hex_idx: int) -> torch.Tensor:
        """Get neighbor indices for a hex."""
        return self.neighbors.get_neighbors(hex_idx)
    
    def get_hexes_within_distance(
        self,
        hex_idx: int,
        max_distance: float,
    ) -> torch.Tensor:
        """Get hexes within distance of given hex."""
        return self.distance_matrix.get_hexes_within_distance(hex_idx, max_distance)
    
    def get_nearest_hexes(
        self,
        hex_idx: int,
        k: int,
        exclude_self: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get k nearest hexes."""
        return self.distance_matrix.get_nearest_hexes(hex_idx, k, exclude_self)
    
    def hex_id_to_idx(self, hex_id: str) -> int:
        """Convert hex ID to index."""
        return self._hex_to_idx.get(hex_id, -1)
    
    def idx_to_hex_id(self, idx: int) -> str:
        """Convert index to hex ID."""
        if 0 <= idx < len(self._hex_ids):
            return self._hex_ids[idx]
        return ""
    
    def batch_hex_id_to_idx(self, hex_ids: List[str]) -> torch.Tensor:
        """Convert list of hex IDs to indices tensor."""
        indices = [self._hex_to_idx.get(h, -1) for h in hex_ids]
        return torch.tensor(indices, dtype=torch.long, device=self.device)
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Get adjacency matrix for GCN."""
        return self.neighbors.to_adjacency_matrix()
    
    def get_normalized_adjacency(self) -> torch.Tensor:
        """Get symmetrically normalized adjacency for GCN."""
        adj = self.get_adjacency_matrix()
        
        # Add self-loops
        adj = adj + torch.eye(self.num_hexes, device=self.device)
        
        # Compute degree matrix
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # D^(-1/2) A D^(-1/2)
        norm_adj = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
        
        return norm_adj
    
    def get_hex_centers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get (latitudes, longitudes) tensors."""
        return self._latitudes, self._longitudes
    
    def random_hex_indices(self, n: int) -> torch.Tensor:
        """Get n random hex indices."""
        return torch.randint(0, self.num_hexes, (n,), device=self.device)
    
    def distribute_vehicles(
        self,
        num_vehicles: int,
        method: str = "uniform",
    ) -> torch.Tensor:
        """
        Distribute vehicles across hexes.
        
        Args:
            num_vehicles: Number of vehicles
            method: "uniform" or "random"
            
        Returns:
            Tensor of hex indices for each vehicle
        """
        if method == "uniform":
            # Distribute evenly, cycling through hexes
            indices = torch.arange(num_vehicles, device=self.device) % self.num_hexes
        else:
            # Random distribution
            indices = torch.randint(0, self.num_hexes, (num_vehicles,), device=self.device)
        
        return indices
