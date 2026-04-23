"""Hex neighbor lookup utilities."""

import torch
from typing import Optional, List, Dict, Tuple


class HexNeighbors:
    """
    Pre-computed hex neighbor lookup table.
    
    Enables O(1) neighbor queries during simulation
    by caching all k-ring neighbors.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self._neighbors: Optional[torch.Tensor] = None
        self._neighbor_counts: Optional[torch.Tensor] = None
        self._hex_ids: Optional[List[str]] = None
        self._hex_to_idx: Optional[Dict[str, int]] = None
        self._num_hexes: int = 0
        self._max_neighbors: int = 0
    
    @property
    def num_hexes(self) -> int:
        return self._num_hexes
    
    def compute(
        self,
        hex_ids: List[str],
        k: int = 1,
    ) -> None:
        """
        Compute k-ring neighbors for all hexes.
        
        Args:
            hex_ids: List of H3 hex IDs
            k: Number of rings (1 = immediate neighbors)
        """
        try:
            import h3
        except ImportError:
            raise ImportError("h3 library required for neighbor computation")
        
        n = len(hex_ids)
        self._num_hexes = n
        self._hex_ids = hex_ids
        self._hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
        
        # Get neighbors for each hex
        all_neighbors = []
        for hex_id in hex_ids:
            ring = h3.grid_ring(hex_id, k)
            neighbor_indices = []
            for neighbor_id in ring:
                if neighbor_id in self._hex_to_idx:
                    neighbor_indices.append(self._hex_to_idx[neighbor_id])
            all_neighbors.append(neighbor_indices)
        
        # Find max neighbors for padding
        self._max_neighbors = max(len(n) for n in all_neighbors) if all_neighbors else 0
        
        # Create padded tensor
        neighbors = torch.full(
            (n, self._max_neighbors), -1, dtype=torch.long, device=self.device
        )
        counts = torch.zeros(n, dtype=torch.int32, device=self.device)
        
        for i, neighbor_list in enumerate(all_neighbors):
            k_n = len(neighbor_list)
            counts[i] = k_n
            if k_n > 0:
                neighbors[i, :k_n] = torch.tensor(neighbor_list, dtype=torch.long)
        
        self._neighbors = neighbors
        self._neighbor_counts = counts
    
    def compute_from_adjacency(
        self,
        adjacency_matrix: torch.Tensor,
        hex_ids: Optional[List[str]] = None,
    ) -> None:
        """Compute neighbors from adjacency matrix."""
        n = adjacency_matrix.shape[0]
        self._num_hexes = n
        self._hex_ids = hex_ids
        if hex_ids:
            self._hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
        
        # Find max neighbors
        neighbor_counts = (adjacency_matrix > 0).sum(dim=1)
        self._max_neighbors = neighbor_counts.max().item()
        
        # Build neighbor tensor
        neighbors = torch.full(
            (n, self._max_neighbors), -1, dtype=torch.long, device=self.device
        )
        
        for i in range(n):
            neighbor_indices = torch.where(adjacency_matrix[i] > 0)[0]
            k_n = len(neighbor_indices)
            if k_n > 0:
                neighbors[i, :k_n] = neighbor_indices
        
        self._neighbors = neighbors.to(self.device)
        self._neighbor_counts = neighbor_counts.to(self.device).to(torch.int32)
    
    def get_neighbors(self, hex_idx: int) -> torch.Tensor:
        """Get neighbor indices for a hex (excludes -1 padding)."""
        count = self._neighbor_counts[hex_idx].item()
        return self._neighbors[hex_idx, :count]
    
    def get_neighbors_batch(self, hex_indices: torch.Tensor) -> torch.Tensor:
        """Get neighbors for batch of hexes [batch_size, max_neighbors]."""
        return self._neighbors[hex_indices]
    
    def get_neighbor_counts(self, hex_indices: torch.Tensor) -> torch.Tensor:
        """Get neighbor counts for batch of hexes."""
        return self._neighbor_counts[hex_indices]
    
    def get_all_neighbors(self) -> torch.Tensor:
        """Get full neighbor tensor [num_hexes, max_neighbors]."""
        return self._neighbors
    
    def get_valid_neighbor_mask(self) -> torch.Tensor:
        """Get mask of valid neighbors [num_hexes, max_neighbors]."""
        return self._neighbors >= 0
    
    def to_adjacency_matrix(self) -> torch.Tensor:
        """Convert neighbor list to adjacency matrix."""
        adj = torch.zeros(
            self._num_hexes, self._num_hexes,
            dtype=torch.float32, device=self.device
        )
        
        for i in range(self._num_hexes):
            neighbors = self.get_neighbors(i)
            adj[i, neighbors] = 1.0
        
        return adj
    
    @staticmethod
    def compute_khop_mask(
        adjacency: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Compute K-hop neighborhood boolean mask from adjacency matrix.

        Returns mask[i,j] = True if hex j is reachable from hex i within k hops.
        Self-loops are included (mask[i,i] = True).

        Args:
            adjacency: [H, H] binary or weighted adjacency (non-normalized).
                       If using normalized adjacency, pass (adj > 0).float() instead.
            k: Number of hops (e.g., 4 → ~61 neighbors on a hex grid).

        Returns:
            [H, H] bool tensor.
        """
        H = adjacency.shape[0]
        device = adjacency.device
        # Start with identity (self-loop) + 1-hop adjacency
        A = (adjacency > 0).float()
        reachable = torch.eye(H, device=device) + A
        A_power = A.clone()
        for _ in range(2, k + 1):
            A_power = A_power @ A
            reachable = reachable + A_power
        return reachable > 0  # [H, H] bool

    @staticmethod
    def khop_to_padded_indices(
        khop_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Convert K-hop boolean mask to padded neighbor index tensor.

        Args:
            khop_mask: [H, H] bool — output of compute_khop_mask.

        Returns:
            neighbor_indices: [H, max_K] long — padded with -1.
            neighbor_counts:  [H] int32 — valid neighbor count per hex.
            max_K: int — maximum number of K-hop neighbors across all hexes.
        """
        H = khop_mask.shape[0]
        device = khop_mask.device
        # Exclude self-loop from reposition targets (can't reposition to own hex)
        mask_no_self = khop_mask.clone()
        mask_no_self.fill_diagonal_(False)

        counts = mask_no_self.sum(dim=1).to(torch.int32)  # [H]
        max_K = int(counts.max().item())

        indices = torch.full((H, max_K), -1, dtype=torch.long, device=device)
        for i in range(H):
            neighbors = torch.where(mask_no_self[i])[0]
            n = neighbors.shape[0]
            if n > 0:
                indices[i, :n] = neighbors

        return indices, counts, max_K

    def save(self, path: str) -> None:
        """Save neighbor data to file."""
        data = {
            "neighbors": self._neighbors.cpu(),
            "neighbor_counts": self._neighbor_counts.cpu(),
            "hex_ids": self._hex_ids,
            "num_hexes": self._num_hexes,
            "max_neighbors": self._max_neighbors,
        }
        torch.save(data, path)
    
    def load(self, path: str) -> None:
        """Load neighbor data from file."""
        data = torch.load(path)
        self._neighbors = data["neighbors"].to(self.device)
        self._neighbor_counts = data["neighbor_counts"].to(self.device)
        self._hex_ids = data.get("hex_ids")
        self._num_hexes = data["num_hexes"]
        self._max_neighbors = data["max_neighbors"]
        
        if self._hex_ids:
            self._hex_to_idx = {h: i for i, h in enumerate(self._hex_ids)}
