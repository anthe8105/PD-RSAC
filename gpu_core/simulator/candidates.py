"""Candidate sets shared by the executor, the MILP projection and its fallback (D6, F57, Q3).

- orders: an order can be served by vehicles whose cell is within the 1-ring of its pickup
  (grid distance <= 1, Q3a) and that keep E >= E_min after pickup + trip (Eq. 4 / Eq. 16);
- stations: the ``num_station_candidates`` nearest stations of the vehicle's cell (5);
- reposition: 91 slots = the H3 k-ring 5 (Q3b). Slot k is a fixed local-IJ offset, so it has
  the same geometric meaning ("2 cells north-east") in every cell; slot 0 is the cell itself
  (stay). Slots that fall outside the domain are -1. The projection only considers the
  actor's target and staying (D7).
"""

from typing import Optional, Sequence, Tuple

import h3

import numpy as np
import torch

from .physics import EPS, Physics


class CandidateGenerator:

    def __init__(self, physics: Physics, distances: torch.Tensor, ring1: torch.Tensor,
                 station_cells: torch.Tensor, num_station_candidates: int = 5, repos_rings: int = 5):
        self.physics = physics
        self.dist = distances.detach().cpu().numpy().astype(np.float64)       # [H, H] km
        self.ring1 = ring1.detach().cpu().numpy().astype(bool)                 # [H, H], diagonal set
        self.station_cells = station_cells.detach().cpu().numpy().astype(np.int64)
        H = self.dist.shape[0]
        S = len(self.station_cells)
        k = min(num_station_candidates, S)
        d_hs = self.dist[:, self.station_cells]                                # [H, S]
        self.near_stations = np.argsort(d_hs, axis=1, kind="stable")[:, :k]   # [H, k] station ids
        self.repos_rings = repos_rings
        self.slot_cells: Optional[np.ndarray] = None     # [H, K] target cell per slot, -1 outside
        self.slot_offsets: Optional[np.ndarray] = None   # [K, 2] local-IJ offsets, row 0 = (0, 0)
        # Before the slot table exists, bound reposition distances by graph hops.
        reach = np.eye(H, dtype=bool)
        for _ in range(repos_rings):
            reach = (reach.astype(np.int32) @ self.ring1.astype(np.int32)) > 0
        self.max_repos_km = float(self.dist[reach].max()) if H else 0.0

    def build_slots(self, cells: Sequence[str]) -> None:
        """Slot table from H3 local IJ coordinates (anchor = first cell; NYC has no pentagon)."""
        anchor = cells[0]
        ij = [h3.cell_to_local_ij(anchor, c) for c in cells]
        lookup = {xy: i for i, xy in enumerate(ij)}
        centre = cells[0]
        ij0 = h3.cell_to_local_ij(anchor, centre)
        offs = sorted(((a - ij0[0], b - ij0[1]) for a, b in
                       (h3.cell_to_local_ij(anchor, x) for x in h3.grid_disk(centre, self.repos_rings))),
                      key=lambda o: (h3.grid_distance(centre, h3.local_ij_to_cell(anchor, ij0[0] + o[0], ij0[1] + o[1])), o))
        assert offs[0] == (0, 0)
        table = np.array([[lookup.get((x + a, y + b), -1) for a, b in offs] for x, y in ij], dtype=np.int64)
        for h, c in enumerate(cells):              # the offsets must reproduce the true k-ring
            want = {lookup[x] for x in (h3.cell_to_local_ij(anchor, d) for d in h3.grid_disk(c, self.repos_rings)) if x in lookup}
            if set(table[h][table[h] >= 0].tolist()) != want:
                raise ValueError(f"local-IJ slots do not match the {self.repos_rings}-ring at cell {c}")
        self.slot_cells = table
        self.slot_offsets = np.array(offs, dtype=np.int64)
        valid = table >= 0
        rows = np.repeat(np.arange(len(cells))[:, None], table.shape[1], axis=1)
        self.max_repos_km = float(self.dist[rows[valid], table[valid]].max())
        # slot of (cell, target), -1 when the target is not a slot of that cell
        self.slot_of = np.full((len(cells), len(cells)), -1, dtype=np.int64)
        self.slot_of[rows[valid], table[valid]] = np.nonzero(valid)[1]

    @property
    def num_slots(self) -> int:
        return 0 if self.slot_cells is None else self.slot_cells.shape[1]

    def station_candidates(self, cells: np.ndarray) -> np.ndarray:
        return self.near_stations[cells]

    def serve_pairs(self, veh_cells: np.ndarray, veh_energy: np.ndarray, pickups: np.ndarray,
                    trip_km: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(vehicle positions, order positions, total km) of every admissible pair."""
        if len(veh_cells) == 0 or len(pickups) == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e, np.empty(0)
        near = self.ring1[veh_cells][:, pickups]                                 # [n_v, n_o]
        km = self.dist[veh_cells][:, pickups] + trip_km[None, :]
        ok = near & (veh_energy[:, None] - self.physics.energy_per_km * km >= self.physics.reserve_kwh - EPS)
        vi, oi = np.nonzero(ok)
        return vi, oi, km[vi, oi]
