"""Reward of paper Eq. (8), per vehicle and for the fleet (workstream C5, decisions Q2, Q8, Q10).

    r_t = sum fares accepted - c_drv * km - c_elec * kWh bought - lambda_wait * #waiting - lambda_drop * #dropped

- the fare and the driving cost of a trip are booked when the order is accepted, to the vehicle
  that accepts it; reposition and station-trip driving costs when the move starts; electricity
  for every charging step, to the charging vehicle;
- the wait and drop penalties of order o are split equally over the vehicles that decided at
  this epoch within the 1-ring of o's pickup (or, if there are none, over all vehicles that
  decided; or, if none decided, over the whole fleet), so the per-vehicle rewards always sum
  to the fleet reward;
- no shaping terms; the result is divided by a constant ``scale_factor``.
"""

import torch


class RewardComputer:

    def __init__(self, config, ring1: torch.Tensor, device):
        r = config.reward
        self.c_drv = float(r.driving_cost_per_km)
        self.c_elec = float(r.electricity_cost_per_kwh)
        self.wait_penalty = float(r.wait_penalty_per_step)
        self.drop_penalty = float(r.drop_penalty_per_order)
        self.scale = float(r.scale_factor)
        if self.scale <= 0:
            raise ValueError("reward.scale_factor must be positive")
        self.max_wait_steps = int(r.max_wait_steps)
        self.ring1 = ring1.to(device).float()          # [H, H] symmetric, 1 on the diagonal
        self.device = device

    def split_order_penalties(self, pickup_cells: torch.Tensor, penalty: torch.Tensor,
                              vehicle_cells: torch.Tensor, deciders: torch.Tensor) -> torch.Tensor:
        """Per-vehicle share [V] of the order penalties ``penalty`` [O] (unscaled)."""
        V = vehicle_cells.numel()
        share = torch.zeros(V, device=self.device)
        if pickup_cells.numel() == 0:
            return share
        H = self.ring1.shape[0]
        n_cell = torch.zeros(H, device=self.device).index_add_(0, vehicle_cells[deciders], torch.ones(int(deciders.sum()), device=self.device))
        pen_cell = torch.zeros(H, device=self.device).index_add_(0, pickup_cells, penalty)
        n_near = self.ring1 @ n_cell                                   # deciders within the 1-ring of each cell
        local = n_near > 0
        per_vehicle_by_cell = self.ring1 @ torch.where(local, pen_cell / n_near.clamp(min=1.0), torch.zeros_like(pen_cell))
        share[deciders] = per_vehicle_by_cell[vehicle_cells[deciders]]
        pooled = pen_cell[~local].sum()
        if pooled > 0:
            pool = deciders if deciders.any() else torch.ones(V, dtype=torch.bool, device=self.device)
            share[pool] += pooled / pool.sum()
        return share
