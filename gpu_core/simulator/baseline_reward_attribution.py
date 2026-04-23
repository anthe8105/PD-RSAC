import torch
from typing import Optional


class BaselinePerVehicleRewardAttributor:
    """Build per-vehicle reward tensors for MAPPO/MADDPG baselines only."""

    def __init__(self, num_vehicles: int, device: torch.device):
        self.num_vehicles = int(num_vehicles)
        self.device = device

    def _scalar(self, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return torch.tensor(0.0, device=self.device)
            return value.reshape(-1)[0].to(self.device, dtype=torch.float32)
        return torch.tensor(float(value), device=self.device, dtype=torch.float32)

    def _add_share(
        self,
        rewards: torch.Tensor,
        amount,
        *,
        mask: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> None:
        amount_t = self._scalar(amount)
        if abs(float(amount_t.item())) < 1e-12:
            return

        if indices is not None:
            if indices.numel() > 0:
                rewards[indices] += amount_t / float(indices.numel())
                return

        if mask is not None:
            mask_b = mask.to(torch.bool)
            count = int(mask_b.sum().item())
            if count > 0:
                rewards[mask_b] += amount_t / float(count)
                return

        rewards += amount_t / float(self.num_vehicles)

    def _difference_indices(self, base: torch.Tensor, remove: Optional[torch.Tensor]) -> torch.Tensor:
        if base.numel() == 0:
            return base
        if remove is None or remove.numel() == 0:
            return base
        keep = ~torch.isin(base, remove)
        return base[keep]

    def attribute_step(
        self,
        *,
        total_reward,
        ongoing_serve_revenue,
        ongoing_charge_cost,
        serve_driving_cost,
        charge_travel_cost,
        reposition_cost,
        repos_dispatch_bonus,
        reposition_bonus,
        wait_penalty,
        drop_penalty,
        low_soc_penalty,
        serve_fail_penalty,
        charge_fail_penalty,
        high_soc_penalty,
        very_high_soc_penalty,
        serve_vehicle_indices: torch.Tensor,
        charge_vehicle_indices: torch.Tensor,
        reposition_mask: torch.Tensor,
        reposition_failed_indices: Optional[torch.Tensor],
        ongoing_serving_mask: torch.Tensor,
        ongoing_charging_mask: torch.Tensor,
        failed_serve_mask: Optional[torch.Tensor] = None,
        failed_charge_mask: Optional[torch.Tensor] = None,
        high_soc_charge_mask: Optional[torch.Tensor] = None,
        very_high_soc_charge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rewards = torch.zeros(self.num_vehicles, dtype=torch.float32, device=self.device)

        attempted_reposition_indices = reposition_mask.nonzero(as_tuple=True)[0]
        successful_reposition_indices = self._difference_indices(
            attempted_reposition_indices,
            reposition_failed_indices,
        )

        # Action-linked terms.
        self._add_share(rewards, ongoing_serve_revenue, mask=ongoing_serving_mask)
        self._add_share(rewards, -self._scalar(ongoing_charge_cost), mask=ongoing_charging_mask)
        self._add_share(rewards, -self._scalar(serve_driving_cost), indices=serve_vehicle_indices)
        self._add_share(rewards, -self._scalar(charge_travel_cost), indices=charge_vehicle_indices)
        self._add_share(rewards, -self._scalar(reposition_cost), indices=successful_reposition_indices)
        self._add_share(rewards, repos_dispatch_bonus, indices=attempted_reposition_indices)

        # Reposition completion bonus is delayed and global in this environment.
        self._add_share(rewards, reposition_bonus)

        # Global penalties.
        self._add_share(rewards, -self._scalar(wait_penalty))
        self._add_share(rewards, -self._scalar(drop_penalty))
        self._add_share(rewards, -self._scalar(low_soc_penalty))

        # Failure penalties.
        self._add_share(rewards, -self._scalar(serve_fail_penalty), mask=failed_serve_mask)
        self._add_share(rewards, -self._scalar(charge_fail_penalty), mask=failed_charge_mask)

        # SOC charge penalties.
        self._add_share(rewards, -self._scalar(high_soc_penalty), mask=high_soc_charge_mask)
        self._add_share(rewards, -self._scalar(very_high_soc_penalty), mask=very_high_soc_charge_mask)

        # Conservation correction: make per-vehicle sum match scalar reward exactly.
        target = self._scalar(total_reward)
        residual = target - rewards.sum()
        rewards += residual / float(self.num_vehicles)
        return rewards
