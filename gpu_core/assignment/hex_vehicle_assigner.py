"""Deterministic hex-to-vehicle assignment layer.

Converts fleet-level per-hex action allocations into per-vehicle actions
compatible with env.step(action_type, reposition_target).

No learnable parameters — pure torch operations.
"""

import torch
from typing import Tuple


# Action type constants (must match environment/action_processor)
SERVE = 0
CHARGE = 1
REPOSITION = 2


class HexVehicleAssigner:
    """Maps per-hex action allocations to per-vehicle actions.

    Given allocation_probs [H, 3] (P(serve), P(charge), P(repos) per hex),
    assigns each idle vehicle to an action based on:
      1. The hex-level allocation fractions.
      2. SOC-priority ordering (low SOC → CHARGE first).
      3. Safety override: vehicles with SOC < threshold forced to CHARGE.

    Args:
        soc_low_threshold: Vehicles below this SOC are forced to CHARGE.
        soc_priority: If True, sort vehicles by SOC for assignment priority.
    """

    def __init__(
        self,
        soc_low_threshold: float = 20.0,
        soc_priority: bool = True,
        idle_force_charge_steps: int = 24,
    ):
        self.soc_low_threshold = soc_low_threshold
        self.soc_priority = soc_priority
        self.idle_force_charge_steps = max(1, int(idle_force_charge_steps))

    @torch.no_grad()
    def assign(
        self,
        allocation_probs: torch.Tensor,       # [H, 3]
        repos_sampled_targets: torch.Tensor,   # [H] target hex per hex
        charge_power: torch.Tensor,            # [H] continuous power in (0, 1)
        vehicle_hex_ids: torch.Tensor,         # [V] current hex per vehicle
        vehicle_socs: torch.Tensor,            # [V] current SOC (0-100)
        vehicle_status: torch.Tensor,          # [V] int8 status
        idle_steps: torch.Tensor = None,       # [V] consecutive idle steps
        idle_status: int = 0,                  # VehicleStatus.IDLE
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign per-vehicle actions from hex-level allocations.

        Args:
            allocation_probs: [H, 3] P(serve), P(charge), P(repos) per hex.
            repos_sampled_targets: [H] sampled reposition target hex per hex.
            charge_power: [H] continuous charging power fraction per hex.
            vehicle_hex_ids: [V] hex index each vehicle is in.
            vehicle_socs: [V] current SOC values.
            vehicle_status: [V] vehicle status (only IDLE=0 are assigned).
            idle_status: Integer value for IDLE status.

        Returns:
            action_type: [V] int — SERVE=0, CHARGE=1, REPOSITION=2.
            reposition_target: [V] long — target hex for REPOSITION vehicles.
            vehicle_charge_power: [V] float — charging power for CHARGE vehicles.
        """
        V = vehicle_hex_ids.shape[0]
        H = allocation_probs.shape[0]
        device = vehicle_hex_ids.device

        # Output tensors
        action_type = torch.full((V,), SERVE, dtype=torch.long, device=device)
        reposition_target = torch.zeros(V, dtype=torch.long, device=device)
        vehicle_charge_power = torch.zeros(V, dtype=torch.float32, device=device)

        # Only assign idle vehicles
        idle_mask = (vehicle_status == idle_status)
        idle_indices = idle_mask.nonzero(as_tuple=True)[0]  # [N_idle]

        if idle_indices.numel() == 0:
            return action_type, reposition_target, vehicle_charge_power

        idle_hex_ids = vehicle_hex_ids[idle_indices]   # [N_idle]
        idle_socs = vehicle_socs[idle_indices]         # [N_idle]
        idle_idle_steps = idle_steps[idle_indices] if idle_steps is not None else torch.zeros_like(idle_indices, dtype=torch.int64)

        # --- Safety overrides ---
        # Low-SOC vehicles must charge; long-idle vehicles should be pushed to reposition
        # so they do not keep burning serve quota in demand deserts.
        force_charge_mask = idle_socs < self.soc_low_threshold
        force_repos_mask = idle_idle_steps >= self.idle_force_charge_steps

        # --- Process each occupied hex ---
        # Find unique hexes with idle vehicles
        unique_hexes = idle_hex_ids.unique()

        for hex_id in unique_hexes:
            h = hex_id.item()
            # Vehicles in this hex
            in_hex_mask = idle_hex_ids == hex_id  # [N_idle] bool
            in_hex_local = in_hex_mask.nonzero(as_tuple=True)[0]  # indices into idle_indices
            n_in_hex = in_hex_local.numel()

            if n_in_hex == 0:
                continue

            # Fair integer apportionment from fractional allocations:
            #   1) floor all ideal counts
            #   2) distribute leftover by largest remainder
            # Tie-break priority for equal remainders:
            #   REPOSITION > CHARGE > SERVE
            probs = allocation_probs[h].float().clamp(min=0.0)
            prob_sum = probs.sum().item()
            if prob_sum <= 0:
                probs = torch.tensor([0.0, 0.0, 1.0], device=device)  # fallback to REPOSITION
            else:
                probs = probs / prob_sum

            ideal_counts = probs * float(n_in_hex)  # [3]
            base_counts = torch.floor(ideal_counts).long()  # [3]
            leftovers = int(n_in_hex - base_counts.sum().item())

            # Largest remainder with deterministic tie-break.
            remainders = ideal_counts - base_counts.float()
            priority_bias = torch.tensor([0.0, 1e-6, 2e-6], device=device)
            scores = remainders + priority_bias
            if leftovers > 0:
                top_actions = scores.argsort(descending=True)
                for i in range(leftovers):
                    a = top_actions[i % 3].item()
                    base_counts[a] += 1

            n_serve = int(base_counts[SERVE].item())
            n_charge = int(base_counts[CHARGE].item())
            n_repos = int(base_counts[REPOSITION].item())

            # Get global indices for vehicles in this hex
            global_indices = idle_indices[in_hex_local]  # [n_in_hex] into V
            hex_socs = idle_socs[in_hex_local]           # [n_in_hex]
            hex_force_charge = force_charge_mask[in_hex_local]  # [n_in_hex]
            hex_force_repos = force_repos_mask[in_hex_local]    # [n_in_hex]

            # --- SOC-priority sorting ---
            if self.soc_priority:
                sorted_order = hex_socs.argsort()  # ascending SOC
            else:
                sorted_order = torch.arange(n_in_hex, device=device)

            sorted_global = global_indices[sorted_order]
            sorted_force_charge = hex_force_charge[sorted_order]
            sorted_force_repos = hex_force_repos[sorted_order] & (~sorted_force_charge)

            # Count forced action vehicles
            n_forced_charge = int(sorted_force_charge.sum().item())
            n_forced_repos = int(sorted_force_repos.sum().item())

            # Forced charge/reposition claims reduce serve first.
            actual_n_charge = max(n_charge, n_forced_charge)
            actual_n_charge = min(actual_n_charge, n_in_hex)
            remaining_after_charge = n_in_hex - actual_n_charge
            actual_n_repos = max(n_repos, n_forced_repos)
            actual_n_repos = min(actual_n_repos, remaining_after_charge)

            # Build exact forced action sets first
            forced_charge_vehicles = sorted_global[sorted_force_charge]
            remaining_after_charge_force = sorted_global[~sorted_force_charge]
            remaining_force_repos = sorted_force_repos[~sorted_force_charge]

            extra_charge_needed = max(0, actual_n_charge - forced_charge_vehicles.numel())
            extra_charge_vehicles = remaining_after_charge_force[:extra_charge_needed]
            charge_vehicles = torch.cat([forced_charge_vehicles, extra_charge_vehicles], dim=0)

            remaining_vehicles = remaining_after_charge_force[extra_charge_needed:]
            remaining_force_repos = remaining_force_repos[extra_charge_needed:]

            forced_repos_vehicles = remaining_vehicles[remaining_force_repos]
            non_forced_remaining = remaining_vehicles[~remaining_force_repos]
            extra_repos_needed = max(0, actual_n_repos - forced_repos_vehicles.numel())
            extra_repos_vehicles = non_forced_remaining[-extra_repos_needed:] if extra_repos_needed > 0 else non_forced_remaining[:0]
            repos_vehicles = torch.cat([forced_repos_vehicles, extra_repos_vehicles], dim=0)

            if extra_repos_needed > 0:
                serve_vehicles = non_forced_remaining[:-extra_repos_needed]
            else:
                serve_vehicles = non_forced_remaining

            # Write assignments
            if charge_vehicles.numel() > 0:
                action_type[charge_vehicles] = CHARGE
                vehicle_charge_power[charge_vehicles] = charge_power[h]
            if repos_vehicles.numel() > 0:
                action_type[repos_vehicles] = REPOSITION
                reposition_target[repos_vehicles] = repos_sampled_targets[h]
            if serve_vehicles.numel() > 0:
                action_type[serve_vehicles] = SERVE

        return action_type, reposition_target, vehicle_charge_power
