"""
Reward Computation for EV Fleet Environment.

Handles all reward and penalty calculations based on the paper's formulation.
Separated from main environment for clarity and modularity.

Paper Eq. 8:
r_t = trip_revenue - driving_cost - electricity_cost - wait_penalty - drop_penalty
"""

import torch
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..state import TensorFleetState, TensorTripState


class RewardComputer:
    """
    Computes rewards and penalties for the EV Fleet RL environment.
    
    Based on Paper Eq. 8:
    r_t = Σ R(pickup, dropoff) - Σ c^drv * d - Σ p^elec * p * Δt - λ_wait * Σ wait - λ_drop * |dropped|
    
    Additional penalties for suboptimal behavior:
    - Low SOC penalty: Encourage proactive charging
    - IDLE penalty: REMOVED (IDLE action removed from policy)
    - High SOC charge penalty: Discourage unnecessary charging
    """
    
    def __init__(
        self,
        config: 'Config',
        device: torch.device,
        adjacency_matrix: 'torch.Tensor' = None,
    ):
        self.config = config
        self.device = device
        
        # Cache reward config values
        self.driving_cost_per_km = config.reward.driving_cost_per_km
        self.electricity_cost_per_kwh = config.reward.electricity_cost_per_kwh
        self.wait_penalty_per_step = config.reward.wait_penalty_per_step
        self.drop_penalty_per_order = config.reward.drop_penalty_per_order
        self.scale_factor = config.reward.scale_factor
        self.max_wait_steps = config.reward.max_wait_steps
        self.serve_bonus = getattr(config.reward, 'serve_bonus', 0.0)
        
        # Action penalties
        self.reposition_penalty = getattr(config.reward, 'reposition_penalty', 0.0)
        # idle_penalty removed — IDLE is no longer a valid policy action
        self.serve_fail_penalty = getattr(config.reward, 'serve_fail_penalty', 3.0)
        self.high_soc_charge_penalty = getattr(config.reward, 'high_soc_charge_penalty', 0.0)
        self.very_high_soc_charge_penalty = getattr(config.reward, 'very_high_soc_charge_penalty', 0.0)
        
        # Demand-aware reposition rewards
        self.enable_demand_reposition_bonus = getattr(config.reward, 'enable_demand_reposition_bonus', True)
        self.reposition_success_bonus = getattr(config.reward, 'reposition_success_bonus', 0.5)
        self.reposition_nearby_decay = getattr(config.reward, 'reposition_nearby_decay', 0.5)
        self.reposition_action_bonus = getattr(config.reward, 'reposition_action_bonus', 0.0)
        self.reposition_dispatch_bonus = getattr(config.reward, 'reposition_dispatch_bonus', 0.0)
        self.reposition_dispatch_demand_cap = getattr(config.reward, 'reposition_dispatch_demand_cap', 25.0)
        self.reposition_dispatch_per_vehicle_cap = getattr(config.reward, 'reposition_dispatch_per_vehicle_cap', 2.0)
        
        # Adjacency matrix reference for nearby-hex demand lookup (no copy, zero memory)
        self._adjacency_matrix = adjacency_matrix
        
        # SOC thresholds
        self.soc_low_threshold = config.vehicle.soc_low_threshold
    
    def compute_low_soc_penalty(
        self,
        fleet_state: 'TensorFleetState'
    ) -> torch.Tensor:
        """
        Penalty for vehicles with low SOC to encourage proactive charging.
        
        This prevents the fleet from depleting batteries and becoming unable to serve.
        Only penalizes very low SOC to not discourage serving trips.
        """
        socs = fleet_state.socs  # [num_vehicles]
        
        # Only penalize critically low SOC to avoid over-charging behavior
        # Below 15%: penalty to encourage charging
        # Below 10%: critical penalty
        
        critical_mask = socs < 10.0  # Critical: almost dead
        low_mask = (socs >= 10.0) & (socs < 15.0)  # Low: needs charge soon
        
        penalty = torch.zeros(1, device=self.device)
        
        # Penalty rates (per vehicle per step) - reduced to avoid over-charging
        critical_penalty = 3.0  # Penalty for nearly dead batteries
        low_penalty = 0.5       # Small penalty
        
        penalty += critical_mask.sum() * critical_penalty
        penalty += low_mask.sum() * low_penalty
        
        return penalty
    
    def compute_wait_penalty(
        self,
        trip_state: 'TensorTripState'
    ) -> torch.Tensor:
        """
        Compute penalty for waiting trips.
        Paper Eq. 8: λ_wait * sum(wait_o)
        """
        unassigned_mask = trip_state.get_unassigned_mask()
        total_wait_steps = trip_state.wait_steps[unassigned_mask].sum().item()
        penalty = total_wait_steps * self.wait_penalty_per_step
        return torch.tensor(penalty, device=self.device)
    
    def compute_drop_penalty(
        self,
        trip_state: 'TensorTripState'
    ) -> Tuple[torch.Tensor, int]:
        """
        Drop expired trips and compute penalty.
        Paper Eq. 8: λ_drop * |dropped_t|
        
        Returns:
            Tuple of (penalty_tensor, num_trips_dropped)
        """
        trips_dropped = trip_state.drop_expired(self.max_wait_steps)
        penalty = trips_dropped * self.drop_penalty_per_order
        return torch.tensor(penalty, device=self.device), trips_dropped
    
    def compute_action_penalties(
        self,
        action_type: torch.Tensor,
        fleet_state: 'TensorFleetState',
        trip_state: 'TensorTripState',
    ) -> torch.Tensor:
        """
        Compute penalties for suboptimal action choices.
        
        Args:
            action_type: [num_vehicles] action indices (0=SERVE, 1=CHARGE, 2=REPOSITION)
            fleet_state: Current fleet state
            trip_state: Current trip state
            
        Returns:
            Total penalty tensor
        """
        penalty = torch.zeros(1, device=self.device)
        
        # IDLE action removed — no IDLE penalty block needed
        _ = trip_state.get_unassigned_mask().any()  # kept for symmetry with drop penalty
        
        # REPOSITION penalty: small discouragement for unnecessary movement (REPOSITION=2)
        if self.reposition_penalty > 0:
            reposition_mask = action_type == 2
            penalty += reposition_mask.sum() * self.reposition_penalty
        
        # High SOC charge penalty: discourage charging when battery is already sufficient (CHARGE=1)
        if self.high_soc_charge_penalty > 0:
            charge_mask = action_type == 1
            high_soc_charge = charge_mask & (fleet_state.socs > 60.0)
            penalty += high_soc_charge.sum() * self.high_soc_charge_penalty
        
        # Very high SOC charge penalty: stronger penalty for charging above 80% (CHARGE=1)
        if self.very_high_soc_charge_penalty > 0:
            charge_mask = action_type == 1
            very_high_soc_charge = charge_mask & (fleet_state.socs > 80.0)
            penalty += very_high_soc_charge.sum() * self.very_high_soc_charge_penalty
        
        return penalty
    
    def compute_total_reward(
        self,
        trip_revenue: float,
        driving_cost: float,
        charge_cost: float,
        reposition_cost: float,
        wait_penalty: float,
        drop_penalty: float,
        low_soc_penalty: float,
        action_penalty: float,
        trips_served: int,
    ) -> float:
        """
        Compute total step reward.
        
        Paper Eq. 8:
        r_t = trip_revenue - driving_cost - electricity_cost - wait_penalty - drop_penalty
        
        Plus additional bonuses/penalties for learning signal.
        """
        # Base reward from paper
        reward = trip_revenue - driving_cost - charge_cost - reposition_cost
        
        # Penalties from paper
        reward -= wait_penalty
        reward -= drop_penalty
        
        # Additional learning signals
        reward -= low_soc_penalty
        reward -= action_penalty
        
        # Serve bonus to encourage taking trips
        if self.serve_bonus > 0:
            reward += trips_served * self.serve_bonus
        
        # Scale reward
        if self.scale_factor > 0:
            reward = reward / self.scale_factor
        
        return reward
    
    def compute_reposition_dispatch_bonus(
        self,
        reposition_mask: torch.Tensor,
        reposition_target: torch.Tensor,
        trip_state: 'TensorTripState',
    ) -> torch.Tensor:
        """
        Immediate demand-proportional bonus at dispatch time.

        Fires when a vehicle CHOOSES reposition, not when it arrives.
        Bonus per vehicle = dispatch_bonus * min(demand_at_target, demand_cap) / demand_cap

        - High demand at target → high bonus (we need vehicles there)
        - Zero demand at target → zero bonus (pointless reposition)
        - Saturates at demand_cap to prevent extreme values

        This captures opportunity cost: the agent is rewarded for recognising
        an undersupplied hex and dispatching toward it immediately, solving
        the multi-step credit-assignment delay of the arrival bonus.
        """
        if not self.enable_demand_reposition_bonus or not reposition_mask.any():
            return torch.zeros(1, device=self.device)
        if self.reposition_dispatch_bonus <= 0.0:
            return torch.zeros(1, device=self.device)

        repos_indices = reposition_mask.nonzero(as_tuple=True)[0]
        if repos_indices.numel() == 0:
            return torch.zeros(1, device=self.device)

        targets = reposition_target[repos_indices].long()

        unassigned_mask = trip_state.get_unassigned_mask()
        if not unassigned_mask.any():
            return torch.zeros(1, device=self.device)

        pickup_hexes = trip_state.pickup_hex[unassigned_mask].long()

        if self._adjacency_matrix is not None:
            num_hexes = self._adjacency_matrix.shape[0]
        else:
            num_hexes = int(max(pickup_hexes.max().item(), targets.max().item())) + 1

        pickup_hexes = pickup_hexes.clamp(0, num_hexes - 1)
        targets = targets.clamp(0, num_hexes - 1)

        hex_demand = torch.zeros(num_hexes, device=self.device)
        hex_demand.scatter_add_(0, pickup_hexes, torch.ones_like(pickup_hexes, dtype=torch.float32))

        demand_at_target = hex_demand[targets]  # [N_repos]
        # Scale proportionally to demand (no demand ceiling), cap the per-vehicle contribution
        bonus_raw = self.reposition_dispatch_bonus * demand_at_target / self.reposition_dispatch_demand_cap
        bonus_per_vehicle = bonus_raw.clamp(max=self.reposition_dispatch_per_vehicle_cap)
        total_bonus = bonus_per_vehicle.sum()

        return total_bonus

    def compute_reposition_bonus(
        self,
        completed_reposition_mask: torch.Tensor,
        fleet_state: 'TensorFleetState',
        trip_state: 'TensorTripState'
    ) -> torch.Tensor:
        """
        Bonus for repositioning to high-demand hexes (exact + nearby).
        
        Uses adjacency matrix to count demand from nearby hexes (~3km),
        making repositioning near trip clusters attractive even if the
        exact target hex has no trips.
        
        bonus = success_bonus * (demand_exact + nearby_decay * demand_neighbors)
        
        Memory: Zero additional allocation — reuses adjacency matrix from GCN.
        
        Args:
            completed_reposition_mask: [num_vehicles] mask of vehicles that just completed reposition
            fleet_state: Current fleet state (to get positions)
            trip_state: Current trip state (to compute demand)
            
        Returns:
            Bonus tensor (positive reward)
        """
        if not self.enable_demand_reposition_bonus or not completed_reposition_mask.any():
            return torch.zeros(1, device=self.device)
        
        # Get positions of vehicles that completed reposition
        completed_indices = torch.nonzero(completed_reposition_mask, as_tuple=True)[0]
        target_hexes = fleet_state.positions[completed_indices].long()
        
        # Build hex demand vector: count unassigned trips per hex
        unassigned_mask = trip_state.get_unassigned_mask()
        if not unassigned_mask.any():
            return torch.zeros(1, device=self.device)
        
        pickup_hexes = trip_state.pickup_hex[unassigned_mask].long()
        
        # Determine num_hexes from adjacency matrix if available, else infer
        if self._adjacency_matrix is not None:
            num_hexes = self._adjacency_matrix.shape[0]
        else:
            all_max = max(fleet_state.positions.max().item(), 0)
            if pickup_hexes.numel() > 0:
                all_max = max(all_max, pickup_hexes.max().item())
            num_hexes = int(all_max) + 1
        
        # Safety clamp indices to bounds
        pickup_hexes = pickup_hexes.clamp(0, num_hexes - 1)
        target_hexes = target_hexes.clamp(0, num_hexes - 1)
        
        if target_hexes.numel() == 0:
            return torch.zeros(1, device=self.device)
        
        # Scatter-add trips into hex_demand vector [num_hexes]
        hex_demand = torch.zeros(num_hexes, device=self.device)
        if pickup_hexes.numel() > 0:
            hex_demand.scatter_add_(0, pickup_hexes, torch.ones_like(pickup_hexes, dtype=torch.float32))
        
        # Exact-hex demand for each repositioned vehicle
        demand_exact = hex_demand[target_hexes]  # [N_repos]
        
        # Nearby-hex demand via adjacency matrix (memory-free: just matmul on existing tensors)
        if self._adjacency_matrix is not None and self.reposition_nearby_decay > 0:
            # adj_matrix is [num_hexes, num_hexes], already normalized for GCN
            # adj[target_hexes] gives [N_repos, num_hexes] neighbor weights
            # Multiply by hex_demand to get weighted nearby demand
            nearby_demand = self._adjacency_matrix[target_hexes] @ hex_demand  # [N_repos]
            # Subtract exact-hex contribution to avoid double-counting
            nearby_demand = nearby_demand - demand_exact
            nearby_demand = nearby_demand.clamp(min=0)
            effective_demand = demand_exact + self.reposition_nearby_decay * nearby_demand
        else:
            effective_demand = demand_exact
        
        # Scale bonus with number of repositioning vehicles to prevent explosive rewards
        num_repositioning = completed_indices.numel()
        # Apply diminishing returns: total reward saturates instead of growing superlinearly.
        # This formula ensures that as N increases, the per-vehicle reward drops.
        # scaling_cap = (S * N) / (1.0 + 0.002 * N)
        scaling_cap = (self.reposition_success_bonus * num_repositioning) / (1.0 + 0.002 * num_repositioning)
        
        total_bonus = (effective_demand * self.reposition_success_bonus).sum()
        total_bonus = torch.clamp(total_bonus, max=scaling_cap)
        
        # Debug logging (5% of the time)
        if completed_indices.numel() > 0 and torch.rand(1).item() < 0.05 and not getattr(self, 'suppress_debug_logs', False):
            nearby_str = ""
            if self._adjacency_matrix is not None and self.reposition_nearby_decay > 0:
                nearby_str = f", nearby: {nearby_demand.sum().item():.0f}"
            uncapped_bonus = (effective_demand * self.reposition_success_bonus).sum().item()
            capped_bonus = total_bonus.item()
            cap_hit = "(CAPPED)" if uncapped_bonus > capped_bonus else ""
            print(f"    [ReposBonus] {completed_indices.numel()} vehicles, "
                  f"exact demand: {demand_exact.sum().item():.0f}{nearby_str}, "
                  f"bonus: {capped_bonus:.2f} {cap_hit}")
        
        return total_bonus
    
    def compute_opportunity_cost(
        self,
        charge_mask: torch.Tensor,
        idle_mask: torch.Tensor = None  # kept for API compat; IDLE action removed
    ) -> torch.Tensor:
        """
        Penalty for wasting serving opportunities.
        
        DISABLED: This penalty was causing issues with reward scaling.
        The agent already gets penalized via wait_penalty and drop_penalty
        when it fails to serve trips.
        """
        return torch.zeros(1, device=self.device)
