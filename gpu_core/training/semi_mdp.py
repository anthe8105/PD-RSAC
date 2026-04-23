"""
Semi-MDP Duration Handling per paper Section 2.

Implements:
- Variable action durations (paper Eq. 8)
- Duration-aware discounting γ^Δ (paper Eq. 12)
- Semi-MDP backup (paper Eq. 13)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ActionDuration:
    """Duration information for semi-MDP actions."""
    serve_duration: torch.Tensor      # Steps to complete serve (pickup + trip)
    reposition_duration: torch.Tensor  # Steps to reposition to target hex
    charge_duration: torch.Tensor      # Always 1 step for charging
    
    @property
    def duration(self) -> torch.Tensor:
        """Get the duration tensor for the selected action."""
        return self.serve_duration  # Default, should be overridden based on action type


class SemiMDPHandler:
    """
    Handles Semi-MDP duration calculations per paper Section 2.3.
    
    Action durations (in steps) per paper Eq. 8:
    - Serve: Δ_serve = T(h_i, h_pu) + T(h_pu, h_do)
    - Reposition: Δ_reb = T(h_i, g)
    - Charge: Δ_chg = 1 (fixed single step)
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        step_duration_minutes: float = 5.0,
        avg_speed_kmh: float = 30.0,
        device: str = "cuda"
    ):
        self.gamma = gamma
        self.step_duration_minutes = step_duration_minutes
        self.avg_speed_kmh = avg_speed_kmh
        self.device = torch.device(device)
        
        # Precompute distance per step
        self.km_per_step = (avg_speed_kmh * step_duration_minutes) / 60.0
    
    def compute_action_duration(
        self,
        action_type: torch.Tensor,      # [num_vehicles] - 0=SERVE, 1=CHARGE, 2=REPOSITION (IDLE removed)
        vehicle_hexes: torch.Tensor,    # [num_vehicles] - current hex for each vehicle
        target_hexes: torch.Tensor,     # [num_vehicles] - target hex (for reposition)
        trip_pickup_hexes: Optional[torch.Tensor] = None,   # [num_vehicles] - pickup hex for serve
        trip_dropoff_hexes: Optional[torch.Tensor] = None,  # [num_vehicles] - dropoff hex for serve
        travel_time_matrix: Optional[torch.Tensor] = None   # [num_hexes, num_hexes] - travel times in steps
    ) -> torch.Tensor:
        """
        Compute action duration for each vehicle.
        
        Returns:
            duration: [num_vehicles] - number of steps for each vehicle's action
        """
        num_vehicles = action_type.size(0)
        duration = torch.ones(num_vehicles, device=self.device, dtype=torch.float32)
        
        # CHARGE has duration 1 (fixed single step)
        # SERVE=0, CHARGE=1, REPOSITION=2
        charge_mask = (action_type == 1)
        # duration already 1 for charge
        
        # SERVE duration
        serve_mask = (action_type == 0)
        if serve_mask.any() and travel_time_matrix is not None:
            if trip_pickup_hexes is not None and trip_dropoff_hexes is not None:
                # Duration = pickup_time + trip_time
                pickup_time = travel_time_matrix[vehicle_hexes[serve_mask], trip_pickup_hexes[serve_mask]]
                trip_time = travel_time_matrix[trip_pickup_hexes[serve_mask], trip_dropoff_hexes[serve_mask]]
                duration[serve_mask] = (pickup_time + trip_time).float().clamp(min=1.0)
        
        # REPOSITION duration
        reposition_mask = (action_type == 2)  # REPOSITION=2 in new scheme
        if reposition_mask.any() and travel_time_matrix is not None:
            reposition_time = travel_time_matrix[vehicle_hexes[reposition_mask], target_hexes[reposition_mask]]
            duration[reposition_mask] = reposition_time.float().clamp(min=1.0)
        
        return duration
    
    def compute_discounted_factor(self, duration: torch.Tensor) -> torch.Tensor:
        """
        Compute semi-MDP discount factor γ^Δ per paper Eq. 12.
        
        Args:
            duration: [batch] or [num_vehicles] - action durations
            
        Returns:
            discount: [batch] or [num_vehicles] - γ^duration
        """
        return torch.pow(self.gamma, duration)
    
    def compute_semi_mdp_target(
        self,
        rewards: torch.Tensor,           # [batch]
        next_values: torch.Tensor,       # [batch]
        durations: torch.Tensor,         # [batch]
        dones: torch.Tensor,             # [batch]
        entropy_bonus: Optional[torch.Tensor] = None  # [batch]
    ) -> torch.Tensor:
        """
        Compute Semi-MDP soft backup target per paper Eq. 13.
        
        y_t = r_t + γ^Δ * (V(s_{t+Δ}) - α * log π)
        
        Args:
            rewards: Immediate rewards
            next_values: V(s_{t+Δ}) value estimates
            durations: Action durations Δ
            dones: Episode termination flags
            entropy_bonus: Optional entropy bonus (α * log π term)
            
        Returns:
            target: Semi-MDP backup target
        """
        discount = self.compute_discounted_factor(durations)
        
        target = rewards + (1 - dones.float()) * discount * next_values
        
        if entropy_bonus is not None:
            target = target - entropy_bonus
        
        return target


class DurationPredictor(nn.Module):
    """
    Neural network to predict action durations.
    
    Used when exact travel time matrix is not available.
    """
    
    def __init__(
        self,
        hex_embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_hexes: int = 1300
    ):
        super().__init__()
        self.num_hexes = num_hexes
        
        # Hex embeddings for source and target
        self.hex_embedding = nn.Embedding(num_hexes, hex_embedding_dim)
        
        # Duration predictor MLP
        self.predictor = nn.Sequential(
            nn.Linear(hex_embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive duration
        )
    
    def forward(
        self,
        source_hex: torch.Tensor,  # [batch]
        target_hex: torch.Tensor   # [batch]
    ) -> torch.Tensor:
        """Predict travel duration in steps."""
        source_emb = self.hex_embedding(source_hex)
        target_emb = self.hex_embedding(target_hex)
        
        combined = torch.cat([source_emb, target_emb], dim=-1)
        duration = self.predictor(combined).squeeze(-1)
        
        # Minimum duration of 1 step
        return duration.clamp(min=1.0)


def integrate_semi_mdp_in_sac_update(
    critic_loss_fn,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    durations: torch.Tensor,
    gamma: float,
    value_network,
    alpha: float
):
    """
    Helper function to integrate semi-MDP discounting into SAC critic update.
    
    Instead of using fixed γ, use γ^Δ where Δ is the action duration.
    """
    # Compute γ^Δ for each transition
    variable_discount = torch.pow(gamma, durations)
    
    # Get next state values
    with torch.no_grad():
        next_values = value_network(next_states)
    
    # Semi-MDP target: r + γ^Δ * V(s')
    target = rewards + (1 - dones.float()) * variable_discount * next_values
    
    return target
