"""
Assignment Loss Module for Training serve_scores and charge_scores.

This module provides auxiliary losses that enable the actor to learn
optimal trip and station assignments through gradient-based optimization.

Two main approaches:
1. Supervised: Use environment's actual successful assignments as targets
2. Contrastive: Push scores higher for successful assignments, lower for failed

Author: Claude
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, NamedTuple
from dataclasses import dataclass


@dataclass
class AssignmentInfo:
    """Information about assignments made during a step.
    
    Stored in replay buffer to provide supervision signal.
    """
    # SERVE assignments
    serve_vehicle_indices: torch.Tensor  # [num_served] - which vehicles served
    serve_trip_indices: torch.Tensor     # [num_served] - which trips they got
    num_serve_attempted: int
    num_serve_success: int
    
    # CHARGE assignments  
    charge_vehicle_indices: torch.Tensor  # [num_charged] - which vehicles charged
    charge_station_indices: torch.Tensor  # [num_charged] - which stations they got
    num_charge_attempted: int
    num_charge_success: int


class AssignmentLoss(nn.Module):
    """
    Compute auxiliary losses for serve_scores and charge_scores.
    
    These losses provide direct gradient signal to the assignment heads,
    enabling the actor to learn optimal target selection.
    
    Loss types:
    1. Cross-entropy loss: Treat actual assignment as target
    2. Margin loss: Push assigned targets higher than non-assigned
    3. Distance-weighted loss: Higher reward for closer assignments
    """
    
    def __init__(
        self,
        serve_loss_weight: float = 0.1,
        charge_loss_weight: float = 0.1,
        margin: float = 1.0,
        use_margin_loss: bool = True,
        use_distance_reward: bool = True,
    ):
        super().__init__()
        self.serve_loss_weight = serve_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.margin = margin
        self.use_margin_loss = use_margin_loss
        self.use_distance_reward = use_distance_reward
    
    def compute_serve_loss(
        self,
        serve_scores: torch.Tensor,        # [batch, num_vehicles, max_trips]
        serve_vehicle_indices: torch.Tensor,  # [batch, max_served]
        serve_trip_indices: torch.Tensor,     # [batch, max_served]
        serve_mask: Optional[torch.Tensor] = None,  # [batch, max_served] - valid assignments
        trip_mask: Optional[torch.Tensor] = None,   # [batch, max_trips] - valid trips
    ) -> torch.Tensor:
        """
        Compute loss for serve_scores to encourage selecting assigned trips.
        
        For each vehicle that successfully served, we want its score for
        the assigned trip to be high (and higher than other trips).
        
        Args:
            serve_scores: Actor's trip preference scores [B, V, T]
            serve_vehicle_indices: Vehicle indices that served [B, M]
            serve_trip_indices: Trip indices they were assigned [B, M]
            serve_mask: Which assignments are valid (padding mask) [B, M]
            trip_mask: Which trips are valid [B, T]
        
        Returns:
            Scalar loss value
        """
        if serve_scores is None or serve_vehicle_indices.numel() == 0:
            return torch.tensor(0.0, device=serve_scores.device if serve_scores is not None else 'cuda')
        
        batch_size = serve_scores.shape[0]
        device = serve_scores.device
        
        # Handle single sample case
        if serve_vehicle_indices.dim() == 1:
            serve_vehicle_indices = serve_vehicle_indices.unsqueeze(0)
            serve_trip_indices = serve_trip_indices.unsqueeze(0)
            if serve_mask is not None:
                serve_mask = serve_mask.unsqueeze(0)
        
        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0
        
        for b in range(batch_size):
            v_indices = serve_vehicle_indices[b]
            t_indices = serve_trip_indices[b]
            
            if serve_mask is not None:
                valid = serve_mask[b]
                v_indices = v_indices[valid]
                t_indices = t_indices[valid]
            
            if len(v_indices) == 0:
                continue
            
            # Get scores for serving vehicles
            # serve_scores[b]: [num_vehicles, max_trips]
            vehicle_scores = serve_scores[b, v_indices]  # [num_served, max_trips]
            
            # Target: the trip that was actually assigned
            # Cross-entropy: treat as classification problem
            if self.use_margin_loss:
                # Margin loss: score[assigned] > score[others] + margin
                assigned_scores = vehicle_scores[torch.arange(len(v_indices)), t_indices]  # [num_served]
                
                # Mask out the assigned trip for comparison
                mask = torch.ones_like(vehicle_scores, dtype=torch.bool)
                mask[torch.arange(len(v_indices)), t_indices] = False
                
                # Max of non-assigned trips
                other_scores = vehicle_scores.clone()
                other_scores[~mask] = float('-inf')
                max_other_scores = other_scores.max(dim=1)[0]  # [num_served]
                
                # Hinge loss: max(0, margin + max_other - assigned)
                margin_loss = F.relu(self.margin + max_other_scores - assigned_scores)
                total_loss = total_loss + margin_loss.mean()
            else:
                # Cross-entropy loss
                ce_loss = F.cross_entropy(vehicle_scores, t_indices, reduction='mean')
                total_loss = total_loss + ce_loss
            
            num_valid += 1
        
        if num_valid > 0:
            total_loss = total_loss / num_valid
        
        return total_loss * self.serve_loss_weight
    
    def compute_charge_loss(
        self,
        charge_scores: torch.Tensor,          # [batch, num_vehicles, num_stations]
        charge_vehicle_indices: torch.Tensor,  # [batch, max_charged]
        charge_station_indices: torch.Tensor,  # [batch, max_charged]
        charge_mask: Optional[torch.Tensor] = None,  # [batch, max_charged]
        station_mask: Optional[torch.Tensor] = None,  # [batch, num_stations]
    ) -> torch.Tensor:
        """
        Compute loss for charge_scores to encourage selecting assigned stations.
        
        Similar to serve_loss but for charging assignments.
        """
        if charge_scores is None or charge_vehicle_indices.numel() == 0:
            return torch.tensor(0.0, device=charge_scores.device if charge_scores is not None else 'cuda')
        
        batch_size = charge_scores.shape[0]
        device = charge_scores.device
        
        # Handle single sample case
        if charge_vehicle_indices.dim() == 1:
            charge_vehicle_indices = charge_vehicle_indices.unsqueeze(0)
            charge_station_indices = charge_station_indices.unsqueeze(0)
            if charge_mask is not None:
                charge_mask = charge_mask.unsqueeze(0)
        
        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0
        
        for b in range(batch_size):
            v_indices = charge_vehicle_indices[b]
            s_indices = charge_station_indices[b]
            
            if charge_mask is not None:
                valid = charge_mask[b]
                v_indices = v_indices[valid]
                s_indices = s_indices[valid]
            
            if len(v_indices) == 0:
                continue
            
            # Get scores for charging vehicles
            vehicle_scores = charge_scores[b, v_indices]  # [num_charged, num_stations]
            
            if self.use_margin_loss:
                # Margin loss
                assigned_scores = vehicle_scores[torch.arange(len(v_indices)), s_indices]
                
                mask = torch.ones_like(vehicle_scores, dtype=torch.bool)
                mask[torch.arange(len(v_indices)), s_indices] = False
                
                other_scores = vehicle_scores.clone()
                other_scores[~mask] = float('-inf')
                max_other_scores = other_scores.max(dim=1)[0]
                
                margin_loss = F.relu(self.margin + max_other_scores - assigned_scores)
                total_loss = total_loss + margin_loss.mean()
            else:
                ce_loss = F.cross_entropy(vehicle_scores, s_indices, reduction='mean')
                total_loss = total_loss + ce_loss
            
            num_valid += 1
        
        if num_valid > 0:
            total_loss = total_loss / num_valid
        
        return total_loss * self.charge_loss_weight
    
    def forward(
        self,
        serve_scores: Optional[torch.Tensor],
        charge_scores: Optional[torch.Tensor],
        assignment_info: AssignmentInfo,
        serve_mask: Optional[torch.Tensor] = None,
        charge_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute both serve and charge assignment losses.
        
        Returns:
            Dict with 'serve_loss', 'charge_loss', 'total_loss'
        """
        serve_loss = self.compute_serve_loss(
            serve_scores,
            assignment_info.serve_vehicle_indices,
            assignment_info.serve_trip_indices,
            serve_mask
        )
        
        charge_loss = self.compute_charge_loss(
            charge_scores,
            assignment_info.charge_vehicle_indices,
            assignment_info.charge_station_indices,
            charge_mask
        )
        
        total_loss = serve_loss + charge_loss
        
        return {
            'serve_loss': serve_loss,
            'charge_loss': charge_loss,
            'assignment_loss': total_loss
        }


def create_assignment_info(
    serve_vehicle_indices: torch.Tensor,
    serve_trip_indices: torch.Tensor,
    num_serve_attempted: int,
    num_serve_success: int,
    charge_vehicle_indices: torch.Tensor,
    charge_station_indices: torch.Tensor,
    num_charge_attempted: int,
    num_charge_success: int,
) -> AssignmentInfo:
    """Factory function to create AssignmentInfo."""
    return AssignmentInfo(
        serve_vehicle_indices=serve_vehicle_indices,
        serve_trip_indices=serve_trip_indices,
        num_serve_attempted=num_serve_attempted,
        num_serve_success=num_serve_success,
        charge_vehicle_indices=charge_vehicle_indices,
        charge_station_indices=charge_station_indices,
        num_charge_attempted=num_charge_attempted,
        num_charge_success=num_charge_success,
    )
