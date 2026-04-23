import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        num_vehicles: int = 500,
        max_trips: int = 1000,
        num_stations: int = 150,
        max_serve_per_step: int = 200,
        max_charge_per_step: int = 100,
        use_assignment_encoding: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_vehicles = num_vehicles
        self.max_trips = max_trips
        self.num_stations = num_stations
        self.max_serve = max_serve_per_step
        self.max_charge = max_charge_per_step
        self.use_assignment_encoding = use_assignment_encoding
        
        # Compute action encoding dimension
        base_action_dim = action_dim
        if use_assignment_encoding:
            # Add assignment encoding: embedding dim for trips and stations
            self.assignment_embed_dim = 64
            # Embed trips (not vehicle-trip pairs) to preserve per-vehicle info
            self.serve_embed = nn.Embedding(max_trips, self.assignment_embed_dim)
            self.charge_embed = nn.Embedding(num_stations, self.assignment_embed_dim)
            action_encoding_dim = base_action_dim + self.assignment_embed_dim
        else:
            action_encoding_dim = base_action_dim
        
        input_dim = state_dim + action_encoding_dim
        dims = [input_dim] + hidden_dims + [1]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        serve_vehicle_idx: Optional[torch.Tensor] = None,
        serve_trip_idx: Optional[torch.Tensor] = None,
        num_served: Optional[torch.Tensor] = None,
        charge_vehicle_idx: Optional[torch.Tensor] = None,
        charge_station_idx: Optional[torch.Tensor] = None,
        num_charged: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional assignment encoding.
        
        Args:
            state: [batch, state_dim]
            action: [batch, num_vehicles] or [batch, num_vehicles, action_dim]
            serve_vehicle_idx: [batch, max_serve] vehicle indices for serve assignments
            serve_trip_idx: [batch, max_serve] trip indices for serve assignments
            num_served: [batch] number of actual serves per batch
            charge_vehicle_idx: [batch, max_charge] vehicle indices for charge assignments
            charge_station_idx: [batch, max_charge] station indices for charge assignments
            num_charged: [batch] number of actual charges per batch
        """
        # Encode action type
        if action.dim() == 1:
            action_one_hot = F.one_hot(action.long(), self.action_dim).float()
        else:
            # action is [batch, num_vehicles] - take mode of actions as aggregate
            batch_size = action.size(0)
            # Simple aggregation: count of each action type (normalized)
            action_counts = torch.zeros(batch_size, self.action_dim, device=action.device)
            for i in range(self.action_dim):
                action_counts[:, i] = (action[:, :, 0] == i).sum(dim=1).float()
            action_one_hot = action_counts / self.num_vehicles
        
        # Encode assignments if enabled and provided
        if self.use_assignment_encoding:
            assignment_encoding = self._encode_assignments(
                action_one_hot.size(0),
                serve_vehicle_idx,
                serve_trip_idx,
                num_served,
                charge_vehicle_idx,
                charge_station_idx,
                num_charged,
            )
            action_encoding = torch.cat([action_one_hot, assignment_encoding], dim=-1)
        else:
            action_encoding = action_one_hot
        
        x = torch.cat([state, action_encoding], dim=-1)
        return self.network(x).squeeze(-1)
    
    def _encode_assignments(
        self,
        batch_size: int,
        serve_vehicle_idx: Optional[torch.Tensor],
        serve_trip_idx: Optional[torch.Tensor],
        num_served: Optional[torch.Tensor],
        charge_vehicle_idx: Optional[torch.Tensor],
        charge_station_idx: Optional[torch.Tensor],
        num_charged: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode assignments into per-vehicle vector.
        
        
        Returns:
            [batch, num_vehicles, assignment_embed_dim] encoding of assignments
        """
        device = serve_vehicle_idx.device if serve_vehicle_idx is not None else torch.device('cuda')
        # Per-vehicle encoding instead of batch-level average
        encoding = torch.zeros(batch_size, self.num_vehicles, self.assignment_embed_dim, device=device)
        
        # Encode serve assignments with per-vehicle positioning (VECTORIZED)
        if serve_vehicle_idx is not None and serve_trip_idx is not None:
            # Create valid mask for all assignments at once
            valid_mask = (serve_vehicle_idx >= 0) & (serve_trip_idx >= 0)
            
            if valid_mask.any():
                # Get batch indices for scatter
                batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(serve_vehicle_idx)
                
                # Filter valid entries
                valid_batch = batch_idx[valid_mask]
                valid_veh = serve_vehicle_idx[valid_mask]
                valid_trip = serve_trip_idx[valid_mask]
                
                # Clamp vehicle and trip IDs to valid range
                valid_veh = valid_veh.clamp(0, self.num_vehicles - 1)
                valid_trip = valid_trip.clamp(0, self.max_trips - 1)
                
                # Embed all trips at once
                trip_embeds = self.serve_embed(valid_trip)  # [num_valid, embed_dim]
                
                # Scatter to per-vehicle positions using advanced indexing
                # encoding[batch_idx, veh_idx] = trip_embed
                encoding[valid_batch, valid_veh] = trip_embeds
        
        # Encode charge assignments with per-vehicle positioning 
        if charge_vehicle_idx is not None and charge_station_idx is not None:
            valid_mask = (charge_vehicle_idx >= 0) & (charge_station_idx >= 0)
            
            if valid_mask.any():
                batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(charge_vehicle_idx)
                
                valid_batch = batch_idx[valid_mask]
                valid_veh = charge_vehicle_idx[valid_mask]
                valid_station = charge_station_idx[valid_mask]
                
                # Clamp to valid range
                valid_veh = valid_veh.clamp(0, self.num_vehicles - 1)
                valid_station = valid_station.clamp(0, self.num_stations - 1)
                
                # Embed all stations at once
                station_embeds = self.charge_embed(valid_station)
                
                # Add to encoding (in case vehicle both serves and charges)
                encoding[valid_batch, valid_veh] += station_embeds
        
        # Pool to fixed size for critic input (sum across vehicles, normalized)
        # This preserves information about WHO got WHAT, unlike naive averaging
        pooled_encoding = encoding.sum(dim=1) / (self.num_vehicles + 1e-6)  # [batch, embed_dim]
        
        return pooled_encoding


class TwinCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        **critic_kwargs
    ):
        super().__init__()
        self.critic1 = Critic(state_dim, action_dim, hidden_dims, dropout, **critic_kwargs)
        self.critic2 = Critic(state_dim, action_dim, hidden_dims, dropout, **critic_kwargs)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        **assignment_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.critic1(state, action, **assignment_kwargs), self.critic2(state, action, **assignment_kwargs)
    
    def q1(self, state: torch.Tensor, action: torch.Tensor, **assignment_kwargs) -> torch.Tensor:
        return self.critic1(state, action, **assignment_kwargs)
    
    def min_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        **assignment_kwargs
    ) -> torch.Tensor:
        q1, q2 = self.forward(state, action, **assignment_kwargs)
        return torch.min(q1, q2)


class ValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1
    ):
        super().__init__()
        
        dims = [state_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class DistributionalCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        num_atoms: int = 51,
        v_min: float = -100.0,
        v_max: float = 100.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, num_atoms)
        )
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        input_dim = state_dim + action_dim
        dims = [input_dim] + hidden_dims + [num_atoms]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        if action.dim() == 1:
            action_one_hot = F.one_hot(action.long(), self.action_dim).float()
        else:
            action_one_hot = action
        
        x = torch.cat([state, action_one_hot], dim=-1)
        logits = self.network(x)
        return F.softmax(logits, dim=-1)
    
    def get_q_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        probs = self.forward(state, action)
        return (probs * self.support).sum(dim=-1)


class QRCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        num_quantiles: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        
        self.register_buffer(
            'tau',
            (torch.arange(num_quantiles) + 0.5) / num_quantiles
        )
        
        input_dim = state_dim + action_dim
        dims = [input_dim] + hidden_dims + [num_quantiles]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        if action.dim() == 1:
            action_one_hot = F.one_hot(action.long(), self.action_dim).float()
        else:
            action_one_hot = action
        
        x = torch.cat([state, action_one_hot], dim=-1)
        return self.network(x)
    
    def get_q_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        quantiles = self.forward(state, action)
        return quantiles.mean(dim=-1)


class DuelingCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        dims = [state_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.shared = nn.Sequential(*layers)
        
        self.value_head = nn.Linear(hidden_dims[-1], 1)
        self.advantage_head = nn.Linear(hidden_dims[-1] + action_dim, 1)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        features = self.shared(state)
        
        value = self.value_head(features)
        
        if action.dim() == 1:
            action_one_hot = F.one_hot(action.long(), self.action_dim).float()
        else:
            action_one_hot = action
        
        advantage_input = torch.cat([features, action_one_hot], dim=-1)
        advantage = self.advantage_head(advantage_input)
        
        return (value + advantage).squeeze(-1)


class MultiHeadCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.critics = nn.ModuleList([
            Critic(state_dim, action_dim, hidden_dims, dropout)
            for _ in range(num_heads)
        ])
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        q_values = torch.stack([
            critic(state, action) for critic in self.critics
        ], dim=-1)
        return q_values
    
    def mean_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(state, action).mean(dim=-1)
    
    def min_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(state, action).min(dim=-1)[0]
    
    def std_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(state, action).std(dim=-1)
