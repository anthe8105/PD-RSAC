import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, List


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with batch support. Supports both sparse and dense adjacency.

        Args:
            x: Node features [batch, num_nodes, in_features] or [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes] (sparse or dense)

        Returns:
            output: [batch, num_nodes, out_features] or [num_nodes, out_features]
        """
        support = self.linear(x)  # [batch, num_nodes, out_features] or [num_nodes, out_features]

        if adj.dim() != 2:
            raise ValueError(f"GCN adjacency must be 2D [N, N], got shape {tuple(adj.shape)}")
        if adj.shape[0] != adj.shape[1]:
            raise ValueError(f"GCN adjacency must be square, got shape {tuple(adj.shape)}")

        if support.dim() == 3:
            num_nodes = support.shape[1]
            if adj.shape[0] != num_nodes:
                raise ValueError(
                    f"GCN shape mismatch: adjacency {tuple(adj.shape)} vs node features {tuple(support.shape)}"
                )
            # Batched: support is [B, N, F], adj is [N, N]
            output = torch.einsum('ij,bjf->bif', adj, support)
        else:
            num_nodes = support.shape[0]
            if adj.shape[0] != num_nodes:
                raise ValueError(
                    f"GCN shape mismatch: adjacency {tuple(adj.shape)} vs node features {tuple(support.shape)}"
                )
            # Non-batched: support is [N, F]
            output = torch.matmul(adj, support)

        return output


class GCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        self.activation = self._get_activation(activation)
        
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(len(dims) - 1):
            self.layers.append(GCNLayer(dims[i], dims[i + 1]))
            if use_batch_norm and i < len(dims) - 2:
                self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))
    
    def _get_activation(self, name: str):
        activations = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu,
            'tanh': torch.tanh,
            'silu': F.silu
        }
        return activations.get(name, F.relu)
    
    def _forward_layer(self, layer_idx: int, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward single GCN layer (checkpointable)."""
        x = self.layers[layer_idx](x, adj)
        if self.use_batch_norm and layer_idx < len(self.layers) - 1:
            if x.dim() == 3:
                batch_size, num_nodes, features = x.shape
                x = x.reshape(-1, features)
                x = self.batch_norms[layer_idx](x)
                x = x.reshape(batch_size, num_nodes, features)
            else:
                x = self.batch_norms[layer_idx](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        layer_outputs = []

        for i in range(len(self.layers) - 1):
            if self.training and x.requires_grad:
                # Gradient checkpointing: trade compute for memory
                x = checkpoint(self._forward_layer, i, x, adj, use_reentrant=False)
            else:
                x = self._forward_layer(i, x, adj)
            layer_outputs.append(x)

        # Last layer (no activation/batchnorm)
        x = self.layers[-1](x, adj)
        layer_outputs.append(x)

        if return_all_layers:
            return x, layer_outputs
        return x


class HexEncoder(nn.Module):
    def __init__(
        self,
        num_hexes: int,
        embedding_dim: int,
        gcn_hidden_dims: List[int],
        gcn_output_dim: int,
        feature_dim: int = 0,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.num_hexes = num_hexes
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        
        self.hex_embedding = nn.Embedding(num_hexes, embedding_dim)
        
        gcn_input_dim = embedding_dim + feature_dim
        self.gcn = GCNEncoder(
            input_dim=gcn_input_dim,
            hidden_dims=gcn_hidden_dims,
            output_dim=gcn_output_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
    
    def forward(
        self,
        adj: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        node_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if node_ids is None:
            node_ids = torch.arange(self.num_hexes, device=adj.device)
        
        embeddings = self.hex_embedding(node_ids)
        
        if embeddings.dim() == 2 and adj.dim() == 3:
            batch_size = adj.shape[0]
            embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        if node_features is not None:
            x = torch.cat([embeddings, node_features], dim=-1)
        else:
            x = embeddings
        
        return self.gcn(x, adj)


class VehicleEncoder(nn.Module):
    def __init__(
        self,
        vehicle_feature_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        dims = [vehicle_feature_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, vehicle_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(vehicle_features)


class CombinedEncoder(nn.Module):
    def __init__(
        self,
        num_hexes: int,
        hex_embedding_dim: int,
        hex_feature_dim: int,
        gcn_hidden_dims: List[int],
        gcn_output_dim: int,
        vehicle_feature_dim: int,
        vehicle_hidden_dims: List[int],
        vehicle_output_dim: int,
        global_feature_dim: int,
        combined_output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hex_encoder = HexEncoder(
            num_hexes=num_hexes,
            embedding_dim=hex_embedding_dim,
            gcn_hidden_dims=gcn_hidden_dims,
            gcn_output_dim=gcn_output_dim,
            feature_dim=hex_feature_dim,
            dropout=dropout
        )
        
        self.vehicle_encoder = VehicleEncoder(
            vehicle_feature_dim=vehicle_feature_dim,
            hidden_dims=vehicle_hidden_dims,
            output_dim=vehicle_output_dim,
            dropout=dropout
        )
        
        combined_input_dim = gcn_output_dim + vehicle_output_dim + global_feature_dim
        self.output_projection = nn.Sequential(
            nn.Linear(combined_input_dim, combined_output_dim),
            nn.LayerNorm(combined_output_dim),
            nn.ReLU()
        )
        
        self.gcn_output_dim = gcn_output_dim
        self.vehicle_output_dim = vehicle_output_dim
    
    def forward(
        self,
        adj: torch.Tensor,
        hex_features: torch.Tensor,  # φ_h: hex features for spatial reasoning
        vehicle_features: torch.Tensor,
        vehicle_hex_ids: torch.Tensor,
        global_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with hex features φ_h (phi_h) from paper."""
        hex_encodings = self.hex_encoder(adj, hex_features)
        
        if hex_encodings.dim() == 3:
            batch_size, num_hexes, _ = hex_encodings.shape
            batch_indices = torch.arange(batch_size, device=vehicle_hex_ids.device)
            batch_indices = batch_indices.unsqueeze(1).expand_as(vehicle_hex_ids)
            vehicle_hex_context = hex_encodings[batch_indices, vehicle_hex_ids]
        else:
            vehicle_hex_context = hex_encodings[vehicle_hex_ids]
        
        vehicle_encodings = self.vehicle_encoder(vehicle_features)
        
        combined = torch.cat([
            vehicle_encodings,
            vehicle_hex_context,
            global_features.unsqueeze(-2).expand(-1, vehicle_features.shape[-2], -1)
            if global_features.dim() == 2 else global_features
        ], dim=-1)
        
        return self.output_projection(combined)
