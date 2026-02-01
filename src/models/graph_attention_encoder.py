"""Graph Attention Encoder for T-GAT lateral movement detection.

This module implements the GraphAttentionEncoder class that:
- Uses multi-head GAT layers for learning node embeddings
- Implements graph pooling for graph-level embeddings
- Supports configurable layers, heads, and hidden dimensions
- Provides attention weights for interpretability

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool


class GraphAttentionEncoder(nn.Module):
    """Graph Attention Network encoder for learning node and graph embeddings.
    
    Uses multi-head attention mechanism to aggregate neighbor information
    weighted by learned attention scores. Produces both node-level and
    graph-level embeddings.
    
    Attributes:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        num_layers: Number of GAT layers
        heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        """Initialize GraphAttentionEncoder.
        
        Args:
            in_channels: Dimension of input node features
            hidden_channels: Dimension of hidden layers (default: 128)
            out_channels: Dimension of output embeddings (default: 64)
            num_layers: Number of GAT layers, 2-4 recommended (default: 2)
            heads: Number of attention heads, 4-8 recommended (default: 4)
            dropout: Dropout rate for regularization (default: 0.1)
            edge_dim: Dimension of edge features, None if not used (default: None)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        # Store attention weights for interpretability
        self._attention_weights: List[Tensor] = []
        
        # Build GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer: in_channels -> hidden_channels
        self.convs.append(
            GATConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=True,  # Concatenate multi-head outputs
            )
        )
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Middle layers: hidden_channels * heads -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_channels=hidden_channels * heads,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Last layer: hidden_channels * heads -> out_channels
        # Use concat=False for final layer to get single output per node
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    in_channels=hidden_channels * heads,
                    out_channels=out_channels,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=False,  # Average multi-head outputs
                )
            )
            self.norms.append(nn.LayerNorm(out_channels))
        
        # Graph-level pooling projection
        self.graph_projection = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )

    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through GAT layers.
        
        Computes node embeddings using multi-head attention mechanism,
        then aggregates to produce graph-level embedding.
        
        Property 6: Attention weights sum to 1 per neighborhood
        Property 7: Output dimensions match configuration
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
            batch: Optional batch assignment for multiple graphs [num_nodes]
            return_attention_weights: Whether to store attention weights
            
        Returns:
            Tuple of:
            - node_embeddings: [num_nodes, out_channels]
            - graph_embedding: [batch_size, out_channels]
        """
        # Clear previous attention weights
        self._attention_weights = []
        
        # Handle empty graphs
        if x.size(0) == 0:
            batch_size = 1 if batch is None else batch.max().item() + 1
            empty_node = torch.zeros((0, self.out_channels), device=x.device, dtype=x.dtype)
            empty_graph = torch.zeros((batch_size, self.out_channels), device=x.device, dtype=x.dtype)
            return empty_node, empty_graph
        
        # Create batch tensor if not provided (single graph)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Forward through GAT layers
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Apply GAT convolution with attention
            if return_attention_weights or not self.training:
                # Get attention weights for interpretability
                h_new, (edge_index_out, attention_weights) = conv(
                    h, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
                self._attention_weights.append(attention_weights)
            else:
                h_new = conv(h, edge_index, edge_attr=edge_attr)
            
            # Apply layer normalization
            h_new = norm(h_new)
            
            # Apply activation (LeakyReLU as per requirement 2.2)
            # Skip activation on last layer
            if i < len(self.convs) - 1:
                h_new = F.leaky_relu(h_new, negative_slope=0.2)
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            h = h_new
        
        node_embeddings = h
        
        # Graph-level pooling (mean pooling over nodes)
        # Requirement 2.5: produce graph-level embedding through aggregation
        graph_embedding = global_mean_pool(node_embeddings, batch)
        
        # Project graph embedding
        graph_embedding = self.graph_projection(graph_embedding)
        
        return node_embeddings, graph_embedding
    
    def get_attention_weights(self) -> List[Tensor]:
        """Return attention weights from the last forward pass.
        
        Attention weights can be used for interpretability to understand
        which edges the model considers most important.
        
        Property 6: Each weight is in [0, 1] and sums to 1 per neighborhood
        
        Returns:
            List of attention weight tensors, one per GAT layer.
            Each tensor has shape [num_edges, heads] or [num_edges * heads]
            depending on the layer configuration.
        """
        return self._attention_weights
    
    def encode_nodes(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode only node embeddings without graph pooling.
        
        Useful when only node-level representations are needed.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        node_embeddings, _ = self.forward(x, edge_index, edge_attr)
        return node_embeddings
    
    def encode_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode only graph-level embedding.
        
        Useful when only graph-level representation is needed.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
            batch: Optional batch assignment [num_nodes]
            
        Returns:
            Graph embedding [batch_size, out_channels]
        """
        _, graph_embedding = self.forward(x, edge_index, edge_attr, batch)
        return graph_embedding
