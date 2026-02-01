"""Temporal Aggregator for T-GAT lateral movement detection.

This module implements the TemporalAggregator class that:
- Uses LSTM-based sequence model to capture temporal dependencies
- Supports hidden state passing for streaming inference
- Supports configurable hidden size and number of layers

Requirements: 3.1, 3.2, 3.3, 3.4
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class TemporalAggregator(nn.Module):
    """LSTM-based temporal aggregator for capturing cross-window dependencies.
    
    Processes sequences of graph embeddings to capture temporal patterns
    that evolve over time. Supports streaming inference by maintaining
    hidden state across windows.
    
    Attributes:
        input_dim: Dimension of input graph embeddings
        hidden_dim: Dimension of LSTM hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout rate for regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize TemporalAggregator.
        
        Args:
            input_dim: Dimension of input graph embeddings
            hidden_dim: Dimension of LSTM hidden state, 256-512 recommended (default: 256)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate for regularization (default: 0.1)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # LSTM for temporal sequence modeling
        # Requirement 3.1: Use LSTM layers to capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # Unidirectional for streaming inference
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        graph_embeddings: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Process sequence of graph embeddings.
        
        Property 8: Permuting sequence order changes output (temporal sensitivity)
        Property 9: Split processing with state equals full sequence processing
        
        Args:
            graph_embeddings: Sequence of graph embeddings [batch, seq_len, input_dim]
            hidden: Optional initial hidden state tuple (h_0, c_0)
                   Each tensor has shape [num_layers, batch, hidden_dim]
                   
        Returns:
            Tuple of:
            - output: Temporal embedding [batch, hidden_dim]
            - hidden: Updated hidden state tuple (h_n, c_n) for streaming
        """
        batch_size = graph_embeddings.size(0)
        seq_len = graph_embeddings.size(1)
        
        # Handle empty sequences
        if seq_len == 0:
            device = graph_embeddings.device
            dtype = graph_embeddings.dtype
            
            # Return zero output and initialize hidden state if not provided
            output = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            if hidden is None:
                h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
                c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
                hidden = (h_0, c_0)
            return output, hidden
        
        # Initialize hidden state if not provided
        # Requirement 3.3: Maintain hidden state across windows for streaming
        if hidden is None:
            device = graph_embeddings.device
            dtype = graph_embeddings.dtype
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
            hidden = (h_0, c_0)
        
        # Process through LSTM
        # Requirement 3.1: LSTM captures temporal dependencies
        lstm_out, hidden_out = self.lstm(graph_embeddings, hidden)
        # lstm_out: [batch, seq_len, hidden_dim]
        # hidden_out: tuple of (h_n, c_n), each [num_layers, batch, hidden_dim]
        
        # Take the last timestep output as the temporal representation
        # Requirement 3.4: Output temporal-aware representation for classification
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Apply layer normalization
        normalized = self.layer_norm(last_output)
        
        # Apply output projection
        output = self.output_projection(normalized)
        
        return output, hidden_out
    
    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Initialize hidden state for streaming inference.
        
        Args:
            batch_size: Number of sequences in batch
            device: Device for tensors (default: CPU)
            dtype: Data type for tensors (default: float32)
            
        Returns:
            Tuple of (h_0, c_0) initial hidden states
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
            
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
        
        return (h_0, c_0)
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the temporal aggregator.
        
        Returns:
            Output dimension (same as hidden_dim)
        """
        return self.hidden_dim
