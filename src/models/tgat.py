"""Temporal Graph Attention Network (T-GAT) for lateral movement detection.

This module implements the full T-GAT model that integrates:
- GraphAttentionEncoder: Spatial graph attention for node embeddings
- TemporalAggregator: LSTM for temporal dependencies across windows
- BayesianUncertaintyModule: MC Dropout for uncertainty estimation
- PhysicsConstraintModule: Domain constraints and energy-based anomaly scores
- DetectionHead: Final classification layer

Requirements: 2.5, 3.4, 4.4, 5.2, 6.1, 9.4
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data, Batch

from src.data.models import DetectionResult, UncertaintyOutput, TimeWindow
from src.models.graph_attention_encoder import GraphAttentionEncoder
from src.models.temporal_aggregator import TemporalAggregator
from src.models.bayesian_uncertainty import BayesianUncertaintyModule
from src.models.physics_constraint import PhysicsConstraintModule
from src.models.detection_head import DetectionHead


class TGAT(nn.Module):
    """Temporal Graph Attention Network for lateral movement detection.
    
    Combines spatial graph attention with temporal sequence modeling and
    Bayesian uncertainty quantification for detecting lateral movement
    attacks in enterprise networks.
    
    Architecture:
    1. GraphAttentionEncoder: Encodes each time window's graph into embeddings
    2. TemporalAggregator: Captures temporal dependencies across windows
    3. BayesianUncertaintyModule: Provides calibrated uncertainty estimates
    4. PhysicsConstraintModule: Enforces domain constraints
    5. DetectionHead: Produces final lateral movement probability
    
    Attributes:
        node_feature_dim: Dimension of input node features
        edge_feature_dim: Dimension of input edge features
        hidden_dim: Hidden dimension for encoder and aggregator
        num_gat_layers: Number of GAT layers
        num_attention_heads: Number of attention heads in GAT
        num_lstm_layers: Number of LSTM layers
        dropout: Dropout rate
        num_mc_samples: Number of MC Dropout samples for uncertainty
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: Optional[int] = None,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_gat_layers: int = 2,
        num_attention_heads: int = 4,
        num_lstm_layers: int = 2,
        lstm_hidden_dim: int = 256,
        dropout: float = 0.1,
        num_mc_samples: int = 50,
        mc_dropout_rate: float = 0.2,
        min_auth_delta: float = 0.1,
        energy_weight: float = 0.1,
        constraint_weight: float = 0.01,
        class_weights: Optional[Tensor] = None,
        detection_threshold: float = 0.5,
    ):
        """Initialize T-GAT model.
        
        Args:
            node_feature_dim: Dimension of input node features
            edge_feature_dim: Dimension of edge features (optional)
            hidden_dim: Hidden dimension for GAT encoder (default: 128)
            output_dim: Output dimension for graph embeddings (default: 64)
            num_gat_layers: Number of GAT layers (default: 2)
            num_attention_heads: Number of attention heads (default: 4)
            num_lstm_layers: Number of LSTM layers (default: 2)
            lstm_hidden_dim: Hidden dimension for LSTM (default: 256)
            dropout: Dropout rate (default: 0.1)
            num_mc_samples: Number of MC Dropout samples (default: 50)
            mc_dropout_rate: Dropout rate for MC Dropout (default: 0.2)
            min_auth_delta: Minimum time between authentications (default: 0.1)
            energy_weight: Weight for energy term in loss (default: 0.1)
            constraint_weight: Weight for constraint violations (default: 0.01)
            class_weights: Optional class weights for imbalanced data
            detection_threshold: Decision threshold for classification (default: 0.5)
        """
        super().__init__()
        
        # Store configuration
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_mc_samples = num_mc_samples
        self.detection_threshold = detection_threshold
        
        # Component 1: Graph Attention Encoder
        # Requirement 2.5: Produce graph-level embedding through aggregation
        self.graph_encoder = GraphAttentionEncoder(
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=num_gat_layers,
            heads=num_attention_heads,
            dropout=dropout,
            edge_dim=edge_feature_dim,
        )
        
        # Component 2: Temporal Aggregator
        # Requirement 3.4: Output temporal-aware representation
        self.temporal_aggregator = TemporalAggregator(
            input_dim=output_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout,
        )
        
        # Component 3: Bayesian Uncertainty Module
        # Requirement 4.4: Provide epistemic and aleatoric uncertainty
        self.uncertainty_module = BayesianUncertaintyModule(
            input_dim=lstm_hidden_dim,
            num_samples=num_mc_samples,
            dropout_rate=mc_dropout_rate,
        )
        
        # Component 4: Physics Constraint Module
        # Requirement 5.2: Compute energy measuring deviation from normal
        self.physics_module = PhysicsConstraintModule(
            min_auth_delta=min_auth_delta,
            energy_weight=energy_weight,
            constraint_weight=constraint_weight,
        )
        
        # Component 5: Detection Head
        # Requirement 6.1: Output probability of lateral movement
        self.detection_head = DetectionHead(
            input_dim=lstm_hidden_dim,
            hidden_dim=hidden_dim,
            class_weights=class_weights,
            threshold=detection_threshold,
            dropout=dropout,
        )
        
        # Store normal embedding for energy computation
        # This will be updated during training
        self.register_buffer(
            'normal_embedding',
            torch.zeros(lstm_hidden_dim)
        )

    def forward(
        self,
        graphs: Union[List[Data], Batch],
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
        return_uncertainty: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, UncertaintyOutput, Tuple[Tensor, Tensor]]]:
        """Forward pass through all T-GAT components.
        
        Processes a sequence of graphs through the full pipeline:
        1. Encode each graph using GAT
        2. Aggregate temporal information using LSTM
        3. Optionally compute uncertainty using MC Dropout
        4. Produce final detection probability
        
        Args:
            graphs: List of PyTorch Geometric Data objects or a Batch object
                   representing a sequence of temporal windows
            hidden: Optional LSTM hidden state for streaming inference
            return_uncertainty: Whether to compute and return uncertainty estimates
            
        Returns:
            If return_uncertainty is False:
                predictions: Detection probabilities [batch_size, 1]
            If return_uncertainty is True:
                Tuple of (predictions, uncertainty_output, hidden_state)
        """
        # Handle empty input
        if isinstance(graphs, list) and len(graphs) == 0:
            device = next(self.parameters()).device
            empty_pred = torch.zeros(1, 1, device=device)
            if return_uncertainty:
                empty_uncertainty = UncertaintyOutput(
                    mean_prediction=0.0,
                    epistemic_uncertainty=0.0,
                    aleatoric_uncertainty=0.0,
                    samples=torch.zeros(1, 1, device=device),
                )
                empty_hidden = self.temporal_aggregator.init_hidden(1, device)
                return empty_pred, empty_uncertainty, empty_hidden
            return empty_pred
        
        # Step 1: Encode each graph using GAT
        # Requirement 2.5: Produce graph-level embedding
        graph_embeddings = self._encode_graphs(graphs)
        
        # Step 2: Process through temporal aggregator
        # Requirement 3.4: Output temporal-aware representation
        # graph_embeddings: [batch_size, seq_len, output_dim]
        temporal_embedding, new_hidden = self.temporal_aggregator(
            graph_embeddings, hidden
        )
        
        # Step 3: Detection head produces probability
        # Requirement 6.1: Output probability of lateral movement
        predictions = self.detection_head(temporal_embedding)
        
        if return_uncertainty:
            # Step 4: Compute uncertainty using MC Dropout
            # Requirement 4.4: Provide epistemic and aleatoric uncertainty
            uncertainty_output = self.uncertainty_module(temporal_embedding)
            return predictions, uncertainty_output, new_hidden
        
        return predictions
    
    def _encode_graphs(
        self,
        graphs: Union[List[Data], Batch],
    ) -> Tensor:
        """Encode a sequence of graphs into embeddings.
        
        Args:
            graphs: List of Data objects or Batch object
            
        Returns:
            Graph embeddings [batch_size, seq_len, output_dim]
        """
        if isinstance(graphs, Batch):
            # Single batched graph - encode directly
            _, graph_embedding = self.graph_encoder(
                graphs.x,
                graphs.edge_index,
                edge_attr=graphs.edge_attr if hasattr(graphs, 'edge_attr') else None,
                batch=graphs.batch,
            )
            # Add sequence dimension
            return graph_embedding.unsqueeze(1)
        
        # List of graphs - encode each one
        embeddings = []
        for graph in graphs:
            _, graph_embedding = self.graph_encoder(
                graph.x,
                graph.edge_index,
                edge_attr=graph.edge_attr if hasattr(graph, 'edge_attr') else None,
                batch=graph.batch if hasattr(graph, 'batch') else None,
            )
            embeddings.append(graph_embedding)
        
        # Stack into sequence: [batch_size, seq_len, output_dim]
        # Assuming each graph is a single sample, stack along seq dimension
        if len(embeddings) > 0:
            stacked = torch.stack(embeddings, dim=1)
            return stacked
        
        device = next(self.parameters()).device
        return torch.zeros(1, 0, self.output_dim, device=device)
    
    def inference(
        self,
        graphs: Union[List[Data], Batch],
        window_id: str,
        timestamp: float,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
        auth_timestamps: Optional[Tensor] = None,
    ) -> Tuple[DetectionResult, Optional[Tuple[Tensor, Tensor]]]:
        """Perform inference and return DetectionResult.
        
        Full inference pipeline that returns a complete DetectionResult
        with all required fields for alerting.
        
        Property 19: Alert Field Completeness
        Requirement 9.4: Include prediction probability, epistemic and aleatoric uncertainty
        
        Args:
            graphs: Sequence of graphs representing temporal windows
            window_id: Unique identifier for the time window
            timestamp: Timestamp of the detection
            hidden: Optional LSTM hidden state for streaming
            auth_timestamps: Optional authentication timestamps for constraint checking
            
        Returns:
            Tuple of (DetectionResult, updated_hidden_state)
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass with uncertainty
            predictions, uncertainty_output, new_hidden = self.forward(
                graphs, hidden, return_uncertainty=True
            )
            
            # Get probability value
            probability = predictions.mean().item()
            
            # Compute energy score using physics module
            # Requirement 5.2: Compute energy measuring deviation from normal
            temporal_embedding = self._get_temporal_embedding(graphs, hidden)
            energy = self.physics_module.compute_energy(
                temporal_embedding,
                self.normal_embedding,
            )
            energy_score = energy.mean().item() if energy.numel() > 0 else 0.0
            
            # Check constraint violations
            constraint_violations = []
            if auth_timestamps is not None:
                violations = self.physics_module.check_temporal_ordering(auth_timestamps)
                if violations.sum() > 0:
                    violation_descs = self.physics_module.get_violation_descriptions(
                        auth_timestamps
                    )
                    constraint_violations.extend(violation_descs)
            
            # Create DetectionResult with all required fields
            # Property 19: All required fields present
            result = DetectionResult(
                window_id=window_id,
                timestamp=timestamp,
                probability=probability,
                epistemic_uncertainty=uncertainty_output.epistemic_uncertainty,
                aleatoric_uncertainty=uncertainty_output.aleatoric_uncertainty,
                energy_score=energy_score,
                constraint_violations=constraint_violations,
            )
            
            return result, new_hidden
    
    def _get_temporal_embedding(
        self,
        graphs: Union[List[Data], Batch],
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Get temporal embedding without full forward pass.
        
        Args:
            graphs: Sequence of graphs
            hidden: Optional LSTM hidden state
            
        Returns:
            Temporal embedding tensor
        """
        graph_embeddings = self._encode_graphs(graphs)
        temporal_embedding, _ = self.temporal_aggregator(graph_embeddings, hidden)
        return temporal_embedding

    def compute_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        graphs: Optional[Union[List[Data], Batch]] = None,
        auth_timestamps: Optional[Tensor] = None,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Compute total loss including detection and constraint losses.
        
        Args:
            predictions: Model predictions [batch_size, 1]
            targets: Ground truth labels [batch_size]
            graphs: Optional graphs for constraint computation
            auth_timestamps: Optional timestamps for temporal constraints
            hidden: Optional LSTM hidden state
            
        Returns:
            Dictionary containing:
            - 'total_loss': Combined loss
            - 'detection_loss': Binary cross-entropy loss
            - 'constraint_loss': Physics constraint loss
        """
        # Detection loss
        detection_loss = self.detection_head.compute_loss(predictions, targets)
        
        # Constraint loss
        constraint_loss = torch.tensor(0.0, device=predictions.device)
        
        if graphs is not None:
            temporal_embedding = self._get_temporal_embedding(graphs, hidden)
            
            constraints = {}
            
            # Energy constraint
            energy = self.physics_module.compute_energy(
                temporal_embedding,
                self.normal_embedding,
            )
            constraints['energy'] = energy
            
            # Temporal ordering constraint
            if auth_timestamps is not None:
                violations = self.physics_module.check_temporal_ordering(auth_timestamps)
                constraints['temporal_violations'] = violations
            
            constraint_loss = self.physics_module.compute_constraint_loss(
                predictions, constraints
            )
        
        total_loss = detection_loss + constraint_loss
        
        return {
            'total_loss': total_loss,
            'detection_loss': detection_loss,
            'constraint_loss': constraint_loss,
        }
    
    def update_normal_embedding(self, embedding: Tensor) -> None:
        """Update the reference normal embedding for energy computation.
        
        Should be called during training to update the baseline normal
        pattern embedding.
        
        Args:
            embedding: New normal embedding [hidden_dim]
        """
        self.normal_embedding.copy_(embedding.detach())
    
    def set_detection_threshold(self, threshold: float) -> None:
        """Set the decision threshold for classification.
        
        Args:
            threshold: New threshold in [0, 1]
        """
        self.detection_threshold = threshold
        self.detection_head.set_threshold(threshold)
    
    def get_attention_weights(self) -> List[Tensor]:
        """Get attention weights from the graph encoder.
        
        Returns:
            List of attention weight tensors from GAT layers
        """
        return self.graph_encoder.get_attention_weights()
    
    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Initialize LSTM hidden state for streaming inference.
        
        Args:
            batch_size: Number of sequences in batch
            device: Device for tensors
            
        Returns:
            Tuple of (h_0, c_0) initial hidden states
        """
        if device is None:
            device = next(self.parameters()).device
        return self.temporal_aggregator.init_hidden(batch_size, device)
    
    def get_config(self) -> Dict:
        """Get model configuration as dictionary.
        
        Returns:
            Dictionary of model configuration parameters
        """
        return {
            'node_feature_dim': self.node_feature_dim,
            'edge_feature_dim': self.edge_feature_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_mc_samples': self.num_mc_samples,
            'detection_threshold': self.detection_threshold,
        }

