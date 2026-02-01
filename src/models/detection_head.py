"""Detection Head for T-GAT lateral movement detection.

This module implements the DetectionHead class that:
- Implements classification layer with sigmoid output
- Implements weighted binary cross-entropy loss
- Supports configurable decision thresholds

Requirements: 6.1, 6.2, 6.3
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DetectionHead(nn.Module):
    """Detection head for lateral movement classification.
    
    Final classification layer that produces lateral movement probability
    from temporal embeddings. Supports class weighting for handling
    imbalanced datasets and configurable decision thresholds.
    
    Attributes:
        input_dim: Dimension of input temporal embeddings
        hidden_dim: Dimension of hidden layer
        class_weights: Optional weights for handling class imbalance
        threshold: Decision threshold for classification
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        class_weights: Optional[Tensor] = None,
        threshold: float = 0.5,
        dropout: float = 0.1,
    ):
        """Initialize DetectionHead.
        
        Args:
            input_dim: Dimension of input temporal embeddings
            hidden_dim: Dimension of hidden layer (default: 64)
            class_weights: Optional tensor of shape [2] with weights for
                          [negative_class, positive_class] (default: None)
            threshold: Decision threshold for classification (default: 0.5)
            dropout: Dropout rate for regularization (default: 0.1)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.dropout_rate = dropout
        
        # Store class weights for weighted loss
        # Requirement 6.3: Handle class imbalance through weighted loss
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', None)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Produce lateral movement probability.
        
        Property 14: Output is always in [0, 1]
        Requirement 6.1: Output probability of lateral movement
        
        Args:
            x: Temporal embedding tensor [batch_size, input_dim]
            
        Returns:
            Probability tensor [batch_size, 1] with values in [0, 1]
        """
        # Pass through classification layers
        logits = self.classifier(x)
        
        # Apply sigmoid to get probability
        # Requirement 6.1: Output probability of lateral movement
        probability = torch.sigmoid(logits)
        
        return probability
    
    def compute_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        class_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute weighted binary cross-entropy loss.
        
        Property 15: Changing weights changes loss value
        Requirement 6.3: Handle class imbalance through weighted loss
        
        Args:
            predictions: Predicted probabilities [batch_size, 1] or [batch_size]
            targets: Ground truth labels [batch_size, 1] or [batch_size]
            class_weights: Optional override for class weights [2]
                          Format: [weight_for_class_0, weight_for_class_1]
            
        Returns:
            Scalar loss tensor
        """
        # Flatten predictions and targets if needed
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()
        
        # Use provided weights or stored weights
        weights = class_weights if class_weights is not None else self.class_weights
        
        if weights is not None:
            # Compute per-sample weights based on target class
            # weights[0] for negative class, weights[1] for positive class
            sample_weights = torch.where(
                targets == 1,
                weights[1],
                weights[0]
            )
            
            # Compute weighted BCE loss
            # Using reduction='none' to apply per-sample weights
            bce_loss = F.binary_cross_entropy(
                predictions,
                targets,
                reduction='none'
            )
            
            # Apply sample weights and compute mean
            weighted_loss = (bce_loss * sample_weights).mean()
            return weighted_loss
        else:
            # Standard BCE loss without weighting
            return F.binary_cross_entropy(predictions, targets)
    
    def predict(self, x: Tensor) -> Tensor:
        """Make binary predictions using the decision threshold.
        
        Requirement 6.2: Support configurable decision thresholds
        
        Args:
            x: Temporal embedding tensor [batch_size, input_dim]
            
        Returns:
            Binary predictions [batch_size, 1] (0 or 1)
        """
        probabilities = self.forward(x)
        predictions = (probabilities >= self.threshold).float()
        return predictions
    
    def set_threshold(self, threshold: float) -> None:
        """Set the decision threshold for classification.
        
        Requirement 6.2: Support configurable decision thresholds
        
        Args:
            threshold: New decision threshold in [0, 1]
            
        Raises:
            ValueError: If threshold is not in [0, 1]
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
    
    def set_class_weights(self, class_weights: Tensor) -> None:
        """Set class weights for weighted loss computation.
        
        Args:
            class_weights: Tensor of shape [2] with weights for
                          [negative_class, positive_class]
                          
        Raises:
            ValueError: If class_weights doesn't have shape [2]
        """
        if class_weights.shape != torch.Size([2]):
            raise ValueError(f"class_weights must have shape [2], got {class_weights.shape}")
        
        # Update the buffer
        self.register_buffer('class_weights', class_weights)
    
    def get_threshold(self) -> float:
        """Get the current decision threshold.
        
        Returns:
            Current decision threshold
        """
        return self.threshold
    
    def get_class_weights(self) -> Optional[Tensor]:
        """Get the current class weights.
        
        Returns:
            Current class weights tensor or None if not set
        """
        return self.class_weights
