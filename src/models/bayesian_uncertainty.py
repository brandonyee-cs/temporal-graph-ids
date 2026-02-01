"""Bayesian Uncertainty Module for T-GAT lateral movement detection.

This module implements the BayesianUncertaintyModule class that:
- Implements MC Dropout for uncertainty estimation
- Computes epistemic uncertainty as entropy of expected prediction
- Computes aleatoric uncertainty as expected entropy
- Returns UncertaintyOutput with all fields

Requirements: 4.1, 4.2, 4.3, 4.4
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.data.models import UncertaintyOutput


class BayesianUncertaintyModule(nn.Module):
    """Bayesian uncertainty module using MC Dropout.
    
    Provides calibrated uncertainty estimates by performing multiple
    forward passes with dropout enabled during inference.
    
    Attributes:
        input_dim: Dimension of input features
        num_samples: Number of MC Dropout samples for uncertainty estimation
        dropout_rate: Dropout rate for MC Dropout
    """
    
    def __init__(
        self,
        input_dim: int,
        num_samples: int = 50,
        dropout_rate: float = 0.2,
    ):
        """Initialize BayesianUncertaintyModule.
        
        Args:
            input_dim: Dimension of input features
            num_samples: Number of MC Dropout samples (default: 50)
            dropout_rate: Dropout rate for uncertainty estimation (default: 0.2)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        
        # Dropout layer for MC Dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Projection layers with dropout for uncertainty
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim),
        )
    
    def forward(
        self,
        x: Tensor,
        num_samples: Optional[int] = None,
    ) -> UncertaintyOutput:
        """Compute prediction with uncertainty estimates.
        
        Property 10: MC Dropout Sample Variance
        Property 11: Uncertainty Decomposition Validity
        
        Args:
            x: Input tensor [batch_size, input_dim]
            num_samples: Optional override for number of MC samples
            
        Returns:
            UncertaintyOutput containing mean prediction, uncertainties, and samples
        """
        n_samples = num_samples if num_samples is not None else self.num_samples
        batch_size = x.size(0)
        
        # Collect MC Dropout samples
        samples = []
        
        # Enable dropout during inference for MC Dropout
        self.train()  # Enable dropout
        
        for _ in range(n_samples):
            # Apply dropout and projection
            dropped = self.dropout(x)
            projected = self.projection(dropped)
            # Apply sigmoid to get probability
            prob = torch.sigmoid(projected.mean(dim=-1))
            samples.append(prob)
        
        # Stack samples: [num_samples, batch_size]
        samples_tensor = torch.stack(samples, dim=0)
        
        # Compute mean prediction
        mean_prediction = samples_tensor.mean(dim=0)
        
        # Compute epistemic uncertainty (entropy of expected prediction)
        epistemic = self.compute_epistemic(samples_tensor)
        
        # Compute aleatoric uncertainty (expected entropy)
        aleatoric = self.compute_aleatoric(samples_tensor)
        
        # Return single UncertaintyOutput for batch (using mean across batch)
        return UncertaintyOutput(
            mean_prediction=mean_prediction.mean().item(),
            epistemic_uncertainty=epistemic.mean().item(),
            aleatoric_uncertainty=aleatoric.mean().item(),
            samples=samples_tensor,
        )
    
    def compute_epistemic(self, samples: Tensor) -> Tensor:
        """Compute epistemic uncertainty from MC samples.
        
        Epistemic uncertainty is computed as the entropy of the expected
        prediction (variance in model predictions).
        
        Args:
            samples: MC samples [num_samples, batch_size]
            
        Returns:
            Epistemic uncertainty [batch_size]
        """
        # Mean prediction across samples
        mean_pred = samples.mean(dim=0)
        
        # Entropy of mean prediction: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-7
        mean_pred = torch.clamp(mean_pred, eps, 1 - eps)
        entropy = -mean_pred * torch.log(mean_pred) - (1 - mean_pred) * torch.log(1 - mean_pred)
        
        return entropy
    
    def compute_aleatoric(self, samples: Tensor) -> Tensor:
        """Compute aleatoric uncertainty from MC samples.
        
        Aleatoric uncertainty is computed as the expected entropy
        (average uncertainty across samples).
        
        Args:
            samples: MC samples [num_samples, batch_size]
            
        Returns:
            Aleatoric uncertainty [batch_size]
        """
        eps = 1e-7
        samples = torch.clamp(samples, eps, 1 - eps)
        
        # Entropy of each sample
        entropies = -samples * torch.log(samples) - (1 - samples) * torch.log(1 - samples)
        
        # Expected entropy (mean across samples)
        expected_entropy = entropies.mean(dim=0)
        
        return expected_entropy
    
    def get_samples(
        self,
        x: Tensor,
        num_samples: Optional[int] = None,
    ) -> Tensor:
        """Get raw MC Dropout samples without computing uncertainties.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            num_samples: Optional override for number of MC samples
            
        Returns:
            Samples tensor [num_samples, batch_size]
        """
        n_samples = num_samples if num_samples is not None else self.num_samples
        
        samples = []
        self.train()  # Enable dropout
        
        for _ in range(n_samples):
            dropped = self.dropout(x)
            projected = self.projection(dropped)
            prob = torch.sigmoid(projected.mean(dim=-1))
            samples.append(prob)
        
        return torch.stack(samples, dim=0)

