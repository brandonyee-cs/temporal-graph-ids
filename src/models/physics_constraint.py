"""Physics Constraint Module for T-GAT lateral movement detection.

This module implements the PhysicsConstraintModule class that:
- Enforces temporal ordering constraints (minimum time between sequential authentications)
- Computes energy function measuring deviation from normal authentication patterns
- Adds constraint violations as soft regularization terms to the loss function
- Supports configurable constraint weights

Requirements: 5.1, 5.2, 5.3, 5.4
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PhysicsConstraintModule(nn.Module):
    """Physics-inspired constraint module for authentication pattern validation.
    
    Enforces domain-specific rules encoding physical limitations on authentication
    patterns and computes energy-based anomaly scores for deviation from normal
    behavior.
    
    Attributes:
        min_auth_delta: Minimum time (seconds) between sequential authentications
        energy_weight: Weight for energy term in loss computation
        constraint_weight: Weight for constraint violation penalty
    """
    
    def __init__(
        self,
        min_auth_delta: float = 0.1,
        energy_weight: float = 0.1,
        constraint_weight: float = 0.01,
    ):
        """Initialize PhysicsConstraintModule.
        
        Args:
            min_auth_delta: Minimum time between sequential authentications in seconds.
                           Authentications happening faster than this are flagged as
                           constraint violations (default: 0.1 seconds)
            energy_weight: Weight for energy term in loss computation (default: 0.1)
            constraint_weight: Weight for constraint violation penalty (default: 0.01)
        """
        super().__init__()
        
        self.min_auth_delta = min_auth_delta
        self.energy_weight = energy_weight
        self.constraint_weight = constraint_weight
        
        # Learnable parameters for energy computation
        # These help adapt the energy function to the data distribution
        self.energy_scale = nn.Parameter(torch.tensor(1.0))
        self.energy_bias = nn.Parameter(torch.tensor(0.0))
    
    def check_temporal_ordering(self, auth_sequence: Tensor) -> Tensor:
        """Check temporal ordering constraint violations.
        
        Validates that sequential authentications respect the minimum time delta.
        Authentications happening too quickly (faster than min_auth_delta) are
        flagged as violations.
        
        Property 12: Temporal Ordering Constraint Detection
        - Violations are detected when auth happens too quickly
        
        Args:
            auth_sequence: Tensor of authentication timestamps [batch, seq_len]
                          or [seq_len] for single sequence. Timestamps should be
                          in seconds.
                          
        Returns:
            Tensor of violation flags [batch, seq_len-1] or [seq_len-1]
            where 1.0 indicates a violation (time delta < min_auth_delta)
            and 0.0 indicates no violation.
        """
        # Handle empty or single-element sequences
        if auth_sequence.numel() == 0:
            return torch.zeros(0, device=auth_sequence.device, dtype=auth_sequence.dtype)
        
        # Ensure at least 2D for consistent processing
        if auth_sequence.dim() == 1:
            auth_sequence = auth_sequence.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Handle sequences with 0 or 1 elements
        if auth_sequence.size(1) <= 1:
            batch_size = auth_sequence.size(0)
            result = torch.zeros(batch_size, 0, device=auth_sequence.device, dtype=auth_sequence.dtype)
            if squeeze_output:
                result = result.squeeze(0)
            return result
        
        # Compute time deltas between consecutive authentications
        # delta[i] = auth_sequence[i+1] - auth_sequence[i]
        time_deltas = auth_sequence[:, 1:] - auth_sequence[:, :-1]
        
        # Flag violations where delta < min_auth_delta
        # A violation occurs when authentication happens too quickly
        violations = (time_deltas < self.min_auth_delta).float()
        
        if squeeze_output:
            violations = violations.squeeze(0)
        
        return violations
    
    def compute_energy(
        self,
        graph_embedding: Tensor,
        normal_embedding: Tensor,
    ) -> Tensor:
        """Compute energy as deviation from normal authentication patterns.
        
        The energy function measures how much a graph embedding deviates from
        the expected normal pattern. Higher energy indicates more anomalous
        behavior.
        
        Property 13: Energy Function Properties
        - Energy is non-negative
        - Energy is zero for identical embeddings (within floating-point tolerance)
        
        Args:
            graph_embedding: Current graph embedding [batch, embed_dim] or [embed_dim]
            normal_embedding: Reference normal pattern embedding [batch, embed_dim] 
                            or [embed_dim]
                            
        Returns:
            Energy values [batch] or scalar, always non-negative.
            Zero when graph_embedding equals normal_embedding.
        """
        # Handle 1D inputs
        if graph_embedding.dim() == 1:
            graph_embedding = graph_embedding.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        if normal_embedding.dim() == 1:
            normal_embedding = normal_embedding.unsqueeze(0)
        
        # Broadcast normal_embedding if needed
        if normal_embedding.size(0) == 1 and graph_embedding.size(0) > 1:
            normal_embedding = normal_embedding.expand(graph_embedding.size(0), -1)
        
        # Compute squared L2 distance (deviation from normal)
        # This ensures energy is always non-negative
        diff = graph_embedding - normal_embedding
        squared_distance = torch.sum(diff ** 2, dim=-1)
        
        # Apply learnable scaling and ensure non-negativity
        # Using softplus to ensure energy_scale is positive
        scaled_energy = F.softplus(self.energy_scale) * squared_distance + F.relu(self.energy_bias)
        
        # Ensure non-negativity (should already be non-negative, but explicit for safety)
        energy = F.relu(scaled_energy)
        
        if squeeze_output:
            energy = energy.squeeze(0)
        
        return energy
    
    def compute_constraint_loss(
        self,
        predictions: Tensor,
        constraints: Dict[str, Tensor],
    ) -> Tensor:
        """Compute soft constraint violation loss for training.
        
        Combines multiple constraint violations into a single loss term that
        can be added to the main training loss. This implements soft constraints
        as regularization rather than hard constraints.
        
        Args:
            predictions: Model predictions [batch] or [batch, 1]
            constraints: Dictionary containing constraint-related tensors:
                - 'temporal_violations': Output from check_temporal_ordering()
                - 'energy': Output from compute_energy()
                - 'auth_timestamps': Optional timestamps for additional checks
                
        Returns:
            Scalar loss value representing total constraint violation penalty.
        """
        total_loss = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        
        # Temporal ordering constraint loss
        if 'temporal_violations' in constraints:
            violations = constraints['temporal_violations']
            if violations.numel() > 0:
                # Mean violation rate weighted by constraint_weight
                temporal_loss = self.constraint_weight * violations.mean()
                total_loss = total_loss + temporal_loss
        
        # Energy-based constraint loss
        if 'energy' in constraints:
            energy = constraints['energy']
            if energy.numel() > 0:
                # Mean energy weighted by energy_weight
                energy_loss = self.energy_weight * energy.mean()
                total_loss = total_loss + energy_loss
        
        # Additional constraint: high confidence predictions with violations
        # Penalize confident predictions that violate physical constraints
        if 'temporal_violations' in constraints and predictions.numel() > 0:
            violations = constraints['temporal_violations']
            if violations.numel() > 0:
                # Flatten predictions if needed
                preds = predictions.view(-1)
                
                # If we have per-sequence violations, aggregate them
                if violations.dim() > 1:
                    # Mean violation per sequence
                    seq_violations = violations.mean(dim=-1)
                else:
                    seq_violations = violations
                
                # Match dimensions
                if seq_violations.size(0) == preds.size(0):
                    # Penalize high-confidence normal predictions when violations exist
                    # This encourages the model to be uncertain when physics is violated
                    confidence_penalty = (1 - preds) * seq_violations
                    total_loss = total_loss + self.constraint_weight * confidence_penalty.mean()
        
        return total_loss
    
    def forward(
        self,
        graph_embedding: Tensor,
        normal_embedding: Tensor,
        auth_timestamps: Optional[Tensor] = None,
        predictions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass computing all constraint-related outputs.
        
        Convenience method that computes all constraint checks and returns
        them in a dictionary suitable for loss computation.
        
        Args:
            graph_embedding: Current graph embedding [batch, embed_dim]
            normal_embedding: Reference normal pattern embedding [embed_dim] or [batch, embed_dim]
            auth_timestamps: Optional authentication timestamps [batch, seq_len]
            predictions: Optional model predictions [batch]
            
        Returns:
            Dictionary containing:
            - 'energy': Energy values from compute_energy()
            - 'temporal_violations': Violation flags if auth_timestamps provided
            - 'constraint_loss': Total constraint loss if predictions provided
        """
        result = {}
        
        # Compute energy
        energy = self.compute_energy(graph_embedding, normal_embedding)
        result['energy'] = energy
        
        # Check temporal ordering if timestamps provided
        if auth_timestamps is not None:
            violations = self.check_temporal_ordering(auth_timestamps)
            result['temporal_violations'] = violations
        
        # Compute constraint loss if predictions provided
        if predictions is not None:
            constraints = {'energy': energy}
            if auth_timestamps is not None:
                constraints['temporal_violations'] = result['temporal_violations']
            
            constraint_loss = self.compute_constraint_loss(predictions, constraints)
            result['constraint_loss'] = constraint_loss
        
        return result
    
    def get_violation_descriptions(
        self,
        auth_timestamps: Tensor,
        entity_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> List[str]:
        """Get human-readable descriptions of constraint violations.
        
        Useful for generating interpretable alerts that explain why
        a detection was flagged.
        
        Args:
            auth_timestamps: Authentication timestamps [seq_len]
            entity_pairs: Optional list of (source, dest) entity pairs
                         corresponding to each timestamp
                         
        Returns:
            List of violation description strings
        """
        violations = self.check_temporal_ordering(auth_timestamps)
        descriptions = []
        
        if violations.dim() == 0 or violations.numel() == 0:
            return descriptions
        
        # Flatten if needed
        violations = violations.view(-1)
        timestamps = auth_timestamps.view(-1)
        
        for i, is_violation in enumerate(violations):
            if is_violation > 0.5:  # Threshold for binary violation
                time_delta = timestamps[i + 1] - timestamps[i]
                
                if entity_pairs and i + 1 < len(entity_pairs):
                    src, dst = entity_pairs[i + 1]
                    desc = (
                        f"Temporal violation: Authentication to {dst} from {src} "
                        f"occurred {time_delta:.3f}s after previous auth "
                        f"(minimum: {self.min_auth_delta}s)"
                    )
                else:
                    desc = (
                        f"Temporal violation at index {i + 1}: "
                        f"Time delta {time_delta:.3f}s < minimum {self.min_auth_delta}s"
                    )
                descriptions.append(desc)
        
        return descriptions
