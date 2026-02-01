"""Tests for DetectionHead module.

Tests verify:
- Property 14: Detection Head Probability Output - output is always in [0, 1]
- Property 15: Class Weight Effect on Loss - changing weights changes loss value
- Requirement 6.1: Output probability of lateral movement
- Requirement 6.2: Support configurable decision thresholds
- Requirement 6.3: Handle class imbalance through weighted loss
"""

import pytest
import torch
from hypothesis import given, settings, strategies as st

from src.models.detection_head import DetectionHead


class TestDetectionHeadProbabilityOutput:
    """Tests for Property 14: Detection Head Probability Output."""
    
    def test_output_in_valid_range(self):
        """Test that output is always in [0, 1]."""
        head = DetectionHead(input_dim=64, hidden_dim=32)
        x = torch.randn(8, 64)
        
        probs = head(x)
        
        assert (probs >= 0).all(), "Output contains values < 0"
        assert (probs <= 1).all(), "Output contains values > 1"
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        head = DetectionHead(input_dim=128, hidden_dim=64)
        batch_size = 16
        x = torch.randn(batch_size, 128)
        
        probs = head(x)
        
        assert probs.shape == torch.Size([batch_size, 1])
    
    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        input_dim=st.integers(min_value=16, max_value=256),
    )
    @settings(max_examples=100, deadline=None)
    def test_output_always_valid_probability(self, batch_size, input_dim):
        """Property 14: For any valid input, output is in [0, 1]."""
        # Feature: temporal-gat-lateral-movement, Property 14: Detection Head Probability Output
        head = DetectionHead(input_dim=input_dim, hidden_dim=32)
        head.eval()
        
        x = torch.randn(batch_size, input_dim)
        probs = head(x)
        
        assert probs.shape == torch.Size([batch_size, 1])
        assert (probs >= 0).all() and (probs <= 1).all()
    
    def test_extreme_inputs(self):
        """Test with extreme input values."""
        head = DetectionHead(input_dim=64, hidden_dim=32)
        
        # Very large positive values
        x_large = torch.ones(4, 64) * 100
        probs_large = head(x_large)
        assert (probs_large >= 0).all() and (probs_large <= 1).all()
        
        # Very large negative values
        x_neg = torch.ones(4, 64) * -100
        probs_neg = head(x_neg)
        assert (probs_neg >= 0).all() and (probs_neg <= 1).all()


class TestClassWeightEffect:
    """Tests for Property 15: Class Weight Effect on Loss."""
    
    def test_weights_change_loss(self):
        """Test that changing weights changes loss value."""
        head = DetectionHead(input_dim=64, hidden_dim=32)
        x = torch.randn(8, 64)
        probs = head(x)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
        
        # Loss with weights [1, 1]
        weights1 = torch.tensor([1.0, 1.0])
        loss1 = head.compute_loss(probs, targets, class_weights=weights1)
        
        # Loss with weights [1, 10]
        weights2 = torch.tensor([1.0, 10.0])
        loss2 = head.compute_loss(probs, targets, class_weights=weights2)
        
        # Losses should be different
        assert loss1.item() != loss2.item(), "Changing weights should change loss"
    
    @given(
        batch_size=st.integers(min_value=2, max_value=32),
        input_dim=st.integers(min_value=16, max_value=128),
        weight_scale=st.floats(min_value=1.5, max_value=20.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_class_weight_effect_property(self, batch_size, input_dim, weight_scale):
        """Property 15: For any prediction/target pair, changing weights changes loss.
        
        Feature: temporal-gat-lateral-movement, Property 15: Class Weight Effect on Loss
        
        This property verifies that changing class weights changes the computed loss
        value (unless the prediction perfectly matches the target).
        """
        head = DetectionHead(input_dim=input_dim, hidden_dim=32)
        head.eval()
        
        # Generate random predictions (simulating model output)
        # Use values away from 0 and 1 to avoid perfect predictions
        torch.manual_seed(batch_size * input_dim)  # Reproducible but varied
        probs = torch.sigmoid(torch.randn(batch_size, 1)) * 0.8 + 0.1  # Range [0.1, 0.9]
        
        # Generate mixed targets (both classes present)
        targets = torch.zeros(batch_size, dtype=torch.float32)
        targets[::2] = 1.0  # Alternate between 0 and 1
        
        # Compute loss with equal weights
        weights1 = torch.tensor([1.0, 1.0])
        loss1 = head.compute_loss(probs, targets, class_weights=weights1)
        
        # Compute loss with different weights
        weights2 = torch.tensor([1.0, weight_scale])
        loss2 = head.compute_loss(probs, targets, class_weights=weights2)
        
        # Losses should be different when weights are different
        assert not torch.isnan(loss1), "Loss1 should not be NaN"
        assert not torch.isnan(loss2), "Loss2 should not be NaN"
        assert loss1.item() >= 0, "Loss1 should be non-negative"
        assert loss2.item() >= 0, "Loss2 should be non-negative"
        
        # The key property: different weights produce different loss
        # (unless predictions perfectly match targets, which is extremely unlikely)
        assert abs(loss1.item() - loss2.item()) > 1e-6, \
            f"Changing weights should change loss: loss1={loss1.item()}, loss2={loss2.item()}"
    
    def test_higher_positive_weight_increases_loss_for_misclassified_positives(self):
        """Test that higher positive class weight increases loss for FN."""
        head = DetectionHead(input_dim=64, hidden_dim=32)
        
        # Create predictions that are all low (predicting negative)
        probs = torch.tensor([[0.1], [0.1], [0.1], [0.1]])
        # But targets are all positive
        targets = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        
        weights_low = torch.tensor([1.0, 1.0])
        weights_high = torch.tensor([1.0, 10.0])
        
        loss_low = head.compute_loss(probs, targets, class_weights=weights_low)
        loss_high = head.compute_loss(probs, targets, class_weights=weights_high)
        
        # Higher weight for positive class should increase loss for FN
        assert loss_high.item() > loss_low.item()
    
    def test_loss_without_weights(self):
        """Test loss computation without class weights."""
        head = DetectionHead(input_dim=64, hidden_dim=32)
        x = torch.randn(4, 64)
        probs = head(x)
        targets = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        
        loss = head.compute_loss(probs, targets)
        
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"


class TestThresholdConfiguration:
    """Tests for Requirement 6.2: Configurable decision thresholds."""
    
    def test_set_threshold(self):
        """Test setting decision threshold."""
        head = DetectionHead(input_dim=64, threshold=0.5)
        
        head.set_threshold(0.3)
        assert head.get_threshold() == 0.3
        
        head.set_threshold(0.7)
        assert head.get_threshold() == 0.7
    
    def test_threshold_affects_predictions(self):
        """Test that threshold affects binary predictions."""
        head = DetectionHead(input_dim=64, hidden_dim=32)
        
        # Create input that produces probability around 0.5
        torch.manual_seed(42)
        x = torch.randn(10, 64)
        probs = head(x)
        
        # With low threshold, more predictions should be positive
        head.set_threshold(0.3)
        preds_low = head.predict(x)
        
        # With high threshold, fewer predictions should be positive
        head.set_threshold(0.7)
        preds_high = head.predict(x)
        
        # Low threshold should have >= positive predictions than high threshold
        assert preds_low.sum() >= preds_high.sum()
    
    def test_invalid_threshold_raises_error(self):
        """Test that invalid threshold raises ValueError."""
        head = DetectionHead(input_dim=64)
        
        with pytest.raises(ValueError):
            head.set_threshold(-0.1)
        
        with pytest.raises(ValueError):
            head.set_threshold(1.1)
    
    def test_predict_returns_binary(self):
        """Test that predict returns binary values."""
        head = DetectionHead(input_dim=64, hidden_dim=32)
        x = torch.randn(8, 64)
        
        preds = head.predict(x)
        
        assert ((preds == 0) | (preds == 1)).all()


class TestClassWeightConfiguration:
    """Tests for class weight configuration."""
    
    def test_set_class_weights(self):
        """Test setting class weights."""
        head = DetectionHead(input_dim=64)
        weights = torch.tensor([1.0, 5.0])
        
        head.set_class_weights(weights)
        
        stored_weights = head.get_class_weights()
        assert stored_weights is not None
        assert torch.allclose(stored_weights, weights)
    
    def test_invalid_class_weights_raises_error(self):
        """Test that invalid class weights raise ValueError."""
        head = DetectionHead(input_dim=64)
        
        with pytest.raises(ValueError):
            head.set_class_weights(torch.tensor([1.0]))  # Wrong shape
        
        with pytest.raises(ValueError):
            head.set_class_weights(torch.tensor([1.0, 2.0, 3.0]))  # Wrong shape
    
    def test_class_weights_in_constructor(self):
        """Test passing class weights in constructor."""
        weights = torch.tensor([1.0, 10.0])
        head = DetectionHead(input_dim=64, class_weights=weights)
        
        stored_weights = head.get_class_weights()
        assert stored_weights is not None
        assert torch.allclose(stored_weights, weights)
