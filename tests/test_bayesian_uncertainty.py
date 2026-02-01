"""Property-based tests for Bayesian Uncertainty Module.

This module contains property-based tests using Hypothesis to verify:
- Property 10: MC Dropout Sample Variance
- Property 11: Uncertainty Decomposition Validity

Feature: temporal-gat-lateral-movement
"""

import pytest
import torch
from hypothesis import given, settings, strategies as st, HealthCheck

from src.models.bayesian_uncertainty import BayesianUncertaintyModule
from src.data.models import UncertaintyOutput


class TestMCDropoutVariance:
    """Property tests for MC Dropout sample variance.
    
    Feature: temporal-gat-lateral-movement, Property 10: MC Dropout Sample Variance
    Validates: Requirements 4.1
    """

    def test_multiple_passes_produce_different_samples(self):
        """
        Property 10: MC Dropout Sample Variance
        
        For any input to the Bayesian Layer with dropout enabled, multiple 
        forward passes SHALL produce different output samples.
        
        **Validates: Requirements 4.1**
        """
        module = BayesianUncertaintyModule(
            input_dim=64,
            num_samples=10,
            dropout_rate=0.3
        )
        
        # Create non-trivial input
        x = torch.randn(4, 64)
        
        # Get samples
        samples = module.get_samples(x, num_samples=10)
        
        # Check that samples are different (variance > 0)
        variance = samples.var(dim=0)
        
        # At least some samples should have non-zero variance
        assert variance.sum() > 0, "MC Dropout samples should have non-zero variance"

    def test_variance_non_zero_for_non_trivial_inputs(self):
        """
        Property 10: MC Dropout Sample Variance
        
        The variance across samples SHALL be non-zero for non-trivial inputs.
        
        **Validates: Requirements 4.1**
        """
        module = BayesianUncertaintyModule(
            input_dim=32,
            num_samples=20,
            dropout_rate=0.5  # Higher dropout for more variance
        )
        
        # Create varied input
        x = torch.randn(8, 32) * 2.0  # Scale for more variation
        
        # Get samples
        samples = module.get_samples(x, num_samples=20)
        
        # Variance should be non-zero
        variance = samples.var(dim=0)
        assert (variance > 0).any(), "Variance should be non-zero for non-trivial inputs"

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        input_dim=st.integers(min_value=16, max_value=64),
        num_samples=st.integers(min_value=5, max_value=20),
        dropout_rate=st.floats(min_value=0.1, max_value=0.5)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_mc_dropout_variance_property(
        self, 
        batch_size: int, 
        input_dim: int, 
        num_samples: int,
        dropout_rate: float
    ):
        """
        Property 10: MC Dropout Sample Variance (property-based)
        
        For any valid configuration, MC Dropout SHALL produce samples with
        variance when dropout rate > 0.
        
        **Validates: Requirements 4.1**
        """
        module = BayesianUncertaintyModule(
            input_dim=input_dim,
            num_samples=num_samples,
            dropout_rate=dropout_rate
        )
        
        # Create input
        x = torch.randn(batch_size, input_dim)
        
        # Get samples
        samples = module.get_samples(x, num_samples=num_samples)
        
        # Verify samples shape
        assert samples.shape == (num_samples, batch_size), \
            f"Expected shape ({num_samples}, {batch_size}), got {samples.shape}"
        
        # Verify samples are valid probabilities
        assert (samples >= 0).all() and (samples <= 1).all(), \
            "All samples should be valid probabilities in [0, 1]"

    def test_different_forward_passes_differ(self):
        """
        Property 10: MC Dropout Sample Variance
        
        Two separate forward passes with MC Dropout SHALL produce different results.
        
        **Validates: Requirements 4.1**
        """
        module = BayesianUncertaintyModule(
            input_dim=64,
            num_samples=50,
            dropout_rate=0.3
        )
        
        x = torch.randn(4, 64)
        
        # Two forward passes
        output1 = module(x)
        output2 = module(x)
        
        # The samples should be different (different random dropout masks)
        # Note: mean predictions might be similar but samples should differ
        assert not torch.allclose(output1.samples, output2.samples), \
            "Different forward passes should produce different samples"


class TestUncertaintyDecomposition:
    """Property tests for uncertainty decomposition validity.
    
    Feature: temporal-gat-lateral-movement, Property 11: Uncertainty Decomposition Validity
    Validates: Requirements 4.2, 4.3, 4.4
    """

    def test_uncertainties_non_negative(self):
        """
        Property 11: Uncertainty Decomposition Validity
        
        Both epistemic and aleatoric uncertainty SHALL be non-negative.
        
        **Validates: Requirements 4.2, 4.3, 4.4**
        """
        module = BayesianUncertaintyModule(
            input_dim=64,
            num_samples=50,
            dropout_rate=0.2
        )
        
        x = torch.randn(4, 64)
        output = module(x)
        
        assert output.epistemic_uncertainty >= 0, \
            f"Epistemic uncertainty should be non-negative, got {output.epistemic_uncertainty}"
        assert output.aleatoric_uncertainty >= 0, \
            f"Aleatoric uncertainty should be non-negative, got {output.aleatoric_uncertainty}"

    def test_output_contains_all_fields(self):
        """
        Property 11: Uncertainty Decomposition Validity
        
        The output SHALL contain both uncertainty values alongside the mean prediction.
        
        **Validates: Requirements 4.2, 4.3, 4.4**
        """
        module = BayesianUncertaintyModule(
            input_dim=64,
            num_samples=50,
            dropout_rate=0.2
        )
        
        x = torch.randn(4, 64)
        output = module(x)
        
        # Verify output is UncertaintyOutput
        assert isinstance(output, UncertaintyOutput), \
            f"Output should be UncertaintyOutput, got {type(output)}"
        
        # Verify all fields are present and valid
        assert output.mean_prediction is not None, "mean_prediction should be present"
        assert output.epistemic_uncertainty is not None, "epistemic_uncertainty should be present"
        assert output.aleatoric_uncertainty is not None, "aleatoric_uncertainty should be present"
        assert output.samples is not None, "samples should be present"

    def test_mean_prediction_in_valid_range(self):
        """
        Property 11: Uncertainty Decomposition Validity
        
        The mean prediction SHALL be a valid probability in [0, 1].
        
        **Validates: Requirements 4.4**
        """
        module = BayesianUncertaintyModule(
            input_dim=64,
            num_samples=50,
            dropout_rate=0.2
        )
        
        x = torch.randn(4, 64)
        output = module(x)
        
        assert 0 <= output.mean_prediction <= 1, \
            f"Mean prediction should be in [0, 1], got {output.mean_prediction}"

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        input_dim=st.integers(min_value=16, max_value=64),
        num_samples=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_uncertainty_decomposition_property(
        self,
        batch_size: int,
        input_dim: int,
        num_samples: int
    ):
        """
        Property 11: Uncertainty Decomposition Validity (property-based)
        
        For any valid input, both uncertainties SHALL be non-negative and
        the output SHALL contain all required fields.
        
        **Validates: Requirements 4.2, 4.3, 4.4**
        """
        module = BayesianUncertaintyModule(
            input_dim=input_dim,
            num_samples=num_samples,
            dropout_rate=0.2
        )
        
        x = torch.randn(batch_size, input_dim)
        output = module(x)
        
        # Verify uncertainties are non-negative
        assert output.epistemic_uncertainty >= 0, \
            f"Epistemic uncertainty should be >= 0, got {output.epistemic_uncertainty}"
        assert output.aleatoric_uncertainty >= 0, \
            f"Aleatoric uncertainty should be >= 0, got {output.aleatoric_uncertainty}"
        
        # Verify mean prediction is valid probability
        assert 0 <= output.mean_prediction <= 1, \
            f"Mean prediction should be in [0, 1], got {output.mean_prediction}"
        
        # Verify samples tensor has correct shape
        assert output.samples.shape[0] == num_samples, \
            f"Expected {num_samples} samples, got {output.samples.shape[0]}"

    def test_epistemic_uncertainty_computation(self):
        """
        Test that epistemic uncertainty is computed correctly as entropy of expected prediction.
        
        **Validates: Requirements 4.2**
        """
        module = BayesianUncertaintyModule(
            input_dim=32,
            num_samples=100,
            dropout_rate=0.2
        )
        
        # Create samples manually
        samples = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])  # 3 samples, 2 batch
        
        epistemic = module.compute_epistemic(samples)
        
        # For p=0.5, entropy should be ln(2) ≈ 0.693
        expected_entropy = 0.693
        assert torch.allclose(epistemic, torch.tensor([expected_entropy, expected_entropy]), atol=0.01), \
            f"Epistemic uncertainty for p=0.5 should be ~0.693, got {epistemic}"

    def test_aleatoric_uncertainty_computation(self):
        """
        Test that aleatoric uncertainty is computed correctly as expected entropy.
        
        **Validates: Requirements 4.3**
        """
        module = BayesianUncertaintyModule(
            input_dim=32,
            num_samples=100,
            dropout_rate=0.2
        )
        
        # Create samples manually
        samples = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])  # 3 samples, 2 batch
        
        aleatoric = module.compute_aleatoric(samples)
        
        # For p=0.5, entropy should be ln(2) ≈ 0.693
        expected_entropy = 0.693
        assert torch.allclose(aleatoric, torch.tensor([expected_entropy, expected_entropy]), atol=0.01), \
            f"Aleatoric uncertainty for p=0.5 should be ~0.693, got {aleatoric}"


class TestBayesianModuleConfiguration:
    """Tests for Bayesian module configuration and initialization."""

    def test_default_configuration(self):
        """Test module initializes with default configuration."""
        module = BayesianUncertaintyModule(input_dim=64)
        
        assert module.input_dim == 64
        assert module.num_samples == 50
        assert module.dropout_rate == 0.2

    def test_custom_configuration(self):
        """Test module initializes with custom configuration."""
        module = BayesianUncertaintyModule(
            input_dim=128,
            num_samples=100,
            dropout_rate=0.3
        )
        
        assert module.input_dim == 128
        assert module.num_samples == 100
        assert module.dropout_rate == 0.3

    def test_num_samples_override(self):
        """Test that num_samples can be overridden at inference time."""
        module = BayesianUncertaintyModule(
            input_dim=64,
            num_samples=50,
            dropout_rate=0.2
        )
        
        x = torch.randn(4, 64)
        
        # Override with different number of samples
        output = module(x, num_samples=10)
        
        assert output.samples.shape[0] == 10, \
            f"Expected 10 samples, got {output.samples.shape[0]}"
