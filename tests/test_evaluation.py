"""Tests for EvaluationModule.

Tests the evaluation module functionality including:
- Detection metrics computation (precision, recall, F1, AUC-ROC, AUC-PR)
- Calibration metrics computation (ECE, Brier score, NLL)
- Reliability diagram generation
- Statistical significance testing
"""

import pytest
import numpy as np
import torch
from hypothesis import given, strategies as st, settings

from src.models.evaluation import EvaluationModule, SignificanceResult
from src.data.models import DetectionMetrics, CalibrationMetrics


class TestDetectionMetrics:
    """Tests for detection metrics computation."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        evaluator = EvaluationModule()
        
        predictions = torch.tensor([0.9, 0.8, 0.1, 0.2])
        targets = torch.tensor([1, 1, 0, 0])
        
        metrics = evaluator.compute_detection_metrics(predictions, targets)
        
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.auc_roc == 1.0
    
    def test_all_wrong_predictions(self):
        """Test metrics with completely wrong predictions."""
        evaluator = EvaluationModule()
        
        predictions = torch.tensor([0.1, 0.2, 0.9, 0.8])
        targets = torch.tensor([1, 1, 0, 0])
        
        metrics = evaluator.compute_detection_metrics(predictions, targets)
        
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
    
    def test_f1_formula_correctness(self):
        """Test that F1 = 2 * precision * recall / (precision + recall)."""
        evaluator = EvaluationModule()
        
        # Create predictions that give known precision/recall
        predictions = torch.tensor([0.9, 0.8, 0.6, 0.4, 0.3, 0.2])
        targets = torch.tensor([1, 1, 0, 1, 0, 0])
        
        metrics = evaluator.compute_detection_metrics(predictions, targets)
        
        if metrics.precision + metrics.recall > 0:
            expected_f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            assert abs(metrics.f1_score - expected_f1) < 1e-6
    
    def test_auc_roc_bounds(self):
        """Test that AUC-ROC is in [0, 1]."""
        evaluator = EvaluationModule()
        
        predictions = torch.rand(100)
        targets = torch.randint(0, 2, (100,))
        
        metrics = evaluator.compute_detection_metrics(predictions, targets)
        
        assert 0.0 <= metrics.auc_roc <= 1.0
    
    def test_auc_pr_bounds(self):
        """Test that AUC-PR is in [0, 1]."""
        evaluator = EvaluationModule()
        
        predictions = torch.rand(100)
        targets = torch.randint(0, 2, (100,))
        
        metrics = evaluator.compute_detection_metrics(predictions, targets)
        
        assert 0.0 <= metrics.auc_pr <= 1.0
    
    def test_empty_predictions(self):
        """Test handling of empty predictions."""
        evaluator = EvaluationModule()
        
        predictions = torch.tensor([])
        targets = torch.tensor([])
        
        metrics = evaluator.compute_detection_metrics(predictions, targets)
        
        # Should handle gracefully without errors
        assert isinstance(metrics, DetectionMetrics)
    
    def test_all_positive_targets(self):
        """Test with all positive targets."""
        evaluator = EvaluationModule()
        
        predictions = torch.tensor([0.9, 0.8, 0.7])
        targets = torch.tensor([1, 1, 1])
        
        metrics = evaluator.compute_detection_metrics(predictions, targets)
        
        assert metrics.recall == 1.0
        assert metrics.auc_roc == 0.5  # No negatives to discriminate
    
    def test_all_negative_targets(self):
        """Test with all negative targets."""
        evaluator = EvaluationModule()
        
        predictions = torch.tensor([0.1, 0.2, 0.3])
        targets = torch.tensor([0, 0, 0])
        
        metrics = evaluator.compute_detection_metrics(predictions, targets)
        
        assert metrics.precision == 0.0  # No true positives possible
        assert metrics.auc_roc == 0.5  # No positives to discriminate


class TestCalibrationMetrics:
    """Tests for calibration metrics computation."""
    
    def test_perfect_calibration(self):
        """Test ECE with perfectly calibrated predictions."""
        evaluator = EvaluationModule()
        
        # Perfectly calibrated: 80% confidence, 80% accuracy
        predictions = torch.tensor([0.8] * 10)
        targets = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        
        metrics = evaluator.compute_calibration_metrics(predictions, targets)
        
        # ECE should be close to 0 for perfect calibration
        assert metrics.ece < 0.1
    
    def test_ece_non_negative(self):
        """Test that ECE is always non-negative."""
        evaluator = EvaluationModule()
        
        predictions = torch.rand(100)
        targets = torch.randint(0, 2, (100,))
        
        metrics = evaluator.compute_calibration_metrics(predictions, targets)
        
        assert metrics.ece >= 0.0
    
    def test_brier_score_bounds(self):
        """Test that Brier score is in [0, 1]."""
        evaluator = EvaluationModule()
        
        predictions = torch.rand(100)
        targets = torch.randint(0, 2, (100,)).float()
        
        metrics = evaluator.compute_calibration_metrics(predictions, targets)
        
        assert 0.0 <= metrics.brier_score <= 1.0
    
    def test_brier_score_perfect(self):
        """Test Brier score with perfect predictions."""
        evaluator = EvaluationModule()
        
        predictions = torch.tensor([1.0, 1.0, 0.0, 0.0])
        targets = torch.tensor([1, 1, 0, 0])
        
        metrics = evaluator.compute_calibration_metrics(predictions, targets)
        
        assert metrics.brier_score < 0.01  # Should be very close to 0
    
    def test_nll_positive(self):
        """Test that NLL is positive for non-perfect predictions."""
        evaluator = EvaluationModule()
        
        predictions = torch.tensor([0.7, 0.6, 0.3, 0.4])
        targets = torch.tensor([1, 1, 0, 0])
        
        metrics = evaluator.compute_calibration_metrics(predictions, targets)
        
        assert metrics.nll > 0.0


class TestReliabilityDiagram:
    """Tests for reliability diagram generation."""
    
    def test_diagram_structure(self):
        """Test that reliability diagram returns correct structure."""
        evaluator = EvaluationModule()
        
        predictions = torch.rand(100)
        targets = torch.randint(0, 2, (100,))
        
        diagram = evaluator.generate_reliability_diagram(predictions, targets)
        
        assert "bin_centers" in diagram
        assert "bin_accuracies" in diagram
        assert "bin_confidences" in diagram
        assert "bin_counts" in diagram
        assert "perfect_calibration" in diagram
        assert "num_bins" in diagram
    
    def test_bin_count_consistency(self):
        """Test that bin counts sum to total samples."""
        evaluator = EvaluationModule()
        
        n_samples = 100
        predictions = torch.rand(n_samples)
        targets = torch.randint(0, 2, (n_samples,))
        
        diagram = evaluator.generate_reliability_diagram(predictions, targets)
        
        assert sum(diagram["bin_counts"]) == n_samples
    
    def test_custom_num_bins(self):
        """Test reliability diagram with custom number of bins."""
        evaluator = EvaluationModule()
        
        predictions = torch.rand(100)
        targets = torch.randint(0, 2, (100,))
        
        diagram = evaluator.generate_reliability_diagram(predictions, targets, num_bins=5)
        
        assert len(diagram["bin_centers"]) == 5
        assert diagram["num_bins"] == 5


class TestStatisticalSignificance:
    """Tests for statistical significance testing."""
    
    def test_identical_results(self):
        """Test significance test with identical results."""
        evaluator = EvaluationModule()
        
        results_a = [0.8, 0.82, 0.79, 0.81, 0.80]
        results_b = [0.8, 0.82, 0.79, 0.81, 0.80]
        
        result = evaluator.statistical_significance_test(results_a, results_b)
        
        assert isinstance(result, SignificanceResult)
        assert result.p_value >= 0.05  # Should not be significant
        assert not result.is_significant
    
    def test_clearly_different_results(self):
        """Test significance test with clearly different results."""
        evaluator = EvaluationModule()
        
        results_a = [0.9, 0.91, 0.89, 0.92, 0.90, 0.88, 0.91, 0.89, 0.90, 0.91]
        results_b = [0.5, 0.51, 0.49, 0.52, 0.50, 0.48, 0.51, 0.49, 0.50, 0.51]
        
        result = evaluator.statistical_significance_test(results_a, results_b)
        
        assert result.is_significant
        assert result.p_value < 0.05
    
    def test_confidence_interval_contains_mean(self):
        """Test that CI contains the mean difference."""
        evaluator = EvaluationModule()
        
        results_a = [0.8, 0.82, 0.79, 0.81, 0.80]
        results_b = [0.7, 0.72, 0.69, 0.71, 0.70]
        
        result = evaluator.statistical_significance_test(results_a, results_b)
        
        mean_diff = np.mean(results_a) - np.mean(results_b)
        
        # CI should contain the mean difference (with some tolerance for bootstrap variance)
        assert result.ci_lower <= mean_diff + 0.05
        assert result.ci_upper >= mean_diff - 0.05
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched array lengths raise an error."""
        evaluator = EvaluationModule()
        
        results_a = [0.8, 0.82, 0.79]
        results_b = [0.7, 0.72]
        
        with pytest.raises(ValueError):
            evaluator.statistical_significance_test(results_a, results_b)
    
    def test_single_sample(self):
        """Test handling of single sample."""
        evaluator = EvaluationModule()
        
        results_a = [0.8]
        results_b = [0.7]
        
        result = evaluator.statistical_significance_test(results_a, results_b)
        
        # Should handle gracefully
        assert isinstance(result, SignificanceResult)


class TestConfusionMatrix:
    """Tests for confusion matrix computation."""
    
    def test_confusion_matrix_elements(self):
        """Test confusion matrix element computation."""
        evaluator = EvaluationModule()
        
        predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        targets = torch.tensor([1, 0, 1, 0])
        
        cm = evaluator.compute_confusion_matrix(predictions, targets)
        
        assert cm["true_positives"] == 1
        assert cm["false_positives"] == 1
        assert cm["false_negatives"] == 1
        assert cm["true_negatives"] == 1
    
    def test_confusion_matrix_sum(self):
        """Test that confusion matrix elements sum to total samples."""
        evaluator = EvaluationModule()
        
        n_samples = 100
        predictions = torch.rand(n_samples)
        targets = torch.randint(0, 2, (n_samples,))
        
        cm = evaluator.compute_confusion_matrix(predictions, targets)
        
        total = cm["true_positives"] + cm["false_positives"] + cm["true_negatives"] + cm["false_negatives"]
        assert total == n_samples


class TestDetectionLatency:
    """Tests for detection latency computation."""
    
    def test_zero_latency(self):
        """Test latency when detection is immediate."""
        evaluator = EvaluationModule()
        
        attack_ts = [100.0, 200.0, 300.0]
        detect_ts = [100.0, 200.0, 300.0]
        
        latency = evaluator.compute_detection_latency(attack_ts, detect_ts)
        
        assert latency == 0.0
    
    def test_positive_latency(self):
        """Test latency computation with delays."""
        evaluator = EvaluationModule()
        
        attack_ts = [100.0, 200.0]
        detect_ts = [160.0, 260.0]  # 60 seconds delay each
        
        latency = evaluator.compute_detection_latency(attack_ts, detect_ts)
        
        assert latency == 1.0  # 60 seconds = 1 minute
    
    def test_empty_timestamps(self):
        """Test handling of empty timestamp lists."""
        evaluator = EvaluationModule()
        
        latency = evaluator.compute_detection_latency([], [])
        
        assert latency == 0.0



# Property-Based Tests
# Feature: temporal-gat-lateral-movement, Property 20: Metrics Computation Correctness

class TestMetricsComputationProperty:
    """Property-based tests for metrics computation correctness.
    
    Property 20: Metrics Computation Correctness
    Verify F1 = 2*precision*recall/(precision+recall), AUC in [0,1], ECE >= 0
    **Validates: Requirements 10.1, 10.2**
    """
    
    @given(
        predictions=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=200
        ),
        targets=st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=10,
            max_size=200
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_f1_formula_property(self, predictions, targets):
        """Property: F1 = 2 * precision * recall / (precision + recall).
        
        For any set of predictions and targets, the computed F1 score
        must satisfy the standard F1 formula.
        """
        # Ensure same length
        min_len = min(len(predictions), len(targets))
        if min_len < 2:
            return  # Skip trivial cases
        
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        
        evaluator = EvaluationModule()
        preds_tensor = torch.tensor(predictions)
        targs_tensor = torch.tensor(targets)
        
        metrics = evaluator.compute_detection_metrics(preds_tensor, targs_tensor)
        
        # Verify F1 formula
        if metrics.precision + metrics.recall > 0:
            expected_f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            assert abs(metrics.f1_score - expected_f1) < 1e-6, \
                f"F1 mismatch: got {metrics.f1_score}, expected {expected_f1}"
        else:
            assert metrics.f1_score == 0.0, "F1 should be 0 when precision + recall = 0"
    
    @given(
        predictions=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=200
        ),
        targets=st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=10,
            max_size=200
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_auc_bounds_property(self, predictions, targets):
        """Property: AUC-ROC and AUC-PR must be in [0, 1].
        
        For any set of predictions and targets, both AUC metrics
        must be valid probabilities in the range [0, 1].
        """
        # Ensure same length
        min_len = min(len(predictions), len(targets))
        if min_len < 2:
            return  # Skip trivial cases
        
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        
        evaluator = EvaluationModule()
        preds_tensor = torch.tensor(predictions)
        targs_tensor = torch.tensor(targets)
        
        metrics = evaluator.compute_detection_metrics(preds_tensor, targs_tensor)
        
        # Verify AUC-ROC bounds
        assert 0.0 <= metrics.auc_roc <= 1.0, \
            f"AUC-ROC out of bounds: {metrics.auc_roc}"
        
        # Verify AUC-PR bounds
        assert 0.0 <= metrics.auc_pr <= 1.0, \
            f"AUC-PR out of bounds: {metrics.auc_pr}"
    
    @given(
        probabilities=st.lists(
            st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=200
        ),
        targets=st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=10,
            max_size=200
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_ece_non_negative_property(self, probabilities, targets):
        """Property: ECE must be non-negative.
        
        For any set of probabilities and targets, the Expected Calibration
        Error must be >= 0.
        """
        # Ensure same length
        min_len = min(len(probabilities), len(targets))
        if min_len < 2:
            return  # Skip trivial cases
        
        probabilities = probabilities[:min_len]
        targets = targets[:min_len]
        
        evaluator = EvaluationModule()
        probs_tensor = torch.tensor(probabilities)
        targs_tensor = torch.tensor(targets)
        
        metrics = evaluator.compute_calibration_metrics(probs_tensor, targs_tensor)
        
        # Verify ECE is non-negative
        assert metrics.ece >= 0.0, f"ECE is negative: {metrics.ece}"
    
    @given(
        probabilities=st.lists(
            st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=200
        ),
        targets=st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=10,
            max_size=200
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_brier_score_bounds_property(self, probabilities, targets):
        """Property: Brier score must be in [0, 1].
        
        For any set of probabilities and targets, the Brier score
        must be a valid value in [0, 1].
        """
        # Ensure same length
        min_len = min(len(probabilities), len(targets))
        if min_len < 2:
            return  # Skip trivial cases
        
        probabilities = probabilities[:min_len]
        targets = targets[:min_len]
        
        evaluator = EvaluationModule()
        probs_tensor = torch.tensor(probabilities)
        targs_tensor = torch.tensor(targets)
        
        metrics = evaluator.compute_calibration_metrics(probs_tensor, targs_tensor)
        
        # Verify Brier score bounds
        assert 0.0 <= metrics.brier_score <= 1.0, \
            f"Brier score out of bounds: {metrics.brier_score}"
