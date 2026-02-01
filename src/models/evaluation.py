"""Evaluation module for T-GAT lateral movement detection.

This module implements the EvaluationModule class that provides:
- Detection metrics computation (precision, recall, F1, AUC-ROC, AUC-PR)
- Calibration metrics computation (ECE, Brier score, NLL)
- Reliability diagram generation
- Statistical significance testing with bootstrap confidence intervals
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch import Tensor
from scipy import stats

from src.data.models import DetectionMetrics, CalibrationMetrics


@dataclass
class SignificanceResult:
    """Result of statistical significance test.
    
    Attributes:
        statistic: Test statistic value
        p_value: P-value of the test
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        is_significant: Whether the difference is statistically significant
        method: Name of the statistical test used
    """
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    method: str


class EvaluationModule:
    """Comprehensive evaluation module for T-GAT model assessment.
    
    Provides methods for computing detection metrics, calibration metrics,
    generating reliability diagrams, and performing statistical significance tests.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize evaluation module.
        
        Args:
            significance_level: Alpha level for significance tests (default: 0.05)
        """
        self.significance_level = significance_level
    
    def compute_detection_metrics(
        self,
        predictions: Tensor,
        targets: Tensor,
        threshold: float = 0.5
    ) -> DetectionMetrics:
        """Compute detection performance metrics.
        
        Computes precision, recall, F1 score, AUC-ROC, and AUC-PR.
        Property 20: F1 = 2 * (precision * recall) / (precision + recall),
        AUC in [0, 1].
        
        Args:
            predictions: Predicted probabilities [N]
            targets: Ground truth binary labels [N]
            threshold: Decision threshold for binary classification
            
        Returns:
            DetectionMetrics containing all computed metrics
        """
        # Convert to numpy for sklearn-like computations
        if isinstance(predictions, Tensor):
            preds = predictions.detach().cpu().numpy()
        else:
            preds = np.array(predictions)
            
        if isinstance(targets, Tensor):
            targs = targets.detach().cpu().numpy()
        else:
            targs = np.array(targets)
        
        # Ensure 1D arrays
        preds = preds.flatten()
        targs = targs.flatten()
        
        # Binary predictions
        binary_preds = (preds >= threshold).astype(int)
        
        # Compute confusion matrix elements
        tp = np.sum((binary_preds == 1) & (targs == 1))
        fp = np.sum((binary_preds == 1) & (targs == 0))
        fn = np.sum((binary_preds == 0) & (targs == 1))
        tn = np.sum((binary_preds == 0) & (targs == 0))
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # AUC-ROC
        auc_roc = self._compute_auc_roc(preds, targs)
        
        # AUC-PR
        auc_pr = self._compute_auc_pr(preds, targs)
        
        # FPR at 90% recall
        fpr_at_90_recall = self._compute_fpr_at_recall(preds, targs, target_recall=0.9)
        
        return DetectionMetrics(
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1_score),
            auc_roc=float(auc_roc),
            auc_pr=float(auc_pr),
            fpr_at_90_recall=float(fpr_at_90_recall),
            detection_latency_minutes=0.0  # Computed separately if needed
        )
    
    def _compute_auc_roc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Area Under ROC Curve.
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth labels
            
        Returns:
            AUC-ROC score in [0, 1]
        """
        # Handle edge cases
        if len(np.unique(targets)) < 2:
            return 0.5  # No positive or no negative samples
        
        # Sort by predictions descending
        sorted_indices = np.argsort(-predictions)
        sorted_targets = targets[sorted_indices]
        
        # Compute TPR and FPR at each threshold
        n_pos = np.sum(targets == 1)
        n_neg = np.sum(targets == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tpr_list = []
        fpr_list = []
        
        tp = 0
        fp = 0
        
        for i, label in enumerate(sorted_targets):
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)
        
        # Add origin point
        tpr_list = [0.0] + tpr_list
        fpr_list = [0.0] + fpr_list
        
        # Compute AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
        
        return np.clip(auc, 0.0, 1.0)
    
    def _compute_auc_pr(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Area Under Precision-Recall Curve.
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth labels
            
        Returns:
            AUC-PR score in [0, 1]
        """
        # Handle edge cases
        if len(np.unique(targets)) < 2:
            return np.mean(targets)  # Baseline
        
        n_pos = np.sum(targets == 1)
        if n_pos == 0:
            return 0.0
        
        # Sort by predictions descending
        sorted_indices = np.argsort(-predictions)
        sorted_targets = targets[sorted_indices]
        
        precision_list = []
        recall_list = []
        
        tp = 0
        fp = 0
        
        for i, label in enumerate(sorted_targets):
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / n_pos
            
            precision_list.append(precision)
            recall_list.append(recall)
        
        # Add starting point (recall=0, precision=1)
        recall_list = [0.0] + recall_list
        precision_list = [1.0] + precision_list
        
        # Compute AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(recall_list)):
            auc += (recall_list[i] - recall_list[i-1]) * (precision_list[i] + precision_list[i-1]) / 2
        
        return np.clip(auc, 0.0, 1.0)
    
    def _compute_fpr_at_recall(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        target_recall: float = 0.9
    ) -> float:
        """Compute False Positive Rate at a target recall level.
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth labels
            target_recall: Target recall level (default: 0.9)
            
        Returns:
            FPR at the target recall level
        """
        n_pos = np.sum(targets == 1)
        n_neg = np.sum(targets == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.0
        
        # Sort by predictions descending
        sorted_indices = np.argsort(-predictions)
        sorted_targets = targets[sorted_indices]
        
        tp = 0
        fp = 0
        
        for label in sorted_targets:
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            recall = tp / n_pos
            if recall >= target_recall:
                fpr = fp / n_neg
                return fpr
        
        return 1.0  # If target recall not achieved

    
    def compute_calibration_metrics(
        self,
        probabilities: Tensor,
        targets: Tensor,
        num_bins: int = 10
    ) -> CalibrationMetrics:
        """Compute calibration metrics.
        
        Computes Expected Calibration Error (ECE), Brier score, and
        Negative Log-Likelihood (NLL).
        Property 20: ECE >= 0.
        
        Args:
            probabilities: Predicted probabilities [N]
            targets: Ground truth binary labels [N]
            num_bins: Number of bins for ECE computation
            
        Returns:
            CalibrationMetrics containing ECE, Brier score, and NLL
        """
        # Convert to numpy
        if isinstance(probabilities, Tensor):
            probs = probabilities.detach().cpu().numpy()
        else:
            probs = np.array(probabilities)
            
        if isinstance(targets, Tensor):
            targs = targets.detach().cpu().numpy()
        else:
            targs = np.array(targets)
        
        # Ensure 1D arrays
        probs = probs.flatten()
        targs = targs.flatten()
        
        # Clip probabilities to avoid numerical issues
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        
        # ECE
        ece = self._compute_ece(probs, targs, num_bins)
        
        # Brier Score
        brier_score = np.mean((probs - targs) ** 2)
        
        # Negative Log-Likelihood
        nll = -np.mean(targs * np.log(probs) + (1 - targs) * np.log(1 - probs))
        
        return CalibrationMetrics(
            ece=float(ece),
            brier_score=float(brier_score),
            nll=float(nll)
        )
    
    def _compute_ece(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error.
        
        ECE measures the difference between predicted confidence and
        actual accuracy across probability bins.
        
        Args:
            probabilities: Predicted probabilities
            targets: Ground truth labels
            num_bins: Number of bins
            
        Returns:
            ECE value (non-negative)
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        n_samples = len(probabilities)
        
        if n_samples == 0:
            return 0.0
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin
            if i == num_bins - 1:
                # Include upper boundary for last bin
                in_bin = (probabilities >= bin_lower) & (probabilities <= bin_upper)
            else:
                in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
            
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                # Average confidence in bin
                avg_confidence = np.mean(probabilities[in_bin])
                # Average accuracy in bin
                avg_accuracy = np.mean(targets[in_bin])
                # Weighted contribution to ECE
                ece += (bin_size / n_samples) * np.abs(avg_accuracy - avg_confidence)
        
        return ece
    
    def generate_reliability_diagram(
        self,
        probabilities: Tensor,
        targets: Tensor,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """Generate data for reliability diagram visualization.
        
        Returns bin-level statistics for plotting a reliability diagram
        that shows calibration quality.
        
        Args:
            probabilities: Predicted probabilities [N]
            targets: Ground truth binary labels [N]
            num_bins: Number of bins for the diagram
            
        Returns:
            Dictionary containing:
            - bin_centers: Center of each bin
            - bin_accuracies: Actual accuracy in each bin
            - bin_confidences: Average confidence in each bin
            - bin_counts: Number of samples in each bin
            - perfect_calibration: Reference line for perfect calibration
        """
        # Convert to numpy
        if isinstance(probabilities, Tensor):
            probs = probabilities.detach().cpu().numpy()
        else:
            probs = np.array(probabilities)
            
        if isinstance(targets, Tensor):
            targs = targets.detach().cpu().numpy()
        else:
            targs = np.array(targets)
        
        # Ensure 1D arrays
        probs = probs.flatten()
        targs = targs.flatten()
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            bin_center = (bin_lower + bin_upper) / 2
            
            # Find samples in this bin
            if i == num_bins - 1:
                in_bin = (probs >= bin_lower) & (probs <= bin_upper)
            else:
                in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
            bin_size = np.sum(in_bin)
            bin_counts.append(int(bin_size))
            bin_centers.append(float(bin_center))
            
            if bin_size > 0:
                bin_accuracies.append(float(np.mean(targs[in_bin])))
                bin_confidences.append(float(np.mean(probs[in_bin])))
            else:
                bin_accuracies.append(None)
                bin_confidences.append(None)
        
        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "perfect_calibration": [0.0, 1.0],  # Line from (0,0) to (1,1)
            "num_bins": num_bins
        }
    
    def statistical_significance_test(
        self,
        results_a: List[float],
        results_b: List[float],
        n_bootstrap: int = 1000
    ) -> SignificanceResult:
        """Perform statistical significance test with bootstrap confidence intervals.
        
        Uses paired t-test for significance and bootstrap for confidence intervals.
        
        Args:
            results_a: Performance metrics from model A
            results_b: Performance metrics from model B
            n_bootstrap: Number of bootstrap samples for CI estimation
            
        Returns:
            SignificanceResult with test statistics and confidence intervals
        """
        results_a = np.array(results_a)
        results_b = np.array(results_b)
        
        if len(results_a) != len(results_b):
            raise ValueError("Results arrays must have the same length for paired test")
        
        if len(results_a) < 2:
            return SignificanceResult(
                statistic=0.0,
                p_value=1.0,
                ci_lower=0.0,
                ci_upper=0.0,
                is_significant=False,
                method="paired_t_test"
            )
        
        # Paired t-test
        differences = results_a - results_b
        t_stat, p_value = stats.ttest_rel(results_a, results_b)
        
        # Handle NaN from t-test (e.g., when all differences are zero)
        if np.isnan(t_stat):
            t_stat = 0.0
        if np.isnan(p_value):
            p_value = 1.0
        
        # Bootstrap confidence interval for the mean difference
        bootstrap_diffs = []
        n_samples = len(differences)
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = differences[indices]
            bootstrap_diffs.append(np.mean(bootstrap_sample))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
        ci_upper = float(np.percentile(bootstrap_diffs, 97.5))
        
        is_significant = p_value < self.significance_level
        
        return SignificanceResult(
            statistic=float(t_stat),
            p_value=float(p_value),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_significant=is_significant,
            method="paired_t_test"
        )
    
    def compute_detection_latency(
        self,
        attack_timestamps: List[float],
        detection_timestamps: List[float]
    ) -> float:
        """Compute average detection latency in minutes.
        
        Args:
            attack_timestamps: Timestamps when attacks started
            detection_timestamps: Timestamps when attacks were detected
            
        Returns:
            Average detection latency in minutes
        """
        if not attack_timestamps or not detection_timestamps:
            return 0.0
        
        latencies = []
        for attack_ts, detect_ts in zip(attack_timestamps, detection_timestamps):
            if detect_ts >= attack_ts:
                latency_seconds = detect_ts - attack_ts
                latencies.append(latency_seconds / 60.0)  # Convert to minutes
        
        if not latencies:
            return 0.0
        
        return float(np.mean(latencies))
    
    def compute_confusion_matrix(
        self,
        predictions: Tensor,
        targets: Tensor,
        threshold: float = 0.5
    ) -> Dict[str, int]:
        """Compute confusion matrix elements.
        
        Args:
            predictions: Predicted probabilities [N]
            targets: Ground truth binary labels [N]
            threshold: Decision threshold
            
        Returns:
            Dictionary with TP, FP, TN, FN counts
        """
        if isinstance(predictions, Tensor):
            preds = predictions.detach().cpu().numpy()
        else:
            preds = np.array(predictions)
            
        if isinstance(targets, Tensor):
            targs = targets.detach().cpu().numpy()
        else:
            targs = np.array(targets)
        
        preds = preds.flatten()
        targs = targs.flatten()
        
        binary_preds = (preds >= threshold).astype(int)
        
        tp = int(np.sum((binary_preds == 1) & (targs == 1)))
        fp = int(np.sum((binary_preds == 1) & (targs == 0)))
        fn = int(np.sum((binary_preds == 0) & (targs == 1)))
        tn = int(np.sum((binary_preds == 0) & (targs == 0)))
        
        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }
