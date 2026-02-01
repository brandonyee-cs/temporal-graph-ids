"""Core data models for T-GAT lateral movement detection.

This module defines the fundamental data structures used throughout the system:
- AuthenticationEvent: Represents a single authentication attempt
- TimeWindow: Groups events within a temporal window
- UncertaintyOutput: Contains prediction with uncertainty estimates
- DetectionResult: Final detection output with all alert fields
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
import torch
from torch import Tensor


@dataclass
class AuthenticationEvent:
    """Represents a single authentication event from the LANL dataset.
    
    Attributes:
        timestamp: Unix timestamp of the authentication event
        source_user: User account initiating the authentication
        source_computer: Machine from which authentication originates
        dest_computer: Target machine for authentication
        auth_type: Authentication protocol used (e.g., Kerberos, NTLM)
        logon_type: Type of logon (interactive, network, batch, etc.)
        success: Whether the authentication attempt succeeded
    """
    timestamp: float
    source_user: str
    source_computer: str
    dest_computer: str
    auth_type: str
    logon_type: str
    success: bool
    
    def to_edge_features(self) -> Tensor:
        """Convert authentication event to edge feature vector.
        
        Returns:
            Tensor containing encoded features:
            - Time of day (normalized 0-1)
            - Day of week (one-hot encoded, 7 dims)
            - Auth type encoding
            - Logon type encoding  
            - Success flag
        """
        import math
        
        # Time of day (normalized to 0-1)
        time_of_day = (self.timestamp % 86400) / 86400.0
        
        # Day of week (one-hot, 7 dimensions)
        day_of_week = int((self.timestamp // 86400) % 7)
        day_one_hot = [0.0] * 7
        day_one_hot[day_of_week] = 1.0

        # Auth type encoding (simple hash-based encoding)
        auth_type_hash = hash(self.auth_type) % 10
        auth_encoding = [0.0] * 10
        auth_encoding[auth_type_hash] = 1.0
        
        # Logon type encoding
        logon_type_hash = hash(self.logon_type) % 5
        logon_encoding = [0.0] * 5
        logon_encoding[logon_type_hash] = 1.0
        
        # Success flag
        success_flag = 1.0 if self.success else 0.0
        
        # Combine all features
        features = (
            [time_of_day] +
            day_one_hot +
            auth_encoding +
            logon_encoding +
            [success_flag]
        )
        
        return torch.tensor(features, dtype=torch.float32)


@dataclass
class TimeWindow:
    """Represents a temporal window containing authentication events.
    
    Attributes:
        start_time: Unix timestamp for window start
        end_time: Unix timestamp for window end
        events: List of authentication events in this window
        label: Binary label (0=normal, 1=lateral movement)
    """
    start_time: float
    end_time: float
    events: List[AuthenticationEvent] = field(default_factory=list)
    label: int = 0
    
    def get_nodes(self) -> Set[str]:
        """Get unique computers and users in this window.
        
        Returns:
            Set of unique entity identifiers (users and computers)
        """
        nodes = set()
        for event in self.events:
            nodes.add(event.source_user)
            nodes.add(event.source_computer)
            nodes.add(event.dest_computer)
        return nodes
    
    def get_edges(self) -> List[Tuple[str, str, "AuthenticationEvent"]]:
        """Get authentication edges with their associated events.
        
        Returns:
            List of tuples (source_node, dest_node, event)
        """
        edges = []
        for event in self.events:
            # Edge from source_computer to dest_computer
            edges.append((event.source_computer, event.dest_computer, event))
        return edges
    
    @property
    def duration(self) -> float:
        """Get window duration in seconds."""
        return self.end_time - self.start_time
    
    def __len__(self) -> int:
        """Return number of events in window."""
        return len(self.events)


@dataclass
class UncertaintyOutput:
    """Contains prediction with Bayesian uncertainty estimates.
    
    Attributes:
        mean_prediction: Expected probability from MC samples
        epistemic_uncertainty: Model uncertainty due to limited data
        aleatoric_uncertainty: Inherent noise/uncertainty in the data
        samples: Raw MC Dropout samples for further analysis
    """
    mean_prediction: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    samples: Tensor
    
    def total_uncertainty(self) -> float:
        """Compute total uncertainty as sum of epistemic and aleatoric.
        
        Returns:
            Total uncertainty value
        """
        return self.epistemic_uncertainty + self.aleatoric_uncertainty
    
    def requires_review(self, threshold: float = 0.5) -> bool:
        """Check if prediction requires human review based on uncertainty.
        
        Args:
            threshold: Uncertainty threshold above which review is needed
            
        Returns:
            True if epistemic uncertainty exceeds threshold
        """
        return self.epistemic_uncertainty > threshold


@dataclass
class DetectionResult:
    """Final detection result with all alert fields.
    
    Attributes:
        window_id: Unique identifier for the time window
        timestamp: Timestamp of the detection
        probability: Predicted probability of lateral movement
        epistemic_uncertainty: Model uncertainty estimate
        aleatoric_uncertainty: Data uncertainty estimate
        energy_score: Physics-based anomaly score
        constraint_violations: List of violated physics constraints
    """
    window_id: str
    timestamp: float
    probability: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    energy_score: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)
    
    def to_alert(self, threshold: float = 0.5) -> Optional["Alert"]:
        """Convert to alert if probability exceeds threshold.
        
        Args:
            threshold: Decision threshold for alerting
            
        Returns:
            Alert object if above threshold, None otherwise
        """
        if self.probability >= threshold:
            return Alert(
                window_id=self.window_id,
                timestamp=self.timestamp,
                probability=self.probability,
                epistemic_uncertainty=self.epistemic_uncertainty,
                aleatoric_uncertainty=self.aleatoric_uncertainty,
                severity=self._compute_severity(),
                constraint_violations=self.constraint_violations
            )
        return None
    
    def _compute_severity(self) -> str:
        """Compute alert severity based on probability and uncertainty.
        
        Returns:
            Severity level: 'low', 'medium', 'high', or 'critical'
        """
        if self.probability >= 0.9 and self.epistemic_uncertainty < 0.3:
            return "critical"
        elif self.probability >= 0.7:
            return "high"
        elif self.probability >= 0.5:
            return "medium"
        return "low"


@dataclass
class Alert:
    """Security alert generated from detection result.
    
    Attributes:
        window_id: Unique identifier for the time window
        timestamp: Timestamp of the alert
        probability: Predicted probability of lateral movement
        epistemic_uncertainty: Model uncertainty estimate
        aleatoric_uncertainty: Data uncertainty estimate
        severity: Alert severity level
        constraint_violations: List of violated physics constraints
    """
    window_id: str
    timestamp: float
    probability: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    severity: str
    constraint_violations: List[str] = field(default_factory=list)


@dataclass
class DetectionMetrics:
    """Container for detection performance metrics.
    
    Attributes:
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        auc_roc: Area under ROC curve
        auc_pr: Area under Precision-Recall curve
        fpr_at_90_recall: False positive rate at 90% recall
        detection_latency_minutes: Average detection latency
    """
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    fpr_at_90_recall: float = 0.0
    detection_latency_minutes: float = 0.0


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics.
    
    Attributes:
        ece: Expected Calibration Error
        brier_score: Brier score
        nll: Negative Log-Likelihood
    """
    ece: float
    brier_score: float
    nll: float
