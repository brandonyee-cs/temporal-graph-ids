"""Inference Engine for T-GAT lateral movement detection.

This module implements the InferenceEngine class that handles:
- Batch inference with uncertainty estimation
- GPU and CPU inference support
- Alert generation from DetectionResults
- Streaming inference with hidden state management

Requirements: 9.1, 9.2, 9.4
"""

from typing import Dict, List, Optional, Tuple, Union
import time

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data, Batch

from src.data.models import (
    Alert,
    DetectionResult,
    TimeWindow,
    UncertaintyOutput,
)
from src.data.graph_constructor import GraphConstructor
from src.models.tgat import TGAT


class InferenceEngine:
    """Inference engine for T-GAT lateral movement detection.
    
    Provides efficient inference capabilities including:
    - Batch processing with uncertainty estimation
    - GPU and CPU inference support
    - Alert generation from detection results
    - Streaming inference with hidden state management
    
    Requirement 9.1: Process each window in less than 1 second
    Requirement 9.2: Support both GPU and optimized CPU inference
    Requirement 9.4: Include prediction probability, epistemic and aleatoric uncertainty
    
    Attributes:
        model: The T-GAT model for inference
        device: Device for inference (CPU or GPU)
        graph_constructor: GraphConstructor for building graphs from windows
        detection_threshold: Threshold for generating alerts
    """
    
    def __init__(
        self,
        model: TGAT,
        device: Optional[Union[str, torch.device]] = None,
        graph_constructor: Optional[GraphConstructor] = None,
        detection_threshold: float = 0.5,
        node_feature_dim: int = 64,
        edge_feature_dim: int = 32,
    ):
        """Initialize InferenceEngine.
        
        Args:
            model: Trained T-GAT model for inference
            device: Device for inference ('cpu', 'cuda', or torch.device)
                   If None, automatically selects GPU if available
            graph_constructor: Optional GraphConstructor instance
                              If None, creates a new one with default settings
            detection_threshold: Threshold for generating alerts (default: 0.5)
            node_feature_dim: Node feature dimension for graph constructor (default: 64)
            edge_feature_dim: Edge feature dimension for graph constructor (default: 32)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Move model to device and set to eval mode
        self.model = model.to(self.device)
        self.model.eval()
        
        # Initialize graph constructor
        if graph_constructor is not None:
            self.graph_constructor = graph_constructor
        else:
            self.graph_constructor = GraphConstructor(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
            )
        
        self.detection_threshold = detection_threshold
        
        # Hidden state for streaming inference
        self._hidden_state: Optional[Tuple[Tensor, Tensor]] = None
    
    def infer(
        self,
        graphs: Union[List[Data], Batch, Data],
        window_id: str = "unknown",
        timestamp: Optional[float] = None,
        return_uncertainty: bool = True,
    ) -> DetectionResult:
        """Perform inference on a single graph or batch.
        
        Requirement 9.1: Process each window in less than 1 second
        Requirement 9.4: Include prediction probability, epistemic and aleatoric uncertainty
        
        Args:
            graphs: Single graph, list of graphs, or batched graphs
            window_id: Unique identifier for the window
            timestamp: Timestamp for the detection (uses current time if None)
            return_uncertainty: Whether to compute uncertainty estimates
            
        Returns:
            DetectionResult with all required fields
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Ensure graphs are in list format
        if isinstance(graphs, Data) and not isinstance(graphs, Batch):
            graphs = [graphs]
        
        # Move graphs to device
        graphs = self._move_to_device(graphs)
        
        # Perform inference
        with torch.no_grad():
            result, _ = self.model.inference(
                graphs=graphs,
                window_id=window_id,
                timestamp=timestamp,
                hidden=None,
            )
        
        return result
    
    def infer_batch(
        self,
        windows: List[TimeWindow],
        return_alerts: bool = True,
    ) -> Tuple[List[DetectionResult], List[Alert]]:
        """Perform batch inference on multiple windows.
        
        Efficiently processes multiple windows and optionally generates alerts.
        
        Requirement 9.1: Process each window in less than 1 second
        Requirement 9.4: Include prediction probability, epistemic and aleatoric uncertainty
        
        Args:
            windows: List of TimeWindow objects to process
            return_alerts: Whether to generate alerts from results
            
        Returns:
            Tuple of (list of DetectionResults, list of Alerts)
        """
        results = []
        alerts = []
        
        for i, window in enumerate(windows):
            # Build graph from window
            graph = self.graph_constructor.build_graph(window)
            
            # Generate window ID
            window_id = f"window_{i}_{int(window.start_time)}"
            
            # Perform inference
            result = self.infer(
                graphs=graph,
                window_id=window_id,
                timestamp=window.start_time,
            )
            results.append(result)
            
            # Generate alert if above threshold
            if return_alerts:
                alert = self.generate_alert(result)
                if alert is not None:
                    alerts.append(alert)
        
        return results, alerts
    
    def infer_streaming(
        self,
        graph: Union[Data, List[Data]],
        window_id: str,
        timestamp: float,
        reset_state: bool = False,
    ) -> Tuple[DetectionResult, Optional[Tuple[Tensor, Tensor]]]:
        """Perform streaming inference with hidden state management.
        
        Maintains LSTM hidden state across calls for efficient streaming
        inference on sequential windows.
        
        Requirement 9.1: Process each window in less than 1 second
        
        Args:
            graph: Graph or list of graphs for the current window
            window_id: Unique identifier for the window
            timestamp: Timestamp for the detection
            reset_state: Whether to reset the hidden state
            
        Returns:
            Tuple of (DetectionResult, updated hidden state)
        """
        if reset_state:
            self._hidden_state = None
        
        # Ensure graph is in list format
        if isinstance(graph, Data) and not isinstance(graph, Batch):
            graphs = [graph]
        else:
            graphs = graph if isinstance(graph, list) else [graph]
        
        # Move to device
        graphs = self._move_to_device(graphs)
        
        # Perform inference with hidden state
        with torch.no_grad():
            result, new_hidden = self.model.inference(
                graphs=graphs,
                window_id=window_id,
                timestamp=timestamp,
                hidden=self._hidden_state,
            )
        
        # Update hidden state
        self._hidden_state = new_hidden
        
        return result, new_hidden
    
    def generate_alert(
        self,
        result: DetectionResult,
        threshold: Optional[float] = None,
    ) -> Optional[Alert]:
        """Generate alert from DetectionResult if above threshold.
        
        Requirement 9.4: Include prediction probability, epistemic and aleatoric uncertainty
        
        Args:
            result: DetectionResult to evaluate
            threshold: Optional override for detection threshold
            
        Returns:
            Alert if probability exceeds threshold, None otherwise
        """
        thresh = threshold if threshold is not None else self.detection_threshold
        return result.to_alert(thresh)
    
    def generate_alerts_batch(
        self,
        results: List[DetectionResult],
        threshold: Optional[float] = None,
    ) -> List[Alert]:
        """Generate alerts from multiple DetectionResults.
        
        Args:
            results: List of DetectionResults to evaluate
            threshold: Optional override for detection threshold
            
        Returns:
            List of Alerts for results exceeding threshold
        """
        alerts = []
        for result in results:
            alert = self.generate_alert(result, threshold)
            if alert is not None:
                alerts.append(alert)
        return alerts

    
    def _move_to_device(
        self,
        graphs: Union[List[Data], Batch],
    ) -> Union[List[Data], Batch]:
        """Move graphs to the inference device.
        
        Args:
            graphs: List of Data objects or Batch object
            
        Returns:
            Graphs moved to the inference device
        """
        if isinstance(graphs, Batch):
            return graphs.to(self.device)
        
        moved_graphs = []
        for graph in graphs:
            moved_graphs.append(graph.to(self.device))
        return moved_graphs
    
    def reset_hidden_state(self) -> None:
        """Reset the hidden state for streaming inference."""
        self._hidden_state = None
    
    def get_hidden_state(self) -> Optional[Tuple[Tensor, Tensor]]:
        """Get the current hidden state.
        
        Returns:
            Current hidden state tuple or None if not initialized
        """
        return self._hidden_state
    
    def set_hidden_state(
        self,
        hidden_state: Optional[Tuple[Tensor, Tensor]],
    ) -> None:
        """Set the hidden state for streaming inference.
        
        Args:
            hidden_state: Hidden state tuple (h, c) or None to reset
        """
        self._hidden_state = hidden_state
    
    def set_detection_threshold(self, threshold: float) -> None:
        """Set the detection threshold for alert generation.
        
        Args:
            threshold: New detection threshold in [0, 1]
            
        Raises:
            ValueError: If threshold is not in [0, 1]
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self.detection_threshold = threshold
        self.model.set_detection_threshold(threshold)
    
    def get_detection_threshold(self) -> float:
        """Get the current detection threshold.
        
        Returns:
            Current detection threshold
        """
        return self.detection_threshold
    
    def to_device(self, device: Union[str, torch.device]) -> "InferenceEngine":
        """Move the inference engine to a different device.
        
        Requirement 9.2: Support both GPU and optimized CPU inference
        
        Args:
            device: Target device ('cpu', 'cuda', or torch.device)
            
        Returns:
            Self for method chaining
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # Move hidden state if it exists
        if self._hidden_state is not None:
            h, c = self._hidden_state
            self._hidden_state = (h.to(self.device), c.to(self.device))
        
        return self
    
    def to_cpu(self) -> "InferenceEngine":
        """Move the inference engine to CPU.
        
        Requirement 9.2: Support both GPU and optimized CPU inference
        
        Returns:
            Self for method chaining
        """
        return self.to_device('cpu')
    
    def to_gpu(self, device_id: int = 0) -> "InferenceEngine":
        """Move the inference engine to GPU.
        
        Requirement 9.2: Support both GPU and optimized CPU inference
        
        Args:
            device_id: GPU device ID (default: 0)
            
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If CUDA is not available
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return self.to_device(f'cuda:{device_id}')
    
    def is_gpu(self) -> bool:
        """Check if inference is running on GPU.
        
        Returns:
            True if using GPU, False otherwise
        """
        return self.device.type == 'cuda'
    
    def get_device(self) -> torch.device:
        """Get the current inference device.
        
        Returns:
            Current torch.device
        """
        return self.device
    
    def benchmark_inference(
        self,
        graph: Data,
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """Benchmark inference performance.
        
        Requirement 9.1: Process each window in less than 1 second
        
        Args:
            graph: Sample graph for benchmarking
            num_iterations: Number of iterations for timing
            
        Returns:
            Dictionary with timing statistics:
            - mean_time: Average inference time in seconds
            - std_time: Standard deviation of inference time
            - min_time: Minimum inference time
            - max_time: Maximum inference time
        """
        import statistics
        
        # Warm-up
        for _ in range(10):
            _ = self.infer(graph, window_id="benchmark", timestamp=0.0)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.infer(graph, window_id="benchmark", timestamp=0.0)
            end = time.time()
            times.append(end - start)
        
        return {
            'mean_time': statistics.mean(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min_time': min(times),
            'max_time': max(times),
            'num_iterations': num_iterations,
        }
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "InferenceEngine":
        """Create InferenceEngine from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device for inference
            **kwargs: Additional arguments for InferenceEngine
            
        Returns:
            InferenceEngine instance with loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get model config from checkpoint
        model_config = checkpoint.get('model_config', {})
        
        # Create model
        model = TGAT(**model_config)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create inference engine
        return cls(model=model, device=device, **kwargs)
    
    def get_model_info(self) -> Dict:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'device': str(self.device),
            'detection_threshold': self.detection_threshold,
            'model_config': self.model.get_config(),
            'has_hidden_state': self._hidden_state is not None,
        }
