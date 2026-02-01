"""Tests for InferenceEngine class.

Tests the inference engine functionality including:
- Batch inference with uncertainty
- GPU and CPU inference support
- Alert generation from DetectionResults
- Streaming inference with hidden state management

Requirements: 9.1, 9.2, 9.4
"""

import pytest
import torch
from torch_geometric.data import Data

from src.data.models import (
    AuthenticationEvent,
    TimeWindow,
    DetectionResult,
    Alert,
)
from src.data.graph_constructor import GraphConstructor
from src.models.tgat import TGAT
from src.models.inference_engine import InferenceEngine


@pytest.fixture
def sample_model():
    """Create a sample T-GAT model for testing."""
    model = TGAT(
        node_feature_dim=64,
        edge_feature_dim=32,
        hidden_dim=32,
        output_dim=16,
        num_gat_layers=2,
        num_attention_heads=2,
        num_lstm_layers=1,
        lstm_hidden_dim=32,
        dropout=0.1,
        num_mc_samples=10,
    )
    return model


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    num_nodes = 5
    num_edges = 8
    
    x = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 32)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([0]),
    )


@pytest.fixture
def sample_window():
    """Create a sample TimeWindow for testing."""
    events = [
        AuthenticationEvent(
            timestamp=1000.0 + i * 10,
            source_user=f"U{i}",
            source_computer=f"C{i}",
            dest_computer=f"C{(i + 1) % 3}",
            auth_type="Kerberos",
            logon_type="Network",
            success=True,
        )
        for i in range(5)
    ]
    
    return TimeWindow(
        start_time=1000.0,
        end_time=2000.0,
        events=events,
        label=0,
    )


@pytest.fixture
def inference_engine(sample_model):
    """Create an InferenceEngine for testing."""
    return InferenceEngine(
        model=sample_model,
        device='cpu',
        detection_threshold=0.5,
    )


class TestInferenceEngineInit:
    """Tests for InferenceEngine initialization."""
    
    def test_init_with_model(self, sample_model):
        """Test initialization with a model."""
        engine = InferenceEngine(model=sample_model, device='cpu')
        
        assert engine.model is not None
        assert engine.device == torch.device('cpu')
        assert engine.detection_threshold == 0.5
    
    def test_init_with_custom_threshold(self, sample_model):
        """Test initialization with custom detection threshold."""
        engine = InferenceEngine(
            model=sample_model,
            device='cpu',
            detection_threshold=0.7,
        )
        
        assert engine.detection_threshold == 0.7
    
    def test_init_auto_device_selection(self, sample_model):
        """Test automatic device selection."""
        engine = InferenceEngine(model=sample_model, device=None)
        
        # Should select CPU or CUDA based on availability
        assert engine.device.type in ['cpu', 'cuda']


class TestInferenceEngineInfer:
    """Tests for single inference."""
    
    def test_infer_single_graph(self, inference_engine, sample_graph):
        """Test inference on a single graph."""
        result = inference_engine.infer(
            graphs=sample_graph,
            window_id="test_window",
            timestamp=1000.0,
        )
        
        assert isinstance(result, DetectionResult)
        assert result.window_id == "test_window"
        assert result.timestamp == 1000.0
        assert 0.0 <= result.probability <= 1.0
        assert result.epistemic_uncertainty >= 0.0
        assert result.aleatoric_uncertainty >= 0.0
    
    def test_infer_list_of_graphs(self, inference_engine, sample_graph):
        """Test inference on a list of graphs."""
        graphs = [sample_graph, sample_graph]
        
        result = inference_engine.infer(
            graphs=graphs,
            window_id="test_window",
            timestamp=1000.0,
        )
        
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.probability <= 1.0
    
    def test_infer_auto_timestamp(self, inference_engine, sample_graph):
        """Test inference with automatic timestamp."""
        result = inference_engine.infer(
            graphs=sample_graph,
            window_id="test_window",
        )
        
        assert result.timestamp > 0  # Should be current time


class TestInferenceEngineBatch:
    """Tests for batch inference."""
    
    def test_infer_batch(self, inference_engine, sample_window):
        """Test batch inference on multiple windows."""
        windows = [sample_window, sample_window]
        
        results, alerts = inference_engine.infer_batch(
            windows=windows,
            return_alerts=True,
        )
        
        assert len(results) == 2
        assert all(isinstance(r, DetectionResult) for r in results)
    
    def test_infer_batch_no_alerts(self, inference_engine, sample_window):
        """Test batch inference without alert generation."""
        windows = [sample_window]
        
        results, alerts = inference_engine.infer_batch(
            windows=windows,
            return_alerts=False,
        )
        
        assert len(results) == 1
        assert len(alerts) == 0


class TestInferenceEngineStreaming:
    """Tests for streaming inference."""
    
    def test_streaming_inference(self, inference_engine, sample_graph):
        """Test streaming inference with hidden state."""
        # First inference
        result1, hidden1 = inference_engine.infer_streaming(
            graph=sample_graph,
            window_id="window_1",
            timestamp=1000.0,
        )
        
        assert isinstance(result1, DetectionResult)
        assert hidden1 is not None
        
        # Second inference should use hidden state
        result2, hidden2 = inference_engine.infer_streaming(
            graph=sample_graph,
            window_id="window_2",
            timestamp=2000.0,
        )
        
        assert isinstance(result2, DetectionResult)
        assert hidden2 is not None
    
    def test_streaming_reset_state(self, inference_engine, sample_graph):
        """Test streaming inference with state reset."""
        # First inference
        _, _ = inference_engine.infer_streaming(
            graph=sample_graph,
            window_id="window_1",
            timestamp=1000.0,
        )
        
        # Reset state
        result, hidden = inference_engine.infer_streaming(
            graph=sample_graph,
            window_id="window_2",
            timestamp=2000.0,
            reset_state=True,
        )
        
        assert isinstance(result, DetectionResult)
    
    def test_hidden_state_management(self, inference_engine, sample_graph):
        """Test hidden state getter and setter."""
        # Initially no hidden state
        assert inference_engine.get_hidden_state() is None
        
        # After inference, should have hidden state
        _, _ = inference_engine.infer_streaming(
            graph=sample_graph,
            window_id="window_1",
            timestamp=1000.0,
        )
        
        assert inference_engine.get_hidden_state() is not None
        
        # Reset should clear hidden state
        inference_engine.reset_hidden_state()
        assert inference_engine.get_hidden_state() is None


class TestAlertGeneration:
    """Tests for alert generation."""
    
    def test_generate_alert_above_threshold(self, inference_engine):
        """Test alert generation when above threshold."""
        result = DetectionResult(
            window_id="test",
            timestamp=1000.0,
            probability=0.8,
            epistemic_uncertainty=0.1,
            aleatoric_uncertainty=0.1,
            energy_score=0.5,
            constraint_violations=[],
        )
        
        alert = inference_engine.generate_alert(result, threshold=0.5)
        
        assert alert is not None
        assert isinstance(alert, Alert)
        assert alert.probability == 0.8
    
    def test_generate_alert_below_threshold(self, inference_engine):
        """Test no alert when below threshold."""
        result = DetectionResult(
            window_id="test",
            timestamp=1000.0,
            probability=0.3,
            epistemic_uncertainty=0.1,
            aleatoric_uncertainty=0.1,
            energy_score=0.5,
            constraint_violations=[],
        )
        
        alert = inference_engine.generate_alert(result, threshold=0.5)
        
        assert alert is None
    
    def test_generate_alerts_batch(self, inference_engine):
        """Test batch alert generation."""
        results = [
            DetectionResult(
                window_id=f"test_{i}",
                timestamp=1000.0 + i,
                probability=0.3 + i * 0.3,  # 0.3, 0.6, 0.9
                epistemic_uncertainty=0.1,
                aleatoric_uncertainty=0.1,
                energy_score=0.5,
                constraint_violations=[],
            )
            for i in range(3)
        ]
        
        alerts = inference_engine.generate_alerts_batch(results, threshold=0.5)
        
        # Should have 2 alerts (0.6 and 0.9 are above 0.5)
        assert len(alerts) == 2


class TestDeviceManagement:
    """Tests for device management."""
    
    def test_to_cpu(self, inference_engine):
        """Test moving to CPU."""
        engine = inference_engine.to_cpu()
        
        assert engine.device == torch.device('cpu')
        assert engine.is_gpu() is False
    
    def test_get_device(self, inference_engine):
        """Test getting current device."""
        device = inference_engine.get_device()
        
        assert isinstance(device, torch.device)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_gpu(self, inference_engine):
        """Test moving to GPU."""
        engine = inference_engine.to_gpu()
        
        assert engine.device.type == 'cuda'
        assert engine.is_gpu() is True
    
    def test_to_gpu_no_cuda(self, inference_engine):
        """Test GPU move when CUDA not available."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available")
        
        with pytest.raises(RuntimeError):
            inference_engine.to_gpu()


class TestThresholdManagement:
    """Tests for threshold management."""
    
    def test_set_detection_threshold(self, inference_engine):
        """Test setting detection threshold."""
        inference_engine.set_detection_threshold(0.7)
        
        assert inference_engine.get_detection_threshold() == 0.7
    
    def test_set_invalid_threshold(self, inference_engine):
        """Test setting invalid threshold."""
        with pytest.raises(ValueError):
            inference_engine.set_detection_threshold(1.5)
        
        with pytest.raises(ValueError):
            inference_engine.set_detection_threshold(-0.1)


class TestModelInfo:
    """Tests for model information."""
    
    def test_get_model_info(self, inference_engine):
        """Test getting model information."""
        info = inference_engine.get_model_info()
        
        assert 'device' in info
        assert 'detection_threshold' in info
        assert 'model_config' in info
        assert 'has_hidden_state' in info
