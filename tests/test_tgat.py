"""Tests for the TGAT model integration.

Tests the full T-GAT model that integrates all components:
- GraphAttentionEncoder
- TemporalAggregator
- BayesianUncertaintyModule
- PhysicsConstraintModule
- DetectionHead
"""

import pytest
import torch
from torch_geometric.data import Data

from src.models.tgat import TGAT
from src.data.models import DetectionResult, UncertaintyOutput


@pytest.fixture
def model():
    """Create a TGAT model for testing."""
    return TGAT(
        node_feature_dim=64,
        edge_feature_dim=32,
        hidden_dim=64,
        output_dim=32,
        num_gat_layers=2,
        num_attention_heads=2,
        num_lstm_layers=1,
        lstm_hidden_dim=64,
        num_mc_samples=5,  # Small for faster tests
    )


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    num_nodes = 5
    num_edges = 8
    x = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class TestTGATInitialization:
    """Tests for TGAT model initialization."""
    
    def test_model_creation(self, model):
        """Test that model can be created successfully."""
        assert model is not None
        assert isinstance(model, TGAT)
    
    def test_model_components_exist(self, model):
        """Test that all components are initialized."""
        assert hasattr(model, 'graph_encoder')
        assert hasattr(model, 'temporal_aggregator')
        assert hasattr(model, 'uncertainty_module')
        assert hasattr(model, 'physics_module')
        assert hasattr(model, 'detection_head')
    
    def test_model_config(self, model):
        """Test that model config is accessible."""
        config = model.get_config()
        assert 'node_feature_dim' in config
        assert 'hidden_dim' in config
        assert 'output_dim' in config


class TestTGATForward:
    """Tests for TGAT forward pass."""
    
    def test_forward_single_graph(self, model, sample_graph):
        """Test forward pass with a single graph."""
        predictions = model([sample_graph])
        
        assert predictions.shape == torch.Size([1, 1])
        assert 0 <= predictions.item() <= 1
    
    def test_forward_multiple_graphs(self, model, sample_graph):
        """Test forward pass with multiple graphs (sequence)."""
        graphs = [sample_graph, sample_graph]
        predictions = model(graphs)
        
        assert predictions.shape == torch.Size([1, 1])
        assert 0 <= predictions.item() <= 1
    
    def test_forward_with_uncertainty(self, model, sample_graph):
        """Test forward pass with uncertainty estimation."""
        predictions, uncertainty, hidden = model(
            [sample_graph], return_uncertainty=True
        )
        
        assert predictions.shape == torch.Size([1, 1])
        assert isinstance(uncertainty, UncertaintyOutput)
        assert uncertainty.epistemic_uncertainty >= 0
        assert uncertainty.aleatoric_uncertainty >= 0
        assert hidden is not None
    
    def test_forward_empty_input(self, model):
        """Test forward pass with empty input."""
        predictions = model([])
        assert predictions.shape == torch.Size([1, 1])


class TestTGATInference:
    """Tests for TGAT inference method."""
    
    def test_inference_returns_detection_result(self, model, sample_graph):
        """Test that inference returns DetectionResult."""
        result, hidden = model.inference(
            [sample_graph],
            window_id='test_window_001',
            timestamp=1234567890.0,
        )
        
        assert isinstance(result, DetectionResult)
        assert hidden is not None
    
    def test_inference_all_fields_present(self, model, sample_graph):
        """Test Property 19: Alert Field Completeness."""
        result, _ = model.inference(
            [sample_graph],
            window_id='test_window_001',
            timestamp=1234567890.0,
        )
        
        # All required fields must be present and non-null
        assert result.window_id is not None
        assert result.timestamp is not None
        assert result.probability is not None
        assert result.epistemic_uncertainty is not None
        assert result.aleatoric_uncertainty is not None
        assert result.energy_score is not None
        assert result.constraint_violations is not None
    
    def test_inference_probability_in_range(self, model, sample_graph):
        """Test that probability is in valid range [0, 1]."""
        result, _ = model.inference(
            [sample_graph],
            window_id='test_001',
            timestamp=1000.0,
        )
        
        assert 0 <= result.probability <= 1
    
    def test_inference_uncertainties_non_negative(self, model, sample_graph):
        """Test that uncertainties are non-negative."""
        result, _ = model.inference(
            [sample_graph],
            window_id='test_001',
            timestamp=1000.0,
        )
        
        assert result.epistemic_uncertainty >= 0
        assert result.aleatoric_uncertainty >= 0
    
    def test_streaming_inference(self, model, sample_graph):
        """Test streaming inference with hidden state."""
        hidden = model.init_hidden(batch_size=1)
        
        result1, hidden = model.inference(
            [sample_graph],
            window_id='window_1',
            timestamp=1000.0,
            hidden=hidden,
        )
        
        result2, hidden = model.inference(
            [sample_graph],
            window_id='window_2',
            timestamp=2000.0,
            hidden=hidden,
        )
        
        assert result1.window_id == 'window_1'
        assert result2.window_id == 'window_2'


class TestTGATLoss:
    """Tests for TGAT loss computation."""
    
    def test_compute_loss(self, model, sample_graph):
        """Test loss computation."""
        predictions = model([sample_graph])
        targets = torch.tensor([1.0])
        
        losses = model.compute_loss(predictions, targets, graphs=[sample_graph])
        
        assert 'total_loss' in losses
        assert 'detection_loss' in losses
        assert 'constraint_loss' in losses
    
    def test_loss_values_valid(self, model, sample_graph):
        """Test that loss values are valid."""
        predictions = model([sample_graph])
        targets = torch.tensor([1.0])
        
        losses = model.compute_loss(predictions, targets, graphs=[sample_graph])
        
        assert not torch.isnan(losses['total_loss'])
        assert not torch.isinf(losses['total_loss'])
        assert losses['detection_loss'] >= 0


class TestTGATUtilities:
    """Tests for TGAT utility methods."""
    
    def test_set_detection_threshold(self, model):
        """Test setting detection threshold."""
        model.set_detection_threshold(0.7)
        assert model.detection_threshold == 0.7
    
    def test_init_hidden(self, model):
        """Test hidden state initialization."""
        hidden = model.init_hidden(batch_size=2)
        
        assert isinstance(hidden, tuple)
        assert len(hidden) == 2
        h, c = hidden
        assert h.shape[1] == 2  # batch_size
        assert c.shape[1] == 2
    
    def test_get_attention_weights(self, model, sample_graph):
        """Test getting attention weights."""
        model.eval()
        _ = model([sample_graph])
        
        weights = model.get_attention_weights()
        assert isinstance(weights, list)

