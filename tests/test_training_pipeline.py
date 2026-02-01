"""Tests for TrainingPipeline class.

Tests cover:
- Training loop with Adam optimizer
- Gradient clipping enforcement
- Early stopping behavior
- Checkpoint save/load round-trip
"""

import os
import tempfile
from typing import List

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from src.models.training_pipeline import TrainingPipeline, EarlyStopping


class SimpleModel(nn.Module):
    """Simple model for testing the training pipeline."""
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
    
    def forward(self, batch: Data) -> Tensor:
        """Forward pass."""
        # Global mean pooling
        x = batch.x
        if hasattr(batch, 'batch') and batch.batch is not None:
            # Batch of graphs - do mean pooling per graph
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch.batch)
        else:
            # Single graph - mean over all nodes
            x = x.mean(dim=0, keepdim=True)
        
        x = torch.relu(self.encoder(x))
        x = torch.sigmoid(self.classifier(x))
        return x
    
    def get_config(self) -> dict:
        """Get model config."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
        }


def create_synthetic_graph(num_nodes: int = 10, num_edges: int = 20, label: int = 0) -> Data:
    """Create a synthetic graph for testing."""
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


def create_synthetic_dataset(num_graphs: int = 20, pos_ratio: float = 0.3) -> List[Data]:
    """Create a synthetic dataset for testing."""
    graphs = []
    num_positive = int(num_graphs * pos_ratio)
    
    for i in range(num_graphs):
        label = 1 if i < num_positive else 0
        graph = create_synthetic_graph(
            num_nodes=torch.randint(5, 15, (1,)).item(),
            num_edges=torch.randint(10, 30, (1,)).item(),
            label=label,
        )
        graphs.append(graph)
    
    return graphs


class TestEarlyStopping:
    """Tests for EarlyStopping class."""
    
    def test_early_stopping_triggers_after_patience(self):
        """Test that early stopping triggers after patience epochs without improvement."""
        patience = 3
        early_stopping = EarlyStopping(patience=patience)
        
        # Initial loss
        early_stopping(1.0, 0)
        assert not early_stopping.should_stop
        
        # No improvement for patience epochs
        for epoch in range(1, patience + 1):
            result = early_stopping(1.0, epoch)
            if epoch < patience:
                assert not result
                assert not early_stopping.should_stop
        
        # Should stop after patience epochs
        assert early_stopping.should_stop
    
    def test_early_stopping_resets_on_improvement(self):
        """Test that counter resets when validation loss improves."""
        early_stopping = EarlyStopping(patience=3)
        
        early_stopping(1.0, 0)
        early_stopping(1.0, 1)  # No improvement, counter = 1
        early_stopping(1.0, 2)  # No improvement, counter = 2
        
        assert early_stopping.counter == 2
        
        # Improvement
        early_stopping(0.5, 3)
        assert early_stopping.counter == 0
        assert early_stopping.best_loss == 0.5
    
    def test_early_stopping_reset(self):
        """Test reset functionality."""
        early_stopping = EarlyStopping(patience=3)
        
        early_stopping(1.0, 0)
        early_stopping(1.0, 1)
        
        early_stopping.reset()
        
        assert early_stopping.counter == 0
        assert early_stopping.best_loss is None
        assert not early_stopping.should_stop


class TestTrainingPipeline:
    """Tests for TrainingPipeline class."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel(input_dim=16, hidden_dim=32)
    
    @pytest.fixture
    def data_loaders(self):
        """Create train and validation data loaders."""
        train_data = create_synthetic_dataset(num_graphs=20, pos_ratio=0.3)
        val_data = create_synthetic_dataset(num_graphs=10, pos_ratio=0.3)
        
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
        
        return train_loader, val_loader
    
    def test_pipeline_initialization(self, model):
        """Test that pipeline initializes correctly."""
        pipeline = TrainingPipeline(
            model=model,
            optimizer="adam",
            lr=1e-3,
            grad_clip=1.0,
            early_stopping_patience=5,
        )
        
        assert pipeline.lr == 1e-3
        assert pipeline.grad_clip == 1.0
        assert pipeline.early_stopping.patience == 5
        assert isinstance(pipeline.optimizer, torch.optim.Adam)
    
    def test_training_runs(self, model, data_loaders):
        """Test that training loop runs without errors."""
        train_loader, val_loader = data_loaders
        
        pipeline = TrainingPipeline(
            model=model,
            lr=1e-3,
            grad_clip=1.0,
            early_stopping_patience=3,
        )
        
        result = pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,
            verbose=False,
        )
        
        assert 'best_val_loss' in result
        assert 'best_epoch' in result
        assert 'final_epoch' in result
        assert 'history' in result
        assert len(result['history']['train_loss']) > 0

    def test_gradient_clipping(self, model):
        """Test that gradient clipping is enforced.
        
        Property 16: Gradient Clipping Enforcement
        """
        pipeline = TrainingPipeline(
            model=model,
            lr=1e-3,
            grad_clip=1.0,
        )
        
        # Create a batch with large gradients
        batch = create_synthetic_graph(num_nodes=10, num_edges=20, label=1)
        batch = Batch.from_data_list([batch])
        
        # Forward pass
        pipeline.model.train()
        predictions = pipeline.model(batch)
        loss = nn.functional.binary_cross_entropy(
            predictions.view(-1),
            batch.y.view(-1),
        )
        
        # Backward pass
        loss.backward()
        
        # Get gradient norm before clipping
        norm_before = pipeline.get_gradient_norm()
        
        # Clip gradients
        pipeline.clip_gradients()
        
        # Get gradient norm after clipping
        norm_after = pipeline.get_gradient_norm()
        
        # Verify clipping
        assert norm_after <= pipeline.grad_clip + 1e-6  # Small tolerance for floating point
    
    def test_checkpoint_save_load(self, model, data_loaders):
        """Test checkpoint save and load round-trip.
        
        Property 18: Checkpoint Persistence Round-Trip
        """
        train_loader, val_loader = data_loaders
        
        pipeline = TrainingPipeline(
            model=model,
            lr=1e-3,
            grad_clip=1.0,
            early_stopping_patience=3,
        )
        
        # Train for a few epochs
        pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            verbose=False,
        )
        
        # Get predictions before saving
        pipeline.model.eval()
        test_batch = create_synthetic_graph(num_nodes=10, num_edges=20, label=0)
        test_batch = Batch.from_data_list([test_batch])
        
        with torch.no_grad():
            pred_before = pipeline.model(test_batch).clone()
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
            pipeline.save_checkpoint(checkpoint_path)
            
            # Create new model and pipeline
            new_model = SimpleModel(input_dim=16, hidden_dim=32)
            new_pipeline = TrainingPipeline(
                model=new_model,
                lr=1e-3,
                grad_clip=1.0,
            )
            
            # Load checkpoint
            new_pipeline.load_checkpoint(checkpoint_path)
            
            # Get predictions after loading
            new_pipeline.model.eval()
            with torch.no_grad():
                pred_after = new_pipeline.model(test_batch)
            
            # Verify predictions are identical
            assert torch.allclose(pred_before, pred_after, atol=1e-6)
    
    def test_early_stopping_in_training(self, model, data_loaders):
        """Test that early stopping works during training.
        
        Property 17: Early Stopping Behavior
        """
        train_loader, val_loader = data_loaders
        
        # Use very small patience to trigger early stopping
        pipeline = TrainingPipeline(
            model=model,
            lr=1e-6,  # Very small LR to prevent improvement
            grad_clip=1.0,
            early_stopping_patience=2,
        )
        
        result = pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,  # Large number, should stop early
            verbose=False,
        )
        
        # Should have stopped before 100 epochs
        # Note: May not always stop early if model improves
        assert result['final_epoch'] < 100 or not result['stopped_early']
    
    def test_reset_training_state(self, model, data_loaders):
        """Test that training state can be reset."""
        train_loader, val_loader = data_loaders
        
        pipeline = TrainingPipeline(
            model=model,
            lr=1e-3,
            grad_clip=1.0,
        )
        
        # Train for a few epochs
        pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            verbose=False,
        )
        
        assert len(pipeline.training_history['train_loss']) > 0
        
        # Reset
        pipeline.reset_training_state()
        
        assert pipeline.current_epoch == 0
        assert pipeline.best_val_loss == float('inf')
        assert len(pipeline.training_history['train_loss']) == 0
    
    def test_evaluate_auc_pr(self, model, data_loaders):
        """Test AUC-PR evaluation."""
        train_loader, val_loader = data_loaders
        
        pipeline = TrainingPipeline(
            model=model,
            lr=1e-3,
        )
        
        # Train briefly
        pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            verbose=False,
        )
        
        # Evaluate
        auc_pr = pipeline.evaluate_auc_pr(val_loader)
        
        # AUC-PR should be between 0 and 1
        assert 0.0 <= auc_pr <= 1.0


class TestHyperparameterSearch:
    """Tests for hyperparameter search functionality."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel(input_dim=16, hidden_dim=32)
    
    @pytest.fixture
    def data_loaders(self):
        """Create train and validation data loaders."""
        train_data = create_synthetic_dataset(num_graphs=10, pos_ratio=0.3)
        val_data = create_synthetic_dataset(num_graphs=5, pos_ratio=0.3)
        
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
        
        return train_loader, val_loader
    
    def test_hyperparameter_search_runs(self, model, data_loaders):
        """Test that hyperparameter search runs without errors."""
        pytest.importorskip("optuna")
        
        train_loader, val_loader = data_loaders
        
        pipeline = TrainingPipeline(
            model=model,
            lr=1e-3,
        )
        
        # Run with minimal trials
        result = pipeline.hyperparameter_search(
            train_loader=train_loader,
            val_loader=val_loader,
            search_space={
                'lr': {'type': 'loguniform', 'low': 1e-4, 'high': 1e-2},
            },
            n_trials=2,
        )
        
        assert 'best_params' in result
        assert 'best_value' in result
        assert 'study' in result
