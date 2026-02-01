"""Tests for MetaLearner class.

Tests the MAML meta-learning implementation for T-GAT.
Requirements: 7.1, 7.2, 7.3
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.meta_learner import MetaLearner, Task, TaskCreator
from src.data.models import TimeWindow, AuthenticationEvent


class SimpleModel(nn.Module):
    """Simple model for testing meta-learning."""
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, data):
        if isinstance(data, Data):
            # Graph data - use mean pooling
            x = data.x.mean(dim=0, keepdim=True)
        else:
            x = data
        
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def create_synthetic_graph(num_nodes: int = 5, label: int = 0) -> Data:
    """Create a synthetic graph for testing."""
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float32))


def create_synthetic_task(task_id: str, support_size: int = 5, query_size: int = 10) -> Task:
    """Create a synthetic task for testing."""
    support_set = [create_synthetic_graph(label=i % 2) for i in range(support_size)]
    query_set = [create_synthetic_graph(label=i % 2) for i in range(query_size)]
    support_labels = torch.tensor([i % 2 for i in range(support_size)], dtype=torch.float32)
    query_labels = torch.tensor([i % 2 for i in range(query_size)], dtype=torch.float32)
    
    return Task(
        task_id=task_id,
        support_set=support_set,
        query_set=query_set,
        support_labels=support_labels,
        query_labels=query_labels,
    )


class TestTask:
    """Tests for Task dataclass."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = create_synthetic_task("test_task")
        
        assert task.task_id == "test_task"
        assert task.num_support == 5
        assert task.num_query == 10
        assert len(task) == 15
    
    def test_task_labels(self):
        """Test task labels are correct."""
        task = create_synthetic_task("test_task", support_size=4, query_size=6)
        
        assert task.support_labels.shape == (4,)
        assert task.query_labels.shape == (6,)


class TestTaskCreator:
    """Tests for TaskCreator class."""
    
    def test_task_creator_initialization(self):
        """Test TaskCreator initialization."""
        creator = TaskCreator(support_size=5, query_size=15)
        
        assert creator.support_size == 5
        assert creator.query_size == 15
    
    def test_create_tasks_from_time_periods(self):
        """Test creating tasks from time periods."""
        creator = TaskCreator(support_size=2, query_size=3)
        
        # Create synthetic windows and graphs
        windows = []
        graphs = []
        base_time = 0
        
        for i in range(20):
            event = AuthenticationEvent(
                timestamp=base_time + i * 3600,
                source_user="user1",
                source_computer="comp1",
                dest_computer="comp2",
                auth_type="Kerberos",
                logon_type="Network",
                success=True,
            )
            window = TimeWindow(
                start_time=base_time + i * 3600,
                end_time=base_time + (i + 1) * 3600,
                events=[event],
                label=i % 2,
            )
            windows.append(window)
            graphs.append(create_synthetic_graph(label=i % 2))
        
        tasks = creator.create_tasks_from_time_periods(
            windows, graphs, period_duration=36000  # 10 hours
        )
        
        # Should create at least one task
        assert len(tasks) >= 1
        
        for task in tasks:
            assert task.num_support == 2
            assert task.num_query == 3
    
    def test_create_tasks_empty_input(self):
        """Test creating tasks with empty input."""
        creator = TaskCreator()
        
        tasks = creator.create_tasks_from_time_periods([], [], period_duration=86400)
        assert tasks == []


class TestMetaLearner:
    """Tests for MetaLearner class."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel(input_dim=16, hidden_dim=32)
    
    @pytest.fixture
    def meta_learner(self, model):
        """Create a MetaLearner instance."""
        return MetaLearner(
            model=model,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=3,
            first_order=True,  # Use first-order for faster testing
        )
    
    def test_meta_learner_initialization(self, meta_learner):
        """Test MetaLearner initialization."""
        assert meta_learner.inner_lr == 0.01
        assert meta_learner.outer_lr == 0.001
        assert meta_learner.num_inner_steps == 3
        assert meta_learner.first_order is True
    
    def test_adapt_returns_model(self, meta_learner):
        """Test that adapt() returns an adapted model.
        
        Requirement 7.3: Few-shot adaptation
        """
        support_set = [create_synthetic_graph(label=i % 2) for i in range(5)]
        support_labels = torch.tensor([i % 2 for i in range(5)], dtype=torch.float32)
        
        adapted_model = meta_learner.adapt(
            support_set=support_set,
            support_labels=support_labels,
            num_steps=3,
            return_model=True,
        )
        
        assert isinstance(adapted_model, nn.Module)
        
        # Test that adapted model can make predictions
        test_graph = create_synthetic_graph()
        with torch.no_grad():
            prediction = adapted_model(test_graph)
        
        assert prediction.shape == (1, 1)
        assert 0 <= prediction.item() <= 1
    
    def test_adapt_returns_params(self, meta_learner):
        """Test that adapt() can return parameters."""
        support_set = [create_synthetic_graph(label=i % 2) for i in range(5)]
        support_labels = torch.tensor([i % 2 for i in range(5)], dtype=torch.float32)
        
        adapted_params = meta_learner.adapt(
            support_set=support_set,
            support_labels=support_labels,
            num_steps=3,
            return_model=False,
        )
        
        assert isinstance(adapted_params, dict)
        assert len(adapted_params) > 0
        
        # Check that parameters are tensors
        for name, param in adapted_params.items():
            assert isinstance(param, torch.Tensor)
    
    def test_meta_train_single_epoch(self, meta_learner):
        """Test meta-training for a single epoch.
        
        Requirement 7.1: MAML meta-training loop
        """
        tasks = [create_synthetic_task(f"task_{i}") for i in range(4)]
        
        result = meta_learner.meta_train(
            tasks=tasks,
            num_epochs=1,
            tasks_per_batch=2,
            verbose=False,
        )
        
        assert 'history' in result
        assert 'final_meta_loss' in result
        assert len(result['history']['meta_train_loss']) == 1
    
    def test_meta_train_multiple_epochs(self, meta_learner):
        """Test meta-training for multiple epochs."""
        tasks = [create_synthetic_task(f"task_{i}") for i in range(4)]
        
        result = meta_learner.meta_train(
            tasks=tasks,
            num_epochs=3,
            tasks_per_batch=2,
            verbose=False,
        )
        
        assert len(result['history']['meta_train_loss']) == 3
    
    def test_meta_train_with_validation(self, meta_learner):
        """Test meta-training with validation tasks."""
        train_tasks = [create_synthetic_task(f"train_{i}") for i in range(4)]
        val_tasks = [create_synthetic_task(f"val_{i}") for i in range(2)]
        
        result = meta_learner.meta_train(
            tasks=train_tasks,
            num_epochs=2,
            tasks_per_batch=2,
            val_tasks=val_tasks,
            verbose=False,
        )
        
        assert len(result['history']['meta_val_loss']) == 2
    
    def test_meta_train_empty_tasks_raises(self, meta_learner):
        """Test that meta_train raises error with empty tasks."""
        with pytest.raises(ValueError, match="No tasks provided"):
            meta_learner.meta_train(tasks=[], num_epochs=1)
    
    def test_create_tasks_from_windows(self, meta_learner):
        """Test creating tasks from windows.
        
        Requirement 7.2: Treat each subnetwork/time period as separate task
        """
        windows = []
        graphs = []
        base_time = 0
        
        # Create more windows to ensure we have enough for task creation
        # Default task creator needs support_size=5 + query_size=15 = 20 samples per period
        for i in range(100):
            event = AuthenticationEvent(
                timestamp=base_time + i * 3600,
                source_user="user1",
                source_computer="comp1",
                dest_computer="comp2",
                auth_type="Kerberos",
                logon_type="Network",
                success=True,
            )
            window = TimeWindow(
                start_time=base_time + i * 3600,
                end_time=base_time + (i + 1) * 3600,
                events=[event],
                label=i % 2,
            )
            windows.append(window)
            graphs.append(create_synthetic_graph(label=i % 2))
        
        # Use smaller support/query sizes for testing
        meta_learner.task_creator = TaskCreator(support_size=3, query_size=5)
        
        tasks = meta_learner.create_tasks_from_windows(
            windows=windows,
            graphs=graphs,
            task_type='time_period',
            period_duration=36000,  # 10 hours = ~10 windows per period
        )
        
        assert len(tasks) >= 1
    
    def test_save_and_load_meta_state(self, meta_learner, tmp_path):
        """Test saving and loading meta-learner state."""
        # Do some training first
        tasks = [create_synthetic_task(f"task_{i}") for i in range(4)]
        meta_learner.meta_train(tasks=tasks, num_epochs=1, verbose=False)
        
        # Save state
        save_path = tmp_path / "meta_state.pt"
        meta_learner.save_meta_state(str(save_path))
        
        assert save_path.exists()
        
        # Create new meta-learner and load state
        new_model = SimpleModel(input_dim=16, hidden_dim=32)
        new_meta_learner = MetaLearner(model=new_model)
        new_meta_learner.load_meta_state(str(save_path))
        
        # Check that state was loaded
        assert new_meta_learner.inner_lr == meta_learner.inner_lr
        assert new_meta_learner.outer_lr == meta_learner.outer_lr
    
    def test_adaptation_produces_different_params(self, meta_learner):
        """Test that adaptation changes model parameters."""
        # Get original parameters
        original_params = {
            name: param.clone()
            for name, param in meta_learner.model.named_parameters()
        }
        
        # Adapt
        support_set = [create_synthetic_graph(label=i % 2) for i in range(5)]
        support_labels = torch.tensor([i % 2 for i in range(5)], dtype=torch.float32)
        
        adapted_params = meta_learner.adapt(
            support_set=support_set,
            support_labels=support_labels,
            num_steps=5,
            return_model=False,
        )
        
        # Check that at least some parameters changed
        params_changed = False
        for name in original_params:
            if not torch.allclose(original_params[name], adapted_params[name], atol=1e-6):
                params_changed = True
                break
        
        assert params_changed, "Adaptation should change model parameters"


class TestMetaLearnerIntegration:
    """Integration tests for MetaLearner with TGAT-like models."""
    
    def test_full_meta_learning_pipeline(self):
        """Test the full meta-learning pipeline."""
        # Create model
        model = SimpleModel(input_dim=16, hidden_dim=32)
        
        # Create meta-learner
        meta_learner = MetaLearner(
            model=model,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=3,
            first_order=True,
        )
        
        # Create tasks
        train_tasks = [create_synthetic_task(f"train_{i}") for i in range(6)]
        val_tasks = [create_synthetic_task(f"val_{i}") for i in range(2)]
        
        # Meta-train
        result = meta_learner.meta_train(
            tasks=train_tasks,
            num_epochs=2,
            tasks_per_batch=2,
            val_tasks=val_tasks,
            verbose=False,
        )
        
        assert 'final_meta_loss' in result
        
        # Test adaptation on new task
        new_task = create_synthetic_task("new_task")
        adapted_model = meta_learner.adapt(
            support_set=new_task.support_set,
            support_labels=new_task.support_labels,
            num_steps=3,
            return_model=True,
        )
        
        # Make predictions on query set
        with torch.no_grad():
            for graph in new_task.query_set:
                pred = adapted_model(graph)
                assert 0 <= pred.item() <= 1
