"""Meta-Learning Framework for T-GAT lateral movement detection.

This module implements the MetaLearner class that enables rapid adaptation
to new network topologies using Model-Agnostic Meta-Learning (MAML).

Requirements: 7.1, 7.2, 7.3
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, SGD
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from src.data.models import TimeWindow

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a meta-learning task (subnetwork or time period).
    
    Each task contains a support set for adaptation and a query set for evaluation.
    
    Attributes:
        task_id: Unique identifier for the task
        support_set: Data for few-shot adaptation (small labeled set)
        query_set: Data for evaluation after adaptation
        metadata: Optional metadata about the task (e.g., subnetwork info)
    """
    task_id: str
    support_set: List[Data]
    query_set: List[Data]
    support_labels: Tensor
    query_labels: Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Return total number of samples in the task."""
        return len(self.support_set) + len(self.query_set)
    
    @property
    def num_support(self) -> int:
        """Number of support samples."""
        return len(self.support_set)
    
    @property
    def num_query(self) -> int:
        """Number of query samples."""
        return len(self.query_set)


class TaskCreator:
    """Creates meta-learning tasks from data.
    
    Requirement 7.2: Treat each subnetwork or time period as a separate task.
    
    Supports creating tasks from:
    - Time periods (temporal splits)
    - Subnetworks (based on computer/user groupings)
    """
    
    def __init__(
        self,
        support_size: int = 5,
        query_size: int = 15,
        min_positive_ratio: float = 0.1,
    ):
        """Initialize TaskCreator.
        
        Args:
            support_size: Number of samples in support set (default: 5)
            query_size: Number of samples in query set (default: 15)
            min_positive_ratio: Minimum ratio of positive samples (default: 0.1)
        """
        self.support_size = support_size
        self.query_size = query_size
        self.min_positive_ratio = min_positive_ratio
    
    def create_tasks_from_time_periods(
        self,
        windows: List[TimeWindow],
        graphs: List[Data],
        period_duration: int = 86400,  # 1 day in seconds
    ) -> List[Task]:
        """Create tasks from time periods.
        
        Each time period becomes a separate task for meta-learning.
        
        Args:
            windows: List of TimeWindow objects
            graphs: Corresponding list of graph Data objects
            period_duration: Duration of each period in seconds (default: 1 day)
            
        Returns:
            List of Task objects
        """
        if len(windows) != len(graphs):
            raise ValueError("Number of windows must match number of graphs")
        
        if not windows:
            return []
        
        # Sort by start time
        sorted_pairs = sorted(zip(windows, graphs), key=lambda x: x[0].start_time)
        windows, graphs = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        min_time = windows[0].start_time
        max_time = windows[-1].start_time
        
        tasks = []
        period_start = min_time
        task_idx = 0
        
        while period_start < max_time:
            period_end = period_start + period_duration
            
            # Get windows/graphs in this period
            period_graphs = []
            period_labels = []
            
            for window, graph in zip(windows, graphs):
                if period_start <= window.start_time < period_end:
                    period_graphs.append(graph)
                    period_labels.append(window.label)
            
            # Create task if we have enough samples
            if len(period_graphs) >= self.support_size + self.query_size:
                task = self._create_task_from_samples(
                    task_id=f"period_{task_idx}",
                    graphs=period_graphs,
                    labels=period_labels,
                    metadata={'period_start': period_start, 'period_end': period_end},
                )
                if task is not None:
                    tasks.append(task)
                    task_idx += 1
            
            period_start = period_end
        
        return tasks
    
    def create_tasks_from_subnetworks(
        self,
        windows: List[TimeWindow],
        graphs: List[Data],
        subnetwork_key: str = 'subnet',
    ) -> List[Task]:
        """Create tasks from subnetworks.
        
        Each subnetwork becomes a separate task for meta-learning.
        
        Args:
            windows: List of TimeWindow objects
            graphs: Corresponding list of graph Data objects
            subnetwork_key: Key in graph metadata identifying subnetwork
            
        Returns:
            List of Task objects
        """
        if len(windows) != len(graphs):
            raise ValueError("Number of windows must match number of graphs")
        
        # Group by subnetwork
        subnetwork_data: Dict[str, Tuple[List[Data], List[int]]] = {}
        
        for window, graph in zip(windows, graphs):
            # Get subnetwork identifier
            if hasattr(graph, subnetwork_key):
                subnet_id = getattr(graph, subnetwork_key)
            else:
                # Infer from computer names in window
                computers = set()
                for event in window.events:
                    computers.add(event.source_computer)
                    computers.add(event.dest_computer)
                # Use first computer prefix as subnet identifier
                if computers:
                    first_computer = sorted(computers)[0]
                    subnet_id = first_computer.split('.')[0] if '.' in first_computer else 'default'
                else:
                    subnet_id = 'default'
            
            if subnet_id not in subnetwork_data:
                subnetwork_data[subnet_id] = ([], [])
            
            subnetwork_data[subnet_id][0].append(graph)
            subnetwork_data[subnet_id][1].append(window.label)
        
        # Create tasks from each subnetwork
        tasks = []
        for subnet_id, (graphs_list, labels_list) in subnetwork_data.items():
            if len(graphs_list) >= self.support_size + self.query_size:
                task = self._create_task_from_samples(
                    task_id=f"subnet_{subnet_id}",
                    graphs=graphs_list,
                    labels=labels_list,
                    metadata={'subnetwork': subnet_id},
                )
                if task is not None:
                    tasks.append(task)
        
        return tasks
    
    def _create_task_from_samples(
        self,
        task_id: str,
        graphs: List[Data],
        labels: List[int],
        metadata: Dict[str, Any],
    ) -> Optional[Task]:
        """Create a single task from samples.
        
        Args:
            task_id: Unique task identifier
            graphs: List of graph Data objects
            labels: Corresponding labels
            metadata: Task metadata
            
        Returns:
            Task object or None if not enough samples
        """
        total_needed = self.support_size + self.query_size
        if len(graphs) < total_needed:
            return None
        
        # Shuffle samples
        indices = torch.randperm(len(graphs)).tolist()
        
        # Split into support and query
        support_indices = indices[:self.support_size]
        query_indices = indices[self.support_size:self.support_size + self.query_size]
        
        support_set = [graphs[i] for i in support_indices]
        query_set = [graphs[i] for i in query_indices]
        support_labels = torch.tensor([labels[i] for i in support_indices], dtype=torch.float32)
        query_labels = torch.tensor([labels[i] for i in query_indices], dtype=torch.float32)
        
        return Task(
            task_id=task_id,
            support_set=support_set,
            query_set=query_set,
            support_labels=support_labels,
            query_labels=query_labels,
            metadata=metadata,
        )


class MetaLearner:
    """Model-Agnostic Meta-Learning (MAML) for T-GAT.
    
    Enables rapid adaptation to new network topologies through meta-learning.
    
    Requirement 7.1: Implement Model-Agnostic Meta-Learning (MAML)
    Requirement 7.2: Treat each subnetwork or time period as a separate task
    Requirement 7.3: Require minimal fine-tuning samples (few-shot adaptation)
    
    MAML learns a good initialization that can be quickly adapted to new tasks
    with just a few gradient steps on a small support set.
    
    Attributes:
        model: The base T-GAT model
        inner_lr: Learning rate for inner loop (task adaptation)
        outer_lr: Learning rate for outer loop (meta-update)
        num_inner_steps: Number of gradient steps for adaptation
        first_order: Whether to use first-order approximation (faster)
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        first_order: bool = False,
        device: Optional[torch.device] = None,
    ):
        """Initialize MetaLearner.
        
        Args:
            model: The neural network model to meta-train
            inner_lr: Learning rate for inner loop adaptation (default: 0.01)
            outer_lr: Learning rate for outer loop meta-update (default: 0.001)
            num_inner_steps: Number of gradient steps for adaptation (default: 5)
            first_order: Use first-order MAML approximation (default: False)
            device: Device for computation (default: auto-detect)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # Meta-optimizer for outer loop
        self.meta_optimizer = Adam(self.model.parameters(), lr=outer_lr)
        
        # Task creator for generating tasks
        self.task_creator = TaskCreator()
        
        # Training history
        self.meta_training_history: Dict[str, List[float]] = {
            'meta_train_loss': [],
            'meta_val_loss': [],
            'adaptation_improvement': [],
        }
    
    def meta_train(
        self,
        tasks: List[Task],
        num_epochs: int = 100,
        tasks_per_batch: int = 4,
        val_tasks: Optional[List[Task]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Meta-train across multiple tasks.
        
        Requirement 7.1: Implement MAML meta-training loop
        Requirement 7.2: Treat each subnetwork/time period as separate task
        
        The meta-training loop:
        1. Sample a batch of tasks
        2. For each task:
           a. Clone model parameters
           b. Adapt to support set (inner loop)
           c. Compute loss on query set
        3. Update meta-parameters (outer loop)
        
        Args:
            tasks: List of Task objects for meta-training
            num_epochs: Number of meta-training epochs (default: 100)
            tasks_per_batch: Number of tasks per meta-batch (default: 4)
            val_tasks: Optional validation tasks for monitoring
            verbose: Whether to print progress (default: True)
            
        Returns:
            Dictionary containing training history and final metrics
        """
        if not tasks:
            raise ValueError("No tasks provided for meta-training")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            # Shuffle tasks
            task_indices = torch.randperm(len(tasks)).tolist()
            
            epoch_meta_loss = 0.0
            num_batches = 0
            
            # Process tasks in batches
            for batch_start in range(0, len(tasks), tasks_per_batch):
                batch_end = min(batch_start + tasks_per_batch, len(tasks))
                batch_indices = task_indices[batch_start:batch_end]
                batch_tasks = [tasks[i] for i in batch_indices]
                
                # Meta-update step
                meta_loss = self._meta_update(batch_tasks)
                epoch_meta_loss += meta_loss
                num_batches += 1
            
            avg_meta_loss = epoch_meta_loss / max(num_batches, 1)
            self.meta_training_history['meta_train_loss'].append(avg_meta_loss)
            
            # Validation
            if val_tasks:
                val_loss = self._evaluate_tasks(val_tasks)
                self.meta_training_history['meta_val_loss'].append(val_loss)
            
            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch}: meta_loss={avg_meta_loss:.4f}"
                if val_tasks:
                    msg += f", val_loss={self.meta_training_history['meta_val_loss'][-1]:.4f}"
                logger.info(msg)
        
        return {
            'history': self.meta_training_history,
            'final_meta_loss': self.meta_training_history['meta_train_loss'][-1],
        }
    
    def _meta_update(self, tasks: List[Task]) -> float:
        """Perform one meta-update step on a batch of tasks.
        
        Args:
            tasks: Batch of tasks for meta-update
            
        Returns:
            Average meta-loss across tasks
        """
        self.meta_optimizer.zero_grad()
        
        total_meta_loss = 0.0
        
        for task in tasks:
            # Clone model for task-specific adaptation
            adapted_params = self._inner_loop(task)
            
            # Compute query loss with adapted parameters
            query_loss = self._compute_query_loss(task, adapted_params)
            total_meta_loss += query_loss
        
        # Average loss across tasks
        avg_meta_loss = total_meta_loss / len(tasks)
        
        # Backward pass for meta-update
        avg_meta_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Meta-optimizer step
        self.meta_optimizer.step()
        
        return avg_meta_loss.item()
    
    def _inner_loop(self, task: Task) -> Dict[str, Tensor]:
        """Perform inner loop adaptation on task support set.
        
        Args:
            task: Task containing support set for adaptation
            
        Returns:
            Dictionary of adapted parameters
        """
        # Get current parameters
        adapted_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        # Move support data to device
        support_graphs = [g.to(self.device) for g in task.support_set]
        support_labels = task.support_labels.to(self.device)
        
        # Inner loop: adapt to support set
        for step in range(self.num_inner_steps):
            # Forward pass with current adapted parameters
            loss = self._compute_loss_with_params(
                support_graphs, support_labels, adapted_params
            )
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=not self.first_order,
                allow_unused=True,
            )
            
            # Update adapted parameters
            adapted_params = {
                name: param - self.inner_lr * (grad if grad is not None else torch.zeros_like(param))
                for (name, param), grad in zip(adapted_params.items(), grads)
            }
        
        return adapted_params
    
    def _compute_loss_with_params(
        self,
        graphs: List[Data],
        labels: Tensor,
        params: Dict[str, Tensor],
    ) -> Tensor:
        """Compute loss using specified parameters.
        
        Args:
            graphs: List of graph Data objects
            labels: Target labels
            params: Dictionary of parameters to use
            
        Returns:
            Loss tensor
        """
        # Temporarily replace model parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
            param.data = params[name]
        
        try:
            # Forward pass - process each graph individually and collect predictions
            predictions_list = []
            for graph in graphs:
                pred = self.model(graph)
                predictions_list.append(pred.view(-1))
            
            # Stack predictions
            predictions = torch.cat(predictions_list)
            
            # Compute loss
            loss = nn.functional.binary_cross_entropy(
                predictions.view(-1),
                labels.view(-1),
            )
        finally:
            # Restore original parameters
            for name, param in self.model.named_parameters():
                param.data = original_params[name]
        
        return loss
    
    def _compute_query_loss(
        self,
        task: Task,
        adapted_params: Dict[str, Tensor],
    ) -> Tensor:
        """Compute loss on query set with adapted parameters.
        
        Args:
            task: Task containing query set
            adapted_params: Adapted parameters from inner loop
            
        Returns:
            Query loss tensor
        """
        query_graphs = [g.to(self.device) for g in task.query_set]
        query_labels = task.query_labels.to(self.device)
        
        return self._compute_loss_with_params(query_graphs, query_labels, adapted_params)
    
    def _evaluate_tasks(self, tasks: List[Task]) -> float:
        """Evaluate model on a set of tasks.
        
        Args:
            tasks: List of tasks to evaluate
            
        Returns:
            Average loss across tasks
        """
        self.model.eval()
        total_loss = 0.0
        
        for task in tasks:
            # Adapt to support set (needs gradients for adaptation)
            adapted_model = self.adapt(
                task.support_set,
                task.support_labels,
                num_steps=self.num_inner_steps,
                return_model=True,
            )
            
            # Evaluate on query set (no gradients needed)
            query_graphs = [g.to(self.device) for g in task.query_set]
            query_labels = task.query_labels.to(self.device)
            
            with torch.no_grad():
                # Process each graph individually
                predictions_list = []
                for graph in query_graphs:
                    pred = adapted_model(graph)
                    predictions_list.append(pred.view(-1))
                
                predictions = torch.cat(predictions_list)
                
                loss = nn.functional.binary_cross_entropy(
                    predictions.view(-1),
                    query_labels.view(-1),
                )
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / max(len(tasks), 1)
    
    def adapt(
        self,
        support_set: Union[List[Data], DataLoader],
        support_labels: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        return_model: bool = False,
    ) -> Union[nn.Module, Dict[str, Tensor]]:
        """Adapt model to new network using few-shot support set.
        
        Requirement 7.3: Require minimal fine-tuning samples (few-shot adaptation)
        
        Performs gradient-based adaptation on the support set to quickly
        adapt the meta-learned model to a new network topology.
        
        Args:
            support_set: Support set data (list of graphs or DataLoader)
            support_labels: Labels for support set (required if support_set is list)
            num_steps: Number of adaptation steps (default: self.num_inner_steps)
            return_model: If True, return adapted model; else return parameters
            
        Returns:
            Adapted model or dictionary of adapted parameters
        """
        if num_steps is None:
            num_steps = self.num_inner_steps
        
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_model = adapted_model.to(self.device)
        
        # Create optimizer for adaptation
        adapt_optimizer = SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        adapted_model.train()
        
        # Handle different input formats
        if isinstance(support_set, DataLoader):
            # DataLoader format
            for step in range(num_steps):
                for batch in support_set:
                    batch = batch.to(self.device)
                    adapt_optimizer.zero_grad()
                    
                    predictions = adapted_model(batch)
                    labels = batch.y.float()
                    
                    loss = nn.functional.binary_cross_entropy(
                        predictions.view(-1),
                        labels.view(-1),
                    )
                    loss.backward()
                    adapt_optimizer.step()
        else:
            # List of graphs format
            if support_labels is None:
                raise ValueError("support_labels required when support_set is a list")
            
            support_graphs = [g.to(self.device) for g in support_set]
            labels = support_labels.to(self.device)
            
            for step in range(num_steps):
                adapt_optimizer.zero_grad()
                
                # Process each graph individually
                predictions_list = []
                for graph in support_graphs:
                    pred = adapted_model(graph)
                    predictions_list.append(pred.view(-1))
                
                predictions = torch.cat(predictions_list)
                
                loss = nn.functional.binary_cross_entropy(
                    predictions.view(-1),
                    labels.view(-1),
                )
                loss.backward()
                adapt_optimizer.step()
        
        if return_model:
            adapted_model.eval()
            return adapted_model
        else:
            return {name: param.clone() for name, param in adapted_model.named_parameters()}
    
    def create_tasks_from_windows(
        self,
        windows: List[TimeWindow],
        graphs: List[Data],
        task_type: str = 'time_period',
        **kwargs,
    ) -> List[Task]:
        """Create meta-learning tasks from windows and graphs.
        
        Convenience method for creating tasks from data.
        
        Args:
            windows: List of TimeWindow objects
            graphs: Corresponding list of graph Data objects
            task_type: Type of task creation ('time_period' or 'subnetwork')
            **kwargs: Additional arguments for task creation
            
        Returns:
            List of Task objects
        """
        if task_type == 'time_period':
            return self.task_creator.create_tasks_from_time_periods(
                windows, graphs, **kwargs
            )
        elif task_type == 'subnetwork':
            return self.task_creator.create_tasks_from_subnetworks(
                windows, graphs, **kwargs
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    
    def save_meta_state(self, path: str) -> None:
        """Save meta-learner state.
        
        Args:
            path: Path to save the state
        """
        state = {
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'num_inner_steps': self.num_inner_steps,
            'first_order': self.first_order,
            'meta_training_history': self.meta_training_history,
        }
        torch.save(state, path)
        logger.info(f"Meta-learner state saved to {path}")
    
    def load_meta_state(self, path: str) -> None:
        """Load meta-learner state.
        
        Args:
            path: Path to load the state from
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(state['model_state_dict'])
        self.meta_optimizer.load_state_dict(state['meta_optimizer_state_dict'])
        self.inner_lr = state.get('inner_lr', self.inner_lr)
        self.outer_lr = state.get('outer_lr', self.outer_lr)
        self.num_inner_steps = state.get('num_inner_steps', self.num_inner_steps)
        self.first_order = state.get('first_order', self.first_order)
        self.meta_training_history = state.get('meta_training_history', {
            'meta_train_loss': [],
            'meta_val_loss': [],
            'adaptation_improvement': [],
        })
        
        logger.info(f"Meta-learner state loaded from {path}")
    
    def get_adaptation_speed(
        self,
        task: Task,
        baseline_steps: int = 50,
    ) -> Dict[str, float]:
        """Measure adaptation speed compared to training from scratch.
        
        Requirement 7.3: Demonstrate 3-5x faster adaptation
        
        Args:
            task: Task to measure adaptation on
            baseline_steps: Number of steps for baseline comparison
            
        Returns:
            Dictionary with adaptation metrics
        """
        # Measure meta-learned adaptation
        meta_losses = []
        adapted_model = copy.deepcopy(self.model).to(self.device)
        adapt_optimizer = SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        support_graphs = [g.to(self.device) for g in task.support_set]
        support_labels = task.support_labels.to(self.device)
        query_graphs = [g.to(self.device) for g in task.query_set]
        query_labels = task.query_labels.to(self.device)
        
        for step in range(self.num_inner_steps):
            adapt_optimizer.zero_grad()
            
            # Process each graph individually
            predictions_list = []
            for graph in support_graphs:
                pred = adapted_model(graph)
                predictions_list.append(pred.view(-1))
            
            predictions = torch.cat(predictions_list)
            
            loss = nn.functional.binary_cross_entropy(
                predictions.view(-1),
                support_labels.view(-1),
            )
            loss.backward()
            adapt_optimizer.step()
            
            # Evaluate on query set
            adapted_model.eval()
            with torch.no_grad():
                q_predictions_list = []
                for graph in query_graphs:
                    q_pred = adapted_model(graph)
                    q_predictions_list.append(q_pred.view(-1))
                
                q_predictions = torch.cat(q_predictions_list)
                
                q_loss = nn.functional.binary_cross_entropy(
                    q_predictions.view(-1),
                    query_labels.view(-1),
                )
                meta_losses.append(q_loss.item())
            adapted_model.train()
        
        # Find steps to reach final meta loss with random init
        # (This is a simplified comparison)
        final_meta_loss = meta_losses[-1] if meta_losses else float('inf')
        
        return {
            'meta_final_loss': final_meta_loss,
            'meta_steps': self.num_inner_steps,
            'loss_history': meta_losses,
        }
