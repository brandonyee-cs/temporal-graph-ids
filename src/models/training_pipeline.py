"""Training Pipeline for T-GAT lateral movement detection.

This module implements the TrainingPipeline class that handles:
- Training loop with Adam optimizer
- Gradient clipping with max norm 1.0
- Early stopping with configurable patience
- Checkpoint saving and loading
- Hyperparameter search with Optuna

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import os
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping handler to stop training when validation loss stops improving.
    
    Property 17: Early Stopping Behavior
    Requirement 8.3: Support early stopping with configurable patience
    
    Attributes:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        counter: Current count of epochs without improvement
        best_loss: Best validation loss seen so far
        should_stop: Whether training should stop
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Initialize EarlyStopping.
        
        Args:
            patience: Number of epochs to wait for improvement (default: 10)
            min_delta: Minimum change to qualify as improvement (default: 0.0)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_epoch = 0


class TrainingPipeline:
    """Training pipeline for T-GAT model.
    
    Orchestrates model training with:
    - Adam optimizer with configurable learning rate
    - Gradient clipping with max norm 1.0
    - Early stopping with configurable patience
    - Checkpoint saving and loading
    - Hyperparameter search with Optuna
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    
    Attributes:
        model: The T-GAT model to train
        optimizer: Adam optimizer instance
        lr: Learning rate
        grad_clip: Maximum gradient norm for clipping
        early_stopping: EarlyStopping handler
        device: Device for training (CPU/GPU)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: str = "adam",
        lr: float = 1e-4,
        grad_clip: float = 1.0,
        early_stopping_patience: int = 10,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize TrainingPipeline.
        
        Requirement 8.1: Use Adam optimizer with configurable learning rate
        
        Args:
            model: The neural network model to train
            optimizer: Optimizer type (currently only "adam" supported)
            lr: Learning rate (default: 1e-4, range: 1e-4 to 1e-3)
            grad_clip: Maximum gradient norm for clipping (default: 1.0)
            early_stopping_patience: Epochs to wait before early stopping (default: 10)
            weight_decay: L2 regularization weight (default: 0.0)
            device: Device for training (default: auto-detect)
        """
        self.model = model
        self.lr = lr
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        # Requirement 8.1: Use Adam optimizer
        if optimizer.lower() == "adam":
            self.optimizer = Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Only 'adam' is supported.")
        
        # Initialize learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # Initialize early stopping
        # Requirement 8.3: Support early stopping with configurable patience
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train model with early stopping.
        
        Property 17: Early Stopping Behavior
        Requirements: 8.1, 8.2, 8.3
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of epochs (default: 100)
            verbose: Whether to print training progress (default: True)
            
        Returns:
            Dictionary containing:
            - 'best_val_loss': Best validation loss achieved
            - 'best_epoch': Epoch with best validation loss
            - 'final_epoch': Final epoch number
            - 'history': Training history dictionary
            - 'stopped_early': Whether training stopped early
        """
        self.model.train()
        stopped_early = False
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            self.training_history['val_loss'].append(val_loss)
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['learning_rate'].append(current_lr)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
            # Check early stopping
            # Property 17: Training stops after patience epochs without improvement
            if self.early_stopping(val_loss, epoch):
                if verbose:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best epoch: {self.early_stopping.best_epoch}"
                    )
                stopped_early = True
                break
            
            if verbose and epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, lr={current_lr:.6f}"
                )
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.early_stopping.best_epoch,
            'final_epoch': self.current_epoch,
            'history': self.training_history,
            'stopped_early': stopped_early,
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch.
        
        Property 16: Gradient Clipping Enforcement
        Requirement 8.2: Implement gradient clipping with max norm 1.0
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                # PyTorch Geometric batch
                predictions = self.model(batch)
                targets = batch.y.float()
            else:
                # Assume it's a tuple/list of (graphs, targets)
                graphs, targets = batch
                predictions = self.model(graphs)
                targets = targets.float().to(self.device)
            
            # Compute loss
            if hasattr(self.model, 'compute_loss'):
                loss_dict = self.model.compute_loss(predictions, targets)
                loss = loss_dict['total_loss']
            else:
                loss = nn.functional.binary_cross_entropy(
                    predictions.view(-1),
                    targets.view(-1),
                )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            # Property 16: Gradient norm is clipped to max value
            # Requirement 8.2: Gradient clipping with max norm 1.0
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip,
                )
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Run one validation epoch.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                    predictions = self.model(batch)
                    targets = batch.y.float()
                else:
                    graphs, targets = batch
                    predictions = self.model(graphs)
                    targets = targets.float().to(self.device)
                
                # Compute loss
                if hasattr(self.model, 'compute_loss'):
                    loss_dict = self.model.compute_loss(predictions, targets)
                    loss = loss_dict['total_loss']
                else:
                    loss = nn.functional.binary_cross_entropy(
                        predictions.view(-1),
                        targets.view(-1),
                    )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _move_to_device(self, batch: Any) -> Any:
        """Move batch data to the training device.
        
        Args:
            batch: Batch data (Data object, tuple, or tensor)
            
        Returns:
            Batch data on the correct device
        """
        if isinstance(batch, Data):
            return batch.to(self.device)
        elif isinstance(batch, (tuple, list)):
            return [self._move_to_device(item) for item in batch]
        elif isinstance(batch, Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        return batch

    def save_checkpoint(self, path: str, include_optimizer: bool = True) -> None:
        """Save model checkpoint and training state.
        
        Property 18: Checkpoint Persistence Round-Trip
        Requirement 8.5: Save model checkpoints and training metrics
        
        Args:
            path: Path to save the checkpoint
            include_optimizer: Whether to include optimizer state (default: True)
        """
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'early_stopping_state': {
                'counter': self.early_stopping.counter,
                'best_loss': self.early_stopping.best_loss,
                'best_epoch': self.early_stopping.best_epoch,
            },
            'lr': self.lr,
            'grad_clip': self.grad_clip,
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save model config if available
        if hasattr(self.model, 'get_config'):
            checkpoint['model_config'] = self.model.get_config()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load model checkpoint and training state.
        
        Property 18: Checkpoint Persistence Round-Trip
        Requirement 8.5: Load model checkpoints
        
        Args:
            path: Path to the checkpoint file
            load_optimizer: Whether to load optimizer state (default: True)
            strict: Whether to strictly enforce state dict matching (default: True)
            
        Returns:
            Dictionary containing loaded checkpoint information
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        })
        
        # Load early stopping state
        es_state = checkpoint.get('early_stopping_state', {})
        self.early_stopping.counter = es_state.get('counter', 0)
        self.early_stopping.best_loss = es_state.get('best_loss', None)
        self.early_stopping.best_epoch = es_state.get('best_epoch', 0)
        
        # Load optimizer state if requested
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_optimizer and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
        
        return {
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'model_config': checkpoint.get('model_config', {}),
        }
    
    def get_gradient_norm(self) -> float:
        """Get the current gradient norm of model parameters.
        
        Useful for monitoring gradient clipping behavior.
        
        Returns:
            Total gradient norm across all parameters
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def clip_gradients(self) -> float:
        """Manually clip gradients and return the norm before clipping.
        
        Property 16: Gradient Clipping Enforcement
        Requirement 8.2: Gradient clipping with max norm 1.0
        
        Returns:
            Gradient norm before clipping
        """
        # Get norm before clipping
        norm_before = self.get_gradient_norm()
        
        # Clip gradients
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip,
            )
        
        return norm_before
    
    def reset_training_state(self) -> None:
        """Reset training state for a fresh training run."""
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
        self.early_stopping.reset()
        
        # Reset optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Reset scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )

    def hyperparameter_search(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        search_space: Optional[Dict[str, Any]] = None,
        n_trials: int = 100,
        model_factory: Optional[Callable[[Dict[str, Any]], nn.Module]] = None,
        timeout: Optional[int] = None,
        study_name: str = "tgat_hpo",
        storage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run Bayesian hyperparameter optimization with Optuna.
        
        Requirement 8.4: Implement hyperparameter search using Bayesian optimization
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            search_space: Dictionary defining the search space (optional)
            n_trials: Number of optimization trials (default: 100)
            model_factory: Function to create model from hyperparameters
            timeout: Maximum time in seconds for optimization (optional)
            study_name: Name for the Optuna study (default: "tgat_hpo")
            storage: Optuna storage URL for persistence (optional)
            
        Returns:
            Dictionary containing:
            - 'best_params': Best hyperparameters found
            - 'best_value': Best AUC-PR achieved
            - 'study': Optuna study object
            
        Raises:
            ImportError: If Optuna is not installed
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter search. "
                "Install it with: pip install optuna"
            )
        
        # Default search space
        if search_space is None:
            search_space = self._get_default_search_space()
        
        def objective(trial: Trial) -> float:
            """Optuna objective function optimizing AUC-PR."""
            # Sample hyperparameters
            params = self._sample_hyperparameters(trial, search_space)
            
            # Create model with sampled hyperparameters
            if model_factory is not None:
                model = model_factory(params)
            else:
                # Use current model with updated learning rate
                model = self.model
            
            model = model.to(self.device)
            
            # Create optimizer with sampled learning rate
            optimizer = Adam(
                model.parameters(),
                lr=params.get('lr', self.lr),
                weight_decay=params.get('weight_decay', 0.0),
            )
            
            # Train for a few epochs
            num_epochs = params.get('num_epochs', 20)
            patience = params.get('early_stopping_patience', 5)
            
            early_stopping = EarlyStopping(patience=patience)
            best_auc_pr = 0.0
            
            for epoch in range(num_epochs):
                # Training
                model.train()
                for batch in train_loader:
                    batch = self._move_to_device(batch)
                    optimizer.zero_grad()
                    
                    if hasattr(batch, 'x'):
                        predictions = model(batch)
                        targets = batch.y.float()
                    else:
                        graphs, targets = batch
                        predictions = model(graphs)
                        targets = targets.float().to(self.device)
                    
                    loss = nn.functional.binary_cross_entropy(
                        predictions.view(-1),
                        targets.view(-1),
                    )
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=params.get('grad_clip', 1.0),
                    )
                    optimizer.step()
                
                # Validation and compute AUC-PR
                model.eval()
                all_preds = []
                all_targets = []
                val_loss = 0.0
                num_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = self._move_to_device(batch)
                        
                        if hasattr(batch, 'x'):
                            predictions = model(batch)
                            targets = batch.y.float()
                        else:
                            graphs, targets = batch
                            predictions = model(graphs)
                            targets = targets.float().to(self.device)
                        
                        loss = nn.functional.binary_cross_entropy(
                            predictions.view(-1),
                            targets.view(-1),
                        )
                        val_loss += loss.item()
                        num_batches += 1
                        
                        all_preds.extend(predictions.view(-1).cpu().numpy())
                        all_targets.extend(targets.view(-1).cpu().numpy())
                
                val_loss /= max(num_batches, 1)
                
                # Compute AUC-PR
                if len(set(all_targets)) > 1:  # Need both classes
                    auc_pr = average_precision_score(all_targets, all_preds)
                    best_auc_pr = max(best_auc_pr, auc_pr)
                
                # Early stopping check
                if early_stopping(val_loss, epoch):
                    break
                
                # Optuna pruning
                trial.report(best_auc_pr, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return best_auc_pr
        
        # Create or load study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",  # Maximize AUC-PR
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best AUC-PR: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study,
        }
    
    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default hyperparameter search space.
        
        Returns:
            Dictionary defining the search space
        """
        return {
            'lr': {
                'type': 'loguniform',
                'low': 1e-5,
                'high': 1e-2,
            },
            'weight_decay': {
                'type': 'loguniform',
                'low': 1e-6,
                'high': 1e-2,
            },
            'grad_clip': {
                'type': 'uniform',
                'low': 0.5,
                'high': 2.0,
            },
            'dropout': {
                'type': 'uniform',
                'low': 0.0,
                'high': 0.5,
            },
            'hidden_dim': {
                'type': 'categorical',
                'choices': [64, 128, 256],
            },
            'num_layers': {
                'type': 'int',
                'low': 1,
                'high': 4,
            },
            'num_heads': {
                'type': 'categorical',
                'choices': [2, 4, 8],
            },
            'early_stopping_patience': {
                'type': 'int',
                'low': 3,
                'high': 15,
            },
        }
    
    def _sample_hyperparameters(
        self,
        trial: "Trial",
        search_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Sample hyperparameters from search space using Optuna trial.
        
        Args:
            trial: Optuna trial object
            search_space: Dictionary defining the search space
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        params = {}
        
        for name, config in search_space.items():
            param_type = config.get('type', 'uniform')
            
            if param_type == 'uniform':
                params[name] = trial.suggest_float(
                    name,
                    config['low'],
                    config['high'],
                )
            elif param_type == 'loguniform':
                params[name] = trial.suggest_float(
                    name,
                    config['low'],
                    config['high'],
                    log=True,
                )
            elif param_type == 'int':
                params[name] = trial.suggest_int(
                    name,
                    config['low'],
                    config['high'],
                )
            elif param_type == 'categorical':
                params[name] = trial.suggest_categorical(
                    name,
                    config['choices'],
                )
        
        return params
    
    def evaluate_auc_pr(
        self,
        data_loader: DataLoader,
    ) -> float:
        """Evaluate model and compute AUC-PR.
        
        Args:
            data_loader: DataLoader for evaluation data
            
        Returns:
            AUC-PR score
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_to_device(batch)
                
                if hasattr(batch, 'x'):
                    predictions = self.model(batch)
                    targets = batch.y.float()
                else:
                    graphs, targets = batch
                    predictions = self.model(graphs)
                    targets = targets.float().to(self.device)
                
                all_preds.extend(predictions.view(-1).cpu().numpy())
                all_targets.extend(targets.view(-1).cpu().numpy())
        
        if len(set(all_targets)) > 1:
            return average_precision_score(all_targets, all_preds)
        return 0.0
