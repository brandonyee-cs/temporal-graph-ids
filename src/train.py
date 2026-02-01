#!/usr/bin/env python
"""Main training script for T-GAT lateral movement detection.

This script wires together the full training pipeline:
DataPreprocessor → GraphConstructor → TGAT → TrainingPipeline

Usage:
    python -m src.train --config configs/default.yaml --data-path /path/to/auth.txt
    python -m src.train --help

Requirements: 8.1, 8.5
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.data.graph_constructor import GraphConstructor
from src.data.models import TimeWindow
from src.models.tgat import TGAT
from src.models.training_pipeline import TrainingPipeline
from src.models.evaluation import EvaluationModule


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str, experiment_name: str) -> None:
    """Setup file logging for experiment tracking.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"{experiment_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")


def windows_to_graphs(
    windows: List[TimeWindow],
    graph_constructor: GraphConstructor,
) -> List[Data]:
    """Convert TimeWindows to PyTorch Geometric Data objects.
    
    Args:
        windows: List of TimeWindow objects
        graph_constructor: GraphConstructor instance
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    graphs = []
    for window in windows:
        graph = graph_constructor.build_graph(window)
        graphs.append(graph)
        # Update history for future windows
        graph_constructor.update_history(window)
    return graphs


def create_data_loaders(
    train_graphs: List[Data],
    val_graphs: List[Data],
    test_graphs: List[Data],
    batch_size: int = 32,
) -> tuple:
    """Create DataLoaders for training, validation, and testing.
    
    Args:
        train_graphs: List of training graphs
        val_graphs: List of validation graphs
        test_graphs: List of test graphs
        batch_size: Batch size for DataLoaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def create_model(config: Dict[str, Any]) -> TGAT:
    """Create T-GAT model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized TGAT model
    """
    data_config = config.get('data', {})
    encoder_config = config.get('encoder', {})
    temporal_config = config.get('temporal', {})
    uncertainty_config = config.get('uncertainty', {})
    physics_config = config.get('physics', {})
    
    model = TGAT(
        node_feature_dim=data_config.get('node_feature_dim', 64),
        edge_feature_dim=data_config.get('edge_feature_dim', 32),
        hidden_dim=encoder_config.get('hidden_channels', 128),
        output_dim=encoder_config.get('out_channels', 64),
        num_gat_layers=encoder_config.get('num_layers', 2),
        num_attention_heads=encoder_config.get('heads', 4),
        num_lstm_layers=temporal_config.get('num_layers', 2),
        lstm_hidden_dim=temporal_config.get('hidden_dim', 256),
        dropout=encoder_config.get('dropout', 0.1),
        num_mc_samples=uncertainty_config.get('num_samples', 50),
        mc_dropout_rate=uncertainty_config.get('dropout_rate', 0.2),
        min_auth_delta=physics_config.get('min_auth_delta', 0.1),
        energy_weight=physics_config.get('energy_weight', 0.1),
        constraint_weight=physics_config.get('constraint_weight', 0.01),
    )
    
    return model


def train(
    config: Dict[str, Any],
    data_path: str,
    redteam_path: Optional[str] = None,
    output_dir: str = "outputs",
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full training pipeline.
    
    Wires together: DataPreprocessor → GraphConstructor → TGAT → TrainingPipeline
    
    Args:
        config: Configuration dictionary
        data_path: Path to authentication data file
        redteam_path: Path to red team ground truth file (optional)
        output_dir: Directory for outputs (checkpoints, logs)
        experiment_name: Name for the experiment
        
    Returns:
        Dictionary containing training results
    """
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"tgat_experiment_{timestamp}"
    
    # Setup directories
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_path / "logs"
    
    # Setup logging
    setup_logging(str(log_dir), experiment_name)
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Output directory: {output_path}")
    
    # Save configuration
    config_save_path = output_path / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Configuration saved to {config_save_path}")
    
    # Get configuration values
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Step 1: Initialize DataPreprocessor
    logger.info("Initializing DataPreprocessor...")
    preprocessor = DataPreprocessor(
        window_size=data_config.get('window_size', 3600),
        overlap=data_config.get('overlap', 1800),
    )
    
    # Step 2: Load and preprocess data
    logger.info(f"Loading data from {data_path}...")
    events_df = preprocessor.load_events(data_path)
    logger.info(f"Loaded {len(events_df)} authentication events")
    
    # Step 3: Create temporal windows
    logger.info("Creating temporal windows...")
    windows = preprocessor.create_windows(events_df)
    logger.info(f"Created {len(windows)} temporal windows")
    
    # Step 4: Extract labels if red team file provided
    if redteam_path and os.path.exists(redteam_path):
        logger.info(f"Extracting labels from {redteam_path}...")
        windows = preprocessor.extract_labels(windows, redteam_path)
        malicious_count = sum(1 for w in windows if w.label == 1)
        logger.info(f"Found {malicious_count} malicious windows out of {len(windows)}")
    
    # Step 5: Create temporal splits
    logger.info("Creating temporal train/val/test splits...")
    
    # Calculate actual number of days in the data
    if windows:
        min_time = min(w.start_time for w in windows)
        max_time = max(w.end_time for w in windows)
        total_days = max(1, int((max_time - min_time) / 86400) + 1)
        
        # Use 70/15/15 split based on actual days
        train_end_day = max(1, int(total_days * 0.7))
        val_end_day = max(train_end_day + 1, int(total_days * 0.85))
        
        train_windows, val_windows, test_windows = preprocessor.split_temporal(
            windows,
            train_days=(1, train_end_day),
            val_days=(train_end_day + 1, val_end_day),
            test_days=(val_end_day + 1, total_days)
        )
        logger.info(f"Data spans {total_days} days. Split: train=days 1-{train_end_day}, val=days {train_end_day+1}-{val_end_day}, test=days {val_end_day+1}-{total_days}")
    else:
        train_windows, val_windows, test_windows = [], [], []
    
    logger.info(f"Split sizes - Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")
    
    # Step 6: Initialize GraphConstructor
    logger.info("Initializing GraphConstructor...")
    graph_constructor = GraphConstructor(
        node_feature_dim=data_config.get('node_feature_dim', 64),
        edge_feature_dim=data_config.get('edge_feature_dim', 32),
    )
    
    # Step 7: Convert windows to graphs
    logger.info("Converting windows to graphs...")
    train_graphs = windows_to_graphs(train_windows, graph_constructor)
    graph_constructor.clear_history()  # Reset for validation
    val_graphs = windows_to_graphs(val_windows, graph_constructor)
    graph_constructor.clear_history()  # Reset for test
    test_graphs = windows_to_graphs(test_windows, graph_constructor)
    
    logger.info(f"Created {len(train_graphs)} training graphs, {len(val_graphs)} validation graphs, {len(test_graphs)} test graphs")
    
    # Step 8: Create DataLoaders
    batch_size = training_config.get('batch_size', 32)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_graphs, val_graphs, test_graphs, batch_size
    )
    
    # Step 9: Create model
    logger.info("Creating T-GAT model...")
    model = create_model(config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Step 10: Initialize TrainingPipeline
    logger.info("Initializing TrainingPipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    pipeline = TrainingPipeline(
        model=model,
        optimizer=training_config.get('optimizer', 'adam'),
        lr=training_config.get('learning_rate', 1e-4),
        grad_clip=training_config.get('grad_clip', 1.0),
        early_stopping_patience=training_config.get('early_stopping_patience', 10),
        device=device,
    )
    
    # Step 11: Train model
    logger.info("Starting training...")
    max_epochs = training_config.get('max_epochs', 100)
    
    training_results = pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=max_epochs,
        verbose=True,
    )
    
    logger.info(f"Training completed at epoch {training_results['final_epoch']}")
    logger.info(f"Best validation loss: {training_results['best_val_loss']:.4f} at epoch {training_results['best_epoch']}")
    
    # Step 12: Save final checkpoint
    checkpoint_path = checkpoint_dir / "final_model.pt"
    pipeline.save_checkpoint(str(checkpoint_path))
    logger.info(f"Final checkpoint saved to {checkpoint_path}")
    
    # Save best checkpoint
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    pipeline.save_checkpoint(str(best_checkpoint_path))
    
    # Step 13: Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluation = EvaluationModule()
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            predictions = model(batch)
            all_predictions.extend(predictions.view(-1).cpu().numpy())
            all_targets.extend(batch.y.view(-1).cpu().numpy())
    
    # Compute metrics
    predictions_tensor = torch.tensor(all_predictions)
    targets_tensor = torch.tensor(all_targets)
    
    detection_metrics = evaluation.compute_detection_metrics(
        predictions_tensor, targets_tensor
    )
    
    logger.info("Test Set Metrics:")
    logger.info(f"  Precision: {detection_metrics.precision:.4f}")
    logger.info(f"  Recall: {detection_metrics.recall:.4f}")
    logger.info(f"  F1 Score: {detection_metrics.f1_score:.4f}")
    logger.info(f"  AUC-ROC: {detection_metrics.auc_roc:.4f}")
    logger.info(f"  AUC-PR: {detection_metrics.auc_pr:.4f}")
    
    # Save metrics
    metrics_path = output_path / "metrics.yaml"
    metrics_dict = {
        'training': {
            'best_val_loss': float(training_results['best_val_loss']),
            'best_epoch': int(training_results['best_epoch']),
            'final_epoch': int(training_results['final_epoch']),
            'stopped_early': training_results['stopped_early'],
        },
        'test': {
            'precision': float(detection_metrics.precision),
            'recall': float(detection_metrics.recall),
            'f1_score': float(detection_metrics.f1_score),
            'auc_roc': float(detection_metrics.auc_roc),
            'auc_pr': float(detection_metrics.auc_pr),
        }
    }
    
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics_dict, f)
    logger.info(f"Metrics saved to {metrics_path}")
    
    return {
        'experiment_name': experiment_name,
        'output_dir': str(output_path),
        'training_results': training_results,
        'test_metrics': detection_metrics,
        'checkpoint_path': str(checkpoint_path),
    }


def run_hyperparameter_search(
    config: Dict[str, Any],
    data_path: str,
    redteam_path: Optional[str] = None,
    output_dir: str = "outputs",
    n_trials: int = 100,
) -> Dict[str, Any]:
    """Run hyperparameter search using Optuna.
    
    Args:
        config: Base configuration dictionary
        data_path: Path to authentication data file
        redteam_path: Path to red team ground truth file
        output_dir: Directory for outputs
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary containing best hyperparameters and results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"tgat_hpo_{timestamp}"
    
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    setup_logging(str(output_path / "logs"), experiment_name)
    
    logger.info(f"Starting hyperparameter search: {experiment_name}")
    
    # Load and preprocess data (same as train function)
    data_config = config.get('data', {})
    
    preprocessor = DataPreprocessor(
        window_size=data_config.get('window_size', 3600),
        overlap=data_config.get('overlap', 1800),
    )
    
    events_df = preprocessor.load_events(data_path)
    windows = preprocessor.create_windows(events_df)
    
    if redteam_path and os.path.exists(redteam_path):
        windows = preprocessor.extract_labels(windows, redteam_path)
    
    train_windows, val_windows, _ = preprocessor.split_temporal(windows)
    
    graph_constructor = GraphConstructor(
        node_feature_dim=data_config.get('node_feature_dim', 64),
        edge_feature_dim=data_config.get('edge_feature_dim', 32),
    )
    
    train_graphs = windows_to_graphs(train_windows, graph_constructor)
    graph_constructor.clear_history()
    val_graphs = windows_to_graphs(val_windows, graph_constructor)
    
    batch_size = config.get('training', {}).get('batch_size', 32)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    
    # Create base model for HPO
    model = create_model(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = TrainingPipeline(model=model, device=device)
    
    # Define model factory for HPO
    def model_factory(params: Dict[str, Any]) -> TGAT:
        return TGAT(
            node_feature_dim=data_config.get('node_feature_dim', 64),
            edge_feature_dim=data_config.get('edge_feature_dim', 32),
            hidden_dim=params.get('hidden_dim', 128),
            num_gat_layers=params.get('num_layers', 2),
            num_attention_heads=params.get('num_heads', 4),
            dropout=params.get('dropout', 0.1),
        )
    
    # Run hyperparameter search
    hpo_results = pipeline.hyperparameter_search(
        train_loader=train_loader,
        val_loader=val_loader,
        n_trials=n_trials,
        model_factory=model_factory,
    )
    
    # Save results
    results_path = output_path / "hpo_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump({
            'best_params': hpo_results['best_params'],
            'best_value': float(hpo_results['best_value']),
        }, f)
    
    logger.info(f"HPO results saved to {results_path}")
    
    return hpo_results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train T-GAT model for lateral movement detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to authentication data file (LANL format)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    
    parser.add_argument(
        "--redteam-path",
        type=str,
        default=None,
        help="Path to red team ground truth file",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for outputs (checkpoints, logs, metrics)",
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for the experiment (auto-generated if not provided)",
    )
    
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Run hyperparameter optimization instead of training",
    )
    
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=100,
        help="Number of HPO trials",
    )
    
    # Override config values
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs from config",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for training",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for training script."""
    args = parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {}
    
    # Override config with command-line arguments
    if args.learning_rate is not None:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    
    if args.max_epochs is not None:
        config.setdefault('training', {})['max_epochs'] = args.max_epochs
    
    if args.device != "auto":
        config.setdefault('inference', {})['device'] = args.device
    
    # Run training or HPO
    if args.hpo:
        logger.info("Running hyperparameter optimization...")
        results = run_hyperparameter_search(
            config=config,
            data_path=args.data_path,
            redteam_path=args.redteam_path,
            output_dir=args.output_dir,
            n_trials=args.hpo_trials,
        )
        logger.info(f"Best AUC-PR: {results['best_value']:.4f}")
        logger.info(f"Best parameters: {results['best_params']}")
    else:
        logger.info("Running training...")
        results = train(
            config=config,
            data_path=args.data_path,
            redteam_path=args.redteam_path,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
        )
        logger.info(f"Training completed. Results saved to {results['output_dir']}")


if __name__ == "__main__":
    main()
