#!/usr/bin/env python
"""Inference script for T-GAT lateral movement detection.

This script wires together the inference pipeline:
Data loading → InferenceEngine → Alert output

Supports both batch and streaming inference modes.

Usage:
    # Batch inference
    python -m src.infer --checkpoint outputs/model.pt --data-path /path/to/auth.txt
    
    # Streaming inference
    python -m src.infer --checkpoint outputs/model.pt --data-path /path/to/auth.txt --streaming
    
    python -m src.infer --help

Requirements: 9.1, 9.4
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator

import torch
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.data.graph_constructor import GraphConstructor
from src.data.models import TimeWindow, Alert, DetectionResult
from src.models.tgat import TGAT
from src.models.inference_engine import InferenceEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> TGAT:
    """Load T-GAT model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device for inference ('cpu', 'cuda', or None for auto)
        
    Returns:
        Loaded TGAT model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    model_config = checkpoint.get('model_config', {})
    
    # Create model with saved config
    model = TGAT(
        node_feature_dim=model_config.get('node_feature_dim', 64),
        edge_feature_dim=model_config.get('edge_feature_dim', 32),
        hidden_dim=model_config.get('hidden_dim', 128),
        output_dim=model_config.get('output_dim', 64),
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"Using device: {device}")
    
    return model


def stream_windows(
    data_path: str,
    preprocessor: DataPreprocessor,
    chunk_size: int = 10000,
) -> Iterator[List[TimeWindow]]:
    """Stream windows from data file in chunks.
    
    Enables processing large files without loading everything into memory.
    
    Args:
        data_path: Path to authentication data file
        preprocessor: DataPreprocessor instance
        chunk_size: Number of events per chunk
        
    Yields:
        Lists of TimeWindow objects
    """
    import pandas as pd
    
    # Read file in chunks
    column_names = [
        'timestamp', 'source_user', 'dest_user', 
        'source_computer', 'dest_computer',
        'auth_type', 'logon_type', 'auth_orientation', 'success'
    ]
    
    for chunk in pd.read_csv(
        data_path, 
        names=column_names, 
        header=None, 
        chunksize=chunk_size
    ):
        # Process chunk
        chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
        chunk['success'] = chunk['success'].apply(
            lambda x: x.lower() == 'success' if isinstance(x, str) else bool(x)
        )
        chunk['source_user'] = chunk['source_user'].apply(
            lambda x: x.split('@')[0] if isinstance(x, str) and '@' in x else str(x)
        )
        chunk = chunk.dropna(subset=['timestamp'])
        
        # Create windows from chunk
        windows = preprocessor.create_windows(chunk)
        
        if windows:
            yield windows


def run_batch_inference(
    engine: InferenceEngine,
    windows: List[TimeWindow],
    output_path: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run batch inference on all windows.
    
    Args:
        engine: InferenceEngine instance
        windows: List of TimeWindow objects
        output_path: Optional path to save results
        threshold: Detection threshold for alerts
        
    Returns:
        Dictionary containing results and statistics
    """
    logger.info(f"Running batch inference on {len(windows)} windows...")
    
    start_time = time.time()
    
    # Run inference
    results, alerts = engine.infer_batch(windows, return_alerts=True)
    
    elapsed_time = time.time() - start_time
    avg_time_per_window = elapsed_time / len(windows) if windows else 0
    
    logger.info(f"Inference completed in {elapsed_time:.2f}s")
    logger.info(f"Average time per window: {avg_time_per_window*1000:.2f}ms")
    logger.info(f"Generated {len(alerts)} alerts")
    
    # Prepare output
    output = {
        'statistics': {
            'total_windows': len(windows),
            'total_alerts': len(alerts),
            'elapsed_time_seconds': elapsed_time,
            'avg_time_per_window_ms': avg_time_per_window * 1000,
            'threshold': threshold,
        },
        'alerts': [
            {
                'window_id': alert.window_id,
                'timestamp': alert.timestamp,
                'probability': alert.probability,
                'epistemic_uncertainty': alert.epistemic_uncertainty,
                'aleatoric_uncertainty': alert.aleatoric_uncertainty,
                'severity': alert.severity,
                'constraint_violations': alert.constraint_violations,
            }
            for alert in alerts
        ],
        'results': [
            {
                'window_id': result.window_id,
                'timestamp': result.timestamp,
                'probability': result.probability,
                'epistemic_uncertainty': result.epistemic_uncertainty,
                'aleatoric_uncertainty': result.aleatoric_uncertainty,
                'energy_score': result.energy_score,
            }
            for result in results
        ]
    }
    
    # Save results if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return output


def run_streaming_inference(
    engine: InferenceEngine,
    data_path: str,
    preprocessor: DataPreprocessor,
    graph_constructor: GraphConstructor,
    output_path: Optional[str] = None,
    threshold: float = 0.5,
    chunk_size: int = 10000,
) -> Dict[str, Any]:
    """Run streaming inference with hidden state management.
    
    Processes data in chunks while maintaining LSTM hidden state
    for temporal continuity.
    
    Args:
        engine: InferenceEngine instance
        data_path: Path to authentication data file
        preprocessor: DataPreprocessor instance
        graph_constructor: GraphConstructor instance
        output_path: Optional path to save results
        threshold: Detection threshold for alerts
        chunk_size: Number of events per chunk
        
    Returns:
        Dictionary containing results and statistics
    """
    logger.info("Running streaming inference...")
    
    all_results = []
    all_alerts = []
    total_windows = 0
    start_time = time.time()
    
    # Reset hidden state at start
    engine.reset_hidden_state()
    
    # Process data in streaming fashion
    for chunk_idx, windows in enumerate(stream_windows(data_path, preprocessor, chunk_size)):
        logger.info(f"Processing chunk {chunk_idx + 1} with {len(windows)} windows...")
        
        for i, window in enumerate(windows):
            # Build graph
            graph = graph_constructor.build_graph(window)
            graph_constructor.update_history(window)
            
            # Generate window ID
            window_id = f"stream_{chunk_idx}_{i}_{int(window.start_time)}"
            
            # Run streaming inference (maintains hidden state)
            result, _ = engine.infer_streaming(
                graph=graph,
                window_id=window_id,
                timestamp=window.start_time,
                reset_state=False,  # Keep hidden state
            )
            
            all_results.append(result)
            total_windows += 1
            
            # Generate alert if above threshold
            alert = engine.generate_alert(result, threshold)
            if alert is not None:
                all_alerts.append(alert)
                logger.info(
                    f"ALERT: {alert.window_id} - "
                    f"prob={alert.probability:.3f}, "
                    f"severity={alert.severity}"
                )
    
    elapsed_time = time.time() - start_time
    avg_time_per_window = elapsed_time / total_windows if total_windows > 0 else 0
    
    logger.info(f"Streaming inference completed in {elapsed_time:.2f}s")
    logger.info(f"Processed {total_windows} windows")
    logger.info(f"Average time per window: {avg_time_per_window*1000:.2f}ms")
    logger.info(f"Generated {len(all_alerts)} alerts")
    
    # Prepare output
    output = {
        'statistics': {
            'total_windows': total_windows,
            'total_alerts': len(all_alerts),
            'elapsed_time_seconds': elapsed_time,
            'avg_time_per_window_ms': avg_time_per_window * 1000,
            'threshold': threshold,
            'mode': 'streaming',
        },
        'alerts': [
            {
                'window_id': alert.window_id,
                'timestamp': alert.timestamp,
                'probability': alert.probability,
                'epistemic_uncertainty': alert.epistemic_uncertainty,
                'aleatoric_uncertainty': alert.aleatoric_uncertainty,
                'severity': alert.severity,
                'constraint_violations': alert.constraint_violations,
            }
            for alert in all_alerts
        ],
    }
    
    # Save results if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return output


def infer(
    checkpoint_path: str,
    data_path: str,
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    threshold: float = 0.5,
    streaming: bool = False,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Main inference function.
    
    Wires together: Data loading → InferenceEngine → Alert output
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to authentication data file
        config_path: Optional path to configuration file
        output_path: Optional path to save results
        threshold: Detection threshold for alerts
        streaming: Whether to use streaming mode
        device: Device for inference
        
    Returns:
        Dictionary containing inference results
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    data_config = config.get('data', {})
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # Initialize components
    preprocessor = DataPreprocessor(
        window_size=data_config.get('window_size', 3600),
        overlap=data_config.get('overlap', 1800),
    )
    
    graph_constructor = GraphConstructor(
        node_feature_dim=data_config.get('node_feature_dim', 64),
        edge_feature_dim=data_config.get('edge_feature_dim', 32),
    )
    
    # Create inference engine
    engine = InferenceEngine(
        model=model,
        device=device,
        graph_constructor=graph_constructor,
        detection_threshold=threshold,
    )
    
    if streaming:
        # Streaming mode with hidden state
        return run_streaming_inference(
            engine=engine,
            data_path=data_path,
            preprocessor=preprocessor,
            graph_constructor=graph_constructor,
            output_path=output_path,
            threshold=threshold,
        )
    else:
        # Batch mode
        logger.info(f"Loading data from {data_path}...")
        events_df = preprocessor.load_events(data_path)
        logger.info(f"Loaded {len(events_df)} events")
        
        windows = preprocessor.create_windows(events_df)
        logger.info(f"Created {len(windows)} windows")
        
        return run_batch_inference(
            engine=engine,
            windows=windows,
            output_path=output_path,
            threshold=threshold,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run T-GAT inference for lateral movement detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )
    
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
        default=None,
        help="Path to configuration YAML file",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save inference results (JSON format)",
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold for generating alerts",
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode with hidden state management",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for inference",
    )
    
    parser.add_argument(
        "--alerts-only",
        action="store_true",
        help="Only output alerts (not all results)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for inference script."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine device
    device = None if args.device == "auto" else args.device
    
    # Generate output path if not provided
    output_path = args.output
    if output_path is None and not args.alerts_only:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"inference_results_{timestamp}.json"
    
    # Run inference
    results = infer(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        config_path=args.config,
        output_path=output_path,
        threshold=args.threshold,
        streaming=args.streaming,
        device=device,
    )
    
    # Print summary
    stats = results['statistics']
    print("\n" + "="*50)
    print("INFERENCE SUMMARY")
    print("="*50)
    print(f"Total windows processed: {stats['total_windows']}")
    print(f"Total alerts generated: {stats['total_alerts']}")
    print(f"Elapsed time: {stats['elapsed_time_seconds']:.2f}s")
    print(f"Avg time per window: {stats['avg_time_per_window_ms']:.2f}ms")
    print(f"Detection threshold: {stats['threshold']}")
    
    if results['alerts']:
        print("\n" + "-"*50)
        print("ALERTS:")
        print("-"*50)
        for alert in results['alerts'][:10]:  # Show first 10 alerts
            print(
                f"  [{alert['severity'].upper()}] {alert['window_id']}: "
                f"prob={alert['probability']:.3f}, "
                f"epistemic={alert['epistemic_uncertainty']:.3f}"
            )
        if len(results['alerts']) > 10:
            print(f"  ... and {len(results['alerts']) - 10} more alerts")
    
    if output_path:
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
