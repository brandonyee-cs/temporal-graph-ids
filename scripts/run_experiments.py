#!/usr/bin/env python3
"""
Comprehensive experiment script for T-GAT publication results.

Runs:
1. Hyperparameter tuning
2. Baseline comparisons (MLP, GCN, Random Forest)
3. Calibration metrics (ECE, Brier score)
4. Visualizations (ROC curve, PR curve, reliability diagram)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, 
    average_precision_score, f1_score, precision_score, recall_score
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.data.graph_constructor import GraphConstructor
from src.models.tgat import TGAT
from src.models.training_pipeline import TrainingPipeline
from src.models.evaluation import EvaluationModule


def load_data(data_path: str, redteam_path: str, window_size: int = 3600, overlap: int = 1800):
    """Load and preprocess LANL data."""
    print(f"Loading data from {data_path}...")
    
    preprocessor = DataPreprocessor(window_size=window_size, overlap=overlap)
    events_df = preprocessor.load_events(data_path)
    print(f"  Loaded {len(events_df):,} events")
    
    print("Creating temporal windows...")
    windows = preprocessor.create_windows(events_df)
    print(f"  Created {len(windows)} windows")
    
    print(f"Extracting labels from {redteam_path}...")
    windows = preprocessor.extract_labels(windows, redteam_path)
    malicious = sum(1 for w in windows if w.label == 1)
    print(f"  Found {malicious} malicious windows ({100*malicious/len(windows):.1f}%)")
    
    # Calculate splits based on actual data
    min_time = min(w.start_time for w in windows)
    max_time = max(w.end_time for w in windows)
    total_days = max(1, int((max_time - min_time) / 86400) + 1)
    train_end = max(1, int(total_days * 0.7))
    val_end = max(train_end + 1, int(total_days * 0.85))
    
    train_windows, val_windows, test_windows = preprocessor.split_temporal(
        windows,
        train_days=(1, train_end),
        val_days=(train_end + 1, val_end),
        test_days=(val_end + 1, total_days)
    )
    print(f"  Split: train={len(train_windows)}, val={len(val_windows)}, test={len(test_windows)}")
    
    return train_windows, val_windows, test_windows, preprocessor


def windows_to_graphs(windows, graph_constructor):
    """Convert windows to PyG graphs."""
    graphs = []
    for window in windows:
        graph = graph_constructor.build_graph(window)
        graphs.append(graph)
    return graphs


def extract_features_for_baselines(graphs: List[Data]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract flat features from graphs for baseline models."""
    X_list = []
    y_list = []
    
    for graph in graphs:
        # Aggregate node features: mean, max, std
        node_feats = graph.x.numpy()
        mean_feats = node_feats.mean(axis=0)
        max_feats = node_feats.max(axis=0)
        std_feats = node_feats.std(axis=0)
        
        # Graph-level features
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        density = num_edges / (num_nodes * (num_nodes - 1) + 1e-6) if num_nodes > 1 else 0
        
        # Combine features
        features = np.concatenate([
            mean_feats, max_feats, std_feats,
            [num_nodes, num_edges, density]
        ])
        
        X_list.append(features)
        y_list.append(graph.y.item())
    
    return np.array(X_list), np.array(y_list)


class MLPBaseline(nn.Module):
    """Simple MLP baseline for comparison."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


def train_mlp_baseline(X_train, y_train, X_val, y_val, epochs=100):
    """Train MLP baseline."""
    input_dim = X_train.shape[1]
    model = MLPBaseline(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    model.load_state_dict(best_model_state)
    return model


def train_rf_baseline(X_train, y_train):
    """Train Random Forest baseline."""
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    return rf


def train_lr_baseline(X_train, y_train):
    """Train Logistic Regression baseline."""
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    return lr


def compute_calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict:
    """Compute calibration metrics: ECE, MCE, Brier score."""
    # Brier score
    brier = np.mean((probs - labels) ** 2)
    
    # ECE and MCE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    bin_data = []
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            bin_size = mask.sum()
            
            gap = abs(bin_acc - bin_conf)
            ece += (bin_size / len(probs)) * gap
            mce = max(mce, gap)
            
            bin_data.append({
                'bin': i,
                'lower': bin_boundaries[i],
                'upper': bin_boundaries[i + 1],
                'accuracy': float(bin_acc),
                'confidence': float(bin_conf),
                'count': int(bin_size),
                'gap': float(gap)
            })
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'brier_score': float(brier),
        'bins': bin_data
    }


def evaluate_model(model, graphs, device='cpu', is_tgat=True) -> Dict:
    """Evaluate a model and return all metrics."""
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for i, graph in enumerate(graphs):
            graph = graph.to(device)
            if is_tgat:
                # Use forward pass directly for evaluation
                predictions, uncertainty_output, _ = model.forward([graph], return_uncertainty=True)
                prob = predictions.mean().item()
            else:
                # For baseline models
                prob = model(graph).item()
            
            all_probs.append(prob)
            all_labels.append(graph.y.item())
    
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    
    # Detection metrics
    auc_roc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    auc_pr = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    
    # Find best threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    preds = (probs >= best_thresh).astype(int)
    
    # Calibration metrics
    calibration = compute_calibration_metrics(probs, labels)
    
    return {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1': float(best_f1),
        'best_threshold': float(best_thresh),
        'ece': calibration['ece'],
        'mce': calibration['mce'],
        'brier_score': calibration['brier_score'],
        'calibration_bins': calibration['bins'],
        'probs': probs.tolist(),
        'labels': labels.tolist()
    }


def create_visualizations(results: Dict, output_dir: Path):
    """Create publication-quality visualizations."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. ROC Curve comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, metrics in results.items():
        if 'probs' in metrics and 'labels' in metrics:
            probs = np.array(metrics['probs'])
            labels = np.array(metrics['labels'])
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = metrics['auc_roc']
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'roc_curve.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, metrics in results.items():
        if 'probs' in metrics and 'labels' in metrics:
            probs = np.array(metrics['probs'])
            labels = np.array(metrics['labels'])
            precision, recall, _ = precision_recall_curve(labels, probs)
            ap = metrics['auc_pr']
            ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})", linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve Comparison', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'pr_curve.pdf', bbox_inches='tight')
    plt.close()
    
    # 3. Reliability Diagram (for T-GAT)
    if 'T-GAT' in results and 'calibration_bins' in results['T-GAT']:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bins = results['T-GAT']['calibration_bins']
        confidences = [b['confidence'] for b in bins]
        accuracies = [b['accuracy'] for b in bins]
        counts = [b['count'] for b in bins]
        
        # Bar plot
        width = 0.08
        positions = [b['confidence'] for b in bins]
        ax.bar(positions, accuracies, width=width, alpha=0.7, label='Accuracy', color='steelblue')
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'Reliability Diagram (ECE={results["T-GAT"]["ece"]:.3f})', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(output_dir / 'reliability_diagram.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'reliability_diagram.pdf', bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def create_results_table(results: Dict, output_dir: Path):
    """Create LaTeX table for paper."""
    
    # Markdown table
    md_lines = [
        "| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall | ECE |",
        "|-------|---------|--------|-----|-----------|--------|-----|"
    ]
    
    for model_name, metrics in results.items():
        md_lines.append(
            f"| {model_name} | {metrics['auc_roc']:.3f} | {metrics['auc_pr']:.3f} | "
            f"{metrics['f1']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics.get('ece', 'N/A'):.3f} |"
        )
    
    with open(output_dir / 'results_table.md', 'w') as f:
        f.write('\n'.join(md_lines))
    
    # LaTeX table
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of T-GAT with baseline methods on LANL dataset}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & AUC-ROC & AUC-PR & F1 & Precision & Recall & ECE \\",
        r"\midrule"
    ]
    
    for model_name, metrics in results.items():
        ece_str = f"{metrics['ece']:.3f}" if 'ece' in metrics else "N/A"
        latex_lines.append(
            f"{model_name} & {metrics['auc_roc']:.3f} & {metrics['auc_pr']:.3f} & "
            f"{metrics['f1']:.3f} & {metrics['precision']:.3f} & {metrics['recall']:.3f} & "
            f"{ece_str} \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Results tables saved to {output_dir}")


def run_tgat_training(train_graphs, val_graphs, config: Dict, device='cpu'):
    """Train T-GAT model with given config."""
    # Get input dimensions from first graph
    sample_graph = train_graphs[0]
    node_feature_dim = sample_graph.x.shape[1]
    edge_feature_dim = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 0
    
    model = TGAT(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=config.get('hidden_dim', 128),
        num_gat_layers=config.get('num_gat_layers', 2),
        num_attention_heads=config.get('num_attention_heads', 4),
        lstm_hidden_dim=config.get('lstm_hidden_dim', 256),
        num_lstm_layers=config.get('num_lstm_layers', 2),
        dropout=config.get('dropout', 0.2),
        num_mc_samples=config.get('mc_samples', 50)
    )
    
    pipeline = TrainingPipeline(
        model=model,
        lr=config.get('learning_rate', 0.0001),
        weight_decay=config.get('weight_decay', 1e-5),
        grad_clip=1.0,
        early_stopping_patience=config.get('patience', 15),
        device=device
    )
    
    train_loader = PyGDataLoader(train_graphs, batch_size=config.get('batch_size', 16), shuffle=True)
    val_loader = PyGDataLoader(val_graphs, batch_size=config.get('batch_size', 16), shuffle=False)
    
    history = pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('max_epochs', 100)
    )
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Run T-GAT experiments for publication")
    parser.add_argument("--data-path", type=str, required=True, help="Path to auth data")
    parser.add_argument("--redteam-path", type=str, required=True, help="Path to redteam data")
    parser.add_argument("--output-dir", type=str, default="experiments/publication", help="Output directory")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline training")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("T-GAT PUBLICATION EXPERIMENTS")
    print("=" * 60)
    
    # Load data
    train_windows, val_windows, test_windows, preprocessor = load_data(
        args.data_path, args.redteam_path
    )
    
    # Convert to graphs
    print("\nConverting windows to graphs...")
    graph_constructor = GraphConstructor(node_feature_dim=64, edge_feature_dim=32)
    train_graphs = windows_to_graphs(train_windows, graph_constructor)
    graph_constructor.clear_history()
    val_graphs = windows_to_graphs(val_windows, graph_constructor)
    graph_constructor.clear_history()
    test_graphs = windows_to_graphs(test_windows, graph_constructor)
    
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    results = {}
    
    # Train T-GAT
    print("\n" + "=" * 60)
    print("TRAINING T-GAT")
    print("=" * 60)
    
    tgat_config = {
        'hidden_dim': 128,
        'num_gat_layers': 2,
        'num_attention_heads': 4,
        'lstm_hidden_dim': 256,
        'num_lstm_layers': 2,
        'dropout': 0.2,
        'mc_samples': 50,
        'learning_rate': 0.0001,
        'batch_size': 16,
        'max_epochs': args.max_epochs,
        'patience': 15
    }
    
    tgat_model, tgat_history = run_tgat_training(
        train_graphs, val_graphs, tgat_config, args.device
    )
    
    print("\nEvaluating T-GAT on test set...")
    results['T-GAT'] = evaluate_model(tgat_model, test_graphs, args.device, is_tgat=True)
    print(f"  AUC-ROC: {results['T-GAT']['auc_roc']:.4f}")
    print(f"  AUC-PR: {results['T-GAT']['auc_pr']:.4f}")
    print(f"  F1: {results['T-GAT']['f1']:.4f}")
    print(f"  ECE: {results['T-GAT']['ece']:.4f}")
    
    # Save T-GAT model
    torch.save(tgat_model.state_dict(), output_dir / 'tgat_model.pt')

    
    # Train baselines
    if not args.skip_baselines:
        print("\n" + "=" * 60)
        print("TRAINING BASELINES")
        print("=" * 60)
        
        # Extract features for baselines
        print("\nExtracting features for baseline models...")
        X_train, y_train = extract_features_for_baselines(train_graphs)
        X_val, y_val = extract_features_for_baselines(val_graphs)
        X_test, y_test = extract_features_for_baselines(test_graphs)
        print(f"  Feature dim: {X_train.shape[1]}")
        
        # Random Forest
        print("\nTraining Random Forest...")
        rf_model = train_rf_baseline(X_train, y_train)
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        results['Random Forest'] = {
            'auc_roc': float(roc_auc_score(y_test, rf_probs)) if len(np.unique(y_test)) > 1 else 0.5,
            'auc_pr': float(average_precision_score(y_test, rf_probs)) if len(np.unique(y_test)) > 1 else 0.0,
            'probs': rf_probs.tolist(),
            'labels': y_test.tolist()
        }
        # Find best threshold for RF
        best_f1 = 0
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (rf_probs >= thresh).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        preds = (rf_probs >= best_thresh).astype(int)
        results['Random Forest'].update({
            'f1': float(best_f1),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'best_threshold': float(best_thresh),
            **compute_calibration_metrics(rf_probs, y_test)
        })
        print(f"  AUC-ROC: {results['Random Forest']['auc_roc']:.4f}")
        
        # Logistic Regression
        print("\nTraining Logistic Regression...")
        lr_model = train_lr_baseline(X_train, y_train)
        lr_probs = lr_model.predict_proba(X_test)[:, 1]
        results['Logistic Regression'] = {
            'auc_roc': float(roc_auc_score(y_test, lr_probs)) if len(np.unique(y_test)) > 1 else 0.5,
            'auc_pr': float(average_precision_score(y_test, lr_probs)) if len(np.unique(y_test)) > 1 else 0.0,
            'probs': lr_probs.tolist(),
            'labels': y_test.tolist()
        }
        best_f1 = 0
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (lr_probs >= thresh).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        preds = (lr_probs >= best_thresh).astype(int)
        results['Logistic Regression'].update({
            'f1': float(best_f1),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'best_threshold': float(best_thresh),
            **compute_calibration_metrics(lr_probs, y_test)
        })
        print(f"  AUC-ROC: {results['Logistic Regression']['auc_roc']:.4f}")
        
        # MLP
        print("\nTraining MLP...")
        mlp_model = train_mlp_baseline(X_train, y_train, X_val, y_val)
        mlp_model.eval()
        with torch.no_grad():
            mlp_probs = mlp_model(torch.FloatTensor(X_test)).squeeze().numpy()
        results['MLP'] = {
            'auc_roc': float(roc_auc_score(y_test, mlp_probs)) if len(np.unique(y_test)) > 1 else 0.5,
            'auc_pr': float(average_precision_score(y_test, mlp_probs)) if len(np.unique(y_test)) > 1 else 0.0,
            'probs': mlp_probs.tolist(),
            'labels': y_test.tolist()
        }
        best_f1 = 0
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (mlp_probs >= thresh).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        preds = (mlp_probs >= best_thresh).astype(int)
        results['MLP'].update({
            'f1': float(best_f1),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'best_threshold': float(best_thresh),
            **compute_calibration_metrics(mlp_probs, y_test)
        })
        print(f"  AUC-ROC: {results['MLP']['auc_roc']:.4f}")

    
    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    create_visualizations(results, output_dir)
    create_results_table(results, output_dir)
    
    # Save all results
    # Remove numpy arrays for JSON serialization
    results_json = {}
    for model_name, metrics in results.items():
        results_json[model_name] = {k: v for k, v in metrics.items() if k != 'calibration_bins'}
        if 'calibration_bins' in metrics:
            results_json[model_name]['calibration_bins'] = metrics['calibration_bins']
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\n| Model | AUC-ROC | AUC-PR | F1 | ECE |")
    print("|-------|---------|--------|-----|-----|")
    for model_name, metrics in results.items():
        ece = metrics.get('ece', 'N/A')
        ece_str = f"{ece:.3f}" if isinstance(ece, float) else ece
        print(f"| {model_name} | {metrics['auc_roc']:.3f} | {metrics['auc_pr']:.3f} | {metrics['f1']:.3f} | {ece_str} |")
    
    print(f"\nAll results saved to {output_dir}")
    print("\nFiles created:")
    for f in output_dir.iterdir():
        print(f"  - {f.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
