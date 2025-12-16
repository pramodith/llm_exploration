#!/usr/bin/env python3
"""
Analyze and visualize training results.

This script loads training logs from both models, generates comparison plots,
computes key metrics, and creates a summary report.
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging
import numpy as np

from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")

def load_training_logs(log_path: Path) -> pd.DataFrame:
    """Load training logs from JSON file."""
    with open(log_path, 'r') as f:
        logs = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(logs)

    # Filter to training steps
    df = df[df['step'].notna()]

    return df

def plot_loss_curves(df_a: pd.DataFrame, df_b: pd.DataFrame, output_path: Path):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Training loss
    ax1.plot(df_a['step'], df_a['loss'], label='Gaussian Init', color='blue', linewidth=2)
    ax1.plot(df_b['step'], df_b['loss'], label='Default Init', color='red', linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation loss
    if 'eval_loss' in df_a.columns and 'eval_loss' in df_b.columns:
        eval_a = df_a.dropna(subset=['eval_loss'])
        eval_b = df_b.dropna(subset=['eval_loss'])

        ax2.plot(eval_a['step'], eval_a['eval_loss'], label='Gaussian Init', color='blue', linewidth=2, marker='o')
        ax2.plot(eval_b['step'], eval_b['eval_loss'], label='Default Init', color='red', linewidth=2, marker='o')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "loss_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Loss curves saved to {output_path / 'loss_curves.png'}")

def plot_gradient_norms(df_a: pd.DataFrame, df_b: pd.DataFrame, output_path: Path):
    """Plot gradient norm evolution."""
    if 'grad_norm' not in df_a.columns:
        logger.warning("Gradient norms not found in logs")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(df_a['step'], df_a['grad_norm'], label='Gaussian Init', color='blue', linewidth=2)
    plt.plot(df_b['step'], df_b['grad_norm'], label='Default Init', color='red', linewidth=2)

    plt.xlabel('Training Steps')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(output_path / "gradient_norms.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Gradient norms plot saved to {output_path / 'gradient_norms.png'}")

def compute_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, Any]:
    """Compute key comparison metrics."""
    metrics = {}

    # Final losses
    final_loss_a = df_a['loss'].iloc[-1]
    final_loss_b = df_b['loss'].iloc[-1]
    metrics['final_training_loss'] = {
        'gaussian_init': final_loss_a,
        'default_init': final_loss_b,
        'improvement': (final_loss_b - final_loss_a) / final_loss_b * 100
    }

    # Steps to reach loss thresholds
    thresholds = [3.0, 2.5, 2.0, 1.5]
    for threshold in thresholds:
        steps_a = df_a[df_a['loss'] <= threshold]['step'].min()
        steps_b = df_b[df_b['loss'] <= threshold]['step'].min()

        if pd.isna(steps_a):
            steps_a = df_a['step'].max()
        if pd.isna(steps_b):
            steps_b = df_b['step'].max()

        metrics[f'steps_to_loss_{threshold}'] = {
            'gaussian_init': int(steps_a) if not pd.isna(steps_a) else None,
            'default_init': int(steps_b) if not pd.isna(steps_b) else None,
            'speedup': (steps_b - steps_a) / steps_b * 100 if steps_b > 0 else 0
        }

    # Area under curve (AUC) for first N steps
    max_steps = min(df_a['step'].max(), df_b['step'].max(), 5000)  # First 5000 steps
    auc_a = np.trapz(df_a[df_a['step'] <= max_steps]['loss'], df_a[df_a['step'] <= max_steps]['step'])
    auc_b = np.trapz(df_b[df_b['step'] <= max_steps]['loss'], df_b[df_b['step'] <= max_steps]['step'])

    metrics['auc_first_5000_steps'] = {
        'gaussian_init': auc_a,
        'default_init': auc_b,
        'improvement': (auc_b - auc_a) / auc_b * 100
    }

    return metrics

def create_summary_report(metrics: Dict[str, Any], output_path: Path):
    """Create a markdown summary report."""
    report = "# Training Comparison Report\n\n"

    report += "## Final Training Loss\n"
    loss_metrics = metrics['final_training_loss']
    report += f"- Gaussian Init: {loss_metrics['gaussian_init']:.4f}\n"
    report += f"- Default Init: {loss_metrics['default_init']:.4f}\n"
    report += f"- Improvement: {loss_metrics['improvement']:.2f}%\n\n"

    report += "## Convergence Speed\n"
    for key, value in metrics.items():
        if key.startswith('steps_to_loss_'):
            threshold = key.split('_')[-1]
            report += f"### Loss ≤ {threshold}\n"
            if value['gaussian_init'] and value['default_init']:
                report += f"- Gaussian Init: {value['gaussian_init']} steps\n"
                report += f"- Default Init: {value['default_init']} steps\n"
                report += f"- Speedup: {value['speedup']:.2f}%\n"
            else:
                report += "- Threshold not reached by both models\n"
            report += "\n"

    report += "## Area Under Curve (First 5000 Steps)\n"
    auc_metrics = metrics['auc_first_5000_steps']
    report += f"- Gaussian Init: {auc_metrics['gaussian_init']:.2f}\n"
    report += f"- Default Init: {auc_metrics['default_init']:.2f}\n"
    report += f"- Improvement: {auc_metrics['improvement']:.2f}%\n\n"

    report += "## Conclusion\n"
    improvement = loss_metrics['improvement']
    if improvement > 0:
        report += f"The Gaussian initialization from source model statistics resulted in {improvement:.2f}% better final training loss and faster convergence.\n"
    else:
        report += f"The default initialization performed better, with {abs(improvement):.2f}% lower final training loss.\n"

    with open(output_path / "analysis_report.md", 'w') as f:
        f.write(report)

    logger.info(f"Summary report saved to {output_path / 'analysis_report.md'}")

def main():
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument("--experiment_dir", type=str, required=True,
                       help="Directory containing training results")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for analysis results")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = Path(args.experiment_dir)

    # Load training logs
    logs_a_path = experiment_dir / "training_logs_gaussian.json"
    logs_b_path = experiment_dir / "training_logs_default.json"

    df_a = load_training_logs(logs_a_path)
    df_b = load_training_logs(logs_b_path)

    logger.info(f"Loaded {len(df_a)} log entries for Gaussian init")
    logger.info(f"Loaded {len(df_b)} log entries for Default init")

    # Generate plots
    plot_loss_curves(df_a, df_b, output_dir)
    plot_gradient_norms(df_a, df_b, output_dir)

    # Compute metrics
    metrics = compute_metrics(df_a, df_b)

    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Create report
    create_summary_report(metrics, output_dir)

    logger.info("Analysis complete")

if __name__ == "__main__":
    main()