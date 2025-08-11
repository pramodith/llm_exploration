"""
Generates a markdown report summarizing the experiment results.
"""
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np


def generate_report(results: List[Dict], report_path: str):
    """
    Generate a markdown report with summary tables and plots.
    """
    # Example: results = [{"query": ..., "precision": ..., "recall": ..., ...}, ...]
    precisions = [r["precision"] for r in results]
    avg_precision = np.mean(precisions)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Multimodal Negation Search Experiment\n\n")
        f.write(f"**Average Precision@k:** {avg_precision:.3f}\n\n")
        f.write(f"**Average Recall@k:** {avg_recall:.3f}\n\n")
        f.write("| Query | Precision@k | Recall@k |\n")
        f.write("|-------|-------------|----------|\n")
        for r in results:
            f.write(f"| {r['query']} | {r['precision']:.2f} | {r['recall']:.2f} |\n")

    # Plot precision/recall histograms
    plt.figure(figsize=(8, 4))
    plt.hist(precisions, bins=10, alpha=0.7, label="Precision@k")
    plt.hist(recalls, bins=10, alpha=0.7, label="Recall@k")
    plt.legend()
    plt.title("Distribution of Precision and Recall")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(report_path.replace(".md", "_hist.png"))
