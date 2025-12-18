#!/usr/bin/env python3
"""
Extract parameter statistics from a pre-trained transformer model.

This script loads a pre-trained model, categorizes its parameters by type,
computes mean and standard deviation for each category, and saves the
statistics to a JSON file.
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime

from utils.model_utils import load_model_and_config, categorize_parameters
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def compute_parameter_stats(model, categories: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Compute mean and std for each parameter category."""
    stats = {}
    category_params = {}

    # Group parameters by category
    for name, param in model.named_parameters():
        category = categories.get(name, "other")
        if category not in category_params:
            category_params[category] = []
        category_params[category].append(param.detach().flatten())

    # Compute stats for each category
    for category, param_list in category_params.items():
        if param_list:
            all_params = torch.cat(param_list)
            mean = all_params.mean().item()
            std = all_params.std().item()
            count = all_params.numel()

            stats[category] = {
                "mean": mean,
                "std": std,
                "count": count
            }

            logger.info(f"{category}: μ={mean:.6f}, σ={std:.6f}, count={count}")

    return stats

def save_stats(stats: Dict[str, Dict], metadata: Dict[str, Any], output_path: Path):
    """Save statistics and metadata to JSON file."""
    output_data = {
        "metadata": metadata,
        "statistics": stats,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Statistics saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract parameter statistics from pre-trained model")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-1.7B",
                       help="HuggingFace model name (e.g., '')")
    parser.add_argument("--output_path", type=str, default="./transformer_from_scratch/distilling_init/param_stats/stats.json",
                       help="Output path for statistics JSON file")
    parser.add_argument("--torch_dtype", type=str, default="float32",
                       choices=["float16", "float32", "float64"],
                       help="Torch dtype for loading model")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Convert dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64
    }
    torch_dtype = dtype_map[args.torch_dtype]

    logger.info(f"Loading model {args.model_name} with dtype {args.torch_dtype}")

    # Load model
    model, config = load_model_and_config(args.model_name, torch_dtype)

    # Categorize parameters
    categories = categorize_parameters(model)
    logger.info(f"Categorized {len(categories)} parameters into {len(set(categories.values()))} categories")

    # Compute statistics
    stats = compute_parameter_stats(model, categories)

    # Create metadata
    metadata = {
        "source_model": args.model_name,
        "num_layers": getattr(config, 'num_hidden_layers', None),
        "hidden_size": getattr(config, 'hidden_size', None),
        "num_attention_heads": getattr(config, 'num_attention_heads', None),
        "vocab_size": getattr(config, 'vocab_size', None),
        "torch_dtype": str(torch_dtype)
    }

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_stats(stats, metadata, output_path)

if __name__ == "__main__":
    main()