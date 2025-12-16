#!/usr/bin/env python3
"""
Initialize target model with transferred statistics.

This script creates a smaller transformer model and initializes it using
statistics extracted from a larger source model, comparing against
default initialization.
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any
import logging
import yaml

from transformers import AutoModelForCausalLM, AutoConfig
from utils.model_utils import initialize_with_stats, create_smaller_config
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def load_stats(stats_path: Path) -> Dict[str, Any]:
    """Load statistics from JSON file."""
    with open(stats_path, 'r') as f:
        data = json.load(f)
    return data

def create_and_initialize_models(
    target_config: Dict[str, Any],
    stats_dict: Dict[str, Dict],
    temperature: float = 1.0,
    device: str = "cpu",
    parent_model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
):
    """Create and initialize two model variants."""
    # Create config from dict
    config = AutoConfig.from_pretrained(parent_model_name, **target_config)

    # Model A: Initialized with transferred statistics
    logger.info("Creating model with transferred statistics initialization")
    model_a = AutoModelForCausalLM.from_config(config).to(device)
    initialize_with_stats(model_a, stats_dict, temperature)

    # Model B: Default initialization
    logger.info("Creating model with default initialization")
    model_b = AutoModelForCausalLM.from_config(config).to(device)

    return model_a, model_b

def save_model(model, path: Path):
    """Save model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def main():
    parser = argparse.ArgumentParser(description="Initialize target model with transferred statistics")
    parser.add_argument("--stats_path", type=str, default="transformer_from_scratch/distilling_init/artifacts/model_stats.json",
                       help="Path to statistics JSON file")
    parser.add_argument("--target_config", type=str, default="transformer_from_scratch/distilling_init/config/default_config.yaml",
                       help="Path to target model config YAML file")
    parser.add_argument("--output_dir", type=str, default="transformer_from_scratch/distilling_init/models",
                       help="Output directory for initialized models")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for initialization variance scaling")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for model initialization")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load stats
    stats_path = Path(args.stats_path)
    stats_data = load_stats(stats_path)
    stats_dict = stats_data["statistics"]
    metadata = stats_data["metadata"]

    logger.info(f"Loaded statistics from {metadata['source_model']}")

    # Load target config
    with open(args.target_config, 'r') as f:
        target_config = yaml.safe_load(f)

    # Extract target model config
    model_config = target_config.get("target_model_config", {})
    if not model_config:
        raise ValueError("target_model_config not found in config file")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and initialize models
    model_a, model_b = create_and_initialize_models(
        model_config, stats_dict, args.temperature, args.device
    )

    # Log parameter counts
    logger.info(f"Model A (Gaussian init) has {model_a.num_parameters()/10**6} million parameters")
    logger.info(f"Model B (Default init) has {model_b.num_parameters()/10**6} million parameters")

    # Save models
    save_model(model_a, output_dir / "model_gaussian_init.pt")
    save_model(model_b, output_dir / "model_default_init.pt")

    # Save config for reference
    with open(output_dir / "target_config.yaml", 'w') as f:
        yaml.dump(target_config, f)

    logger.info("Model initialization complete")

if __name__ == "__main__":
    main()