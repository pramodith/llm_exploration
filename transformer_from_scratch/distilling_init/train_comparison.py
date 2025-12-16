#!/usr/bin/env python3
"""
Train comparison between Gaussian-initialized and default-initialized models.

This script trains both model variants on the same dataset and logs
comprehensive metrics for comparison.
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, Any
import logging
import yaml

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer
)
from trl import SFTConfig, SFTTrainer
from utils.logging_utils import setup_logging
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def load_model_from_checkpoint(config_path: Path, checkpoint_path: Path, device: str = "auto"):
    """Load model from config and checkpoint."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    model_config = config_data["target_model_config"]
    config = AutoConfig.from_dict(model_config)

    model = AutoModelForCausalLM.from_config(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    if device == "auto":
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = model.to(device)

    return model

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    # SFTTrainer handles evaluation differently, this might not be needed
    return {}

def train_model(
    model,
    train_dataset,
    eval_dataset,
    tokenizer_name,
    output_dir: Path,
    training_config: Dict[str, Any],
    model_name: str,
    hub_repo: str = None
):
    """Train a single model using SFTTrainer."""
    training_args = SFTConfig(
        output_dir=str(output_dir / model_name),
        num_train_epochs=1,  # We'll use max_steps
        max_steps=training_config["max_steps"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        learning_rate=training_config["learning_rate"],
        warmup_steps=training_config["warmup_steps"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        seed=training_config["seed"],
        report_to="trackio",  # Disable wandb for now
        packing=True,  # Enable packing for efficient training
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=training_config["context_length"],
    )

    logger.info(f"Starting training for {model_name}")
    trainer.train()

    # Save final model
    trainer.save_model(str(output_dir / model_name / "final"))

    # Push to Hugging Face Hub if repo specified
    if hub_repo:
        logger.info(f"Pushing {model_name} to Hugging Face Hub: {hub_repo}")
        trainer.model.push_to_hub(hub_repo, private=True)
        trainer.tokenizer.push_to_hub(hub_repo, private=True)

    return trainer

def main():
    parser = argparse.ArgumentParser(description="Train models with comparison")
    parser.add_argument("--model_dir", type=str, default="transformer_from_scratch/distilling_init/models",
                       help="Directory containing initialized models")
    parser.add_argument("--config_path", type=str, default="transformer_from_scratch/distilling_init/config/default_config.yaml",
                       help="Path to training config YAML file")
    parser.add_argument("--dataset", type=str, default="trl-lib/Capybara",
                       help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="results/",
                       help="Output directory for training results")
    parser.add_argument("--hub_repo_gaussian", type=str, default="model_init",
                       help="Hugging Face Hub repo ID for Gaussian initialized model")
    parser.add_argument("--hub_repo_default", type=str, default="model_init",
                       help="Hugging Face Hub repo ID for default initialized model")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                       help="Tokenizer name or path")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    training_config = config["training"]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset {args.dataset}")
    dataset = load_dataset("trl-lib/Capybara")

    # Split dataset
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"] if "validation" in dataset else dataset["test"]

    # Load models
    model_dir = Path(args.model_dir)
    config_path = model_dir / "target_config.yaml"

    model_a = AutoModelForCausalLM.from_pretrained(model_dir / "model_gaussian_init.pt")

    # Train both models
    trainer_a = train_model(
        model_a, train_dataset, eval_dataset, args.tokenizer_name,
        output_dir, training_config, "gaussian_init", args.hub_repo_gaussian
    )
    
    del model_a  # Free up memory
    torch.cuda.empty_cache()
    model_b = AutoModelForCausalLM.from_pretrained(model_dir / "model_default_init.pt")

    trainer_b = train_model(
        model_b, train_dataset, eval_dataset, args.tokenizer_name,
        output_dir, training_config, "default_init", args.hub_repo_default
    )

if __name__ == "__main__":
    main()