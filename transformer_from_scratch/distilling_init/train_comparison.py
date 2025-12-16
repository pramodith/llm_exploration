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
import json

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    AutoTokenizer
)
from trl import SFTTrainer
from utils.logging_utils import setup_logging
from utils.data_utils import load_and_preprocess_dataset

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
    model_name: str
):
    """Train a single model using SFTTrainer."""
    training_args = TrainingArguments(
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
        report_to=[],  # Disable wandb for now
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
        packing=True,
    )

    logger.info(f"Starting training for {model_name}")
    trainer.train()

    # Save final model
    trainer.save_model(str(output_dir / model_name / "final"))

    return trainer

def main():
    parser = argparse.ArgumentParser(description="Train models with comparison")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing initialized models")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to training config YAML file")
    parser.add_argument("--dataset", type=str, default="wikitext-2",
                       help="Dataset name")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for training results")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    training_config = config["training"]
    model_config = config["target_model_config"]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset {args.dataset}")
    dataset, tokenizer_name = load_and_preprocess_dataset(
        args.dataset,
        config["model_name"],  # Use source model tokenizer
        training_config["context_length"]
    )

    # Split dataset
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"] if "validation" in dataset else dataset["test"]

    # Load models
    model_dir = Path(args.model_dir)
    config_path = model_dir / "target_config.yaml"

    model_a = load_model_from_checkpoint(
        config_path, model_dir / "model_gaussian_init.pt"
    )
    model_b = load_model_from_checkpoint(
        config_path, model_dir / "model_default_init.pt"
    )

    # Train both models
    trainer_a = train_model(
        model_a, train_dataset, eval_dataset, tokenizer_name,
        output_dir, training_config, "gaussian_init"
    )

    trainer_b = train_model(
        model_b, train_dataset, eval_dataset, tokenizer_name,
        output_dir, training_config, "default_init"
    )

    # Save training logs
    logs_a = trainer_a.state.log_history
    logs_b = trainer_b.state.log_history

    with open(output_dir / "training_logs_gaussian.json", 'w') as f:
        json.dump(logs_a, f, indent=2)

    with open(output_dir / "training_logs_default.json", 'w') as f:
        json.dump(logs_b, f, indent=2)

    logger.info("Training complete")

if __name__ == "__main__":
    main()