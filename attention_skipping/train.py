"""
Training script for attention skipping experiments with gradual layer dropping.
"""
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomIdentityLayer(nn.Module):
    """A custom identity layer that returns zeros for hidden states."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden_states, *args, **kwargs):
        """Return zeros matching the shape of hidden_states."""
        return hidden_states

class LayerDropConfig:
    """Configuration for gradual layer dropping strategy."""
    
    def __init__(
        self,
        initial_layers_to_drop: int = 1,
        max_layers_to_drop: Optional[int] = None,
        drop_interval_steps: int = 1000,
        loss_threshold: float = 0.01,
        alternate_only: bool = True,
    ):
        self.initial_layers_to_drop = initial_layers_to_drop
        self.max_layers_to_drop = max_layers_to_drop
        self.drop_interval_steps = drop_interval_steps
        self.loss_threshold = loss_threshold
        self.alternate_only = alternate_only


class DroppedLayerModel(nn.Module):
    """Model with gradual layer dropping capability."""
    
    def __init__(self, model, num_layers: int, config: LayerDropConfig):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.config = config
        self.current_layers_to_drop = config.initial_layers_to_drop
        self.last_drop_step = 0
        self.dropped_layers = []
        self.layers = self.model.model.layers
    
    def patch_dropped_layers(self, step: int):
        """Determine which layers to drop at current training step."""
        if (step - self.last_drop_step) % self.config.drop_interval_steps == 0:
            if not self.dropped_layers:
                self.dropped_layers.append(1)
            else:
                self.dropped_layers.append(self.dropped_layers[-1] + 2)
            self.layers[self.dropped_layers[-1]] = CustomIdentityLayer()
    
    def forward(self, *args, **kwargs):
        """Forward pass with layer dropping."""
        return self.model(*args, **kwargs)
    
    def get_dropped_layers(self) -> list:
        """Get current set of dropped layer indices."""
        return self.dropped_layers
    
    def get_current_drop_count(self) -> int:
        """Get current number of layers being dropped."""
        return self.current_layers_to_drop


class LayerWiseComparisonLoss(nn.Module):
    """MSE loss between corresponding layers of two models."""
    
    def __init__(self, use_only_last_layer: bool = False):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.use_only_last_layer = use_only_last_layer
    
    def forward(
        self,
        model_active_outputs: list[torch.Tensor],
        model_frozen_outputs: list[torch.Tensor],
        dropped_layer_indices: set,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute MSE loss between layers after dropped layers.
        
        Args:
            model_active_outputs: Hidden states from active model
            model_frozen_outputs: Hidden states from frozen model
            dropped_layer_indices: Indices of dropped layers
        """
        total_loss = 0.0
        loss_count = 0
        layer2loss = {}

        if not self.use_only_last_layer:
            # Compare layers after the dropped layers
            for i in range(1, len(model_active_outputs)):
                if i not in dropped_layer_indices:
                    active_hidden = model_active_outputs[i]
                    frozen_hidden = model_frozen_outputs[i]
                    
                    # Ensure shapes match
                    if active_hidden.shape != frozen_hidden.shape:
                        frozen_hidden = frozen_hidden[:, -active_hidden.shape[1]:, :]
                    
                    loss = self.mse_loss(active_hidden, frozen_hidden.detach())
                    total_loss += loss
                    layer2loss[i] = loss.detach()
                    loss_count += 1
        else:
            # Only compare the last layer
            active_hidden = model_active_outputs[-1]
            frozen_hidden = model_frozen_outputs[-1]
            
            if active_hidden.shape != frozen_hidden.shape:
                frozen_hidden = frozen_hidden[:, -active_hidden.shape[1]:, :]
            
            loss = self.mse_loss(active_hidden, frozen_hidden.detach())
            total_loss += loss
            layer2loss[len(model_active_outputs) - 1] = loss.detach()
            loss_count += 1
            
        return total_loss / max(loss_count, 1), layer2loss


def get_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikimedia/wikipedia",
    subset: str = "20231101.en",
    max_length: int = 512,
    max_samples: Optional[int] = None,

):
    """Load streaming dataset from Hugging Face."""
    logger.info(f"Loading {dataset_name} dataset...")
    
    dataset = load_dataset(
        dataset_name,
        subset,
        split="train",
        streaming=True,
    )
    
    def tokenize_function(examples):
        """Tokenize function for dataset."""
        texts = examples.get("text", examples.get("content", []))
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding="max_length",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
    
    # Remove unnecessary columns
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    if max_samples:
        dataset = dataset.take(max_samples)
    
    return dataset


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


class Trainer:
    """Trainer for the layer dropping experiment."""
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        # Prefer CUDA, then Apple MPS, then CPU
        device: str = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        ),
        num_training_steps: int = 10000,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        checkpoint_dir: str = "./checkpoints",
        num_best_checkpoints: int = 2,
        log_interval: int = 2,
        gradient_accumulation_steps: int = 2,
        use_scheduler: bool = True,
        use_only_last_layer: bool = False,
    ):
        self.device = device
        self.num_training_steps = num_training_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_best_checkpoints = num_best_checkpoints
        self.log_interval = log_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_scheduler = use_scheduler
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_frozen = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device,
        )
        self.model_active = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device,
        )
        
        # Freeze frozen model
        for param in self.model_frozen.parameters():
            param.requires_grad = False
        
        
        num_layers = self.model_active.model.config.num_hidden_layers
        
        # Setup layer dropping
        drop_config = LayerDropConfig(
            initial_layers_to_drop=1,
            max_layers_to_drop=num_layers // 2,
            drop_interval_steps=1000,
        )
        self.dropped_layer_model = DroppedLayerModel(
            self.model_active,
            num_layers,
            drop_config,
        )
        
        # Loss and optimizer
        self.criterion = LayerWiseComparisonLoss(use_only_last_layer=use_only_last_layer)
        self.optimizer = torch.optim.AdamW(
            self.model_active.parameters(),
            lr=learning_rate,
        )
        
        # Learning rate scheduler
        if self.use_scheduler:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=1e-6,
            )
        else:
            self.scheduler = None
        
        # Tracking best checkpoints
        self.best_checkpoints = []  # List of (loss, checkpoint_path)
        
        # Initialize wandb
        wandb.init(
            project="attention_skipping",
            config={
                "model_name": model_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_training_steps": num_training_steps,
                "num_layers": num_layers,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "use_scheduler": use_scheduler,
            },
        )
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.num_training_steps} steps")
        accumulated_layer2loss = defaultdict(float)
        # Load dataset
        dataset = get_dataset(
            tokenizer=self.tokenizer,
            max_length=512,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )
        
        # Create iterator
        data_iter = iter(dataloader)
        
        self.model_active.train()
        
        accumulated_loss = 0.0
        
        for step in range(self.num_training_steps):
            # Update layer dropping strategy
            self.dropped_layer_model.patch_dropped_layers(step)
            dropped_layers = self.dropped_layer_model.get_dropped_layers()
            
            # Get next batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Forward pass through both models
            with torch.no_grad():
                frozen_outputs = self.model_frozen(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                frozen_hidden_states = frozen_outputs.hidden_states
            
            active_outputs = self.model_active(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            active_hidden_states = active_outputs.hidden_states
            
            # Compute loss
            loss, layer2loss = self.criterion(
                active_hidden_states,
                frozen_hidden_states,
                dropped_layers,
            )
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            accumulated_loss += loss.item()
            for k, v in layer2loss.items():
                accumulated_layer2loss[k] += v.item()
            
            # Update weights only after accumulation steps
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Logging
            if (step + 1) % self.log_interval == 0:
                num_dropped = self.dropped_layer_model.get_current_drop_count()
                avg_loss = accumulated_loss / self.log_interval
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step {step + 1}/{self.num_training_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Dropped Layers: {num_dropped} | "
                    f"Layer Losses: {layer2loss}"
                )
                
                wandb.log({
                    "loss": avg_loss,
                    "learning_rate": current_lr,
                    "num_dropped_layers": num_dropped,
                    "step": step + 1,
                    **{f"layer_{k}_loss": v for k, v in layer2loss.items()},
                })
                
                accumulated_layer2loss = defaultdict(float)

                
                # Save checkpoint if in top-2 best
                self._save_best_checkpoint(step, avg_loss)
                
                accumulated_loss = 0.0
        
        logger.info("Training completed!")
        return self.best_checkpoints
    
    def _save_best_checkpoint(self, step: int, loss: float):
        """Save checkpoint if it's in top-2 best."""
        timestamp = datetime.now().isoformat()
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}_{timestamp}.pt"
        
        # Save checkpoint
        torch.save({
            "step": step,
            "loss": loss,
            "model_state": self.model_active.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "num_dropped_layers": self.dropped_layer_model.get_current_drop_count(),
        }, checkpoint_path)
        
        # Update best checkpoints list
        self.best_checkpoints.append((loss, str(checkpoint_path)))
        self.best_checkpoints = sorted(self.best_checkpoints, key=lambda x: x[0])[:self.num_best_checkpoints]
        
        # Clean up old checkpoints not in top-2
        all_checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        best_paths = {Path(path) for _, path in self.best_checkpoints}
        for checkpoint in all_checkpoints:
            if checkpoint not in best_paths:
                checkpoint.unlink()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        logger.info(f"Best checkpoints: {self.best_checkpoints}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train model with gradual layer dropping")
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="Model name from HuggingFace Hub",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./models/results/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        # Match Trainer default: prefer CUDA, then MPS, then CPU
        default="cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"),
        help="Device to use for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of steps to accumulate gradients before updating weights",
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        default=True,
        help="Use cosine annealing learning rate scheduler",
    )
    parser.add_argument(
        "--no_scheduler",
        action="store_false",
        dest="use_scheduler",
        help="Disable learning rate scheduler",
    )
    parser.add_argument(
        "--use_only_last_layer",
        action="store_true",
        default=False,
        help="Use only the last layer for loss calculation",
    )
    
    args = parser.parse_args([])
    
    trainer = Trainer(
        model_name=args.model_name,
        device=args.device,
        num_training_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_scheduler=args.use_scheduler,
        use_only_last_layer=args.use_only_last_layer,
    )
    
    best_checkpoints = trainer.train()
    
    logger.info("Training finished!")
    logger.info(f"Top 2 best checkpoints: {best_checkpoints}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
