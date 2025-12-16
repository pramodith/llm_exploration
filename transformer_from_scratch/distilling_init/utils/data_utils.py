from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from typing import Dict, Any

def load_and_preprocess_dataset(dataset_name: str, tokenizer_name: str, max_length: int = 512):
    """Load and preprocess dataset for language modeling with SFTTrainer."""
    dataset = load_dataset(dataset_name)

    # For SFTTrainer, we need raw text data
    def prepare_text(examples):
        # Concatenate text if it's split
        if "text" in examples:
            return {"text": examples["text"]}
        else:
            # Handle other formats
            return {"text": examples[list(examples.keys())[0]]}

    processed_dataset = dataset.map(prepare_text, remove_columns=dataset["train"].column_names)

    return processed_dataset, tokenizer_name  # Return tokenizer name instead of loaded tokenizer

def create_data_collator(tokenizer):
    """Create data collator for language modeling."""
    # Not needed for SFTTrainer as it handles collation internally
    return None