from transformers import AutoConfig, AutoModelForCausalLM
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_model_and_config(model_name: str, torch_dtype: torch.dtype = torch.float32):
    """Load model and config from HuggingFace."""
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    return model, config

def create_smaller_config(source_config, scale_factor: float = 0.5):
    """Create a smaller config based on source config."""
    small_config = source_config.__class__()

    # Copy relevant attributes
    for attr in dir(source_config):
        if not attr.startswith('_') and not callable(getattr(source_config, attr)):
            try:
                setattr(small_config, attr, getattr(source_config, attr))
            except:
                pass

    # Scale down dimensions
    small_config.num_hidden_layers = max(1, int(source_config.num_hidden_layers * scale_factor))
    small_config.hidden_size = max(1, int(source_config.hidden_size * scale_factor))
    small_config.num_attention_heads = max(1, int(source_config.num_attention_heads * scale_factor))
    small_config.num_key_value_heads = getattr(source_config, 'num_key_value_heads', small_config.num_attention_heads)
    small_config.num_key_value_heads = max(1, int(small_config.num_key_value_heads * scale_factor))
    small_config.intermediate_size = max(1, int(getattr(source_config, 'intermediate_size', source_config.hidden_size * 4) * scale_factor))

    return small_config

def initialize_with_stats(model, stats_dict: Dict[str, Dict], temperature: float = 1.0):
    """Initialize model parameters with statistics from source model."""
    logger.info(f"Initializing model with temperature {temperature}")

    param_categories = categorize_parameters(model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            category = param_categories.get(name, None)
            if category and category in stats_dict:
                stats = stats_dict[category]
                mean = stats["mean"]
                std = stats["std"] * temperature

                # Sample from Gaussian
                with torch.no_grad():
                    param.data = torch.randn_like(param) * std + mean
                logger.debug(f"Initialized {name} with {category}: μ={mean:.6f}, σ={std:.6f}")
            else:
                logger.warning(f"No stats found for {name} (category: {category}), using default initialization")

def categorize_parameters(model) -> Dict[str, str]:
    """Categorize model parameters by type."""
    categories = {}

    for name, _ in model.named_parameters():
        name_lower = name.lower()

        if 'embed' in name_lower and 'token' in name_lower:
            categories[name] = "embedding.token"
        elif 'embed' in name_lower and 'position' in name_lower:
            categories[name] = "embedding.position"
        elif 'q_proj' in name_lower or 'query' in name_lower:
            categories[name] = "attention.q_proj"
        elif 'k_proj' in name_lower or 'key' in name_lower:
            categories[name] = "attention.k_proj"
        elif 'v_proj' in name_lower or 'value' in name_lower:
            categories[name] = "attention.v_proj"
        elif 'o_proj' in name_lower or 'out_proj' in name_lower:
            categories[name] = "attention.out_proj"
        elif ('gate_proj' in name_lower or 'up_proj' in name_lower) and 'weight' in name_lower:
            categories[name] = "mlp.up_proj"
        elif ('down_proj' in name_lower) and 'weight' in name_lower:
            categories[name] = "mlp.down_proj"
        elif 'layernorm' in name_lower or 'ln_' in name_lower:
            if 'weight' in name_lower:
                categories[name] = "layernorm.weight"
            elif 'bias' in name_lower:
                categories[name] = "layernorm.bias"
        elif 'lm_head' in name_lower:
            categories[name] = "lm_head"
        else:
            categories[name] = "other"

    return categories