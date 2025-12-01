import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import subprocess
import shutil
import sys

class IdentityAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden_states, *args, **kwargs):
        # Return zeros so that the residual connection effectively skips the attention block
        # Output shape should match hidden_states
        # LlamaAttention seems to return (attn_output, attn_weights, past_key_value) usually, 
        # but the error "too many values to unpack (expected 2)" suggests it expects 2.
        # It might be (attn_output, past_key_value) or (attn_output, attn_weights).
        # Let's try returning 3 values first, wait, I returned 3 and it failed with "too many values to unpack".
        # So it expects 2.
        return torch.zeros_like(hidden_states), None

def modify_model(model, k):
    """
    Disables attention in layers that are NOT multiples of k.
    If k=1, all layers keep attention.
    If k=2, layers 0, 2, 4... disable attention? 
    User said: "runs through some N consequent FFN layers before attention is used."
    So if K=3, we want: NoAttn, NoAttn, Attn, NoAttn, NoAttn, Attn...
    So we keep attention if (i + 1) % k == 0.
    """
    print(f"Modifying model with K={k}...")
    count = 0
    for i, layer in enumerate(model.model.layers):
        # We want to KEEP attention every K layers.
        # So if (i + 1) % k != 0, we DISABLE it.
        if (i + 1) % k != 0:
            layer.self_attn = IdentityAttention()
            count += 1
    print(f"Disabled attention in {count} out of {len(model.model.layers)} layers.")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1, help="Frequency of attention layers (1 = all attention, 2 = every 2nd layer has attention)")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no benchmarking, just check model)")
    args = parser.parse_args()

    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto")

    if args.k > 1:
        model = modify_model(model, args.k)

    # Save modified model
    model_save_path = os.path.join(args.output_dir, f"modified_k{args.k}")
    if os.path.exists(model_save_path):
        shutil.rmtree(model_save_path)
    
    print(f"Saving modified model to {model_save_path}...")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    if args.test:
        print("Test mode: Generating some text...")
        inputs = tokenizer("Once upon a time", return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        print(tokenizer.decode(outputs[0]))
        return

    # Run lighteval
    # lighteval --model_args "pretrained=..." --tasks "..." --output_dir "..."
    # We use community tasks or standard ones.
    # Common lighteval tasks: "leaderboard|truthfulqa:mc|0|0", etc.
    # User mentioned: hellaswag, arc_easy, piqa
    # Lighteval format: "community|hellaswag|0|0", "community|arc:easy|0|0", "community|piqa|0|0"
    # Or just "hellaswag", "arc:easy", "piqa" if they are built-in.
    # Let's try standard names.
    
    #tasks = "community|hellaswag|0|0,community|piqa|0|0,community|arc:easy|0|0"
    tasks = 
    
    print("Running lighteval...")
    cmd = [
        "lighteval",
        "accelerate",
        f"model_name={model_save_path}",
        tasks,
        f"--output-dir={args.output_dir}",
    ]
    
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
