import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import pearsonr
from tqdm import tqdm
import math
import gc
# Set random seed for reproducibility
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Utility: GPU memory reporting (optional diagnostics)
def report_mem(stage: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[GPU MEM] {stage}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
    else:
        print(f"[GPU MEM] {stage}: CUDA not available")

# Configuration
MODEL_NAME = "openai/gpt-oss-20b"  # Using Qwen3-32B
NUM_PROMPTS = 100
MAX_TOKENS = 512
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
print("Loading TLDR dataset...")
dataset = load_dataset("trl-lib/tldr", split="train")

prompts = [example["prompt"] for example in dataset.select(range(NUM_PROMPTS))]
messages = [
    [
        {"role": "developer", "content":"Summarize the following reddit conversation."},
        {"role": "user", "content": prompt}
    ]
    for prompt in prompts
]

# Step 2: Load model in vLLM with bf16
print("\nLoading model in vLLM (bf16)...")
vllm_model = LLM(
    model=MODEL_NAME,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    enforce_eager=True
)
report_mem("After vLLM init")

# Step 3: Generate completions with vLLM and collect logits
print("\nGenerating completions with vLLM...")
sampling_params = SamplingParams(
    temperature=0.0,  # Greedy decoding for reproducibility
    max_tokens=MAX_TOKENS,
    logprobs=1,  # Request log probabilities
)
vllm_outputs = vllm_model.chat(messages, sampling_params=sampling_params)
report_mem("After vLLM generate")

# Extract completions and logits
completion_ids = []
prompt_ids = []
vllm_probs_list = []

print("\nExtracting vLLM logits...")
for ind, output in tqdm(enumerate(vllm_outputs)):
    prompt_ids.append(output.prompt_token_ids)
    completion_ids.append(output.outputs[0].token_ids)
    # vLLM returns log probabilities, convert to probabilities
    for i in range(len(output.outputs[0].logprobs)):
        for logprob in list(output.outputs[0].logprobs[i].values()):
            vllm_probs_list.append(math.exp(logprob.logprob))
            
# ---- FREE GPU MEMORY USED BY vLLM BEFORE LOADING HF MODEL ----
print("\nReleasing vLLM GPU memory before loading transformers model...")
try:
    # Attempt to gracefully shut down internal executors if present (future-proofing)
    if hasattr(vllm_model, "_executor") and vllm_model._executor is not None:
        pass  # Placeholder: vLLM may manage this internally; no public API yet.
except Exception as e:
    print(f"(Optional vLLM internal cleanup skipped: {e})")

def free_gpu_memory():
    # Delete reference and run garbage collection to release CUDA tensors
    del vllm_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # Reclaim IPC memory blocks

free_gpu_memory()
report_mem("After vLLM cleanup")

# Step 4 & 5: Load model in transformers and convert lm_head to fp32
print("\nLoading model in transformers (bf16, then creating untied fp32 lm_head)...")
transformers_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
report_mem("After HF model load")

def get_transformers_probs(transformers_model, prompt_ids, completion_ids):
    transformers_probs_list = []
    # Step 6: Run forward pass and collect logits
    print("\nRunning forward passes with transformers...")
    transformers_probs_list = []
    transformers_model.eval()
    with torch.no_grad():
        for i, (prompt, completion) in enumerate(tqdm(zip(prompt_ids, completion_ids), total=len(prompts))):
            prompt_completion = prompt + completion
            prompt_completion = torch.LongTensor([prompt_completion]).to("cuda")
            
            # Forward pass
            outputs = transformers_model(input_ids=prompt_completion)
            logits = outputs.logits  # Shape: [1, seq_len, vocab_size]
            
            # Extract per-token probabilities for completion tokens using torch.gather
            prompt_len = len(prompt)
            completion_probs = torch.exp(torch.log_softmax(logits[:, prompt_len-1:-1], -1))[0]  # Shape: [len(completion), vocab_size]
            assert completion_probs.shape[0] == len(completion), "Mismatch in completion length"
            # Use gather to get probabilities for the actual completion token IDs
            completion_token_ids = torch.tensor(completion, device=logits.device).unsqueeze(-1)  # Shape: [len(completion), 1]
            token_probs = torch.gather(completion_probs, dim=-1, index=completion_token_ids).squeeze(-1)  # Shape: [len(completion)]
            
            transformers_probs_list.extend(token_probs.cpu().tolist())
    return transformers_probs_list


bf16_transformers_probs_list = get_transformers_probs(transformers_model, prompt_ids, completion_ids)

assert len(vllm_probs_list) == len(bf16_transformers_probs_list), (
    f"Mismatch in counts: vLLM={len(vllm_probs_list)} vs HF={len(bf16_transformers_probs_list)}"
)


if transformers_model.config.tie_word_embeddings:
    # Break the weight tie between lm_head and embeddings by cloning before changing dtype.
    embed_ptr_before = transformers_model.model.embed_tokens.weight.data_ptr()
    lm_ptr_before = transformers_model.lm_head.weight.data_ptr()

    # Clone + promote to fp32 (new Parameter so it no longer shares storage).
    transformers_model.lm_head.weight = torch.nn.Parameter(transformers_model.lm_head.weight.detach().clone().float())
    if transformers_model.lm_head.bias is not None:
        transformers_model.lm_head.bias = torch.nn.Parameter(transformers_model.lm_head.bias.detach().clone().float())
    print(f"Embedding weight dtype: {transformers_model.model.embed_tokens.weight.dtype}")
    print(f"lm_head weight dtype (after untie & cast): {transformers_model.lm_head.weight.dtype}")
    print(f"Storage shared pre-change? {embed_ptr_before == lm_ptr_before}; shared after? {transformers_model.model.embed_tokens.weight.data_ptr() == transformers_model.lm_head.weight.data_ptr()}")

else:
    transformers_model.lm_head = transformers_model.lm_head.float()

def cast_to_fp32_hook(module, input):
    return input[0].to(torch.float32)

transformers_model.lm_head.register_forward_pre_hook(cast_to_fp32_hook)

fp32_transformers_probs_list = get_transformers_probs(transformers_model, prompt_ids, completion_ids)

# Step 7: Compute correlation
print("\nComputing correlation...")

# Both lists already contain probabilities, so no conversion needed
vllm_probs = np.array(vllm_probs_list)
bf16_transformers_probs = np.array(bf16_transformers_probs_list)
fp32_transformers_probs = np.array(fp32_transformers_probs_list)

def compute_and_print_correlation(vllm_probs, transformers_probs, model_name):
    # Compute correlation
    correlation = pearsonr(vllm_probs, transformers_probs)[0]
    print(f"Correlation: {correlation:.6f}")

    # Step 8: Plot scatter plot
    print("\nCreating scatter plot...")

    # Compute absolute probability differences for coloring
    abs_diff = np.abs(np.array(vllm_probs) - np.array(transformers_probs))
    sum_abs_diff = np.sum(abs_diff)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        vllm_probs, 
        transformers_probs, 
        c=abs_diff, 
        cmap='viridis', 
        alpha=0.6, 
        s=20
    )

    # Plot perfect correlation line
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Precision (y=x)')

    plt.xlabel('Inference Probability', fontsize=14)
    plt.ylabel('Training Probability', fontsize=14)
    plt.title(f'vLLM (bf16) vs Transformers (bf16 lm_head) - {model_name}', fontsize=16, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Probability Difference', fontsize=12)

    # Add correlation text box
    textstr = f'Correlation: {correlation:.4f}, Sum Abs Diff: {sum_abs_diff:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(f'./logits_comparison_{transformers_model.lm_head.weight.dtype}.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as './logits_comparison_{transformers_model.lm_head.weight.dtype}.png'")

    plt.show()

    print("\nDone!")

compute_and_print_correlation(vllm_probs, bf16_transformers_probs, MODEL_NAME)
compute_and_print_correlation(vllm_probs, fp32_transformers_probs, MODEL_NAME)
