from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from rewards import trl_format_reward, trl_correct_answer_reward
from dataset_processor import get_gsm8k_dataset, tokenize_example

# Set model and training parameters
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
BATCH_SIZE = 8
MAX_GEN_TOKENS = 300
TOP_K = 50
TOP_P = 0.9
TEMPERATURE = 0.7
NUM_RESPONSES_PER_EXAMPLE = 8
LEARNING_RATE = 1e-6
BETA = 0.04
EPSILON = 0.2
MAX_STEPS = 100
    
    
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Load and tokenize dataset
    train_dataset, test_dataset = get_gsm8k_dataset()
    train_dataset = train_dataset.map(tokenize_example, fn_kwargs={"tokenizer": tokenizer})
    test_dataset = test_dataset.map(tokenize_example, fn_kwargs={"tokenizer": tokenizer})

    # TRL expects HuggingFace Datasets format, so we use train_dataset directly
    config = GRPOConfig(
        learning_rate=LEARNING_RATE,
        beta=BETA,
        epsilon=EPSILON,
        max_steps=MAX_STEPS,
        per_device_train_batch_size = 8,
        max_completion_length=MAX_GEN_TOKENS,
        top_k=TOP_K,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        num_generations=NUM_RESPONSES_PER_EXAMPLE,
        use_vllm=False,
        loss_type='grpo',
        ref_model_mixup_alpha=1.0,
        ref_model_sync_steps=16,
        logging_steps=10,
    )

    trainer = GRPOTrainer(
        args=config,
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        reward_funcs=[
            trl_format_reward,
            trl_correct_answer_reward,
        ]
    )

    trainer.train()

if __name__ == "__main__":
    main()
