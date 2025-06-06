import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
from datasets import Dataset

from rewards import correct_answer_reward, format_reward, length_reward
from dataset_processor import get_gsm8k_dataset, tokenize_example


def benchmark_model(
    model: Union[str, AutoModelForCausalLM],
    tokenizer: AutoTokenizer,
    test_dataset: Dataset,
    batch_size: int,
    top_k: int,
    top_p: float,
    temperature: float,
    max_completion_length: int,
):
    if isinstance(model, str):
        model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")

    model_name = model.name_or_path
    prompts = [prompt for prompt in test_dataset["prompt"]]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    inputs = inputs.to(model.device)
    responses = []
    for i in tqdm(range(0, len(inputs["input_ids"]), batch_size)):
        input_ids = inputs["input_ids"][i : i + batch_size]
        attention_mask = inputs["attention_mask"][i : i + batch_size]
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_completion_length,
                do_sample=True,
                num_return_sequences=8,
            )

            prompt_length = input_ids.shape[1]
            outputs = outputs[:, prompt_length:]

            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            responses.extend(outputs)

    answer_rewards = torch.tensor(
        correct_answer_reward(responses, [a for a in test_dataset["answer"] for _ in range(8) ])
    )
    print(f"Model: {model_name} has correct answer reward: {answer_rewards.mean()}")
    f_r = torch.tensor(
        format_reward(
            responses,
            r"(<think>)[\s\S]*?(</think>)[\s\S]*?(<answer>)[\s\D]*(\d+)[\s\D]*(</answer>)",
        )
    )
    print(f"Model: {model_name} has format reward: {f_r.mean()}")
    # l_r = torch.tensor(length_reward(responses))
    # print(f"Model: {model_name_or_path} has length reward: {l_r.mean()}")

    all_rewards = torch.stack([answer_rewards, f_r], dim=-1)
    all_rewards = all_rewards.sum(dim=-1)
    print(f"Model : {model_name} has total reward: {all_rewards.mean()}")
    write_rewards_to_csv(
        model_name, answer_rewards, f_r, "./results/gsmk_results_benchmark.csv"
    )
    return all_rewards, responses


def write_rewards_to_csv(model_name, accuracy_rewards, format_rewards, csv_path):
    """
    Write the model name, mean accuracy reward, and mean format reward to a CSV file.
    If the file does not exist, it creates it and writes the header.
    """
    import os
    import csv
    import numpy as np

    # Convert to numpy if tensor
    if hasattr(accuracy_rewards, "cpu"):
        accuracy_rewards = accuracy_rewards.cpu().numpy()
    if hasattr(format_rewards, "cpu"):
        format_rewards = format_rewards.cpu().numpy()
    mean_accuracy = float(np.mean(accuracy_rewards))
    mean_format = float(np.mean(format_rewards))
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(
                ["Model Name", "Mean Accuracy Rewards", "Mean Format Rewards"]
            )
        writer.writerow([model_name, mean_accuracy, mean_format])


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype=torch.bfloat16
    )
    train_dataset, test_dataset = get_gsm8k_dataset()
    test_dataset = test_dataset.map(
        tokenize_example, fn_kwargs={"tokenizer": tokenizer}
    )
    _, responses = benchmark_model(
        model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        batch_size=4,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        max_completion_length=300,
    )
