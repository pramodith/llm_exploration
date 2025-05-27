from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import lightning as pl
from trl import GRPOTrainer
import torch
from rewards import correct_answer_reward, format_reward
from loguru import logger
from typing import List

from dataset_processor import get_gsm8k_dataset, create_dataloader, tokenize_example

logger.add("grpo_trainer.log", rotation="10 MB")
class SimpleGRPOTrainer(pl.LightningModule):
    def __init__(
        self, 
        model_name_or_path: str, 
        num_responses_per_example: int = 4,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.7,
        max_gen_tokens: int = 128, 
        beta: float = 0.1,
    ):
        """
        Initialize the GRPOTrainer with a model.

        Args:
            model_name_or_path (str): Path to the model or model identifier 
                from the Hugging Face Model Hub.
            num_responses_per_example (int): Number of responses per example.
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (float): The cumulative probability threshold for nucleus sampling.
            temperature (float): The value used to module the logits before applying softmax.
            max_gen_tokens (int): The maximum number of tokens to generate.
            beta (float): The scaling factor for the KL divergence loss against the reference model.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.reference_model.eval()
        self.num_responses_per_example = num_responses_per_example
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_gen_tokens = max_gen_tokens
        self.beta = beta
    
    def _get_logit_scores(
        self, 
        inputs: dict[str, torch.LongTensor],
        completion_ids: torch.LongTensor,
        is_policy_model: bool = True
        ):
        if is_policy_model:
            logit_scores = self.policy_model(**inputs).logits
        else:
            with torch.no_grad():
                logit_scores = self.reference_model(**inputs).logits
        logit_scores = logit_scores / self.temperature

        prob_scores = torch.softmax(logit_scores, dim=-1)
        groupwise_prob_scores = []
        
        # Get the probability scores for each response
        for query_id in range(len(completion_ids)):
            for response_num in range(self.num_responses_per_example):
                groupwise_prob_scores.append(
                    prob_scores[query_id].gather(1, completion_ids[query_id][response_num].unsqueeze(-1))
                )

        return groupwise_prob_scores
    
    def compute_rewards(self, sampled_responses: List, answers:List[str]):
        # Repeat the answers for each response num_responses_per_example times
        answers = [answer for answer in answers for _ in range(self.num_responses_per_example)]
        sampled_responses = [response[i] for i in range(self.num_responses_per_example) for response in sampled_responses]
        correct_answer_rewards = correct_answer_reward(
            answers=sampled_responses,
            reference_answer=answers,
        )
        format_rewards = format_reward(
            answers=sampled_responses,
            reference_format_regex=r"(<think>)\w+(</think>)\s*(<answer>)(\d+)(</answer>)",
        )
        return torch.tensor(correct_answer_rewards).reshape(-1, self.num_responses_per_example), \
            torch.tensor(format_rewards).reshape(-1, self.num_responses_per_example)
    
    def compute_advantage_score(self, rewards: torch.Tensor):
        """
        Standardize the rewards. To get the advantage score of each sampled response

        Args:
            rewards (torch.Tensor): The rewards to standardize.

        Returns:
            torch.Tensor: The advantage scores.
        """
        mean_rewards = rewards.mean(dim=1)
        std = rewards.std(dim=1)
        advantage_scores = (rewards - mean_rewards) / (std + 1e-8)
        return advantage_scores
    
    def training_step(self, batch, batch_idx):      
        prompts = [prompt for prompt in batch["prompt"]] 
        
        inputs = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512, 
            padding=True,
            padding_side='left'
        )
        original_prompt_lengths = [torch.sum(inputs["attention_mask"][i]).item() for i in range(len(batch["prompt"]))]
        # Get the completions from the policy model
        sampled_responses = self.policy_model.generate(
            **inputs,
            do_sample=True,
            temperature = self.temperature,
            top_p = self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_gen_tokens,
            num_return_sequences=self.num_responses_per_example,
        )

        # Get rid of the prompt tokens in the response
        completion_ids = [
            sampled_responses[i*self.num_responses_per_example:(i+1)*self.num_responses_per_example, original_prompt_lengths[i]:]
            for i in range(len(batch["prompt"]))
        ]

        # Get the rewards for each response
        completions = [self.tokenizer.batch_decode(completion_ids[i], skip_special_tokens=True) 
            for i in range(len(completion_ids))]
        correct_answer_rewards, format_rewards = self.compute_rewards(completions, batch["answer"])
        correct_answer_rewards = torch.tensor(correct_answer_rewards).view(len(batch["prompt"]), self.num_responses_per_example)
        format_rewards = torch.tensor(format_rewards).view(len(batch["prompt"]), self.num_responses_per_example)
        logger.info(f"Correct answer rewards: {correct_answer_rewards.mean(dim=1)}")
        logger.info(f"Format rewards: {format_rewards.mean(dim=1)}")

        # Compute the forward pass with gradient calculation enabled.
        policy_logit_scores = self._get_logit_scores(inputs, completion_ids, is_policy_model=True)
        # Compute the forward pass with gradient calculation disabled.
        reference_logit_scores = self._get_logit_scores(inputs, completion_ids, is_policy_model=False)    
        


if __name__ == "__main__":
    trainer = SimpleGRPOTrainer(
        model_name_or_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        num_responses_per_example=4,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        max_gen_tokens=128,
    )
    train_dataset, test_dataset = get_gsm8k_dataset()
    train_dataset = train_dataset.map(tokenize_example, fn_kwargs={"tokenizer": trainer.tokenizer})
    test_dataset = test_dataset.map(tokenize_example, fn_kwargs={"tokenizer": trainer.tokenizer})
    train_dataloader = create_dataloader(train_dataset, is_train=False, batch_size=2)
    test_dataloader = create_dataloader(test_dataset, is_train=False, batch_size=2)
    for batch in train_dataloader:
        trainer.training_step(batch, 0)
        break
    
    