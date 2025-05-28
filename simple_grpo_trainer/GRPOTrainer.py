from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import lightning as pl
from trl import GRPOTrainer
import torch
from rewards import correct_answer_reward, format_reward
from loguru import logger
from typing import List

import copy

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
        beta: float = 0.04,
        epsilon: float = 0.2,
        num_steps_to_refresh_old_policy: int = 2
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
            epsilon (float): The epsilon clipping value
            num_steps_to_refresh_old_policy (int): The old policy is updated to match the current policy after the
                corresponding number of steps.
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
        self.epsilon = epsilon
        self.num_steps_to_refresh_old_policy = num_steps_to_refresh_old_policy
    
    def _get_completion_log_prob_scores(
        self, 
        prompt_ids: torch.LongTensor,
        prompt_mask: torch.LongTensor,
        completion_ids: torch.LongTensor,
        completions_mask: torch.LongTensor,
        is_policy_model: bool = True
        ):
        """
        We need to obtain the logit scores of the completions from the sampled responses
        for the current-policy, old-policy and reference model.

        To do this we run a single forward pass through the model with the prompt and completion
        concatenated and get the logit scores for each of the completions.

        Args:
            prompt_ids (torch.LongTensor): The prompt ids of shape (batch_size * num_responses_per_example, seq_len).
            prompt_mask (torch.LongTensor): The prompt mask of shape (batch_size * num_responses_per_example, seq_len).
            completion_ids (torch.LongTensor): The completion ids of shape (batch_size * num_responses_per_example, seq_len).
            completions_mask (torch.LongTensor): The completions mask of shape (batch_size * num_responses_per_example, seq_len).
            is_policy_model (bool): Whether to use the policy model.
        """
        prompt_completion_input = torch.cat([prompt_ids, completion_ids], dim=-1)
        prompt_length = prompt_ids.shape[-1]
        prompt_completion_mask = torch.cat([prompt_mask, completions_mask], dim=-1)
        if is_policy_model:
            logit_scores = self.policy_model(input_ids = prompt_completion_input, attention_mask = prompt_completion_mask).logits
        else:
            with torch.no_grad():
                logit_scores = self.reference_model(input_ids = prompt_completion_input, attention_mask = prompt_completion_mask).logits
        # Logit scores are of shape (batch_size * num_responses_per_example, seq_len + 1, vocab_size)
        # We exclude the logit scores for the prompt and the last token 
        # because it corresponds to the next token prediction
        logit_scores = logit_scores[:, prompt_length-1:-1, :]
        # We only need to keep the logit scores corresponding to the completion tokens
        logit_scores = torch.gather(logit_scores, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)
        # Get log_probs to avoid numerical underflow/overflow
        logit_scores = logit_scores / self.temperature
        log_prob_scores = torch.log_softmax(logit_scores, dim=-1)
        return log_prob_scores.view(-1, self.num_responses_per_example, log_prob_scores.shape[-1])
    

    def _get_completions_mask(self, sampled_responses: torch.LongTensor) -> torch.Tensor:
        """
        Get a mask for identifying all the valid completion tokens.

        Args:
            sampled_responses: The token ids of the sampled responses/completions
        Returns:
            A masked torch tensor with 1s and 0s. 1s correspond to a valid token.
        """
        # sampled_responses: [batch_size, seq_len]
        eos_token_id = self.tokenizer.eos_token_id

        # Find the first occurrence of EOS in each response
        eos_positions = (sampled_responses == eos_token_id).int()
        # Cumulative sum along the sequence dimension
        cumsum_eos = eos_positions.cumsum(dim=1)
        # If you want strictly after (not including the EOS itself):
        after_eos_mask = cumsum_eos > 1
        after_eos_mask = ~after_eos_mask
        # We need to invert the mask to get the valid tokens
        return after_eos_mask.int()

    
    def compute_grpo_loss(
        self, 
        policy_logprob_scores: torch.Tensor,
        old_policy_logprob_scores: torch.Tensor,
        reference_logprob_scores: torch.Tensor, 
        advantage_scores: torch.Tensor,
        completions_mask: torch.Tensor,
        ):
        """
        Compute the GRPO loss.

        Args:
            policy_logprob_scores (torch.Tensor): The probability scores from the policy model 
                of shape (batch_size, num_responses_per_example, completions_seq_len).
            old_logpolicy_prob_scores (torch.Tensor): The probability scores from the old policy model 
                of shape (batch_size, num_responses_per_example, completions_seq_len).
            reference_logprob_scores (torch.Tensor): The probability scores from the reference model 
                of shape (batch_size, num_responses_per_example, completions_seq_len).
            advantage_scores (torch.Tensor): The advantage scores 
                of shape (batch_size, num_responses_per_example).
            completions_mask (torch.Tensor): The mask for the completions 
                of shape (batch_size, num_responses_per_example, completions_seq_len).
        Returns:
            torch.Tensor: The GRPO loss.
        """
        kl_div_loss = self.beta * torch.nn.functional.kl_div(
            input=policy_logprob_scores,
            target=reference_logprob_scores,
            reduction="none",
            log_target=True,
        )
        kl_div_loss = kl_div_loss * completions_mask
        kl_div_loss = self.beta * kl_div_loss.sum(dim=-1)

        completions_length = completions_mask.sum(dim=-1)

        policy_ratio = torch.exp(policy_logprob_scores - old_policy_logprob_scores)
        policy_ratio = policy_ratio * completions_mask
        clipped_policy_loss = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = torch.minimum(policy_ratio * advantage_scores, clipped_policy_loss * advantage_scores)
        policy_loss = policy_loss.sum(dim=-1)
        grpo_loss = policy_loss - kl_div_loss
        grpo_loss /= completions_length
        return grpo_loss.mean()
    
    def compute_rewards(self, sampled_responses: List, answers:List[str]):
        # Repeat the answers for each response num_responses_per_example times
        answers = [answer for answer in answers for _ in range(self.num_responses_per_example)]
        sampled_responses = [
            response[i] for i in range(self.num_responses_per_example) for response in sampled_responses
        ]
        correct_answer_rewards = correct_answer_reward(
            answers=sampled_responses,
            reference_answer=answers,
        )
        format_rewards = format_reward(
            answers=sampled_responses,
            reference_format_regex=r"(<think>)\w+(</think>)\s*(<answer>)(\d+)(</answer>)",
        )
        return torch.tensor(correct_answer_rewards).view(-1, self.num_responses_per_example), \
            torch.tensor(format_rewards).view(-1, self.num_responses_per_example)
    
    def compute_advantage_score(self, rewards: torch.Tensor):
        """
        Standardize the rewards. To get the advantage score of each sampled response

        Args:
            rewards (torch.Tensor): The rewards to standardize.

        Returns:
            torch.Tensor: The advantage scores.
        """
        mean_rewards = rewards.mean(dim=1).unsqueeze(1)
        std = rewards.std(dim=1).unsqueeze(1)
        advantage_scores = (rewards - mean_rewards) / (std + 1e-8)
        return advantage_scores
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        current_step = self.trainer.global_step
        if current_step > 0 and current_step % self.num_steps_to_refresh_old_policy == 0:
            self.old_policy_model = copy.deepcopy(self.policy_model)

    def training_step(self, batch, batch_idx):      
        prompts = [prompt for prompt in batch["prompt"]] 
        
        inputs = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512, 
            padding=True,
            padding_side='left',
            add_special_tokens=False
        )

        prompt_mask = inputs["attention_mask"]
        # Since we pad the prompts, 
        # all the completions will start from the size of the padded input/prompt
        prompt_end_index = inputs["input_ids"].size(1)
        
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
        completion_ids = sampled_responses[:, prompt_end_index:]
        completions_mask = self._get_completions_mask(completion_ids)

        # Get the rewards for each response
        completions = [self.tokenizer.batch_decode(completion_ids[i*self.num_responses_per_example:(i+1)*self.num_responses_per_example], skip_special_tokens=True) 
            for i in range(len(batch["prompt"]))]
        correct_answer_rewards, format_rewards = self.compute_rewards(completions, batch["answer"])

        advantage_scores = self.compute_advantage_score(correct_answer_rewards + format_rewards)
        logger.info(f"Correct answer rewards: {correct_answer_rewards.mean(dim=1)}")
        logger.info(f"Format rewards: {format_rewards.mean(dim=1)}")
        
        # Repeat the prompts for each response
        prompt_ids = inputs["input_ids"].repeat_interleave(self.num_responses_per_example, dim=0)
        prompt_mask = inputs["attention_mask"].repeat_interleave(self.num_responses_per_example, dim=0)

        # Compute the forward pass with gradient calculation enabled.
        policy_prob_scores = self._get_completion_log_prob_scores(
            prompt_ids, prompt_mask, completion_ids, completions_mask, is_policy_model=True
        )
        # Compute the forward pass with gradient calculation disabled.
        reference_prob_scores = self._get_completion_log_prob_scores(
            prompt_ids, prompt_mask, completion_ids, completions_mask, is_policy_model=False
        )

        loss = self.compute_grpo_loss(
            policy_prob_scores,
            policy_prob_scores,
            reference_prob_scores,
            advantage_scores,
            completions_mask.view(-1, self.num_responses_per_example, completion_ids.shape[-1]),
        )
        logger.info(f"Loss is {loss.item()}")
        return loss
        


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
    
    