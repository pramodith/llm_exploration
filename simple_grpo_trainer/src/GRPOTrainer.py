from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from transformers.optimization import get_linear_schedule_with_warmup
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from trl import GRPOTrainer
import torch
from rewards import correct_answer_reward, format_reward, length_reward
from loguru import logger
from typing import Dict, List
from peft import get_peft_model, LoraConfig
import time
import copy
import torch.profiler

from dataset_processor import get_gsm8k_dataset, create_dataloader, tokenize_example
from schemas import ModelType

torch.set_float32_matmul_precision('medium')

logger.add("grpo_trainer.log", rotation="10 MB")
pl.seed_everything(42, workers=True)
class SimpleGRPOModule(pl.LightningModule):
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
        num_steps_to_refresh_old_policy: int = 16,
        learning_rate: float = 1e-6,
        max_steps: int = 1000,
        is_peft: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = ["q_proj", "v_proj"],
        bottom_k_layers_to_train: int = 4
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
            learning_rate (float): The learning rate for the optimizer.
            bottom_k_layers_to_train (int): The number of bottom layers to train.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
        )
        self.val_num_correct_per_group = []
        if is_peft:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self.policy_model = get_peft_model(self.policy_model, lora_config)
            self.policy_model.print_trainable_parameters()
        elif bottom_k_layers_to_train > 0:
            print(f"Model has {len(self.policy_model.model.layers)} num of layers,\
                we'll only train the last {bottom_k_layers_to_train} layers")
            for param in self.policy_model.model.layers[:-bottom_k_layers_to_train].parameters():
                param.requires_grad = False
        
        self.policy_model = self.policy_model.to(self.device)
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.reference_model = self.reference_model.to(self.device)
        self.reference_model.eval()

        self.num_responses_per_example = num_responses_per_example
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_gen_tokens = max_gen_tokens
        self.beta = beta
        self.epsilon = epsilon
        self.num_steps_to_refresh_old_policy = num_steps_to_refresh_old_policy
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self._step = 0
        # Disable dropout after setting the model to train mode
        # self._disable_dropout()
    
    def _get_completion_log_prob_scores(
        self, 
        prompt_ids: torch.LongTensor,
        prompt_mask: torch.LongTensor,
        completion_ids: torch.LongTensor,
        completions_mask: torch.LongTensor,
        model_type: ModelType
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
        if model_type == ModelType.Active:
            self.policy_model.train()
            logit_scores = self.policy_model(input_ids = prompt_completion_input, attention_mask = prompt_completion_mask).logits
        elif model_type == ModelType.Old:
            with torch.no_grad():
                logit_scores = self.old_policy_model(input_ids = prompt_completion_input, attention_mask = prompt_completion_mask).logits
        elif model_type == ModelType.Reference:
            with torch.no_grad():
                logit_scores = self.reference_model(input_ids = prompt_completion_input, attention_mask = prompt_completion_mask).logits
            
        # If you want to use logprobs for prompt tokens, access out.prompt_logprobs
        # Logit scores are of shape (batch_size * num_responses_per_example, seq_len + 1, vocab_size)
        # We exclude the logit scores for the prompt and the last token 
        # because it corresponds to the next token prediction
        logit_scores = logit_scores[:, prompt_length-1:-1, :]
        
        # Get log_probs to avoid numerical underflow/overflow
        logit_scores = logit_scores / self.temperature
        log_prob_scores = torch.log_softmax(logit_scores, dim=-1)
        # We only need to keep the logit scores corresponding to the completion tokens
        log_prob_scores = torch.gather(log_prob_scores, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)
        return log_prob_scores.view(-1, self.num_responses_per_example, log_prob_scores.shape[-1])
        
    
    def _disable_dropout(self):
        """
        Disable dropout layers in the active policy model
        to ensure that the model behaves deterministically during training.
        """
        for module in self.policy_model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
                module.training = False

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
        ) -> torch.Tensor:
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
        # GRPO uses a custom forumation of KL divergence loss that's always positive
        ref_policy_logprob_diff = reference_logprob_scores - policy_logprob_scores
        kl_div_loss = torch.exp(ref_policy_logprob_diff) - ref_policy_logprob_diff - 1
        kl_div_loss = kl_div_loss * completions_mask
        kl_div_loss = self.beta * kl_div_loss.sum(dim=-1)

        completions_length = completions_mask.sum(dim=-1)

        policy_ratio = torch.exp(policy_logprob_scores - old_policy_logprob_scores)
        policy_ratio = policy_ratio * completions_mask
        clipped_policy_loss = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_score = torch.minimum(policy_ratio * advantage_scores.unsqueeze(-1), clipped_policy_loss * advantage_scores.unsqueeze(-1))
        policy_score = policy_score.sum(dim=-1)
        logger.info(f"Policy score: {(policy_score/completions_length).mean()}")
        logger.info(f"KL div loss: {(kl_div_loss/completions_length).mean()}")
        # In the GRPO paper the total reward is policy_score - kl_div_loss which is maximized
        # To convert it to a loss that's minimized we need to negate it
        grpo_loss = -1.0 * (policy_score - kl_div_loss)
        grpo_loss /= completions_length
        return grpo_loss.mean()
    
    def compute_rewards(self, sampled_responses: List, answers:List[str], completions_mask: torch.LongTensor):
        """
        Compute the rewards for the sampled responses.

        Args:
            sampled_responses (List): The sampled responses.
            answers (List[str]): The answers.
            completions_mask (torch.LongTensor): The completions mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The rewards for the sampled responses.
        """
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
            reference_format_regex=r"(<think>)[\s\S]*?(</think>)[\s\S]*?(<answer>)[\s\D]*(\d+)[\s\D]*(</answer>)",
            # reference_format_regex=r"(?is).*(answer).*(\d+)"
        )
        length_rewards = length_reward(completions_mask)
        correct_answer_rewards = torch.tensor(correct_answer_rewards).view(-1, self.num_responses_per_example)
        format_rewards = torch.tensor(format_rewards).view(-1, self.num_responses_per_example)
        length_rewards = torch.tensor(length_rewards).view(-1, self.num_responses_per_example)

        return correct_answer_rewards.to(self.device),\
                format_rewards.to(self.device),\
                length_rewards.to(self.device)
    
    def compute_advantage_score(self, rewards: torch.Tensor):
        """
        Standardize the rewards. To get the advantage score of each sampled response

        Args:
            rewards (torch.Tensor): The rewards to standardize 
                of shape (batch_size, num_sampled_responses).

        Returns:
            torch.Tensor: The advantage scores of shape (batch_size, num_sampled_responses).
        """
        mean_rewards = rewards.mean(dim=1).unsqueeze(1)
        std = rewards.std(dim=1).unsqueeze(1)
        advantage_scores = (rewards - mean_rewards) / (std + 1e-8)
        return advantage_scores
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self._step % self.num_steps_to_refresh_old_policy == 0:
            logger.info("Refreshing old policy model")
            self.old_policy_model = copy.deepcopy(self.policy_model)
            self.old_policy_model.eval()
        return super().on_train_batch_start(batch, batch_idx)
    
    def configure_optimizers(self):
        """
        Configure the optimizer for the GRPOTrainer.
        We use AdamW optimizer
        """
        optimizer = torch.optim.AdamW(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            lr=self.learning_rate, 
            weight_decay=0.0
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=self.max_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def prepare_inputs(self, batch):
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
        inputs = inputs.to(self.device)
        return inputs
    
    def get_responses_from_policy_model(
        self, 
        inputs: Dict[str, torch.Tensor], 
        prompt_end_index: int
    ):
        with torch.no_grad():
            responses = self.policy_model.generate(
                **inputs,
                do_sample=True,
                temperature = self.temperature,
                top_p = self.top_p,
                top_k=self.top_k,
                max_new_tokens=self.max_gen_tokens,
                num_return_sequences=self.num_responses_per_example,
            )

            # Get rid of the prompt tokens in the response
            completion_ids = responses[:, prompt_end_index:]
            # Get the rewards for each response
            completions = [self.tokenizer.batch_decode(completion_ids[i*self.num_responses_per_example:(i+1)*self.num_responses_per_example], skip_special_tokens=True) 
                for i in range(len(inputs["input_ids"]))]
        
        return completions, completion_ids

    def training_step(self, batch, batch_idx):        
        # Get the prompts and answers from the batch
        # The batch is a dictionary with keys "prompt" and "answer"
        # and values are lists of strings.
        inputs = self.prepare_inputs(batch)
        timings = {}
        prompt_mask = inputs["attention_mask"]
        # Since we pad the prompts, 
        # all the completions will start from the size of the padded input/prompt
        prompt_end_index = inputs["input_ids"].size(1)

        
        self.policy_model.eval()
        # Get the completions from the policy model
        # t1 = time.time()
        completions, completion_ids = self.get_responses_from_policy_model(inputs, prompt_end_index)
        # timings['get_responses_from_policy_model'] = time.time() - t1
    
        completions_mask = self._get_completions_mask(completion_ids)

        logger.info(f"Sample question: {batch['question'][0]}")
        logger.info(f"Sample answer: {batch['answer'][0]}") 
        logger.info(f"Sampled responses: {completions[0][0]}")
        # t2 = time.time()
        correct_answer_rewards, format_rewards, length_rewards = self.compute_rewards(
            completions, batch["answer"], completions_mask
        )
        # timings['compute_rewards'] = time.time() - t2

        # Log total rewards per step
        average_rewards = (correct_answer_rewards + format_rewards + length_rewards).mean().item()
        # t3 = time.time()
        advantage_scores = self.compute_advantage_score(correct_answer_rewards + format_rewards + length_rewards)
        # timings['compute_advantage_score'] = time.time() - t3
        logger.info(f"Correct answer rewards: {correct_answer_rewards.mean(dim=1)}")
        logger.info(f"Format rewards: {format_rewards.mean(dim=1)}")
        logger.info(f"Length rewards: {length_rewards.mean(dim=1)}")

        # Repeat the prompts for each response
        prompt_ids = inputs["input_ids"].repeat_interleave(self.num_responses_per_example, dim=0)
        prompt_mask = inputs["attention_mask"].repeat_interleave(self.num_responses_per_example, dim=0)

        # Compute the forward pass with gradient calculation enabled.
        # t4 = time.time()
        policy_prob_scores = self._get_completion_log_prob_scores(
            prompt_ids, prompt_mask, completion_ids, completions_mask, model_type=ModelType.Active
        )
        # timings['policy_logprob'] = time.time() - t4
        if self._step % self.num_steps_to_refresh_old_policy != 0:
            # If we are refreshing the old policy, we need to compute the log probabilities
            # for the old policy model as well.
            # t5 = time.time()
            old_policy_prob_scores = self._get_completion_log_prob_scores(
                prompt_ids, prompt_mask, completion_ids, completions_mask, model_type=ModelType.Old
            )
            # timings['old_policy_logprob'] = time.time() - t5
        else:
            # The old policy model is the same as the current policy model so the outputs would
            # be the same.
            old_policy_prob_scores = policy_prob_scores
            # timings['old_policy_logprob'] = 0.0

        # Compute the forward pass with gradient calculation disabled.
        # t6 = time.time()
        reference_prob_scores = self._get_completion_log_prob_scores(
            prompt_ids, prompt_mask, completion_ids, completions_mask, model_type=ModelType.Reference
        )
        # timings['reference_logprob'] = time.time() - t6

        # t7 = time.time()
        loss = self.compute_grpo_loss(
            policy_prob_scores,
            old_policy_prob_scores,
            reference_prob_scores,
            advantage_scores,
            completions_mask.view(-1, self.num_responses_per_example, completion_ids.shape[-1]),
        )
        # timings['compute_grpo_loss'] = time.time() - t7

        logger.info(f"Loss is {loss}")
        # for k, v in timings.items():
        #     logger.info(f"TIMING: {k}: {v:.4f} seconds")
        self.log_dict(
            {
                "train_loss": loss, 
                "train_average_rewards": average_rewards
            }, 
            on_step=True, on_epoch=False, prog_bar=True
        )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._step += 1
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        inputs = self.prepare_inputs(batch)
        
        # Since we pad the prompts, 
        # all the completions will start from the size of the padded input/prompt
        prompt_end_index = inputs["input_ids"].size(1)
        
        self.policy_model.eval()
        # Get the completions from the policy model
        completions, completion_ids = self.get_responses_from_policy_model(inputs, prompt_end_index)
        correct_answer_rewards, _ = self.compute_rewards(completions, batch["answer"])

        correct_answer_rewards = correct_answer_rewards == 1
        num_correct_per_group = correct_answer_rewards.sum(dim=0)
        self.val_num_correct_per_group.append(num_correct_per_group)

    def on_validation_epoch_end(self):
        num_correct_per_group = torch.stack(self.val_num_correct_per_group).sum(dim=0)
        logger.info(f"Average number of correct responses: {num_correct_per_group}")
        self.log("average_num_correct_responses", num_correct_per_group.float().mean().item())
        self.val_num_correct_per_group.clear()
        return num_correct_per_group.float().mean().item()        


if __name__ == "__main__":
    grpo_module = SimpleGRPOModule(
        model_name_or_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        num_responses_per_example=8,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        max_gen_tokens=300,
        max_steps=100,
        num_steps_to_refresh_old_policy=16,
        is_peft=False,
        bottom_k_layers_to_train=4,
        learning_rate=2e-5
    )
    train_dataset, test_dataset = get_gsm8k_dataset()
    train_dataset = train_dataset.map(tokenize_example, fn_kwargs={"tokenizer": grpo_module.tokenizer})
    test_dataset = test_dataset.map(tokenize_example, fn_kwargs={"tokenizer": grpo_module.tokenizer})
    train_dataloader = create_dataloader(train_dataset, is_train=False, batch_size=1)
    test_dataloader = create_dataloader(test_dataset, is_train=False, batch_size=32)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpointer = ModelCheckpoint(
        monitor="train_loss",
        dirpath="lightning_logs",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
        every_n_train_steps=20,
    )

    grpo_trainer = Trainer(
        max_steps=10,
        accelerator="auto",
        precision="16-mixed",
        callbacks=[lr_monitor, checkpointer],
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        # profiler="pytorch"
    )

    # Pre-training evaluation on the test set
    # print("\n===== Pre-training evaluation on test set =====")
    # pretrain_eval_results = grpo_trainer.validate(model=grpo_module, dataloaders=test_dataloader)
    # print(f"Pre-training evaluation results: {pretrain_eval_results}\n")

    grpo_trainer.fit(
        model=grpo_module, 
        train_dataloaders=train_dataloader, 
    )
