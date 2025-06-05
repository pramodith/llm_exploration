from transformers import AutoModelForCausalLM, AutoTokenizer
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
import torch


from dataset_processor import get_gsm8k_dataset, create_dataloader, repeat_row_n_times, tokenize_example
from schemas import ModelType
from benchmark_model import benchmark_model

torch.set_float32_matmul_precision("medium")

logger.add("grpo_trainer.log", rotation="10 MB")
pl.seed_everything(42, workers=True)


class SimpleGRPOModule(pl.LightningModule):
    def save_policy_parameters(self, filepath: str):
        """
        Save only the policy model's parameters to the specified filepath.
        Args:
            filepath (str): Path to save the policy model parameters (e.g., 'policy_model.pth').
        """
        torch.save(self.policy_model.state_dict(), filepath)

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
        num_iterations: int = 1,
        sync_ref_model_every_n_steps: int = 64,
        ref_model_mixup_alpha: float = 1.0,
        learning_rate: float = 1e-6,
        max_steps: int = 1000,
        is_peft: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = ["q_proj", "v_proj"],
        bottom_k_layers_to_train: int = 4,
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
            num_iterations (int): The number of times to update the policy model for each batch.
            sync_ref_model_every_n_steps (int): The number of steps to wait before syncing the reference model.
            ref_model_mixup_alpha (float): The alpha for the reference model mixup.
            learning_rate (float): The learning rate for the optimizer.
            max_steps (int): The maximum number of steps to train for.
            is_peft (bool): Whether to use PEFT.
            lora_rank (int): The rank of the Lora layers.
            lora_alpha (int): The alpha of the Lora layers.
            lora_dropout (float): The dropout of the Lora layers.
            lora_target_modules (List[str]): The target modules for the Lora layers.
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
            print(
                f"Model has {len(self.policy_model.model.layers)} num of layers,\
                we'll only train the last {bottom_k_layers_to_train} layers"
            )
            for param in self.policy_model.model.layers[
                :-bottom_k_layers_to_train
            ].parameters():
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
        self.num_iterations = num_iterations
        self.sync_ref_model_every_n_steps = sync_ref_model_every_n_steps
        self.ref_model_mixup_alpha = ref_model_mixup_alpha
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self._step = 0
        self.cache = {}
        # Disable dropout after setting the model to train mode
        # self._disable_dropout()

    def _get_completion_log_prob_scores(
        self,
        prompt_ids: torch.LongTensor,
        prompt_mask: torch.LongTensor,
        completion_ids: torch.LongTensor,
        completions_mask: torch.LongTensor,
        model_type: ModelType,
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
            self.policy_model = self.policy_model.train()
            logit_scores = self.policy_model(
                input_ids=prompt_completion_input, attention_mask=prompt_completion_mask
            ).logits
        
        elif model_type == ModelType.Old:
            self.policy_model = self.policy_model.eval()
            logit_scores = self.policy_model(
                input_ids=prompt_completion_input, attention_mask=prompt_completion_mask
            ).logits
        
        elif model_type == ModelType.Reference:
            with torch.no_grad():
                logit_scores = self.reference_model(
                    input_ids=prompt_completion_input,
                    attention_mask=prompt_completion_mask,
                ).logits

        # If you want to use logprobs for prompt tokens, access out.prompt_logprobs
        # Logit scores are of shape (batch_size * num_responses_per_example, seq_len + 1, vocab_size)
        # We exclude the logit scores for the prompt and the last token
        # because it corresponds to the next token prediction
        logit_scores = logit_scores[:, prompt_length - 1 : -1, :]

        # Get log_probs to avoid numerical underflow/overflow
        logit_scores = logit_scores / self.temperature
        log_prob_scores = torch.log_softmax(logit_scores, dim=-1)
        # We only need to keep the logit scores corresponding to the completion tokens
        log_prob_scores = torch.gather(
            log_prob_scores, dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)
        return log_prob_scores.view(
            -1, self.num_responses_per_example, log_prob_scores.shape[-1]
        )

    def _disable_dropout(self):
        """
        Disable dropout layers in the active policy model
        to ensure that the model behaves deterministically during training.
        """
        for module in self.policy_model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
                module.training = False

    def _get_completions_mask(
        self, sampled_responses: torch.LongTensor
    ) -> torch.Tensor:
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
        per_token_kl_div_loss = (
            torch.exp(ref_policy_logprob_diff) - ref_policy_logprob_diff - 1
        )

        completions_length = completions_mask.sum(dim=-1)

        policy_ratio = torch.exp(policy_logprob_scores - old_policy_logprob_scores)
        policy_ratio = policy_ratio * completions_mask
        clipped_policy_loss = torch.clamp(
            policy_ratio, 1 - self.epsilon, 1 + self.epsilon
        )
        policy_score = torch.minimum(
            policy_ratio * advantage_scores.unsqueeze(-1),
            clipped_policy_loss * advantage_scores.unsqueeze(-1),
        )

        logger.info(
            f"Policy score: {((policy_score * completions_mask).sum(dim=-1) / completions_length).mean()}"
        )
        logger.info(
            f"KL div loss: {((per_token_kl_div_loss * completions_mask).sum(dim=-1) / completions_length).mean()}"
        )
        trl_kl_div_loss = (
            per_token_kl_div_loss * completions_mask
        ).sum() / completions_mask.sum()
        logger.info(f"TRL KL div loss: {trl_kl_div_loss.mean()}")
        if trl_kl_div_loss.item() > 1.0:
            logger.warning(f"KL div loss is too high: {trl_kl_div_loss}")

        # In the GRPO paper the total reward is policy_score - kl_div_loss which is maximized
        # To convert it to a loss that's minimized we need to negate it
        grpo_loss = -1.0 * (policy_score - self.beta * per_token_kl_div_loss)
        grpo_loss = grpo_loss * completions_mask
        grpo_loss /= completions_length.unsqueeze(-1)
        return grpo_loss.mean()

    def compute_rewards(
        self,
        sampled_responses: List,
        answers: List[str],
        completions_mask: torch.LongTensor,
    ):
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
        answers = [
            answer for answer in answers for _ in range(self.num_responses_per_example)
        ]
        sampled_responses = [
            response[i]
            for i in range(self.num_responses_per_example)
            for response in sampled_responses
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
        correct_answer_rewards = torch.tensor(correct_answer_rewards).view(
            -1, self.num_responses_per_example
        )
        format_rewards = torch.tensor(format_rewards).view(
            -1, self.num_responses_per_example
        )
        length_rewards = torch.tensor(length_rewards).view(
            -1, self.num_responses_per_example
        )

        return (
            correct_answer_rewards.to(self.device),
            format_rewards.to(self.device),
            length_rewards.to(self.device),
        )

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
        if self_step > 0 and self._step % self.sync_ref_model_every_n_steps == 0:
            logger.info("Syncing reference model")
            for ref_param, policy_param in zip(
                self.reference_model.parameters(), self.policy_model.parameters()
            ):
                ref_param.data.copy_(
                    self.ref_model_mixup_alpha * ref_param.data
                    + (1 - self.ref_model_mixup_alpha) * policy_param.data
                )
        return super().on_train_batch_start(batch, batch_idx)

    def configure_optimizers(self):
        """
        Configure the optimizer for the GRPOTrainer.
        We use AdamW optimizer
        """
        optimizer = torch.optim.AdamW(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.0,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=self.max_steps
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
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        inputs = inputs.to(self.device)
        return inputs

    def get_responses_from_policy_model(
        self, inputs: Dict[str, torch.Tensor], prompt_end_index: int
    ):
        with torch.no_grad():
            responses = self.policy_model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_new_tokens=self.max_gen_tokens,
                num_return_sequences=self.num_responses_per_example,
            )

            # Get rid of the prompt tokens in the response
            completion_ids = responses[:, prompt_end_index:]
            # Get the rewards for each response
            completions = [
                self.tokenizer.batch_decode(
                    completion_ids[
                        i * self.num_responses_per_example : (i + 1)
                        * self.num_responses_per_example
                    ],
                    skip_special_tokens=True,
                )
                for i in range(len(inputs["input_ids"]))
            ]

        return completions, completion_ids

    def training_step(self, batch: Dict[str, List[str]], batch_idx: int):

        """
        The num_iterations parameter is used to specify how many times the policy model
        will be updated for a given batch of prompts.

        Since the old policy is used to generate responses and both the old policy and reference
        model are not updated via backprop. The responses/logit scores from the old policy and reference
        model will be the same for all iterations. So we only generate these values/tensors once
        and use them for all iterations.

        Whereas we get the logit scores for the policy model for each iteration.
        """
        if self._step % self.num_iterations == 0:
            inputs = self.prepare_inputs(batch)
            # Get the prompts and answers from the batch
            # The batch is a dictionary with keys "prompt" and "answer"
            # and values are lists of strings.
            prompt_mask = inputs["attention_mask"]
            # Since we pad the prompts,
            # all the completions will start from the size of the padded input/prompt
            prompt_end_index = inputs["input_ids"].size(1)

            # Get the completions from the policy model
            completions, completion_ids = self.get_responses_from_policy_model(
                inputs, prompt_end_index
            )

            completions_mask = self._get_completions_mask(completion_ids)

            # logger.info(f"Sample question: {batch['question'][0]}")
            # logger.info(f"Sample answer: {batch['answer'][0]}")
            # logger.info(f"Sampled responses: {completions[0][0]}")
            correct_answer_rewards, format_rewards, length_rewards = (
                self.compute_rewards(completions, batch["answer"], completions_mask)
            )

            # Log total rewards per step
            average_rewards = (
                (correct_answer_rewards + format_rewards + length_rewards).mean().item()
            )
            advantage_scores = self.compute_advantage_score(
                correct_answer_rewards + format_rewards + length_rewards
            )
            logger.info(f"Correct answer rewards: {correct_answer_rewards.mean(dim=1)}")
            logger.info(f"Format rewards: {format_rewards.mean(dim=1)}")
            logger.info(f"Length rewards: {length_rewards.mean(dim=1)}")

            # Repeat the prompts for each response
            prompt_ids = inputs["input_ids"].repeat_interleave(
                self.num_responses_per_example, dim=0
            )
            prompt_mask = inputs["attention_mask"].repeat_interleave(
                self.num_responses_per_example, dim=0
            )
            self.cache["advantage_scores"] = advantage_scores
            self.cache["prompt_ids"] = prompt_ids
            self.cache["prompt_mask"] = prompt_mask
            self.cache["completions_length"] = completion_ids.shape[-1]
            self.cache["completions_mask"] = completions_mask

            if self.num_iterations > 1 and self._step % self.num_iterations == 0:
                # If we are refreshing the old policy, we need to compute the log probabilities
                # for the old policy model as well.
                old_policy_prob_scores = self._get_completion_log_prob_scores(
                    self.cache["prompt_ids"],
                    self.cache["prompt_mask"],
                    self.cache["completion_ids"],
                    self.cache["completions_mask"],
                    model_type=ModelType.Old,
                )
            else:
                # The old policy model is the same as the current policy model so the outputs would
                # be the same.
                old_policy_prob_scores = policy_prob_scores.detach()
            self.cache["old_policy_prob_scores"] = old_policy_prob_scores

            # Compute the forward pass with gradient calculation disabled.
            reference_prob_scores = self._get_completion_log_prob_scores(
                self.cache["prompt_ids"],
                self.cache["prompt_mask"],
                self.cache["completion_ids"],
                self.cache["completions_mask"],
                model_type=ModelType.Reference,
            )
            self.cache["reference_prob_scores"] = reference_prob_scores

        # Compute the forward pass with gradient calculation enabled.
        policy_prob_scores = self._get_completion_log_prob_scores(
            self.cache["prompt_ids"],
            self.cache["prompt_mask"],
            self.cache["completion_ids"],
            self.cache["completions_mask"],
            model_type=ModelType.Active,
        )

        loss = self.compute_grpo_loss(
            policy_prob_scores,
            self.cache["old_policy_prob_scores"],
            self.cache["reference_prob_scores"],
            self.cache["advantage_scores"],
            self.cache["completions_mask"].view(
                -1, self.num_responses_per_example, self.completions_length
            ),
        )

        self.log_dict(
            {
                "train_loss": loss,
                "train_average_rewards": average_rewards,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
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
        completions, completion_ids = self.get_responses_from_policy_model(
            inputs, prompt_end_index
        )
        correct_answer_rewards, _ = self.compute_rewards(completions, batch["answer"])

        correct_answer_rewards = correct_answer_rewards == 1
        num_correct_per_group = correct_answer_rewards.sum(dim=0)
        self.val_num_correct_per_group.append(num_correct_per_group)

    def on_validation_epoch_end(self):
        num_correct_per_group = torch.stack(self.val_num_correct_per_group).sum(dim=0)
        logger.info(f"Average number of correct responses: {num_correct_per_group}")
        self.log(
            "average_num_correct_responses", num_correct_per_group.float().mean().item()
        )
        self.val_num_correct_per_group.clear()
        return num_correct_per_group.float().mean().item()


if __name__ == "__main__":
    grpo_module = SimpleGRPOModule(
        model_name_or_path="HuggingFaceTB/SmolLM2-360M-Instruct",
        num_responses_per_example=8,
        top_k=50,
        top_p=0.9,
        temperature=0.9,
        max_gen_tokens=300,
        max_steps=10,
        num_steps_to_refresh_old_policy=64,
        is_peft=False,
        bottom_k_layers_to_train=-1,
        learning_rate=5e-5,
    )
    train_dataset, test_dataset = get_gsm8k_dataset()
    train_dataset = train_dataset.map(
        tokenize_example, fn_kwargs={"tokenizer": grpo_module.tokenizer}
    )
    test_dataset = test_dataset.map(
        tokenize_example, fn_kwargs={"tokenizer": grpo_module.tokenizer}
    )
    # Create num_iterations number of duplicate rows for each dataset
    train_dataset = repeat_row_n_times(train_dataset, grpo_module.num_iterations)
    test_dataset = repeat_row_n_times(test_dataset, grpo_module.num_iterations)

    train_dataloader = create_dataloader(train_dataset, is_train=False, batch_size=1)
    test_dataloader = create_dataloader(test_dataset, is_train=False, batch_size=32)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpointer = ModelCheckpoint(
        monitor="train_loss",
        dirpath="./lightning_logs",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
        every_n_train_steps=100,
    )

    grpo_trainer = Trainer(
        max_steps=10,
        accelerator="auto",
        precision="bf16-mixed",
        callbacks=[lr_monitor, checkpointer],
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
    )

    # Pre-training evaluation on the test set
    # print("\n===== Pre-training evaluation on test set =====")
    # pretrain_eval_results = grpo_trainer.validate(model=grpo_module, dataloaders=test_dataloader)
    # print(f"Pre-training evaluation results: {pretrain_eval_results}\n")

    grpo_trainer.fit(
        model=grpo_module,
        train_dataloaders=train_dataloader,
    )

    ## Load the best checkpoint
    # grpo_module.save_policy_parameters("./lightning_logs/policy_model.pth")

    # tokenizer = grpo_module.tokenizer
    # grpo_module = None
    # benchmark_model(
    #     "./lightning_logs/policy_model.pth",
    #     tokenizer,
    #     test_dataset,
    #     batch_size=16,
    #     top_k=50,
    #     top_p=0.9,
    #     temperature=0.7,
    #     max_completion_length=300,
    #     )
