import torch
import pytest
from .GRPOTrainer import SimpleGRPOTrainer

@pytest.fixture
def trainer():
    # Use a small model for testing or mock if needed
    class DummyModel(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None):
            batch, seq = input_ids.shape
            vocab_size = 10
            logits = torch.randn(batch, seq, vocab_size)
            return type("obj", (), {"logits": logits})

    class DummyTokenizer:
        eos_token_id = 0
        def batch_decode(self, ids, skip_special_tokens=True):
            return ["dummy"] * len(ids)

    trainer = SimpleGRPOTrainer.__new__(SimpleGRPOTrainer)
    trainer.tokenizer = DummyTokenizer()
    trainer.policy_model = DummyModel()
    trainer.reference_model = DummyModel()
    trainer.num_responses_per_example = 2
    trainer.top_k = 5
    trainer.top_p = 0.9
    trainer.temperature = 1.0
    trainer.max_gen_tokens = 5
    trainer.beta = 0.04
    trainer.epsilon = 0.2
    return trainer

def test_compute_grpo_loss_shapes(trainer):
    batch_size = 3
    num_responses = trainer.num_responses_per_example
    seq_len = 4

    # Random logprob scores
    policy_logprob_scores = torch.randn(batch_size, num_responses, seq_len)
    old_policy_logprob_scores = torch.randn(batch_size, num_responses, seq_len)
    reference_logprob_scores = torch.randn(batch_size, num_responses, seq_len)
    advantage_scores = torch.randn(batch_size, num_responses)
    completions_mask = torch.randint(0, 2, (batch_size, num_responses, seq_len)).float()

    loss = trainer.compute_grpo_loss(
        policy_logprob_scores,
        old_policy_logprob_scores,
        reference_logprob_scores,
        advantage_scores,
        completions_mask,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Should be a scalar

def test_compute_grpo_loss_zero_mask(trainer):
    batch_size = 2
    num_responses = trainer.num_responses_per_example
    seq_len = 3

    # All zeros in completions_mask should result in nan or inf loss due to division by zero
    policy_logprob_scores = torch.zeros(batch_size, num_responses, seq_len)
    old_policy_logprob_scores = torch.zeros(batch_size, num_responses, seq_len)
    reference_logprob_scores = torch.zeros(batch_size, num_responses, seq_len)
    advantage_scores = torch.ones(batch_size, num_responses)
    completions_mask = torch.zeros(batch_size, num_responses, seq_len)

    loss = trainer.compute_grpo_loss(
        policy_logprob_scores,
        old_policy_logprob_scores,
        reference_logprob_scores,
        advantage_scores,
        completions_mask,
    )
    assert torch.isnan(loss) or torch.isinf(loss)

def test_compute_grpo_loss_identical_inputs(trainer):
    batch_size = 2
    num_responses = trainer.num_responses_per_example
    seq_len = 3

    # If policy and reference are identical, KL should be zero
    policy_logprob_scores = torch.ones(batch_size, num_responses, seq_len)
    old_policy_logprob_scores = torch.ones(batch_size, num_responses, seq_len)
    reference_logprob_scores = torch.ones(batch_size, num_responses, seq_len)
    advantage_scores = torch.ones(batch_size, num_responses)
    completions_mask = torch.ones(batch_size, num_responses, seq_len)

    loss = trainer.compute_grpo_loss(
        policy_logprob_scores,
        old_policy_logprob_scores,
        reference_logprob_scores,
        advantage_scores,
        completions_mask,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

def test_compute_grpo_loss_gradient(trainer):
    batch_size = 1
    num_responses = trainer.num_responses_per_example
    seq_len = 2

    policy_logprob_scores = torch.randn(batch_size, num_responses, seq_len, requires_grad=True)
    old_policy_logprob_scores = torch.randn(batch_size, num_responses, seq_len)
    reference_logprob_scores = torch.randn(batch_size, num_responses, seq_len)
    advantage_scores = torch.ones(batch_size, num_responses)
    completions_mask = torch.ones(batch_size, num_responses, seq_len)

    loss = trainer.compute_grpo_loss(
        policy_logprob_scores,
        old_policy_logprob_scores,
        reference_logprob_scores,
        advantage_scores,
        completions_mask,
    )
    loss.backward()
    assert policy_logprob_scores.grad is not None