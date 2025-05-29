import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from simple_grpo_trainer.src.GRPOTrainer import SimpleGRPOModule
from simple_grpo_trainer.src.schemas import ModelType

class TestSimpleGRPOModule:
    
    def test_disable_dropout(self):
        """
        Test that _disable_dropout correctly disables all dropout layers
        in the policy model by setting p=0.0 and training=False.
        """
        # Create a mock model with dropout layers
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout1 = nn.Dropout(p=0.5)
                self.dropout2 = nn.Dropout(p=0.3)
                self.linear = nn.Linear(10, 10)
                self.seq = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.Dropout(p=0.2),
                    nn.ReLU()
                )
            
            def forward(self, x):
                return x
        
        # Create a SimpleGRPOModule instance with the mock model
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained'):
                # Set up our mock model
                policy_model = MockModel()
                mock_model.return_value = policy_model
                
                # Initialize the module with our mocked components
                module = SimpleGRPOModule(model_name_or_path="dummy_model")
                
                # Verify initial dropout settings
                dropouts = [m for m in policy_model.modules() if isinstance(m, nn.Dropout)]
                assert len(dropouts) == 3, "Should have 3 dropout layers"
                
                # Verify all dropout layers have been disabled
                for dropout in dropouts:
                    assert dropout.p == 0.0, "Dropout probability should be set to 0.0"
                    assert dropout.training == False, "Dropout should not be in training mode"
    
    def test_disable_dropout_no_dropout_layers(self):
        """
        Test that _disable_dropout works correctly when there are no dropout layers.
        """
        # Create a mock model without dropout layers
        class MockModelNoDropout(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                return x
        
        # Create a SimpleGRPOModule instance with the mock model
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained'):
                # Set up our mock model
                policy_model = MockModelNoDropout()
                mock_model.return_value = policy_model
                
                # Initialize the module with our mocked components
                module = SimpleGRPOModule(model_name_or_path="dummy_model")
                
                # Verify no dropout layers
                dropouts = [m for m in policy_model.modules() if isinstance(m, nn.Dropout)]
                assert len(dropouts) == 0, "Should have no dropout layers"
                
                # Call the method we're testing - should not raise any errors
                module._disable_dropout()
                
                # Test passes if no exceptions are raised

    def test_disable_dropout_nested_modules(self):
        """
        Test that _disable_dropout correctly disables dropout layers
        in deeply nested modules.
        """
        # Create a mock model with nested dropout layers
        class NestedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(p=0.4)
                self.nested = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.Dropout(p=0.6)
                )
            
            def forward(self, x):
                return x
        
        class MockModelNested(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(p=0.5)
                self.nested_module = NestedModule()
            
            def forward(self, x):
                return x
        
        # Create a SimpleGRPOModule instance with the mock model
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained'):
                # Set up our mock model
                policy_model = MockModelNested()
                mock_model.return_value = policy_model
                
                # Initialize the module with our mocked components
                module = SimpleGRPOModule(model_name_or_path="dummy_model")
                
                # Verify initial dropout settings
                dropouts = [m for m in policy_model.modules() if isinstance(m, nn.Dropout)]
                assert len(dropouts) == 3, "Should have 3 dropout layers"
                
                # Call the method we're testing
                module._disable_dropout()
                
                # Verify all dropout layers have been disabled, including nested ones
                for dropout in dropouts:
                    assert dropout.p == 0.0, "Dropout probability should be set to 0.0"
                    assert dropout.training == False, "Dropout should not be in training mode"
                    
    def test_get_completions_mask(self):
        """
        Test that _get_completions_mask correctly identifies valid tokens up to the first EOS token.
        """
        # Create a SimpleGRPOModule instance with a mock model
        with patch('transformers.AutoModelForCausalLM.from_pretrained'):
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                # Set up our mock tokenizer
                mock_tokenizer_instance = MagicMock()
                mock_tokenizer_instance.eos_token_id = 50256  # Common EOS token ID for GPT models
                mock_tokenizer.return_value = mock_tokenizer_instance
                
                # Initialize the module with our mocked components
                module = SimpleGRPOModule(model_name_or_path="dummy_model")
                
                # Test case 1: Single sequence with EOS token
                test_sequence = torch.tensor([[1, 2, 3, 50256, 50256, 50256]])
                expected_mask = torch.tensor([[1, 1, 1, 1, 0, 0]]).int()
                mask = module._get_completions_mask(test_sequence)
                assert torch.all(mask == expected_mask), f"Expected {expected_mask}, got {mask}"
                
                # Test case 2: Multiple sequences with EOS tokens at different positions
                test_sequences = torch.tensor([
                    [1, 2, 50256, 50256, 50256, 50256],
                    [1, 2, 3, 4, 50256, 50256],
                    [1, 2, 3, 4, 5, 50256]
                ])
                expected_masks = torch.tensor([
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1]
                ]).int()
                masks = module._get_completions_mask(test_sequences)
                assert torch.all(masks == expected_masks), f"Expected {expected_masks}, got {masks}"
                
                # Test case 3: Sequence with no EOS token
                test_sequence_no_eos = torch.tensor([[1, 2, 3, 4, 5, 6]])
                expected_mask_no_eos = torch.tensor([[1, 1, 1, 1, 1, 1]]).int()
                mask_no_eos = module._get_completions_mask(test_sequence_no_eos)
                assert torch.all(mask_no_eos == expected_mask_no_eos), f"Expected {expected_mask_no_eos}, got {mask_no_eos}"
                
    def test_compute_advantage_score(self):
        """
        Test that compute_advantage_score standardizes rewards correctly.
        """
        # Patch model loading to avoid heavy dependencies
        with patch('transformers.AutoModelForCausalLM.from_pretrained'):
            with patch('transformers.AutoTokenizer.from_pretrained'):
                module = SimpleGRPOModule(model_name_or_path="dummy_model")
                # Test 1: Simple case
                rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
                advantage = module.compute_advantage_score(rewards)
                np_adv = ((rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)).numpy()
                assert np.allclose(advantage.numpy(), np_adv, atol=1e-6)

                # Test 2: Batched rewards
                rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
                advantage = module.compute_advantage_score(rewards)
                np_adv = ((rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)).numpy()
                assert np.allclose(advantage.numpy(), np_adv, atol=1e-6)

                # Test 3: All rewards are the same (std=0)
                rewards = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
                advantage = module.compute_advantage_score(rewards)
                # Should be all zeros
                assert torch.allclose(advantage, torch.zeros_like(rewards), atol=1e-6)

                # Test 4: Negative rewards
                rewards = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
                advantage = module.compute_advantage_score(rewards)
                np_adv = ((rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)).numpy()
                assert np.allclose(advantage.numpy(), np_adv, atol=1e-6)

                # Test 5: Large batch, random values
                torch.manual_seed(0)
                rewards = torch.randn(8, 4)
                advantage = module.compute_advantage_score(rewards)
                np_adv = ((rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)).numpy()
                assert np.allclose(advantage.numpy(), np_adv, atol=1e-6)

    def test_compute_rewards_basic(self):
        """
        Test compute_rewards returns correct shapes and values for simple cases.
        """
        with patch('transformers.AutoModelForCausalLM.from_pretrained'):
            with patch('transformers.AutoTokenizer.from_pretrained'):
                # Patch correct_answer_reward and format_reward
                with patch('simple_grpo_trainer.src.GRPOTrainer.correct_answer_reward') as mock_correct, \
                     patch('simple_grpo_trainer.src.GRPOTrainer.format_reward') as mock_format:
                    module = SimpleGRPOModule(model_name_or_path="dummy_model", num_responses_per_example=2)
                    # Setup mocks
                    mock_correct.return_value = [1.0, 0.0, 1.0, 0.0]
                    mock_format.return_value = [0.1, 0.2, 0.3, 0.4]
                    sampled_responses = [["answer 1", "answer 2"], ["answer 3", "answer 4"]]
                    answers = ["1", "3"]
                    correct, fmt = module.compute_rewards(sampled_responses, answers)
                    # Should be shape (2, 2)
                    assert correct.shape == (2, 2)
                    assert fmt.shape == (2, 2)
                    # Should match mocked values
                    np.testing.assert_allclose(correct.numpy(), [[1.0, 0.0], [1.0, 0.0]])
                    np.testing.assert_allclose(fmt.numpy(), [[0.1, 0.2], [0.3, 0.4]])

    def test_compute_rewards_repeat_and_flatten(self):
        """
        Test that compute_rewards repeats and flattens inputs as expected.
        """
        with patch('transformers.AutoModelForCausalLM.from_pretrained'):
            with patch('transformers.AutoTokenizer.from_pretrained'):
                with patch('simple_grpo_trainer.src.GRPOTrainer.correct_answer_reward') as mock_correct, \
                     patch('simple_grpo_trainer.src.GRPOTrainer.format_reward') as mock_format:
                    module = SimpleGRPOModule(model_name_or_path="dummy_model", num_responses_per_example=3)
                    # 2 prompts, 3 responses each
                    sampled_responses = [["a1", "a2", "a3"], ["b1", "b2", "b3"]]
                    answers = ["x", "y"]
                    # Should flatten to [a1, b1, a2, b2, a3, b3] and repeat answers
                    def check_args(answers, reference_answer):
                        # answers: [a1, b1, a2, b2, a3, b3]
                        # reference_answer: [x, x, x, y, y, y]
                        assert answers == ["a1", "b1", "a2", "b2", "a3", "b3"]
                        assert reference_answer == ["x", "x", "x", "y", "y", "y"]
                        return [0.0]*6
                    mock_correct.side_effect = check_args
                    mock_format.return_value = [0.0]*6
                    correct, fmt = module.compute_rewards(sampled_responses, answers)
                    assert correct.shape == (2, 3)
                    assert fmt.shape == (2, 3)

    def test_get_completion_log_prob_scores_shapes_and_calls(self):
        """
        Test _get_completion_log_prob_scores returns correct shape and calls the right model.
        """
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained'):
                # Setup dummy models with a forward method
                class DummyModel:
                    def __init__(self):
                        self.logits = None
                    def eval(self):
                        pass
                    def train(self):
                        pass
                    def modules(self):
                        return []
                    def __call__(self, input_ids, attention_mask):
                        # Return logits of shape (batch, seq_len+1, vocab_size)
                        batch, seq = input_ids.shape
                        vocab_size = 4
                        # Fill with increasing values for deterministic output
                        logits = torch.arange(batch * (seq+1) * vocab_size, dtype=torch.float32).reshape(batch, seq+1, vocab_size)
                        self.logits = logits
                        return MagicMock(logits=logits)
                
                """
                Logits will be
                [
                    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                    [[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]],
                    [[32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]],
                    [[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59], [60, 61, 62, 63]
                ]
                """
                dummy = DummyModel()
                mock_model.return_value = dummy
                module = SimpleGRPOModule(model_name_or_path="dummy_model", num_responses_per_example=2)
                module.policy_model = dummy
                module.old_policy_model = dummy
                module.reference_model = dummy
                module.temperature = 1.0
                # Inputs
                prompt_ids = torch.ones((4, 2), dtype=torch.long)
                prompt_mask = torch.ones((4, 2), dtype=torch.long)
                completion_ids = torch.zeros((4, 2), dtype=torch.long)
                completions_mask = torch.ones((4, 2), dtype=torch.long)
                
                logits = torch.log_softmax(dummy(torch.zeros(4, 3), None).logits, dim=-1)
                # The first token of the 3rd sequence in each batch will be 8, 24, 40, 56
                expected_logits = logits[:, 2:4, 0].view(2, 2, 2)
                # Test for each model type
                for model_type in [ModelType.Active, ModelType.Old, ModelType.Reference]:
                    out = module._get_completion_log_prob_scores(
                        prompt_ids, prompt_mask, completion_ids, completions_mask, model_type=model_type
                    )
                    
                    # Should have shape (batch, num_responses_per_example, completion_seq_len)
                    assert out.shape[0] == 2
                    assert out.shape[1] == 2
                    assert out.shape[2] == 2

                    torch.testing.assert_close(
                        out, expected_logits, rtol=1e-4, atol=1e-4,
                        msg=f"{out} does not match expected output {expected_logits} for model type {model_type}"
                    )