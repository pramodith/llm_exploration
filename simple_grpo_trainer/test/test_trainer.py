import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import sys
import os

from simple_grpo_trainer.src.GRPOTrainer import SimpleGRPOModule

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
                test_sequence = torch.tensor([[1, 2, 3, 50256, 5, 6]])
                expected_mask = torch.tensor([[1, 1, 1, 1, 0, 0]]).int()
                mask = module._get_completions_mask(test_sequence)
                assert torch.all(mask == expected_mask), f"Expected {expected_mask}, got {mask}"
                
                # Test case 2: Multiple sequences with EOS tokens at different positions
                test_sequences = torch.tensor([
                    [1, 2, 50256, 4, 5, 6],
                    [1, 2, 3, 4, 50256, 6],
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
                
                # Test case 4: Sequence with multiple EOS tokens
                test_sequence_multi_eos = torch.tensor([[1, 50256, 3, 50256, 5, 6]])
                expected_mask_multi_eos = torch.tensor([[1, 1, 0, 0, 0, 0]]).int()
                mask_multi_eos = module._get_completions_mask(test_sequence_multi_eos)
                assert torch.all(mask_multi_eos == expected_mask_multi_eos), f"Expected {expected_mask_multi_eos}, got {mask_multi_eos}"
