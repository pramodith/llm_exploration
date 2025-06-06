import unittest
from simple_grpo_trainer.src.rewards import (
    correct_answer_reward,
    format_reward,
    trl_format_reward,
    trl_correct_answer_reward,
)
import numpy as np


class TestRewards(unittest.TestCase):
    def test_trl_format_reward_full_match(self):
        """Test trl_format_reward when there's a full match."""
        completions = ["<think>reasoning</think> <answer>42</answer>"]
        # Matches the reference_regex in trl_format_reward
        rewards = trl_format_reward(completions)
        # The regex has 5 groups, so reward should be 0.5
        self.assertEqual(rewards, [0.5])

    def test_trl_format_reward_no_match(self):
        """Test trl_format_reward when there's no match."""
        completions = ["No tags here"]
        rewards = trl_format_reward(completions)
        self.assertEqual(rewards, [0.0])

    def test_trl_format_reward_multiple(self):
        """Test trl_format_reward with multiple completions."""
        completions = [
            "<think>reasoning</think> <answer>42</answer>",
            "<think>foo\n</think>\n<answer>24</answer>",
            "No tags here",
        ]
        rewards = trl_format_reward(completions)
        self.assertEqual(rewards, [0.5, 0.5, 0.0])

    def test_trl_correct_answer_reward_exact_match(self):
        """Test trl_correct_answer_reward with exact match."""
        completions = ["<think>reasoning</think> <answer>42</answer>"]
        rewards = trl_correct_answer_reward(completions, answer=["42"])
        self.assertEqual(rewards, [1.0])

    def test_trl_correct_answer_reward_wrong_number(self):
        """Test trl_correct_answer_reward with wrong number."""
        completions = ["<think>reasoning</think> <answer>24</answer>"]
        rewards = trl_correct_answer_reward(completions, answer=["42"])
        self.assertEqual(rewards, [0.25])

    def test_trl_correct_answer_reward_no_number(self):
        """Test trl_correct_answer_reward with no number in completion."""
        completions = ["No answer here"]
        rewards = trl_correct_answer_reward(completions, answer=["42"])
        self.assertEqual(rewards, [0.0])

    def test_correct_answer_reward_exact_match(self):
        """Test correct_answer_reward when there's an exact match."""
        answers = ["The answer is 42"]
        reference_answer = ["42"]
        rewards = correct_answer_reward(answers, reference_answer)
        self.assertEqual(rewards, [1.0])

    def test_correct_answer_reward_no_match(self):
        """Test correct_answer_reward when there's no match."""
        answers = ["The answer is 24"]
        reference_answer = ["42"]
        rewards = correct_answer_reward(answers, reference_answer)
        # Extracted answer is "24", which is not equal to ref, so 0.25
        self.assertEqual(rewards, [0.25])

    def test_correct_answer_reward_multiple_answers(self):
        """Test correct_answer_reward with multiple answers."""
        answers = [
            "The answer is 42",
            "The answer is 24",
            "Answer: 42",
            "No number here",
        ]
        reference_answer = ["42", "35", "42", "100"]
        rewards = correct_answer_reward(answers, reference_answer)
        # 1.0 (match), 0.0 (wrong number), 1.0 (match), 0.25 (no number extracted)
        self.assertEqual(rewards, [1.0, 0.25, 1.0, 0.0])

    def test_correct_answer_reward_no_number(self):
        """Test correct_answer_reward when there's no number in the answer."""
        answers = ["The answer is unknown"]
        reference_answer = ["42"]
        rewards = correct_answer_reward(answers, reference_answer)
        # Now, if extraction fails, extracted_answer will be ""
        self.assertEqual(rewards, [0.0])

    def test_correct_answer_reward_case_insensitive(self):
        """Test correct_answer_reward is case insensitive."""
        answers = ["THE ANSWER IS 42", "answer: 42"]
        reference_answer = ["42", "42"]
        rewards = correct_answer_reward(answers, reference_answer)
        self.assertEqual(rewards, [1.0, 1.0])

    def test_format_reward_full_match(self):
        """Test format_reward when there's a full match."""
        answers = ["Name: John, Age: 30"]
        reference_format_regex = r"Name: (.*), Age: (\d+)"
        rewards = format_reward(answers, reference_format_regex)
        self.assertEqual(rewards, [0.2])  # 2 groups * 0.1

    def test_format_reward_no_match(self):
        """Test format_reward when there's no match."""
        answers = ["John is 30 years old"]
        reference_format_regex = r"Name: (.*), Age: (\d+)"
        rewards = format_reward(answers, reference_format_regex)
        self.assertEqual(rewards, [0.0])

    def test_format_reward_multiple_answers(self):
        """Test format_reward with multiple answers."""
        answers = ["Name: John Age: 30", "John is 30 years old", "Name: Alice, Age: 25"]
        reference_format_regex = r"Name: (.*), Age: (\d+)"
        rewards = format_reward(answers, reference_format_regex)
        self.assertEqual(rewards, [0.0, 0.0, 0.2])  # 2 groups * 0.1

    def test_format_reward_custom_per_group_reward(self):
        """Test format_reward with custom per_group_reward."""
        answers = ["Name: John, Age: 30"]
        reference_format_regex = r"Name: (.*), Age: (\d+)"
        per_group_reward = 0.5
        rewards = format_reward(answers, reference_format_regex, per_group_reward)
        self.assertEqual(rewards, [1.0])  # 2 groups * 0.5

    def test_format_reward_different_group_counts(self):
        """Test format_reward with different group counts."""
        answers = [
            "Product: Apple, Price: $1.99, Code: A123",
            "Product: Orange, Price: $0.99",
        ]
        reference_format_regex = (
            r"Product: (.*), Price: \$([\d.]+)(?:, Code: ([A-Z\d]+))?"
        )
        rewards = format_reward(answers, reference_format_regex)
        assert np.allclose(rewards, [0.3, 0.2])


if __name__ == "__main__":
    unittest.main()
