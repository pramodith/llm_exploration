from typing import List
import re

def correct_answer_reward(answers: List[str], reference_answer: List[str]):
    """
    Calculate the correct answer reward.

    Args:
        answers (List[str]): The answers to calculate the reward for.
        reference_answer (List[str]): The reference answer.

    Returns:
        List[float]: The correct answer reward.
    """
    return [1.0 if answer == reference_answer else 0.0 for answer in answers]

def format_reward(answers: List[str], reference_format_regex: str, per_group_reward: float = 0.1):
    """
    Calculate the format reward. Gives a reward of `per_group_reward` for each matched group.

    Args:
        answers (List[str]): The answers to calculate the reward for.
        reference_format_regex (str): The reference format regex.
        per_group_reward (float): The reward per matched group.

    Returns:
        List[float]: The format reward.
    """
    matches = [re.match(reference_format_regex, answer) for answer in answers]
    return [len(match.groups()) * per_group_reward if match else 0.0 for match in matches]
    