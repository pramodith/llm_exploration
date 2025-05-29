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
    #matches = [re.match(r"(?is).*answer[\D]*(\d+)", answer) for answer in answers]
    
    matches = []
    for answer in answers:
        match = re.match(r"(<think>)[\s\S]*?(</think>)[\s\S]*?(<answer>)[\s\D]*(\d+)[\s\D]*(</answer>)", answer)
        if match:
            matches.append(match.group(4))
        else:
            match = re.match(r"(?is).*answer[\D]*(\d+)", answer)
            if match:
                matches.append(match.group(1))
            else:
                matches.append(None)

    return [1.0 if answer is not None and float(answer) == float(ref_answer) else 0.25 if answer is not None else 0.0 
        for answer, ref_answer in zip(matches, reference_answer)
    ]

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
    return [
            len([match.group(i) for i in range(1, len(match.groups()) + 1) if match.group(i)]) * per_group_reward if match else 0.0 
            for match in matches
        ]
    