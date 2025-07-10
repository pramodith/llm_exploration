"""
Generates negation queries using an LLM.
"""
from typing import List
import openai
from .config import NEGATION_PROMPT, NEGATION_OBJECTS, N_QUERIES, SEED
import random

def generate_negation_queries(n: int = N_QUERIES, model: str = "gpt-4o") -> List[str]:
    """
    Use an LLM to generate negation queries.
    """
    prompt = NEGATION_PROMPT.format(n=n)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        seed=SEED,
    )
    # Parse numbered list from LLM response
    lines = response["choices"][0]["message"]["content"].split("\n")
    queries = [line.split(". ", 1)[-1].strip() for line in lines if ". " in line]
    return queries[:n]
