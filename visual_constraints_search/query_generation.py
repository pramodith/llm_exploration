"""
Generates negation queries using an LLM.
"""
from typing import List
import litellm
import random
from config import NEGATION_PROMPT, SEED

def generate_negation_queries(
    n_queries: int, keywords: set, model: str = "gpt-4o", max_num_keywords: int = 5,
) -> List[str]:
    """
    Use an LLM to generate negation queries.
    Args:
        n_queries (int): Number of queries to generate.
        keywords (set): Set of keywords to use for query generation.
        model (str): LLM model to use for generation.
        max_num_keywords (int): Maximum number of keywords to include in each query.
    Returns:
        List[str]: List of generated negation queries.
    """
    queries = []
    keywords = list(keywords)
    for _ in range(n_queries):
        # randomly sample keywords
        n = min(max_num_keywords, len(keywords))
        sampled_keywords = random.sample(keywords, n)
        prompt = NEGATION_PROMPT.format(keywords=", ".join(sampled_keywords))
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            seed=SEED,
        )
        # Parse numbered list from LLM response
        query = response["choices"][0]["message"]["content"]
        queries.append(query.strip())
    return queries

