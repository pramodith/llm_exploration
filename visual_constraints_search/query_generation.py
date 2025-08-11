"""
Generates negation queries using an LLM.
"""
from typing import List
from tqdm import tqdm
import litellm
import random
from config import NEGATION_PROMPT, SEED

def llm_generate_negation_queries(
    n_queries: int, keywords: set, model: str = "gpt-4o", max_num_keywords: int = 2,
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
    for _ in tqdm(range(n_queries), desc="Generating queries"):
        # randomly sample keywords
        n = random.randint(2, min(max_num_keywords, len(keywords)))
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

def generate_negation_queries(
    n_queries: int, keywords: set
) -> List[str]:
    """
    Generate negation queries using an LLM.
    Args:
        n_queries (int): Number of queries to generate.
        keywords (set): Set of keywords to use for query generation.
    Returns:
        List[str]: List of generated negation queries.
    """
    queries = []
    keywords = list(keywords)
    for _ in range(n_queries):
        # Randomly select 2 keywords from the set
        sampled_keywords = random.sample(keywords, 2)
        pos_keyword = sampled_keywords[0]
        neg_keyword = sampled_keywords[1]
        query = f"Images of {pos_keyword}, without any {neg_keyword}."
        queries.append(query)

    return queries