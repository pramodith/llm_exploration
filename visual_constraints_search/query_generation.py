"""
Generates negation queries using an LLM.
"""
from typing import List
from tqdm import tqdm
import litellm
import random
from config import NEGATION_PROMPT, SEED, QUERY_REFINEMENT_USER_PROMPT
import time

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
    n_queries: int, keywords: set, model: str = "gpt-4o"
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
    num_queries_generated = 0
    with tqdm(total=n_queries, desc="Generating queries") as pbar:
        while num_queries_generated < n_queries:
            # Randomly select 2 keywords from the set
            sampled_keywords = random.sample(keywords, 2)
            pos_keyword = sampled_keywords[0]
            neg_keyword = sampled_keywords[1]
            query = f"Images of {pos_keyword}, without any {neg_keyword}."
            refinement_prompt = QUERY_REFINEMENT_USER_PROMPT.format(query=query)
            refined_query = litellm.completion(
                model=model,
                messages=[
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0.2,
            )["choices"][0]["message"]["content"].strip()
            if refined_query.lower() != "na":
                queries.append(refined_query)
                num_queries_generated += 1
                pbar.update(1)
            if num_queries_generated % 10 == 0:
                time.sleep(5)
    
    return queries