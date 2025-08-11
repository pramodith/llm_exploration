"""
Configuration and constants for the multimodal negation experiment.
"""
from typing import List

DATASET_NAME: str = "nlphuji/flickr30k"  # Options: 'coco', 'flickr30k', etc.
MAX_IMAGES: int = 1000
EMBEDDING_MODEL: str = "jinaai/jina-clip-v2"  # Options: 'clip', 'nomic', 'jina', 'cohere', etc.
QUERY_MODEL: str = "gpt-4.1-mini"  # LLM for query generation
JUDGE_MODEL: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  # LLM for judging
N_QUERIES: int = 100
TOP_K: int = 3
SEED: int = 42

# Paths
DATA_DIR: str = "data/"
EMBEDDINGS_PATH: str = "embeddings.npy"
QUERIES_PATH: str = "queries.txt"
IMAGE_PATHS_FILE: str = "image_paths.json"
REPORT_PATH: str = "report.md"
KEYWORDS_PATH: str = "keywords.txt"

# Negation prompt template for LLM
NEGATION_PROMPT: str = (
    "You are given a list of keywords from the captions of an image dataset. You should generate a "
    "query that searching for an image with at least one of these keywords followed by a negation constraint."
    "##Example:\n"
    "Keywords: tables, dogs"
    "Query: Find images of tables, but not dogs\n"
    "##Example:\n"
    "Keywords: cars, trees, buildings\n"
    "Query: Images of a car surrounded by buildings with no trees.\n"
    "##Your turn:\n"
    "Keywords: {keywords}\n"
    "Query: "   
)

QUERY_REFINEMENT_PROMPT: str = (
    "You are given a visual search query, if the query isn't realistic or doesn't make sense edit the query to make it more realistic.\n"
    "You are only allowed to edit the query by deleting words.\n"
    "Make sure you the query retains a negation constraint.\n, if the query can't be refined return NA.\n"
    "Do not return any other text, just the refined query.\n, if the query is already realistic, return the query as is.\n"
    "Query: {query}\n"
    "Refined Query: "
)