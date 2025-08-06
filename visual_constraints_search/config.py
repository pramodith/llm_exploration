"""
Configuration and constants for the multimodal negation experiment.
"""
from typing import List

DATASET_NAME: str = "nlphuji/flickr30k"  # Options: 'coco', 'flickr30k', etc.
MAX_IMAGES: int = 1000
EMBEDDING_MODEL: str = "google/siglip2-large-patch16-384"  # Options: 'clip', 'nomic', 'jina', 'cohere', etc.
QUERY_MODEL: str = "gpt-4.1-mini"  # LLM for query generation
JUDGE_MODEL: str = "gpt-4.1"  # LLM for judging
N_QUERIES: int = 100
TOP_K: int = 10
SEED: int = 42

# Paths
DATA_DIR: str = "data/"
EMBEDDINGS_PATH: str = "embeddings.npy"
QUERIES_PATH: str = "queries.txt"
IMAGE_PATHS_FILE: str = "image_paths.json"
REPORT_PATH: str = "report.md"

# Negation prompt template for LLM
NEGATION_PROMPT: str = (
    "Generate {n} diverse natural language queries that describe images NOT containing certain objects or scenes. "
    "Use negations (e.g., 'no dogs', 'without trees', 'not at night'). "
    "Return as a numbered list."
)

# List of objects/scenes to use for negation queries (can be dataset-specific)
NEGATION_OBJECTS: List[str] = [
    "dog", "cat", "tree", "car", "person", "water", "building", "food", "sky", "flower"
]
