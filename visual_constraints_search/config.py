"""
Configuration and constants for the multimodal negation experiment.
"""
from typing import List

DATASET_NAME: str = "nlphuji/flickr30k"  # Options: 'coco', 'flickr30k', etc.
MAX_IMAGES: int = 1000
EMBEDDING_MODEL: str = "jinaai/jina-clip-v2"  # Options: 'clip', 'nomic', 'jina', 'cohere', etc.
QUERY_MODEL: str = "gpt-4.1-mini"  # LLM for query generation
JUDGE_MODEL: str = "Qwen/Qwen2.5-VL-3B-Instruct"  # LLM for judging
N_QUERIES: int = 150
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

QUERY_REFINEMENT_USER_PROMPT: str = (
    "The aim of the project is to measure how well a visual search system can handle negation constraints. "
    "For example, users might want images that don't containt certain objects, colors, shapes, people of a certain demographic, etc. "
    "You will help with deciding if a generated visual query would be meaningful to test the visual search system. "
    "It doesn't matter how realistic the query is, since all we want to do is ensure that the visual search system can handle negation constraints. "
    "You are given a visual search query with a negation constraint. "
    "Update the negation constraint to contain only one visually representable object or concept. "
    "If the query already contains only one negation object or concept, return it unchanged. "
    "If the query is not suitable for visual search, return 'NA'. "
    "Example:\n"
    "Query: Find images of cars, but not trees without any leaves.\n"
    "Updated Query: Find images of cars, without any trees.\n"
    "Query: Find images of tables, but not dogs with a bone in their mouth.\n"
    "Updated Query: Find images of tables, without any dogs.\n"
    "Query: Find images of people partying without any babies crying.\n"
    "Updated Query: Find images of people partying, without any babies.\n"
    "Query: Find images of a football game without a crowd.\n"
    "Updated Query: Find images of a football game without acrowd.\n"
    "Query: {query}\n"
)