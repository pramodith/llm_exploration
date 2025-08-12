"""
Utility functions for LLM configuration and setup.
"""
import os
from enum import Enum

class SearchStrategy(Enum):
    """
    Enum for different search strategies.
    """
    NAIVE = "naive"
    

def configure_litellm(api_key_env: str = "LITELLM_API_KEY"):
    """
    Set up the environment for litellm to use the correct API key.
    """
    if api_key_env in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ[api_key_env]
