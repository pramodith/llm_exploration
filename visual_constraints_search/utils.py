"""
Utility functions for LLM configuration and setup.
"""
import os

def configure_litellm(api_key_env: str = "LITELLM_API_KEY"):
    """
    Set up the environment for litellm to use the correct API key.
    """
    if api_key_env in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ[api_key_env]
