"""
LLM Provider Configuration

Supports Azure OpenAI and Google Gemini. Auto-selects based on available environment variables.
"""

import os
import dspy
from dotenv import load_dotenv

load_dotenv()

# Environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_llm_config():
    """
    Get LLM configuration based on available environment variables.

    Priority: Azure OpenAI > Google Gemini

    Returns:
        dict: Configuration with provider, api_key, and provider-specific fields

    Raises:
        ValueError: If no valid API keys found
    """
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        return {
            "provider": "azure",
            "api_key": AZURE_OPENAI_API_KEY,
            "endpoint": AZURE_OPENAI_ENDPOINT,
            "deployment": AZURE_OPENAI_DEPLOYMENT,
            "api_version": AZURE_OPENAI_API_VERSION
        }
    elif GEMINI_API_KEY:
        return {
            "provider": "gemini",
            "api_key": GEMINI_API_KEY,
            "model": "gemini-1.5-flash"
        }
    else:
        raise ValueError("No valid LLM API keys found (Azure or Gemini). Please check .env file.")


def setup_dspy_lm(config=None):
    """
    Configure DSPy with the appropriate LM based on config.

    Args:
        config: Optional config dict. If None, calls get_llm_config()

    Returns:
        The configured DSPy LM instance
    """
    if config is None:
        config = get_llm_config()

    if config["provider"] == "azure":
        model_name = "azure/" + config["deployment"]
        lm = dspy.LM(
            model_name,
            api_key=config["api_key"],
            api_base=config["endpoint"],
            api_version=config["api_version"],
        )
    else:  # gemini
        lm = dspy.Google(
            model=config["model"],
            api_key=config["api_key"]
        )

    dspy.settings.configure(lm=lm)
    return lm
