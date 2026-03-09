"""
LLM Provider Configuration

Supports:
- Azure OpenAI deployment-style endpoints
- OpenAI-compatible endpoints (including Azure AI Foundry /openai/v1)
- Google Gemini (via LiteLLM-compatible dspy.LM route)
"""

import os
import dspy
from dotenv import load_dotenv

load_dotenv(override=True)


def _temperature():
    return float(os.getenv("LLM_TEMPERATURE", "0.1"))


def get_llm_config():
    """
    Get LLM configuration based on available environment variables.

    Priority:
    1) AZURE_OPENAI_* (deployment or openai-compatible endpoint)
    2) GEMINI_API_KEY

    Returns:
        dict: provider config

    Raises:
        ValueError: If no valid API keys found
    """
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if azure_key and azure_endpoint and azure_deployment:
        endpoint_norm = azure_endpoint.strip()
        # OpenAI-compatible endpoints (e.g., .../openai/v1/) should use raw model name.
        openai_compatible = endpoint_norm.rstrip("/").endswith("/openai/v1")
        return {
            "provider": "azure_openai_compatible" if openai_compatible else "azure",
            "api_key": azure_key,
            "endpoint": endpoint_norm,
            "deployment": azure_deployment,
            "api_version": azure_api_version,
            "temperature": _temperature(),
        }

    if gemini_key:
        return {
            "provider": "gemini",
            "api_key": gemini_key,
            "model": "gemini-1.5-flash",
            "temperature": _temperature(),
        }

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

    provider = config["provider"]

    if provider == "azure":
        # Azure OpenAI deployment route (classic).
        model_name = "azure/" + config["deployment"]
        lm = dspy.LM(
            model_name,
            api_key=config["api_key"],
            api_base=config["endpoint"],
            api_version=config["api_version"],
            temperature=config["temperature"],
        )
    elif provider == "azure_openai_compatible":
        # OpenAI-compatible route (e.g., Azure AI Foundry .../openai/v1/)
        lm = dspy.LM(
            model="openai/" + config["deployment"],
            api_key=config["api_key"],
            api_base=config["endpoint"],
            temperature=config["temperature"],
        )
    else:  # gemini
        # Use LiteLLM-compatible model routing through dspy.LM for DSPy compatibility.
        lm = dspy.LM(
            model="gemini/" + config["model"],
            api_key=config["api_key"],
            temperature=config["temperature"],
        )

    dspy.settings.configure(lm=lm)
    return lm

