import os
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

# Data Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "Data" / "Resume_long1.csv"

# Data Split Config
TRAIN_SIZE = 20
TEST_SIZE = 30

# LLM Config
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Fallback or Alternative

def get_llm_config():
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
