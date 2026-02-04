"""
Resume Extraction Experiment Configuration
"""

from pathlib import Path

# Paths
EXPERIMENT_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"

# Default data path (can be overridden)
DEFAULT_DATA_PATH = PROJECT_ROOT / "Data" / "Resume_long1.csv"

# Data Split Config
TRAIN_SIZE = 20
TEST_SIZE = 30

# Fields to extract
EXTRACTION_FIELDS = ['job_role', 'skills', 'education', 'experience_years']

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
