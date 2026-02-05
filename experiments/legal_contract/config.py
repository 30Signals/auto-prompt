"""
Legal Contract Analysis Experiment Configuration
"""

from pathlib import Path

# Paths
EXPERIMENT_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"

# Dataset source
DATASET_NAME = "theatticusproject/cuad-qa"  # Hugging Face dataset name

# Focus on top clause types for tractability
# CUAD has 41 clause types; we'll focus on the most common/useful ones
CLAUSE_TYPES = [
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Governing Law",
    "Termination For Convenience",
    "Limitation Of Liability",
    "Indemnification",
    "Non-Compete",
    "Confidentiality"
]

# Data Split Config
TRAIN_SIZE = 40
TEST_SIZE = 40

# Fields to extract (single field per example: the clause text for the asked clause type)
EXTRACTION_FIELDS = ['clause_text']

# Statistical rigor settings
NUM_RUNS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1011]
CONFIDENCE_LEVEL = 0.95
STATISTICAL_TEST = 'ttest'

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
