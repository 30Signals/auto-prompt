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

# Weak fields to target with specialized per-clause optimization runs.
WEAK_CLAUSE_TYPES = [
    "Parties",
    "Effective Date",
    "Expiration Date",
    "Governing Law",
]

# Data Split Config
TRAIN_SIZE = 200
TEST_SIZE = 200
PER_CLAUSE_TRAIN_SIZE = 120
PER_CLAUSE_TEST_SIZE = 80

# Keep optimization cheap by default while prompts/data are still stabilizing.
USE_COPRO = False
BOOTSTRAP_CONFIG = {
    "max_bootstrapped_demos": 6,
    "max_labeled_demos": 6,
    "max_rounds": 2,
    "max_errors": 4,
}
COPRO_CONFIG = {
    "breadth": 3,
    "depth": 1,
    "init_temperature": 0.2,
}


# Metadata DSPy optimization config. Kept config-driven to match other experiments.
METADATA_USE_COPRO = False
METADATA_BOOTSTRAP_CONFIGS = {
    "smoke": {
        "max_bootstrapped_demos": 2,
        "max_labeled_demos": 2,
        "max_rounds": 1,
        "max_errors": 2,
    },
    "medium": {
        "max_bootstrapped_demos": 4,
        "max_labeled_demos": 4,
        "max_rounds": 1,
        "max_errors": 4,
    },
    "full": {
        "max_bootstrapped_demos": 8,
        "max_labeled_demos": 8,
        "max_rounds": 2,
        "max_errors": 6,
    },
}
METADATA_COPRO_CONFIGS = {
    "smoke": {
        "breadth": 2,
        "depth": 1,
        "init_temperature": 1.0,
    },
    "medium": {
        "breadth": 2,
        "depth": 1,
        "init_temperature": 1.0,
    },
    "full": {
        "breadth": 2,
        "depth": 1,
        "init_temperature": 1.0,
    },
}

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
