"""
Medical NER Experiment Configuration
"""

from pathlib import Path

# Paths
EXPERIMENT_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"

# Dataset source
DATASET_NAME = "ncbi/ncbi_disease"  # Hugging Face dataset name

# Data Split Config
TRAIN_SIZE = 120
TEST_SIZE = 120
VALIDATION_SIZE = 24

# Optimization strategy
# Medical NER is currently more stable with BootstrapFewShot-only.
USE_COPRO = False
OPTIMIZATION_CANDIDATES = [
    "bootstrap_exact",
    "bootstrap_recall",
]
VALIDATION_RECALL_TOLERANCE = 0.02
REBALANCE_OPT_TRAIN = True
OPT_POSITIVE_TARGET_RATIO = 0.7
REQUIRE_VALIDATION_BEAT_BASELINE = True

# Optimizer configs (passed to shared optimization helpers)
BOOTSTRAP_CONFIG = {
    "max_bootstrapped_demos": 8,
    "max_labeled_demos": 8,
    "max_rounds": 2,
    "max_errors": 3,
}

COPRO_CONFIG = {
    "breadth": 4,
    "depth": 2,
    "init_temperature": 0.1,
}

# Fields to extract
EXTRACTION_FIELDS = ['diseases']

# Statistical rigor settings
NUM_RUNS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1011]
CONFIDENCE_LEVEL = 0.95
STATISTICAL_TEST = 'ttest'

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
