"""
Company Legal Risk Experiment Configuration
"""

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
DECISION_LABELS = ["YES", "NO"]

# Large replay split settings for train/test experiments
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3
SPLIT_SEED = 42

NUM_RUNS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1011]
CONFIDENCE_LEVEL = 0.95
STATISTICAL_TEST = "ttest"

USE_COPRO = True
BOOTSTRAP_CONFIG = {
    "max_bootstrapped_demos": 8,
    "max_labeled_demos": 8,
    "max_rounds": 3,
    "max_errors": 5,
}
COPRO_CONFIG = {
    "breadth": 4,
    "depth": 2,
    "init_temperature": 0.3,
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
