"""
Statistical Analysis for Multi-Run Experiments

Provides functions for aggregating results across multiple runs,
computing confidence intervals, and performing significance tests.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json

try:
    import numpy as np
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .metrics import EvaluationResult


@dataclass
class AggregatedMetrics:
    """Statistical metrics aggregated across multiple runs."""
    mean: float
    std_dev: float
    ci_lower: float
    ci_upper: float
    runs: List[float]

    def to_dict(self):
        return asdict(self)


@dataclass
class AggregatedResult:
    """Aggregated evaluation results across multiple runs."""
    name: str
    field_accuracies: Dict[str, AggregatedMetrics]
    overall_accuracy: AggregatedMetrics
    num_runs: int
    seeds: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        result = {
            'name': self.name,
            'field_accuracies': {
                k: v.to_dict() for k, v in self.field_accuracies.items()
            },
            'overall_accuracy': self.overall_accuracy.to_dict(),
            'num_runs': self.num_runs,
            'seeds': self.seeds,
            'metadata': self.metadata
        }
        return result


def _check_scipy():
    """Check if scipy is available."""
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for statistical analysis. "
            "Install with: pip install scipy>=1.11.0"
        )


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval using t-distribution.

    Args:
        values: List of values
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    _check_scipy()

    if len(values) < 2:
        mean_val = values[0] if values else 0.0
        return (mean_val, mean_val)

    values = np.array(values)
    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of the mean

    # t-distribution critical value
    dof = len(values) - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, dof)

    margin = t_crit * sem
    return (mean - margin, mean + margin)


def aggregate_metrics(values: List[float], confidence: float = 0.95) -> AggregatedMetrics:
    """
    Aggregate a list of metric values into statistical summary.

    Args:
        values: List of metric values
        confidence: Confidence level for interval

    Returns:
        AggregatedMetrics object
    """
    _check_scipy()

    if not values:
        return AggregatedMetrics(
            mean=0.0,
            std_dev=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            runs=[]
        )

    values = np.array(values)
    mean = float(np.mean(values))
    std_dev = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    ci_lower, ci_upper = compute_confidence_interval(values, confidence)

    return AggregatedMetrics(
        mean=mean,
        std_dev=std_dev,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        runs=values.tolist()
    )


def aggregate_runs(
    runs: List[EvaluationResult],
    name: str = None,
    confidence: float = 0.95,
    seeds: List[int] = None
) -> AggregatedResult:
    """
    Aggregate multiple evaluation runs into statistical summary.

    Args:
        runs: List of EvaluationResult objects
        name: Name for the aggregated result
        confidence: Confidence level for intervals
        seeds: List of seeds used (for metadata)

    Returns:
        AggregatedResult object
    """
    _check_scipy()

    if not runs:
        raise ValueError("No runs provided for aggregation")

    name = name or runs[0].name
    num_runs = len(runs)

    # Aggregate overall accuracy
    overall_accuracies = [run.overall_accuracy for run in runs]
    overall_agg = aggregate_metrics(overall_accuracies, confidence)

    # Aggregate field accuracies
    field_names = runs[0].field_accuracies.keys()
    field_aggregates = {}

    for field_name in field_names:
        field_values = [run.field_accuracies.get(field_name, 0.0) for run in runs]
        field_aggregates[field_name] = aggregate_metrics(field_values, confidence)

    return AggregatedResult(
        name=name,
        field_accuracies=field_aggregates,
        overall_accuracy=overall_agg,
        num_runs=num_runs,
        seeds=seeds or list(range(num_runs))
    )


def compute_significance(
    baseline_runs: List[float],
    optimized_runs: List[float],
    test_type: str = 'ttest'
) -> Dict[str, Any]:
    """
    Perform statistical significance test comparing baseline and optimized results.

    Args:
        baseline_runs: List of baseline accuracy values
        optimized_runs: List of optimized accuracy values
        test_type: Type of test ('ttest' or 'mann_whitney')

    Returns:
        Dict with test results including p-value and interpretation
    """
    _check_scipy()

    if len(baseline_runs) != len(optimized_runs):
        raise ValueError("Baseline and optimized runs must have same length")

    if len(baseline_runs) < 2:
        return {
            'test_type': test_type,
            'p_value': None,
            'statistic': None,
            'significant': None,
            'message': 'Insufficient samples for significance testing (need at least 2)'
        }

    baseline_runs = np.array(baseline_runs)
    optimized_runs = np.array(optimized_runs)

    # Perform test
    if test_type == 'ttest':
        # Paired t-test (same samples, different methods)
        statistic, p_value = stats.ttest_rel(optimized_runs, baseline_runs)
        test_name = "Paired t-test"
    elif test_type == 'mann_whitney':
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(optimized_runs, baseline_runs, alternative='greater')
        test_name = "Mann-Whitney U test"
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    # Interpretation
    alpha = 0.05
    significant = p_value < alpha

    if significant:
        if p_value < 0.001:
            stars = "***"
            interpretation = "highly significant"
        elif p_value < 0.01:
            stars = "**"
            interpretation = "very significant"
        else:
            stars = "*"
            interpretation = "significant"
    else:
        stars = "ns"
        interpretation = "not significant"

    return {
        'test_type': test_name,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': significant,
        'significance_level': stars,
        'interpretation': interpretation,
        'alpha': alpha
    }


def compute_effect_size(baseline_runs: List[float], optimized_runs: List[float]) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        baseline_runs: List of baseline accuracy values
        optimized_runs: List of optimized accuracy values

    Returns:
        Cohen's d effect size
    """
    _check_scipy()

    baseline_runs = np.array(baseline_runs)
    optimized_runs = np.array(optimized_runs)

    mean_diff = np.mean(optimized_runs) - np.mean(baseline_runs)

    # Pooled standard deviation
    n1, n2 = len(baseline_runs), len(optimized_runs)
    var1, var2 = np.var(baseline_runs, ddof=1), np.var(optimized_runs, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    cohens_d = mean_diff / pooled_std
    return float(cohens_d)


def save_aggregated_results(
    aggregated_result: AggregatedResult,
    filepath: Path,
    indent: int = 2
) -> None:
    """
    Save aggregated results to JSON file.

    Args:
        aggregated_result: AggregatedResult object
        filepath: Output file path
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(aggregated_result.to_dict(), f, indent=indent, ensure_ascii=False)


def load_trial_results(results_dir: Path, num_trials: int) -> Dict[str, List[EvaluationResult]]:
    """
    Load results from multiple trial directories.

    Args:
        results_dir: Base results directory containing trial_* subdirectories
        num_trials: Number of trials to load

    Returns:
        Dict with 'baseline' and 'optimized' lists of EvaluationResult objects
    """
    baseline_runs = []
    optimized_runs = []

    for trial_idx in range(num_trials):
        trial_dir = results_dir / f"trial_{trial_idx}"

        # Load baseline results
        baseline_path = trial_dir / "baseline_results.json"
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                data = json.load(f)
                baseline_runs.append(EvaluationResult(**data))

        # Load optimized results
        optimized_path = trial_dir / "dspy_results.json"
        if optimized_path.exists():
            with open(optimized_path, 'r') as f:
                data = json.load(f)
                optimized_runs.append(EvaluationResult(**data))

    return {
        'baseline': baseline_runs,
        'optimized': optimized_runs
    }
