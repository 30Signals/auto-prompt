"""
Generic Evaluation Metrics

Reusable evaluation utilities for comparing model outputs across experiments.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    name: str
    results: List[Dict[str, Any]]
    field_accuracies: Dict[str, float]
    overall_accuracy: float
    total_samples: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def compute_field_accuracy(
    predictions: List[Dict],
    ground_truth: List[Dict],
    fields: List[str],
    comparators: Optional[Dict[str, Callable]] = None
) -> Dict[str, float]:
    """
    Compute per-field accuracy.

    Args:
        predictions: List of prediction dicts
        ground_truth: List of ground truth dicts
        fields: List of field names to evaluate
        comparators: Optional dict mapping field names to custom comparison functions.
                    Each function takes (pred, gt) and returns a score in [0, 1].
                    Default: exact string match (case-insensitive)

    Returns:
        Dict mapping field names to accuracy scores
    """
    if comparators is None:
        comparators = {}

    field_scores = {f: [] for f in fields}

    for pred, gt in zip(predictions, ground_truth):
        for field_name in fields:
            pred_val = pred.get(field_name, "")
            gt_val = gt.get(field_name, "")

            if field_name in comparators:
                score = comparators[field_name](pred_val, gt_val)
            else:
                # Default: case-insensitive exact match
                score = 1.0 if str(pred_val).strip().lower() == str(gt_val).strip().lower() else 0.0

            field_scores[field_name].append(score)

    return {
        field_name: sum(scores) / len(scores) if scores else 0.0
        for field_name, scores in field_scores.items()
    }


def compute_overall_accuracy(field_accuracies: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """
    Compute weighted overall accuracy from field accuracies.

    Args:
        field_accuracies: Dict mapping field names to accuracy scores
        weights: Optional dict mapping field names to weights. Default: equal weights.

    Returns:
        Weighted average accuracy
    """
    if not field_accuracies:
        return 0.0

    if weights is None:
        weights = {k: 1.0 for k in field_accuracies}

    total_weight = sum(weights.get(k, 1.0) for k in field_accuracies)
    weighted_sum = sum(
        field_accuracies[k] * weights.get(k, 1.0)
        for k in field_accuracies
    )

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def compare_results(
    baseline_results: EvaluationResult,
    optimized_results: EvaluationResult
) -> Dict[str, Any]:
    """
    Compare baseline and optimized results to identify improvements.

    Args:
        baseline_results: EvaluationResult from baseline model
        optimized_results: EvaluationResult from optimized model

    Returns:
        Comparison dict with improvements, degradations, and summary statistics
    """
    improvements = []
    degradations = []
    unchanged = []

    min_length = min(len(baseline_results.results), len(optimized_results.results))

    for i in range(min_length):
        baseline_score = baseline_results.results[i].get('overall_score', 0)
        optimized_score = optimized_results.results[i].get('overall_score', 0)
        diff = optimized_score - baseline_score

        sample_info = {
            'sample_id': i + 1,
            'baseline_score': baseline_score,
            'optimized_score': optimized_score,
            'difference': abs(diff)
        }

        if diff > 0.01:  # Small threshold to avoid floating point issues
            improvements.append(sample_info)
        elif diff < -0.01:
            degradations.append(sample_info)
        else:
            unchanged.append(sample_info)

    # Field-level comparison
    field_improvements = {}
    for field_name in baseline_results.field_accuracies:
        baseline_acc = baseline_results.field_accuracies.get(field_name, 0)
        optimized_acc = optimized_results.field_accuracies.get(field_name, 0)
        field_improvements[field_name] = {
            'baseline': baseline_acc,
            'optimized': optimized_acc,
            'improvement': optimized_acc - baseline_acc
        }

    return {
        'improvements': improvements,
        'degradations': degradations,
        'unchanged': unchanged,
        'field_comparison': field_improvements,
        'summary': {
            'total_samples': min_length,
            'total_improvements': len(improvements),
            'total_degradations': len(degradations),
            'total_unchanged': len(unchanged),
            'overall_improvement': optimized_results.overall_accuracy - baseline_results.overall_accuracy,
            'baseline_accuracy': baseline_results.overall_accuracy,
            'optimized_accuracy': optimized_results.overall_accuracy
        }
    }


def save_results_json(
    results: Dict[str, Any],
    filepath: Path,
    indent: int = 2
) -> None:
    """
    Save results to JSON file.

    Args:
        results: Results dict or EvaluationResult
        filepath: Output file path
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, EvaluationResult):
        results = results.to_dict()

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=indent, ensure_ascii=False, default=str)
