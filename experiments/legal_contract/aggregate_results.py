"""
Aggregate Multi-Run Results for Legal Contract Experiment.
"""

import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.evaluation.statistics import (
    aggregate_runs,
    compute_significance,
    compute_effect_size,
    save_aggregated_results,
    load_trial_results,
)

from . import config


def aggregate_trial_results(num_trials=None, results_dir=None):
    num_trials = num_trials or config.NUM_RUNS
    results_dir = results_dir or config.RESULTS_DIR

    print("=" * 60)
    print("AGGREGATING MULTI-RUN RESULTS")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Number of trials: {num_trials}")

    print("\n[1/4] Loading trial results...")
    trial_data = load_trial_results(results_dir, num_trials)
    baseline_runs = trial_data["baseline"]
    optimized_runs = trial_data["optimized"]

    if not baseline_runs or not optimized_runs:
        print("Error: Could not find trial results")
        return None

    print(f"      Loaded {len(baseline_runs)} baseline runs")
    print(f"      Loaded {len(optimized_runs)} optimized runs")

    print("[2/4] Computing aggregated statistics...")
    seeds = config.RANDOM_SEEDS[:num_trials]

    baseline_agg = aggregate_runs(
        baseline_runs,
        name="Baseline",
        confidence=config.CONFIDENCE_LEVEL,
        seeds=seeds,
    )
    optimized_agg = aggregate_runs(
        optimized_runs,
        name="DSPy Optimized",
        confidence=config.CONFIDENCE_LEVEL,
        seeds=seeds,
    )

    print(
        f"      Baseline accuracy: {baseline_agg.overall_accuracy.mean:.2%} +/- {baseline_agg.overall_accuracy.std_dev:.2%}"
    )
    print(
        f"      Optimized accuracy: {optimized_agg.overall_accuracy.mean:.2%} +/- {optimized_agg.overall_accuracy.std_dev:.2%}"
    )

    print("[3/4] Performing significance testing...")
    baseline_values = [r.overall_accuracy for r in baseline_runs]
    optimized_values = [r.overall_accuracy for r in optimized_runs]

    significance = compute_significance(
        baseline_values,
        optimized_values,
        test_type=config.STATISTICAL_TEST,
    )
    effect_size = compute_effect_size(baseline_values, optimized_values)

    print(f"      Test: {significance['test_type']}")
    print(f"      p-value: {significance['p_value']:.4f}")
    print(
        f"      Result: {significance['interpretation']} ({significance['significance_level']})"
    )
    print(f"      Effect size (Cohen's d): {effect_size:.3f}")

    print("[4/4] Saving aggregated results...")
    aggregated_dir = results_dir / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    save_aggregated_results(baseline_agg, aggregated_dir / "baseline_aggregated.json")
    save_aggregated_results(optimized_agg, aggregated_dir / "dspy_aggregated.json")

    comparison = {
        "baseline": baseline_agg.to_dict(),
        "optimized": optimized_agg.to_dict(),
        "statistical_test": significance,
        "effect_size": effect_size,
        "improvement": {
            "mean": optimized_agg.overall_accuracy.mean
            - baseline_agg.overall_accuracy.mean,
            "significant": significance["significant"],
        },
        "field_comparison": {},
    }

    for field_name in config.EXTRACTION_FIELDS:
        baseline_field = baseline_agg.field_accuracies.get(field_name)
        optimized_field = optimized_agg.field_accuracies.get(field_name)
        if baseline_field and optimized_field:
            comparison["field_comparison"][field_name] = {
                "baseline_mean": baseline_field.mean,
                "baseline_ci": [baseline_field.ci_lower, baseline_field.ci_upper],
                "optimized_mean": optimized_field.mean,
                "optimized_ci": [optimized_field.ci_lower, optimized_field.ci_upper],
                "improvement": optimized_field.mean - baseline_field.mean,
            }

    with open(aggregated_dir / "comparison_aggregated.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nAggregated results saved to {aggregated_dir}/")

    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS SUMMARY")
    print("=" * 60)
    print("\nBaseline Overall Accuracy:")
    print(f"  Mean: {baseline_agg.overall_accuracy.mean:.2%}")
    print(f"  Std Dev: {baseline_agg.overall_accuracy.std_dev:.2%}")
    print(
        f"  95% CI: [{baseline_agg.overall_accuracy.ci_lower:.2%}, {baseline_agg.overall_accuracy.ci_upper:.2%}]"
    )

    print("\nDSPy Optimized Overall Accuracy:")
    print(f"  Mean: {optimized_agg.overall_accuracy.mean:.2%}")
    print(f"  Std Dev: {optimized_agg.overall_accuracy.std_dev:.2%}")
    print(
        f"  95% CI: [{optimized_agg.overall_accuracy.ci_lower:.2%}, {optimized_agg.overall_accuracy.ci_upper:.2%}]"
    )

    print(f"\nImprovement: {comparison['improvement']['mean']:.2%}")
    print(
        f"Statistically Significant: {significance['significant']} ({significance['significance_level']})"
    )
    print(f"Effect Size (Cohen's d): {effect_size:.3f}")

    print("\nField-wise Results (Mean +/- Std Dev):")
    print("-" * 60)
    for field_name in config.EXTRACTION_FIELDS:
        baseline_field = baseline_agg.field_accuracies.get(field_name)
        optimized_field = optimized_agg.field_accuracies.get(field_name)
        if baseline_field and optimized_field:
            print(f"  {field_name:20}")
            print(
                f"    Baseline:  {baseline_field.mean:.2%} +/- {baseline_field.std_dev:.2%}"
            )
            print(
                f"    Optimized: {optimized_field.mean:.2%} +/- {optimized_field.std_dev:.2%}"
            )
            print(
                f"    Improvement: {optimized_field.mean - baseline_field.mean:+.2%}"
            )

    return {
        "baseline_aggregated": baseline_agg,
        "optimized_aggregated": optimized_agg,
        "comparison": comparison,
        "significance": significance,
        "effect_size": effect_size,
    }


if __name__ == "__main__":
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else None
    aggregate_trial_results(num_trials=num_trials)
