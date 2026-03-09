"""
Aggregate Multi-Run Results

Aggregates results from multiple trial runs, computes statistics,
and performs significance testing.
"""

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
    load_trial_results
)

from . import config


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def aggregate_trial_results(num_trials=None, results_dir=None):
    """
    Aggregate results from multiple trials.

    Args:
        num_trials: Number of trials to aggregate. Default: config.NUM_RUNS
        results_dir: Base results directory. Default: config.RESULTS_DIR

    Returns:
        Dict with aggregated results and statistical analysis
    """
    num_trials = num_trials or config.NUM_RUNS
    results_dir = results_dir or config.RESULTS_DIR

    print("=" * 60)
    print("AGGREGATING MULTI-RUN RESULTS")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Number of trials: {num_trials}")

    # Load trial results
    print("\n[1/4] Loading trial results...")
    trial_data = load_trial_results(results_dir, num_trials)

    baseline_runs = trial_data['baseline']
    optimized_runs = trial_data['optimized']

    if not baseline_runs or not optimized_runs:
        print("Error: Could not find trial results")
        return None

    print(f"      Loaded {len(baseline_runs)} baseline runs")
    print(f"      Loaded {len(optimized_runs)} optimized runs")

    # Aggregate results
    print("[2/4] Computing aggregated statistics...")
    seeds = config.RANDOM_SEEDS[:num_trials]

    baseline_agg = aggregate_runs(
        baseline_runs,
        name="Baseline",
        confidence=config.CONFIDENCE_LEVEL,
        seeds=seeds
    )

    optimized_agg = aggregate_runs(
        optimized_runs,
        name="DSPy Optimized",
        confidence=config.CONFIDENCE_LEVEL,
        seeds=seeds
    )

    print(f"      Baseline accuracy: {baseline_agg.overall_accuracy.mean:.2%} ± {baseline_agg.overall_accuracy.std_dev:.2%}")
    print(f"      Optimized accuracy: {optimized_agg.overall_accuracy.mean:.2%} ± {optimized_agg.overall_accuracy.std_dev:.2%}")

    # Significance testing
    print("[3/4] Performing significance testing...")
    baseline_values = [r.overall_accuracy for r in baseline_runs]
    optimized_values = [r.overall_accuracy for r in optimized_runs]

    significance = compute_significance(
        baseline_values,
        optimized_values,
        test_type=config.STATISTICAL_TEST
    )

    effect_size = compute_effect_size(baseline_values, optimized_values)

    print(f"      Test: {significance['test_type']}")
    print(f"      p-value: {significance['p_value']:.4f}")
    print(f"      Result: {significance['interpretation']} ({significance['significance_level']})")
    print(f"      Effect size (Cohen's d): {effect_size:.3f}")

    # Save aggregated results
    print("[4/4] Saving aggregated results...")
    aggregated_dir = results_dir / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    save_aggregated_results(baseline_agg, aggregated_dir / "baseline_aggregated.json")
    save_aggregated_results(optimized_agg, aggregated_dir / "dspy_aggregated.json")

    # Save comparison with significance
    comparison = {
        'baseline': baseline_agg.to_dict(),
        'optimized': optimized_agg.to_dict(),
        'statistical_test': significance,
        'effect_size': effect_size,
        'improvement': {
            'mean': optimized_agg.overall_accuracy.mean - baseline_agg.overall_accuracy.mean,
            'significant': significance['significant']
        },
        'field_comparison': {},
        'skills_prf': {}
    }

    baseline_skill_micro_p = []
    baseline_skill_micro_r = []
    baseline_skill_micro_f1 = []
    optimized_skill_micro_p = []
    optimized_skill_micro_r = []
    optimized_skill_micro_f1 = []

    for run in baseline_runs:
        skills_meta = (run.metadata or {}).get('skills_metrics', {})
        if 'micro_precision' in skills_meta:
            baseline_skill_micro_p.append(skills_meta['micro_precision'])
            baseline_skill_micro_r.append(skills_meta['micro_recall'])
            baseline_skill_micro_f1.append(skills_meta['micro_f1'])

    for run in optimized_runs:
        skills_meta = (run.metadata or {}).get('skills_metrics', {})
        if 'micro_precision' in skills_meta:
            optimized_skill_micro_p.append(skills_meta['micro_precision'])
            optimized_skill_micro_r.append(skills_meta['micro_recall'])
            optimized_skill_micro_f1.append(skills_meta['micro_f1'])

    if baseline_skill_micro_f1 and optimized_skill_micro_f1:
        comparison['skills_prf'] = {
            'baseline_micro_precision_mean': _mean(baseline_skill_micro_p),
            'baseline_micro_recall_mean': _mean(baseline_skill_micro_r),
            'baseline_micro_f1_mean': _mean(baseline_skill_micro_f1),
            'optimized_micro_precision_mean': _mean(optimized_skill_micro_p),
            'optimized_micro_recall_mean': _mean(optimized_skill_micro_r),
            'optimized_micro_f1_mean': _mean(optimized_skill_micro_f1),
            'micro_f1_improvement': _mean(optimized_skill_micro_f1) - _mean(baseline_skill_micro_f1),
        }

    # Field-level comparison
    for field_name in config.EXTRACTION_FIELDS:
        baseline_field = baseline_agg.field_accuracies.get(field_name)
        optimized_field = optimized_agg.field_accuracies.get(field_name)

        if baseline_field and optimized_field:
            comparison['field_comparison'][field_name] = {
                'baseline_mean': baseline_field.mean,
                'baseline_ci': [baseline_field.ci_lower, baseline_field.ci_upper],
                'optimized_mean': optimized_field.mean,
                'optimized_ci': [optimized_field.ci_lower, optimized_field.ci_upper],
                'improvement': optimized_field.mean - baseline_field.mean
            }

    import json
    with open(aggregated_dir / "comparison_aggregated.json", 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nAggregated results saved to {aggregated_dir}/")

    # Print summary
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nBaseline Overall Accuracy:")
    print(f"  Mean: {baseline_agg.overall_accuracy.mean:.2%}")
    print(f"  Std Dev: {baseline_agg.overall_accuracy.std_dev:.2%}")
    print(f"  95% CI: [{baseline_agg.overall_accuracy.ci_lower:.2%}, {baseline_agg.overall_accuracy.ci_upper:.2%}]")

    print(f"\nDSPy Optimized Overall Accuracy:")
    print(f"  Mean: {optimized_agg.overall_accuracy.mean:.2%}")
    print(f"  Std Dev: {optimized_agg.overall_accuracy.std_dev:.2%}")
    print(f"  95% CI: [{optimized_agg.overall_accuracy.ci_lower:.2%}, {optimized_agg.overall_accuracy.ci_upper:.2%}]")

    print(f"\nImprovement: {comparison['improvement']['mean']:.2%}")
    print(f"Statistically Significant: {significance['significant']} ({significance['significance_level']})")
    print(f"Effect Size (Cohen's d): {effect_size:.3f}")

    print("\nField-wise Results (Mean ± Std Dev):")
    print("-" * 60)
    for field_name in config.EXTRACTION_FIELDS:
        baseline_field = baseline_agg.field_accuracies.get(field_name)
        optimized_field = optimized_agg.field_accuracies.get(field_name)
        if baseline_field and optimized_field:
            print(f"  {field_name:20}")
            print(f"    Baseline:  {baseline_field.mean:.2%} ± {baseline_field.std_dev:.2%}")
            print(f"    Optimized: {optimized_field.mean:.2%} ± {optimized_field.std_dev:.2%}")
            print(f"    Improvement: {optimized_field.mean - baseline_field.mean:+.2%}")

    if comparison['skills_prf']:
        print("\nSkills Micro PR/F1 (Mean across runs):")
        print("-" * 60)
        print(f"  Baseline Precision:  {comparison['skills_prf']['baseline_micro_precision_mean']:.2%}")
        print(f"  Baseline Recall:     {comparison['skills_prf']['baseline_micro_recall_mean']:.2%}")
        print(f"  Baseline F1:         {comparison['skills_prf']['baseline_micro_f1_mean']:.2%}")
        print(f"  Optimized Precision: {comparison['skills_prf']['optimized_micro_precision_mean']:.2%}")
        print(f"  Optimized Recall:    {comparison['skills_prf']['optimized_micro_recall_mean']:.2%}")
        print(f"  Optimized F1:        {comparison['skills_prf']['optimized_micro_f1_mean']:.2%}")
        print(f"  F1 Improvement:      {comparison['skills_prf']['micro_f1_improvement']:+.2%}")

    return {
        'baseline_aggregated': baseline_agg,
        'optimized_aggregated': optimized_agg,
        'comparison': comparison,
        'significance': significance,
        'effect_size': effect_size
    }


if __name__ == "__main__":
    # Check for command-line arguments
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else None
    aggregate_trial_results(num_trials=num_trials)
