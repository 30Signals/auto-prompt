"""
Aggregate Multi-Run Results for Medical NER Experiment.
"""

import json
import sys
from datetime import datetime
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


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def display_metadata_metrics(baseline_agg, optimized_agg, comparison, effect_size, metadata=None):
    """Display metadata and accuracy metrics in formatted output."""
    
    print("\n" + "=" * 70)
    print("METADATA + ACCURACY METRICS")
    print("=" * 70)
    
    print("\nMETADATA")
    print("-" * 70)
    if metadata:
        print(f"Experiment: {metadata.get('experiment', 'N/A')}")
        print(f"Dataset: {metadata.get('dataset', 'N/A')}")
        print(f"Training Size: {metadata.get('train_size', 'N/A')} | Test Size: {metadata.get('test_size', 'N/A')}")
        print(f"Number of Runs: {metadata.get('num_runs', 'N/A')}")
        print(f"Random Seeds: {metadata.get('random_seeds', 'N/A')}")
        print(f"Confidence Level: {metadata.get('confidence_level', 'N/A')}")
        print(f"Statistical Test: {metadata.get('statistical_test', 'N/A')}")
        print(f"Extraction Fields: {metadata.get('extraction_fields', 'N/A')}")
        print(f"Use COPRO: {metadata.get('use_copro', 'N/A')}")
        print(f"Aggregation Time: {metadata.get('aggregation_timestamp', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("BASELINE ACCURACIES")
    print("=" * 70)
    baseline_acc = baseline_agg.overall_accuracy
    print(f"Mean Accuracy: {baseline_acc.mean:.4f} ({baseline_acc.mean*100:.2f}%)")
    print(f"Std Dev: {baseline_acc.std_dev:.4f} ({baseline_acc.std_dev*100:.2f}%)")
    print(f"95% CI: [{baseline_acc.ci_lower:.4f}, {baseline_acc.ci_upper:.4f}]")
    print(f"        [{baseline_acc.ci_lower*100:.2f}%, {baseline_acc.ci_upper*100:.2f}%]")
    print(f"Per-run accuracies: {[f'{x*100:.2f}%' for x in baseline_acc.runs]}")
    
    print("\n" + "=" * 70)
    print("DSPY OPTIMIZED ACCURACIES")
    print("=" * 70)
    opt_acc = optimized_agg.overall_accuracy
    print(f"Mean Accuracy: {opt_acc.mean:.4f} ({opt_acc.mean*100:.2f}%)")
    print(f"Std Dev: {opt_acc.std_dev:.4f} ({opt_acc.std_dev*100:.2f}%)")
    print(f"95% CI: [{opt_acc.ci_lower:.4f}, {opt_acc.ci_upper:.4f}]")
    print(f"        [{opt_acc.ci_lower*100:.2f}%, {opt_acc.ci_upper*100:.2f}%]")
    print(f"Per-run accuracies: {[f'{x*100:.2f}%' for x in opt_acc.runs]}")
    
    print("\n" + "=" * 70)
    print("FIELD-WISE ACCURACIES (diseases)")
    print("=" * 70)
    baseline_diseases = baseline_agg.field_accuracies.get("diseases")
    optimized_diseases = optimized_agg.field_accuracies.get("diseases")
    if baseline_diseases and optimized_diseases:
        print(f"Baseline Mean: {baseline_diseases.mean*100:.2f}% +/- {baseline_diseases.std_dev*100:.2f}%")
        print(f"Optimized Mean: {optimized_diseases.mean*100:.2f}% +/- {optimized_diseases.std_dev*100:.2f}%")
        print(f"Improvement: {(optimized_diseases.mean - baseline_diseases.mean)*100:+.2f}%")
    
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    sig = comparison.get("statistical_test", {})
    print(f"Test Type: {sig.get('test_type', 'N/A')}")
    print(f"t-statistic: {sig.get('statistic', 'N/A')}")
    print(f"p-value: {sig.get('p_value', 'N/A')}")
    print(f"Significant: {sig.get('significant', 'N/A')}")
    print(f"Significance Level: {sig.get('significance_level', 'N/A')}")
    print(f"Effect Size (Cohen's d): {effect_size:.4f}")
    print(f"Overall Improvement: {comparison.get('improvement', {}).get('mean', 0)*100:+.2f}%")

    micro_prf = comparison.get("micro_prf", {})
    if micro_prf:
        print("\n" + "=" * 70)
        print("ENTITY-LEVEL MICRO METRICS (MEAN ACROSS RUNS)")
        print("=" * 70)
        print(f"Baseline Precision:  {micro_prf.get('baseline_micro_precision_mean', 0):.2%}")
        print(f"Baseline Recall:     {micro_prf.get('baseline_micro_recall_mean', 0):.2%}")
        print(f"Baseline F1:         {micro_prf.get('baseline_micro_f1_mean', 0):.2%}")
        print(f"Optimized Precision: {micro_prf.get('optimized_micro_precision_mean', 0):.2%}")
        print(f"Optimized Recall:    {micro_prf.get('optimized_micro_recall_mean', 0):.2%}")
        print(f"Optimized F1:        {micro_prf.get('optimized_micro_f1_mean', 0):.2%}")
        print(f"F1 Improvement:      {micro_prf.get('micro_f1_improvement', 0):+.2%}")


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

    # Create metadata for individual results
    metadata = {
        "experiment": "medical_ner",
        "aggregation_timestamp": datetime.now().isoformat(),
        "num_runs": num_trials,
        "random_seeds": config.RANDOM_SEEDS[:num_trials],
        "train_size": config.TRAIN_SIZE,
        "test_size": config.TEST_SIZE,
        "dataset": config.DATASET_NAME,
        "extraction_fields": config.EXTRACTION_FIELDS,
        "confidence_level": config.CONFIDENCE_LEVEL,
        "statistical_test": config.STATISTICAL_TEST,
        "use_copro": config.USE_COPRO,
    }

    # Save baseline aggregated with metadata
    baseline_dict = baseline_agg.to_dict()
    baseline_dict["metadata"] = metadata
    with open(aggregated_dir / "baseline_aggregated.json", "w", encoding="utf-8") as f:
        json.dump(baseline_dict, f, indent=2, default=str)

    # Save optimized aggregated with metadata
    optimized_dict = optimized_agg.to_dict()
    optimized_dict["metadata"] = metadata
    with open(aggregated_dir / "dspy_aggregated.json", "w", encoding="utf-8") as f:
        json.dump(optimized_dict, f, indent=2, default=str)

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
        "micro_prf": {},
        "metadata": metadata,
    }

    baseline_micro_p = []
    baseline_micro_r = []
    baseline_micro_f1 = []
    optimized_micro_p = []
    optimized_micro_r = []
    optimized_micro_f1 = []

    for run in baseline_runs:
        run_meta = run.metadata or {}
        if "micro_f1" in run_meta:
            baseline_micro_p.append(run_meta.get("micro_precision", 0.0))
            baseline_micro_r.append(run_meta.get("micro_recall", 0.0))
            baseline_micro_f1.append(run_meta.get("micro_f1", 0.0))

    for run in optimized_runs:
        run_meta = run.metadata or {}
        if "micro_f1" in run_meta:
            optimized_micro_p.append(run_meta.get("micro_precision", 0.0))
            optimized_micro_r.append(run_meta.get("micro_recall", 0.0))
            optimized_micro_f1.append(run_meta.get("micro_f1", 0.0))

    if baseline_micro_f1 and optimized_micro_f1:
        comparison["micro_prf"] = {
            "baseline_micro_precision_mean": _mean(baseline_micro_p),
            "baseline_micro_recall_mean": _mean(baseline_micro_r),
            "baseline_micro_f1_mean": _mean(baseline_micro_f1),
            "optimized_micro_precision_mean": _mean(optimized_micro_p),
            "optimized_micro_recall_mean": _mean(optimized_micro_r),
            "optimized_micro_f1_mean": _mean(optimized_micro_f1),
            "micro_f1_improvement": _mean(optimized_micro_f1) - _mean(baseline_micro_f1),
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

    if comparison["micro_prf"]:
        print("\nEntity-level Micro PR/F1 (Mean across runs):")
        print("-" * 60)
        print(f"  Baseline Precision:  {comparison['micro_prf']['baseline_micro_precision_mean']:.2%}")
        print(f"  Baseline Recall:     {comparison['micro_prf']['baseline_micro_recall_mean']:.2%}")
        print(f"  Baseline F1:         {comparison['micro_prf']['baseline_micro_f1_mean']:.2%}")
        print(f"  Optimized Precision: {comparison['micro_prf']['optimized_micro_precision_mean']:.2%}")
        print(f"  Optimized Recall:    {comparison['micro_prf']['optimized_micro_recall_mean']:.2%}")
        print(f"  Optimized F1:        {comparison['micro_prf']['optimized_micro_f1_mean']:.2%}")
        print(f"  F1 Improvement:      {comparison['micro_prf']['micro_f1_improvement']:+.2%}")

    return {
        "baseline_aggregated": baseline_agg,
        "optimized_aggregated": optimized_agg,
        "comparison": comparison,
        "significance": significance,
        "effect_size": effect_size,
    }


if __name__ == "__main__":
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else None
    num_trials = num_trials or config.NUM_RUNS
    
    results = aggregate_trial_results(num_trials=num_trials)
    
    # Display metadata and accuracy metrics
    if results:
        metadata = {
            "experiment": "medical_ner",
            "aggregation_timestamp": datetime.now().isoformat(),
            "num_runs": num_trials,
            "random_seeds": config.RANDOM_SEEDS[:num_trials],
            "train_size": config.TRAIN_SIZE,
            "test_size": config.TEST_SIZE,
            "dataset": config.DATASET_NAME,
            "extraction_fields": config.EXTRACTION_FIELDS,
            "confidence_level": config.CONFIDENCE_LEVEL,
            "statistical_test": config.STATISTICAL_TEST,
            "use_copro": config.USE_COPRO,
        }
        
        display_metadata_metrics(
            results["baseline_aggregated"],
            results["optimized_aggregated"],
            results["comparison"],
            results["effect_size"],
            metadata
        )
