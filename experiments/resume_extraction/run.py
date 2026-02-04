"""
Resume Extraction Experiment Runner

Runs the full experiment: baseline evaluation, DSPy optimization, and comparison.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.llm_providers import setup_dspy_lm
from shared.optimization import run_two_stage_optimization
from shared.evaluation import compare_results, save_results_json, EvaluationResult

from . import config
from .loader import load_data
from .modules import BaselineModule, StudentModule
from .metrics import validate_resume_output
from .evaluation import detailed_evaluation


def run_experiment(data_path=None, save_results=True):
    """
    Run the full resume extraction experiment.

    Args:
        data_path: Optional path to data CSV. Default: uses config.
        save_results: Whether to save results to JSON files.

    Returns:
        Dict with baseline_results, optimized_results, and comparison
    """
    print("=" * 60)
    print("RESUME EXTRACTION EXPERIMENT")
    print("Baseline (Handcrafted Prompt) vs DSPy (Optimized)")
    print("=" * 60)

    # 1. Setup LLM
    print("\n[1/5] Setting up LLM...")
    setup_dspy_lm()

    # 2. Load Data
    print("[2/5] Loading data...")
    trainset, testset = load_data(data_path)
    print(f"      Train: {len(trainset)} samples, Test: {len(testset)} samples")

    # 3. Baseline Evaluation
    print("[3/5] Evaluating baseline model...")
    baseline = BaselineModule()
    baseline_results = detailed_evaluation(baseline, testset, "Baseline")
    print(f"      Baseline accuracy: {baseline_results.overall_accuracy:.2%}")

    # 4. Optimization
    print("[4/5] Optimizing with DSPy...")
    student = StudentModule()
    optimized_student = run_two_stage_optimization(
        student_module=student,
        trainset=trainset,
        metric=validate_resume_output
    )

    # 5. Optimized Evaluation
    print("[5/5] Evaluating optimized model...")
    optimized_results = detailed_evaluation(optimized_student, testset, "DSPy")
    print(f"      DSPy accuracy: {optimized_results.overall_accuracy:.2%}")

    # Compare results
    comparison = compare_results(baseline_results, optimized_results)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline Overall Accuracy:  {baseline_results.overall_accuracy:.2%}")
    print(f"DSPy Overall Accuracy:      {optimized_results.overall_accuracy:.2%}")
    print(f"Improvement:                {comparison['summary']['overall_improvement']:.2%}")

    print("\nField-wise Accuracy:")
    print("-" * 40)
    for field in config.EXTRACTION_FIELDS:
        baseline_acc = baseline_results.field_accuracies.get(field, 0)
        optimized_acc = optimized_results.field_accuracies.get(field, 0)
        improvement = optimized_acc - baseline_acc
        print(f"  {field:20} Baseline: {baseline_acc:6.2%}  DSPy: {optimized_acc:6.2%}  ({improvement:+.2%})")

    print(f"\nSamples improved: {comparison['summary']['total_improvements']}")
    print(f"Samples degraded: {comparison['summary']['total_degradations']}")
    print(f"Samples unchanged: {comparison['summary']['total_unchanged']}")

    # Save results
    if save_results:
        results_dir = config.RESULTS_DIR
        save_results_json(baseline_results, results_dir / "baseline_results.json")
        save_results_json(optimized_results, results_dir / "dspy_results.json")
        save_results_json(comparison, results_dir / "comparison_results.json")
        optimized_student.save(str(results_dir / "optimized_module.json"))
        print(f"\nResults saved to {results_dir}/")

    return {
        'baseline_results': baseline_results,
        'optimized_results': optimized_results,
        'comparison': comparison,
        'optimized_module': optimized_student
    }


if __name__ == "__main__":
    run_experiment()
