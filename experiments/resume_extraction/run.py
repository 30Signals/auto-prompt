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
from shared.evaluation.prompt_utils import (
    save_baseline_prompt,
    extract_optimized_prompt,
    generate_prompt_comparison
)

from . import config
from .loader import load_data
from .modules import BaselineModule, StudentModule
from .metrics import validate_resume_output
from .evaluation import detailed_evaluation

# The run_experiment function orchestrates the entire workflow of the resume extraction experiment, including setting up the LLM, loading data, evaluating the baseline model, optimizing with DSPy, evaluating the optimized model, comparing results, and saving outputs for analysis and reproducibility.
def run_experiment(data_path=None, save_results=True, seed=None, results_dir=None):
    """
    Run the full resume extraction experiment.

    Args:
        data_path: Optional path to data CSV. Default: uses config.
        save_results: Whether to save results to JSON files.
        seed: Random seed for reproducibility. Default: None.
        results_dir: Directory to save results. Default: config.RESULTS_DIR

    Returns:
        Dict with baseline_results, optimized_results, and comparison
    """
    print("=" * 60)
    print("RESUME EXTRACTION EXPERIMENT")
    print("Baseline (Handcrafted Prompt) vs DSPy (Optimized)")
    if seed is not None:
        print(f"Seed: {seed}")
    print("=" * 60)

    # 1. Setup LLM
    print("\n[1/5] Setting up LLM...")
    setup_dspy_lm()

    # 2. Load Data
    print("[2/5] Loading data...")
    trainset, testset = load_data(data_path, seed=seed)
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
        output_dir = results_dir or config.RESULTS_DIR
        save_results_json(baseline_results, output_dir / "baseline_results.json")
        save_results_json(optimized_results, output_dir / "dspy_results.json")
        save_results_json(comparison, output_dir / "comparison_results.json")
        optimized_student.save(str(output_dir / "optimized_module.json"))

        # Save prompts
        baseline_prompt_path = config.PROMPTS_DIR / "baseline.txt"
        save_baseline_prompt(baseline_prompt_path, output_dir)
        extract_optimized_prompt(optimized_student, output_dir)
        generate_prompt_comparison(output_dir)

        print(f"\nResults saved to {output_dir}/")

    return {
        'baseline_results': baseline_results,
        'optimized_results': optimized_results,
        'comparison': comparison,
        'optimized_module': optimized_student
    }


def run_multiple_trials(num_runs=None, seeds=None, data_path=None):
    """
    Run multiple trials of the experiment with different seeds.

    Args:
        num_runs: Number of trials to run. Default: config.NUM_RUNS
        seeds: List of random seeds. Default: config.RANDOM_SEEDS
        data_path: Optional path to data CSV. Default: uses config.

    Returns:
        Dict with trial results and paths to aggregated results
    """
    num_runs = num_runs or config.NUM_RUNS
    seeds = seeds or config.RANDOM_SEEDS[:num_runs]

    if len(seeds) < num_runs:
        raise ValueError(f"Not enough seeds provided. Need {num_runs}, got {len(seeds)}")

    print("=" * 70)
    print(f"MULTI-TRIAL EXPERIMENT: {num_runs} runs")
    print("=" * 70)

    trial_results = []
    base_results_dir = config.RESULTS_DIR

    # Run each trial
    for trial_idx, seed in enumerate(seeds[:num_runs]):
        print(f"\n{'=' * 70}")
        print(f"TRIAL {trial_idx + 1}/{num_runs} (seed={seed})")
        print("=" * 70)

        # Create trial-specific directory
        trial_dir = base_results_dir / f"trial_{trial_idx}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Run experiment for this trial
        result = run_experiment(
            data_path=data_path,
            save_results=True,
            seed=seed,
            results_dir=trial_dir
        )

        trial_results.append({
            'trial_idx': trial_idx,
            'seed': seed,
            'baseline_results': result['baseline_results'],
            'optimized_results': result['optimized_results'],
            'comparison': result['comparison']
        })

    print(f"\n{'=' * 70}")
    print("ALL TRIALS COMPLETE")
    print("=" * 70)
    print(f"Trial results saved to: {base_results_dir}/trial_*/")
    print("\nNext steps:")
    print("- Run statistical aggregation to compute means and confidence intervals")
    print("- Generate plots with error bars")

    return {
        'trials': trial_results,
        'num_runs': num_runs,
        'seeds': seeds[:num_runs],
        'results_dir': base_results_dir
    }


if __name__ == "__main__":
    import sys

    # Check if user wants multi-run mode
    if len(sys.argv) > 1 and sys.argv[1] == "--multi-run":
        # Multi-run mode
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else None
        run_multiple_trials(num_runs=num_runs)
    else:
        # Single run mode (backward compatible)
        run_experiment()
