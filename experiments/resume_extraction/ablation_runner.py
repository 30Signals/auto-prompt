"""
Ablation Study Runner

Run ablation studies to test different optimization configurations:
- Optimizer-only ablation (BootstrapFewShot vs COPRO+BootstrapFewShot)
- Demo count ablation (4, 8, 16, 32 demos)
- Optimizer comparison (MIPROv2, BootstrapFewShot, SignatureOptimizer)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy
from dspy.teleprompt import BootstrapFewShot, COPRO

from shared.llm_providers import setup_dspy_lm
from shared.evaluation import compare_results, save_results_json

from . import config
from .loader import load_data
from .modules import StudentModule
from .metrics import validate_resume_output
from .evaluation import detailed_evaluation


# Ablation study configurations
ABLATION_CONFIGS = {
    'demo_counts': [4, 8, 16, 32],
    'optimizers': {
        'BootstrapFewShot': {
            'type': 'bootstrap',
            'max_bootstrapped_demos': 16,
            'max_labeled_demos': 16,
            'max_rounds': 4
        },
        'BootstrapFewShot+COPRO': {
            'type': 'two_stage',
            'bootstrap_config': {
                'max_bootstrapped_demos': 20,
                'max_labeled_demos': 20,
                'max_rounds': 6
            },
            'copro_config': {
                'breadth': 10,
                'depth': 3
            }
        }
    }
}


def run_demo_count_ablation(
    demo_counts: List[int] = None,
    data_path: Path = None,
    seed: int = 42,
    results_dir: Path = None
) -> Dict[str, Any]:
    """
    Run ablation study varying the number of demonstrations.

    Args:
        demo_counts: List of demo counts to test
        data_path: Path to data CSV
        seed: Random seed
        results_dir: Directory to save results

    Returns:
        Dict with results for each demo count
    """
    demo_counts = demo_counts or ABLATION_CONFIGS['demo_counts']
    results_dir = results_dir or config.RESULTS_DIR / "ablation_demo_count"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDY: Demo Count Variation")
    print("=" * 70)
    print(f"Testing demo counts: {demo_counts}")

    # Setup
    print("\n[Setup] Loading LLM and data...")
    setup_dspy_lm()
    trainset, testset = load_data(data_path, seed=seed)

    results = {}

    for demo_count in demo_counts:
        print(f"\n{'=' * 60}")
        print(f"Testing demo_count = {demo_count}")
        print("=" * 60)

        # Create and optimize student
        student = StudentModule()

        teleprompter = BootstrapFewShot(
            metric=validate_resume_output,
            max_bootstrapped_demos=demo_count,
            max_labeled_demos=demo_count,
            max_rounds=4,
            max_errors=10
        )

        optimized = teleprompter.compile(student, trainset=trainset)

        # Evaluate
        eval_results = detailed_evaluation(optimized, testset, f"demo_{demo_count}")
        print(f"  Accuracy: {eval_results.overall_accuracy:.2%}")

        # Save
        demo_dir = results_dir / f"demo_{demo_count}"
        demo_dir.mkdir(parents=True, exist_ok=True)
        save_results_json(eval_results, demo_dir / "results.json")
        optimized.save(str(demo_dir / "optimized_module.json"))

        results[demo_count] = {
            'demo_count': demo_count,
            'overall_accuracy': eval_results.overall_accuracy,
            'field_accuracies': eval_results.field_accuracies
        }

    # Save summary
    with open(results_dir / "ablation_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY: Demo Count")
    print("=" * 70)
    for demo_count, res in results.items():
        print(f"  {demo_count} demos: {res['overall_accuracy']:.2%}")

    return results


def run_optimizer_ablation(
    data_path: Path = None,
    seed: int = 42,
    results_dir: Path = None
) -> Dict[str, Any]:
    """
    Run ablation study comparing different optimizers.

    Args:
        data_path: Path to data CSV
        seed: Random seed
        results_dir: Directory to save results

    Returns:
        Dict with results for each optimizer
    """
    results_dir = results_dir or config.RESULTS_DIR / "ablation_optimizer"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDY: Optimizer Comparison")
    print("=" * 70)

    # Setup
    print("\n[Setup] Loading LLM and data...")
    setup_dspy_lm()
    trainset, testset = load_data(data_path, seed=seed)

    results = {}

    for optimizer_name, optimizer_config in ABLATION_CONFIGS['optimizers'].items():
        print(f"\n{'=' * 60}")
        print(f"Testing optimizer: {optimizer_name}")
        print("=" * 60)

        student = StudentModule()

        if optimizer_config['type'] == 'bootstrap':
            teleprompter = BootstrapFewShot(
                metric=validate_resume_output,
                max_bootstrapped_demos=optimizer_config['max_bootstrapped_demos'],
                max_labeled_demos=optimizer_config['max_labeled_demos'],
                max_rounds=optimizer_config['max_rounds'],
                max_errors=10
            )
            optimized = teleprompter.compile(student, trainset=trainset)

        elif optimizer_config['type'] == 'two_stage':
            # Stage 1: BootstrapFewShot
            bootstrap_teleprompter = BootstrapFewShot(
                metric=validate_resume_output,
                **optimizer_config['bootstrap_config'],
                max_errors=10
            )
            stage1 = bootstrap_teleprompter.compile(student, trainset=trainset)

            # Stage 2: COPRO
            try:
                copro_teleprompter = COPRO(
                    metric=validate_resume_output,
                    **optimizer_config['copro_config']
                )
                optimized = copro_teleprompter.compile(stage1, trainset=trainset)
            except Exception as e:
                print(f"  COPRO failed, using BootstrapFewShot only: {e}")
                optimized = stage1

        # Evaluate
        eval_results = detailed_evaluation(optimized, testset, optimizer_name)
        print(f"  Accuracy: {eval_results.overall_accuracy:.2%}")

        # Save
        opt_dir = results_dir / optimizer_name.replace('+', '_')
        opt_dir.mkdir(parents=True, exist_ok=True)
        save_results_json(eval_results, opt_dir / "results.json")
        optimized.save(str(opt_dir / "optimized_module.json"))

        results[optimizer_name] = {
            'optimizer': optimizer_name,
            'overall_accuracy': eval_results.overall_accuracy,
            'field_accuracies': eval_results.field_accuracies
        }

    # Save summary
    with open(results_dir / "ablation_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY: Optimizer Comparison")
    print("=" * 70)
    for opt_name, res in results.items():
        print(f"  {opt_name}: {res['overall_accuracy']:.2%}")

    return results


def run_full_ablation_study(
    data_path: Path = None,
    seed: int = 42,
    results_dir: Path = None
) -> Dict[str, Any]:
    """
    Run all ablation studies.

    Args:
        data_path: Path to data CSV
        seed: Random seed
        results_dir: Base directory for results

    Returns:
        Dict with all ablation results
    """
    results_dir = results_dir or config.RESULTS_DIR / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL ABLATION STUDY")
    print("=" * 70)

    results = {}

    # Demo count ablation
    print("\n" + "=" * 70)
    print("PHASE 1: Demo Count Ablation")
    print("=" * 70)
    results['demo_count'] = run_demo_count_ablation(
        data_path=data_path,
        seed=seed,
        results_dir=results_dir / "demo_count"
    )

    # Optimizer ablation
    print("\n" + "=" * 70)
    print("PHASE 2: Optimizer Ablation")
    print("=" * 70)
    results['optimizer'] = run_optimizer_ablation(
        data_path=data_path,
        seed=seed,
        results_dir=results_dir / "optimizer"
    )

    # Save combined results
    with open(results_dir / "full_ablation_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {results_dir}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--demo-count", action="store_true", help="Run demo count ablation only")
    parser.add_argument("--optimizer", action="store_true", help="Run optimizer ablation only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.demo_count:
        run_demo_count_ablation(seed=args.seed)
    elif args.optimizer:
        run_optimizer_ablation(seed=args.seed)
    else:
        run_full_ablation_study(seed=args.seed)
