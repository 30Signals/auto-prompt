"""
Medical NER Experiment Runner

Runs the full experiment: baseline evaluation, DSPy optimization, and comparison.
"""

import sys
import random
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.llm_providers import enforce_usage_budget, save_lm_history_artifacts, setup_dspy_lm
from shared.optimization import run_bootstrap_optimization, run_two_stage_optimization
from shared.evaluation import compare_results, save_results_json
from shared.evaluation.prompt_utils import (
    save_baseline_prompt,
    extract_optimized_prompt,
    generate_prompt_comparison
)

from . import config
from .loader import load_data
from .modules import BaselineModule, StudentModule
from .metrics import parse_diseases, validate_disease_output_exact, validate_disease_output_exact_recall
from .evaluation import detailed_evaluation, print_evaluation_summary


def _resolve_run_sizes(smoke_run=False):
    if smoke_run:
        return 24, 12
    return None, None


def _split_train_validation(trainset, validation_size=24, seed=None):
    """Split trainset into optimization-train and validation subsets."""
    if not trainset:
        return [], []

    n_total = len(trainset)
    if n_total < 2:
        return trainset, trainset

    val_size = min(validation_size, n_total - 1)
    rng = random.Random(seed if seed is not None else 0)
    indices = list(range(n_total))
    rng.shuffle(indices)
    val_idx = set(indices[:val_size])

    opt_train = [ex for i, ex in enumerate(trainset) if i not in val_idx]
    val_set = [ex for i, ex in enumerate(trainset) if i in val_idx]
    return opt_train, val_set


def _rebalance_optimization_trainset(trainset, seed=None):
    """Upweight entity-positive examples by downsampling negatives."""
    if not trainset:
        return trainset

    positives = [ex for ex in trainset if parse_diseases(ex.diseases)]
    negatives = [ex for ex in trainset if not parse_diseases(ex.diseases)]
    if not positives or not negatives:
        return trainset

    target_ratio = config.OPT_POSITIVE_TARGET_RATIO
    max_negatives = int((len(positives) * (1.0 - target_ratio)) / max(target_ratio, 1e-9))
    max_negatives = max(1, max_negatives)

    rng = random.Random(seed if seed is not None else 0)
    rng.shuffle(negatives)
    selected_negatives = negatives[:max_negatives]

    mixed = positives + selected_negatives
    rng.shuffle(mixed)
    return mixed


def _optimize_candidate(strategy_name, trainset):
    """Build optimized module for a specific strategy."""
    student = StudentModule()

    if strategy_name == "bootstrap_exact":
        return run_bootstrap_optimization(
            student_module=student,
            trainset=trainset,
            metric=validate_disease_output_exact,
            **config.BOOTSTRAP_CONFIG,
        )

    if strategy_name == "bootstrap_recall":
        return run_bootstrap_optimization(
            student_module=student,
            trainset=trainset,
            metric=validate_disease_output_exact_recall,
            **config.BOOTSTRAP_CONFIG,
        )

    if strategy_name == "two_stage_exact":
        return run_two_stage_optimization(
            student_module=student,
            trainset=trainset,
            metric=validate_disease_output_exact,
            bootstrap_config=config.BOOTSTRAP_CONFIG,
            copro_config=config.COPRO_CONFIG,
        )

    if strategy_name == "two_stage_recall":
        return run_two_stage_optimization(
            student_module=student,
            trainset=trainset,
            metric=validate_disease_output_exact_recall,
            bootstrap_config=config.BOOTSTRAP_CONFIG,
            copro_config=config.COPRO_CONFIG,
        )

    raise ValueError(f"Unknown optimization strategy: {strategy_name}")


def _select_best_optimized_module(trainset, seed=None):
    """
    Try several optimization strategies and keep the best on held-out validation.
    Selection target: micro F1, with macro F1 as tiebreaker.
    """
    opt_train, val_set = _split_train_validation(
        trainset,
        validation_size=config.VALIDATION_SIZE,
        seed=seed,
    )
    if config.REBALANCE_OPT_TRAIN:
        opt_train = _rebalance_optimization_trainset(opt_train, seed=seed)

    candidates = list(config.OPTIMIZATION_CANDIDATES)
    if not config.USE_COPRO:
        candidates = [c for c in candidates if not c.startswith("two_stage")]

    print(f"      Optimization train size: {len(opt_train)}, validation size: {len(val_set)}")
    print(f"      Candidate strategies: {candidates}")

    baseline_val_result = detailed_evaluation(BaselineModule(), val_set, "Val/Baseline")
    baseline_micro_recall = baseline_val_result.metadata.get("micro_recall", 0.0)
    recall_floor = max(0.0, baseline_micro_recall - config.VALIDATION_RECALL_TOLERANCE)
    print(
        f"      Baseline validation micro recall: {baseline_micro_recall:.2%} "
        f"(floor for candidates: {recall_floor:.2%})"
    )

    best = {
        "strategy": "baseline_fallback",
        "module": BaselineModule(),
        "micro_f1": baseline_val_result.metadata.get("micro_f1", 0.0),
        "macro_f1": baseline_val_result.metadata.get("f1_score", baseline_val_result.overall_accuracy),
        "micro_recall": baseline_micro_recall,
        "meets_recall_floor": True,
    }

    for strategy in candidates:
        print(f"      - Trying strategy: {strategy}")
        try:
            candidate_module = _optimize_candidate(strategy, opt_train)
            val_result = detailed_evaluation(candidate_module, val_set, f"Val/{strategy}")
            micro_f1 = val_result.metadata.get("micro_f1", 0.0)
            macro_f1 = val_result.metadata.get("f1_score", val_result.overall_accuracy)
            micro_recall = val_result.metadata.get("micro_recall", 0.0)
            meets_recall_floor = micro_recall >= recall_floor
            print(
                f"        Validation micro F1: {micro_f1:.2%} | "
                f"micro recall: {micro_recall:.2%} | macro F1: {macro_f1:.2%} | "
                f"recall floor ok: {meets_recall_floor}"
            )

            is_better = (
                (meets_recall_floor and not best["meets_recall_floor"]) or
                (
                    meets_recall_floor == best["meets_recall_floor"] and
                    (
                        (micro_f1 > best["micro_f1"]) or
                        (micro_f1 == best["micro_f1"] and micro_recall > best["micro_recall"]) or
                        (
                            micro_f1 == best["micro_f1"] and
                            micro_recall == best["micro_recall"] and
                            macro_f1 > best["macro_f1"]
                        )
                    )
                )
            )
            if is_better:
                best = {
                    "strategy": strategy,
                    "module": candidate_module,
                    "micro_f1": micro_f1,
                    "macro_f1": macro_f1,
                    "micro_recall": micro_recall,
                    "meets_recall_floor": meets_recall_floor,
                }
        except Exception as e:
            print(f"        Strategy failed: {type(e).__name__}: {e}")

    if config.REQUIRE_VALIDATION_BEAT_BASELINE:
        baseline_micro_f1 = baseline_val_result.metadata.get("micro_f1", 0.0)
        if best["micro_f1"] <= baseline_micro_f1:
            print(
                "      No candidate beat baseline on validation micro F1. "
                "Using baseline fallback."
            )
            best = {
                "strategy": "baseline_fallback",
                "module": BaselineModule(),
                "micro_f1": baseline_micro_f1,
                "macro_f1": baseline_val_result.metadata.get(
                    "f1_score", baseline_val_result.overall_accuracy
                ),
                "micro_recall": baseline_micro_recall,
                "meets_recall_floor": True,
            }

    if best["module"] is None:
        raise RuntimeError("All optimization strategies failed during model selection.")

    print(
        f"      Selected strategy: {best['strategy']} "
        f"(val micro F1={best['micro_f1']:.2%}, micro recall={best['micro_recall']:.2%}, "
        f"macro F1={best['macro_f1']:.2%}, recall floor ok={best['meets_recall_floor']})"
    )
    return best


def run_experiment(save_results=True, seed=None, results_dir=None, smoke_run=False):
    """
    Run the full medical NER experiment.

    Args:
        save_results: Whether to save results to JSON files.
        seed: Random seed for reproducibility. Default: None.
        results_dir: Directory to save results. Default: config.RESULTS_DIR

    Returns:
        Dict with baseline_results, optimized_results, and comparison
    """
    print("=" * 60)
    print("MEDICAL NER EXPERIMENT - NCBI Disease Corpus")
    print("Baseline (Handcrafted Prompt) vs DSPy (Optimized)")
    if smoke_run:
        print("Smoke Run: enabled")
    if seed is not None:
        print(f"Seed: {seed}")
    print("=" * 60)

    # 1. Setup LLM
    print("\n[1/5] Setting up LLM...")
    setup_dspy_lm()

    # 2. Load Data
    print("[2/5] Loading NCBI Disease Corpus...")
    train_size, test_size = _resolve_run_sizes(smoke_run=smoke_run)
    trainset, testset = load_data(train_size=train_size, test_size=test_size, seed=seed)
    print(f"      Train: {len(trainset)} samples, Test: {len(testset)} samples")

    # 3. Baseline Evaluation
    print("[3/5] Evaluating baseline model...")
    baseline = BaselineModule()
    baseline_results = detailed_evaluation(baseline, testset, "Baseline")
    print_evaluation_summary(baseline_results)

    # 4. Optimization
    print("[4/5] Optimizing with DSPy...")
    best_candidate = _select_best_optimized_module(trainset, seed=seed)
    optimized_student = best_candidate["module"]

    # 5. Optimized Evaluation
    print("[5/5] Evaluating optimized model...")
    optimized_results = detailed_evaluation(
        optimized_student,
        testset,
        f"DSPy ({best_candidate['strategy']})",
    )
    optimized_results.metadata["selected_strategy"] = best_candidate["strategy"]
    optimized_results.metadata["validation_micro_f1"] = best_candidate["micro_f1"]
    optimized_results.metadata["validation_macro_f1"] = best_candidate["macro_f1"]
    print_evaluation_summary(optimized_results)

    # Compare results
    comparison = compare_results(baseline_results, optimized_results)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    baseline_f1 = baseline_results.metadata.get('f1_score', baseline_results.overall_accuracy)
    optimized_f1 = optimized_results.metadata.get('f1_score', optimized_results.overall_accuracy)
    baseline_micro_f1 = baseline_results.metadata.get('micro_f1', 0.0)
    optimized_micro_f1 = optimized_results.metadata.get('micro_f1', 0.0)
    print(f"Baseline F1:   {baseline_f1:.2%}")
    print(f"DSPy F1:       {optimized_f1:.2%}")
    improvement = optimized_f1 - baseline_f1
    print(f"Improvement:   {improvement:+.2%}")
    print(f"Baseline Micro F1: {baseline_micro_f1:.2%}")
    print(f"DSPy Micro F1:     {optimized_micro_f1:.2%}")
    print(f"Micro Improvement: {optimized_micro_f1 - baseline_micro_f1:+.2%}")
    print(f"Selected Strategy: {best_candidate['strategy']}")
    print(f"Validation Micro F1: {best_candidate['micro_f1']:.2%}")

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
        usage_summary = save_lm_history_artifacts(output_dir)
        if usage_summary:
            print(
                "      LM usage:"
                f" calls={usage_summary['calls']},"
                f" total_tokens={usage_summary['total_tokens']},"
                f" estimated_cost={usage_summary['estimated_cost']:.6f}"
            )
            enforce_usage_budget(usage_summary)

        print(f"\nResults saved to {output_dir}/")

    return {
        'baseline_results': baseline_results,
        'optimized_results': optimized_results,
        'comparison': comparison,
        'optimized_module': optimized_student
    }


def run_multiple_trials(num_runs=None, seeds=None, smoke_run=False):
    """
    Run multiple trials of the experiment with different seeds.

    Args:
        num_runs: Number of trials to run. Default: config.NUM_RUNS
        seeds: List of random seeds. Default: config.RANDOM_SEEDS

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
            save_results=True,
            seed=seed,
            results_dir=trial_dir,
            smoke_run=smoke_run
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

    return {
        'trials': trial_results,
        'num_runs': num_runs,
        'seeds': seeds[:num_runs],
        'results_dir': base_results_dir
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run medical_ner experiments")
    parser.add_argument("--multi-run", type=int, default=None, help="Run N trials instead of a single run")
    parser.add_argument("--smoke-run", action="store_true", help="Run a tiny low-cost dataset slice for debugging")
    args = parser.parse_args()

    if args.multi_run is not None:
        run_multiple_trials(num_runs=args.multi_run, smoke_run=args.smoke_run)
    else:
        run_experiment(smoke_run=args.smoke_run)
