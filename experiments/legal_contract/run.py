"""
Legal Contract Analysis Experiment Runner

Runs the full experiment: baseline evaluation, DSPy optimization, and comparison.
"""

import sys
from pathlib import Path
from typing import Iterable

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
from .loader import load_data, load_reviewed_examples_file
from .modules import BaselineModule, StudentModule
from .metrics import validate_clause_extraction
from .evaluation import detailed_evaluation, print_evaluation_summary
from .validate_reviewed_dataset import validate as validate_reviewed_dataset
from .evaluate_metadata import evaluate_metadata, DEFAULT_FIELDS


def _safe_dir_name(name: str) -> str:
    return (
        str(name or "")
        .strip()
        .lower()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )




def _assert_jsonl_path(path_str: str, arg_name: str):
    p = Path(path_str)
    if p.suffix.lower() != ".jsonl":
        raise ValueError(
            f"{arg_name} must be a .jsonl reviewed dataset for clause-extraction run.py. "
            f"Got: {p}. For metadata CSV evaluation, use: "
            "python -m experiments.legal_contract.evaluate_metadata --gold-csv ... --pred-csv ..."
        )

def _normalize_clause_types(clause_types):
    if not clause_types:
        return None
    return [str(x).strip() for x in clause_types if str(x).strip()]


def run_experiment(
    save_results=True,
    seed=None,
    results_dir=None,
    reviewed_file=None,
    train_reviewed_file=None,
    test_reviewed_file=None,
    strict_reviewed_validation=False,
    allow_source_gold=False,
    clause_types=None,
    train_size=None,
    test_size=None,
):
    """
    Run the full legal contract analysis experiment.

    Args:
        save_results: Whether to save results to JSON files.
        seed: Random seed for reproducibility. Default: None.
        results_dir: Directory to save results. Default: config.RESULTS_DIR
        reviewed_file: Optional JSONL file with reviewed labels
        allow_source_gold: Allow source_gold fallback for non-reviewed rows
        clause_types: Optional subset of clause types to train/evaluate

    Returns:
        Dict with baseline_results, optimized_results, and comparison
    """
    print("=" * 60)
    print("LEGAL CONTRACT ANALYSIS EXPERIMENT - CUAD Dataset")
    print("Baseline (Handcrafted Prompt) vs DSPy (Optimized)")
    if seed is not None:
        print(f"Seed: {seed}")
    clause_types = _normalize_clause_types(clause_types)
    if clause_types:
        print(f"Clause filter: {clause_types}")
    print("=" * 60)

    # 1. Setup LLM
    print("\n[1/5] Setting up LLM...")
    setup_dspy_lm()

    if reviewed_file:
        _assert_jsonl_path(reviewed_file, "--reviewed-file")
        report = validate_reviewed_dataset(
            Path(reviewed_file),
            strict=strict_reviewed_validation,
            # Reviewed files can validly contain more clause types than the
            # current run filter (e.g., per-clause runs on an all-types file).
            clause_types=None,
        )
        problems = list(report.get("problems", []))
        # If this run is filtered to a subset, ignore unrelated strict issues.
        selected_clause_types = set(clause_types or [])
        if selected_clause_types and "effective_expiration_overlap" in problems:
            needs_date_overlap_check = (
                "Effective Date" in selected_clause_types
                and "Expiration Date" in selected_clause_types
            )
            if not needs_date_overlap_check:
                problems = [p for p in problems if p != "effective_expiration_overlap"]

        if problems:
            raise ValueError(
                "Reviewed dataset failed strict validation. "
                f"Problems: {problems}"
            )
    if train_reviewed_file:
        _assert_jsonl_path(train_reviewed_file, "--train-reviewed-file")
        report = validate_reviewed_dataset(
            Path(train_reviewed_file),
            strict=strict_reviewed_validation,
            clause_types=None,
        )
        if report.get("problems"):
            raise ValueError(
                "Train reviewed dataset failed strict validation. "
                f"Problems: {report['problems']}"
            )
    if test_reviewed_file:
        _assert_jsonl_path(test_reviewed_file, "--test-reviewed-file")
        report = validate_reviewed_dataset(
            Path(test_reviewed_file),
            strict=strict_reviewed_validation,
            clause_types=None,
        )
        if report.get("problems"):
            raise ValueError(
                "Test reviewed dataset failed strict validation. "
                f"Problems: {report['problems']}"
            )

    # 2. Load Data
    print("[2/5] Loading dataset...")
    if train_reviewed_file and test_reviewed_file:
        selected_clause_types = clause_types or config.CLAUSE_TYPES
        trainset = load_reviewed_examples_file(
            reviewed_file=train_reviewed_file,
            clause_types=selected_clause_types,
            allow_source_gold=allow_source_gold,
            seed=seed,
        )
        testset = load_reviewed_examples_file(
            reviewed_file=test_reviewed_file,
            clause_types=selected_clause_types,
            allow_source_gold=allow_source_gold,
            seed=seed,
        )
    else:
        trainset, testset = load_data(
            train_size=train_size,
            test_size=test_size,
            seed=seed,
            reviewed_file=reviewed_file,
            allow_source_gold=allow_source_gold,
            clause_types=clause_types,
        )
    if not trainset or not testset:
        raise ValueError(
            "No data available after filtering. "
            "Check reviewed file and clause types."
        )
    print(f"      Train: {len(trainset)} samples, Test: {len(testset)} samples")
    if reviewed_file:
        _assert_jsonl_path(reviewed_file, "--reviewed-file")
        print(f"      Source: reviewed labels from {reviewed_file}")
    if train_reviewed_file and test_reviewed_file:
        print(f"      Train source: {train_reviewed_file}")
        print(f"      Test source:  {test_reviewed_file}")
    
    # Show clause type distribution
    clause_dist = {}
    for ex in trainset + testset:
        clause_dist[ex.clause_type] = clause_dist.get(ex.clause_type, 0) + 1
    print(f"      Clause types: {clause_dist}")

    # 3. Baseline Evaluation
    print("[3/5] Evaluating baseline model...")
    baseline = BaselineModule()
    baseline_results = detailed_evaluation(baseline, testset, "Baseline")
    print_evaluation_summary(baseline_results)

    # 4. Optimization
    print("[4/5] Optimizing with DSPy...")
    student = StudentModule()
    optimized_student = run_two_stage_optimization(
        student_module=student,
        trainset=trainset,
        metric=validate_clause_extraction
    )

    # 5. Optimized Evaluation
    print("[5/5] Evaluating optimized model...")
    optimized_results = detailed_evaluation(optimized_student, testset, "DSPy")
    print_evaluation_summary(optimized_results)

    # Compare results
    comparison = compare_results(baseline_results, optimized_results)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    baseline_f1 = baseline_results.metadata.get('f1_score', baseline_results.overall_accuracy)
    optimized_f1 = optimized_results.metadata.get('f1_score', optimized_results.overall_accuracy)
    print(f"Baseline F1:   {baseline_f1:.2%}")
    print(f"DSPy F1:       {optimized_f1:.2%}")
    improvement = optimized_f1 - baseline_f1
    print(f"Improvement:   {improvement:+.2%}")

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


def run_multiple_trials(
    num_runs=None,
    seeds=None,
    reviewed_file=None,
    train_reviewed_file=None,
    test_reviewed_file=None,
    strict_reviewed_validation=False,
    allow_source_gold=False,
    clause_types=None,
    base_results_dir=None,
    train_size=None,
    test_size=None,
):
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
    base_results_dir = Path(base_results_dir) if base_results_dir else config.RESULTS_DIR

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
            reviewed_file=reviewed_file,
            train_reviewed_file=train_reviewed_file,
            test_reviewed_file=test_reviewed_file,
            strict_reviewed_validation=strict_reviewed_validation,
            allow_source_gold=allow_source_gold,
            clause_types=clause_types,
            train_size=train_size,
            test_size=test_size,
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


def run_per_clause_experiments(
    clause_types=None,
    multi_run=None,
    reviewed_file=None,
    train_reviewed_file=None,
    test_reviewed_file=None,
    strict_reviewed_validation=False,
    allow_source_gold=False,
    train_size=None,
    test_size=None,
):
    """
    Run specialized experiments independently for each clause type.
    Results are saved under results/per_clause/<clause_type>/...
    """
    clause_types = _normalize_clause_types(clause_types) or list(config.WEAK_CLAUSE_TYPES)
    print("=" * 70)
    print("PER-CLAUSE EXPERIMENTS")
    print(f"Clause types: {clause_types}")
    print("=" * 70)

    outputs = {}
    for clause_type in clause_types:
        clause_dir = config.RESULTS_DIR / "per_clause" / _safe_dir_name(clause_type)
        clause_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"CLAUSE: {clause_type}")
        print("=" * 70)

        if multi_run is not None:
            outputs[clause_type] = run_multiple_trials(
                num_runs=multi_run,
                reviewed_file=reviewed_file,
                train_reviewed_file=train_reviewed_file,
                test_reviewed_file=test_reviewed_file,
                strict_reviewed_validation=strict_reviewed_validation,
                allow_source_gold=allow_source_gold,
                clause_types=[clause_type],
                base_results_dir=clause_dir,
                train_size=train_size or config.PER_CLAUSE_TRAIN_SIZE,
                test_size=test_size or config.PER_CLAUSE_TEST_SIZE,
            )
        else:
            outputs[clause_type] = run_experiment(
                reviewed_file=reviewed_file,
                train_reviewed_file=train_reviewed_file,
                test_reviewed_file=test_reviewed_file,
                strict_reviewed_validation=strict_reviewed_validation,
                allow_source_gold=allow_source_gold,
                clause_types=[clause_type],
                results_dir=clause_dir,
                train_size=train_size or config.PER_CLAUSE_TRAIN_SIZE,
                test_size=test_size or config.PER_CLAUSE_TEST_SIZE,
            )

    return outputs


def run_metadata_evaluation(
    gold_csv,
    pred_csv,
    output_json=None,
    id_col="contract_id",
    fields=None,
    no_llm=False,
):
    """Run contract metadata CSV evaluation (LLM-judge aware)."""
    print("=" * 60)
    print("LEGAL CONTRACT METADATA EVALUATION")
    print("=" * 60)

    if not no_llm:
        print("\n[1/2] Setting up LLM...")
        setup_dspy_lm()

    print("[2/2] Evaluating metadata CSV...")
    report = evaluate_metadata(
        gold_csv=Path(gold_csv),
        pred_csv=Path(pred_csv),
        id_col=id_col,
        fields=fields or DEFAULT_FIELDS,
        use_llm=not no_llm,
    )

    out_path = Path(output_json or (config.RESULTS_DIR / "metadata_eval.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(__import__("json").dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Compared contracts: {report['contracts_compared']}")
    print(f"Overall accuracy: {report['overall_accuracy']:.2%}")
    print("Per-field accuracy:")
    for f, score in sorted(report['per_field_accuracy'].items()):
        print(f"  {f:28} {score:.2%}")
    print(f"Saved: {out_path}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run legal_contract experiments")
    parser.add_argument(
        "--multi-run",
        type=int,
        default=None,
        help="Run N trials instead of a single run",
    )
    parser.add_argument(
        "--reviewed-file",
        default=None,
        help="Optional reviewed JSONL file for curated labels",
    )
    parser.add_argument(
        "--train-reviewed-file",
        default=None,
        help="Optional fixed train reviewed JSONL file",
    )
    parser.add_argument(
        "--test-reviewed-file",
        default=None,
        help="Optional fixed test reviewed JSONL file",
    )
    parser.add_argument(
        "--strict-reviewed-validation",
        action="store_true",
        help="Enable strict quality validation checks on reviewed JSONL files",
    )
    parser.add_argument(
        "--allow-source-gold",
        action="store_true",
        help="Use source_gold rows when reviewed_gold is not filled yet",
    )
    parser.add_argument(
        "--clause-types",
        nargs="+",
        default=None,
        help="Optional clause type subset (for filtered runs)",
    )
    parser.add_argument(
        "--per-clause",
        action="store_true",
        help="Run one independent experiment per clause type",
    )
    parser.add_argument(
        "--weak-only",
        action="store_true",
        help="Use config.WEAK_CLAUSE_TYPES as clause filter",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Optional train size override",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Optional test size override",
    )
    parser.add_argument(
        "--metadata-gold-csv",
        default=None,
        help="Gold metadata CSV path (enables metadata eval mode)",
    )
    parser.add_argument(
        "--metadata-pred-csv",
        default=None,
        help="Predicted metadata CSV path (enables metadata eval mode)",
    )
    parser.add_argument(
        "--metadata-id-col",
        default="contract_id",
        help="Metadata ID column name",
    )
    parser.add_argument(
        "--metadata-fields",
        nargs="+",
        default=None,
        help="Optional metadata fields list",
    )
    parser.add_argument(
        "--metadata-no-llm",
        action="store_true",
        help="Use lexical-only metadata eval (disable LLM judge)",
    )
    parser.add_argument(
        "--metadata-output-json",
        default=None,
        help="Output JSON path for metadata eval report",
    )
    args = parser.parse_args()


    if args.metadata_gold_csv or args.metadata_pred_csv:
        if not (args.metadata_gold_csv and args.metadata_pred_csv):
            raise ValueError("Provide both --metadata-gold-csv and --metadata-pred-csv")
        run_metadata_evaluation(
            gold_csv=args.metadata_gold_csv,
            pred_csv=args.metadata_pred_csv,
            output_json=args.metadata_output_json,
            id_col=args.metadata_id_col,
            fields=args.metadata_fields,
            no_llm=args.metadata_no_llm,
        )
        raise SystemExit(0)

    selected_clause_types = (
        list(config.WEAK_CLAUSE_TYPES) if args.weak_only else args.clause_types
    )

    if args.per_clause:
        run_per_clause_experiments(
            clause_types=selected_clause_types,
            multi_run=args.multi_run,
            reviewed_file=args.reviewed_file,
            train_reviewed_file=args.train_reviewed_file,
            test_reviewed_file=args.test_reviewed_file,
            strict_reviewed_validation=args.strict_reviewed_validation,
            allow_source_gold=args.allow_source_gold,
            train_size=args.train_size,
            test_size=args.test_size,
        )
        raise SystemExit(0)

    if args.multi_run is not None:
        run_multiple_trials(
            num_runs=args.multi_run,
            reviewed_file=args.reviewed_file,
            train_reviewed_file=args.train_reviewed_file,
            test_reviewed_file=args.test_reviewed_file,
            strict_reviewed_validation=args.strict_reviewed_validation,
            allow_source_gold=args.allow_source_gold,
            clause_types=selected_clause_types,
            train_size=args.train_size,
            test_size=args.test_size,
        )
    else:
        run_experiment(
            reviewed_file=args.reviewed_file,
            train_reviewed_file=args.train_reviewed_file,
            test_reviewed_file=args.test_reviewed_file,
            strict_reviewed_validation=args.strict_reviewed_validation,
            allow_source_gold=args.allow_source_gold,
            clause_types=selected_clause_types,
            train_size=args.train_size,
            test_size=args.test_size,
        )
