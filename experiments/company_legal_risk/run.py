"""
Company Legal Risk Experiment Runner

Uses large labeled replay data and performs a seeded train/test split.
"""

import json
import random
import sys
from pathlib import Path

import dspy

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.llm_providers import setup_dspy_lm
from shared.optimization import run_two_stage_optimization
from shared.evaluation import compare_results, save_results_json
from shared.evaluation.prompt_utils import (
    save_baseline_prompt,
    extract_optimized_prompt,
    generate_prompt_comparison,
)

from . import config
from .modules import BaselineModule, StudentModule
from .metrics import validate_company_risk_decision
from .evaluation import detailed_evaluation, print_evaluation_summary
from .searx_client import (
    SearXNGClient,
    collect_company_evidence,
    build_search_context,
    normalize_retrieved_docs,
)
from .replay_logging import utc_now_iso


DEFAULT_REPLAY_FILE = config.DATA_DIR / "v1" / "retrieval_replay_labeling_queue_v1.json"


def _build_context(results, max_items=8):
    if not results:
        return "No search evidence retrieved."

    lines = []
    for i, item in enumerate(results[:max_items], start=1):
        lines.append(
            "\n".join(
                [
                    f"[{i}] Title: {str(item.get('title', '')).strip()}",
                    f"Source: {str(item.get('source', '')).strip()}",
                    f"URL: {str(item.get('url', '')).strip()}",
                    f"Snippet: {str(item.get('content', '')).strip()}",
                ]
            )
        )
    return "\n\n".join(lines)


def _load_labeled_examples(replay_file):
    with open(replay_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    examples = []

    for case in cases:
        gold = case.get("gold") or case.get("labels") or {}
        risk = str(gold.get("risk_level", "")).strip().upper()
        decision = str(gold.get("should_work_with_company", "")).strip().upper()
        if not risk or not decision:
            continue

        findings = gold.get("expected_findings", gold.get("key_findings", []))
        if isinstance(findings, list):
            findings = ", ".join(str(x).strip() for x in findings if str(x).strip())
        else:
            findings = str(findings or "").strip()

        results = case.get("retrieved_results", case.get("retrieved_docs", []))
        examples.append(
            dspy.Example(
                company_name=str(case.get("company_name", "")).strip(),
                search_context=_build_context(results),
                risk_level=risk,
                should_work_with_company=decision,
                expected_findings=findings,
            ).with_inputs("company_name", "search_context")
        )

    return examples


def _split_examples(examples, train_ratio, seed):
    if not examples:
        raise ValueError("No labeled replay examples found.")
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    split_idx = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
    trainset = shuffled[:split_idx]
    testset = shuffled[split_idx:]
    return trainset, testset


def run_experiment(
    save_results=True,
    seed=None,
    results_dir=None,
    replay_file=None,
    train_ratio=None,
):
    seed = config.SPLIT_SEED if seed is None else seed
    replay_file = Path(replay_file) if replay_file else DEFAULT_REPLAY_FILE
    train_ratio = config.TRAIN_RATIO if train_ratio is None else train_ratio

    print("=" * 60)
    print("COMPANY LEGAL RISK EXPERIMENT (LARGE REPLAY SPLIT)")
    print("Baseline (Handcrafted Prompt) vs DSPy (Optimized)")
    print(f"Seed: {seed}")
    print("=" * 60)

    print("\n[1/5] Setting up LLM...")
    setup_dspy_lm()

    print("[2/5] Loading labeled replay dataset...")
    all_examples = _load_labeled_examples(replay_file)
    trainset, testset = _split_examples(all_examples, train_ratio=train_ratio, seed=seed)
    print(
        f"      Total: {len(all_examples)} | Train: {len(trainset)} ({train_ratio:.0%}) | Test: {len(testset)} ({1-train_ratio:.0%})"
    )

    print("[3/5] Evaluating baseline model on test split...")
    baseline = BaselineModule()
    baseline_results = detailed_evaluation(baseline, testset, "Baseline")
    print_evaluation_summary(baseline_results)

    print("[4/5] Optimizing DSPy model on train split...")
    student = StudentModule()
    optimized_student = run_two_stage_optimization(
        student_module=student,
        trainset=trainset,
        metric=validate_company_risk_decision,
        bootstrap_config=config.BOOTSTRAP_CONFIG,
        copro_config=config.COPRO_CONFIG,
    )

    print("[5/5] Evaluating optimized model on test split...")
    optimized_results = detailed_evaluation(optimized_student, testset, "DSPy")
    print_evaluation_summary(optimized_results)

    comparison = compare_results(baseline_results, optimized_results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline Score:  {baseline_results.overall_accuracy:.2%}")
    print(f"DSPy Score:      {optimized_results.overall_accuracy:.2%}")
    print(f"Improvement:     {optimized_results.overall_accuracy - baseline_results.overall_accuracy:+.2%}")

    if save_results:
        output_dir = results_dir or config.RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        save_results_json(baseline_results, output_dir / "baseline_results.json")
        save_results_json(optimized_results, output_dir / "dspy_results.json")
        save_results_json(comparison, output_dir / "comparison_results.json")
        optimized_student.save(str(output_dir / "optimized_module.json"))

        baseline_prompt_path = config.PROMPTS_DIR / "baseline.txt"
        save_baseline_prompt(baseline_prompt_path, output_dir)
        extract_optimized_prompt(optimized_student, output_dir)
        generate_prompt_comparison(output_dir)

        split_meta = {
            "replay_file": str(replay_file),
            "seed": seed,
            "train_ratio": train_ratio,
            "test_ratio": 1 - train_ratio,
            "total_examples": len(all_examples),
            "train_size": len(trainset),
            "test_size": len(testset),
        }
        save_results_json(split_meta, output_dir / "split_metadata.json")

    return {
        "baseline_results": baseline_results,
        "optimized_results": optimized_results,
        "comparison": comparison,
        "optimized_module": optimized_student,
    }


def run_multiple_trials(num_runs=None, seeds=None, replay_file=None, train_ratio=None):
    num_runs = num_runs or config.NUM_RUNS
    seeds = seeds or config.RANDOM_SEEDS[:num_runs]
    replay_file = replay_file or DEFAULT_REPLAY_FILE
    train_ratio = config.TRAIN_RATIO if train_ratio is None else train_ratio

    trial_results = []
    base_results_dir = config.RESULTS_DIR / "replay_split_trials"

    for trial_idx, seed in enumerate(seeds[:num_runs]):
        print(f"\n{'=' * 70}")
        print(f"TRIAL {trial_idx + 1}/{num_runs} (seed={seed})")
        print("=" * 70)

        trial_dir = base_results_dir / f"trial_{trial_idx}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        result = run_experiment(
            save_results=True,
            seed=seed,
            results_dir=trial_dir,
            replay_file=replay_file,
            train_ratio=train_ratio,
        )
        trial_results.append(
            {
                "trial_idx": trial_idx,
                "seed": seed,
                "baseline_results": result["baseline_results"],
                "optimized_results": result["optimized_results"],
                "comparison": result["comparison"],
            }
        )

    print(f"\nResults saved to: {base_results_dir}/trial_*/")
    return {
        "trials": trial_results,
        "num_runs": num_runs,
        "seeds": seeds[:num_runs],
        "results_dir": base_results_dir,
    }


def run_live_assessment(company_name, use_optimized=True, module_path=None, return_replay_record=False):
    setup_dspy_lm()

    module = StudentModule() if use_optimized else BaselineModule()
    if use_optimized and module_path:
        module.load(module_path)

    client = SearXNGClient()
    if not client.is_configured():
        raise ValueError("SearXNG is not configured. Set SEARXNG_BASE_URL and optional credentials.")

    evidence, query_log = collect_company_evidence(client, company_name, return_query_log=True)
    context = build_search_context(evidence)

    pred = module(company_name=company_name, search_context=context)
    model_output = {
        "risk_level": pred.risk_level,
        "should_work_with_company": pred.should_work_with_company,
        "summary": pred.summary,
        "key_findings": pred.key_findings,
    }

    result = {
        "company_name": company_name,
        **model_output,
        "evidence_count": len(evidence),
    }

    if return_replay_record:
        record = {
            "timestamp": utc_now_iso(),
            "company_name": company_name,
            "query_log": query_log,
            "retrieved_docs": normalize_retrieved_docs(evidence),
            "model_output": model_output,
        }
        return result, record

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--multi-run":
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else None
        run_multiple_trials(num_runs=num_runs)
    else:
        run_experiment()
