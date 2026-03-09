"""
Retrieval Replay Evaluation for Company Legal Risk Agent

Evaluates model decisions on fixed retrieved evidence.
"""

import argparse
import json
import sys
from pathlib import Path

import dspy

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.llm_providers import setup_dspy_lm
from shared.evaluation import save_results_json, compare_results

from .modules import BaselineModule, StudentModule
from .evaluation import detailed_evaluation, print_evaluation_summary

# This script evaluates the company legal risk agent on a set of replay examples, comparing the baseline and optimized modules using the detailed evaluation function and printing summaries of the results. It also supports saving the results to JSON files for further analysis and comparison.

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

# The _load_replay_examples function reads a JSON file containing replay examples, processes each case to extract the company name, search context, risk level, decision, and expected findings, and constructs a list of dspy.Example objects that can be used for evaluation. It supports different input formats and ensures that only fully labeled examples are included in the test set for evaluation.

def _load_replay_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cases = payload.get("cases", []) if isinstance(payload, dict) else payload

    examples = []
    for case in cases:
        # Supports both formats:
        # 1) replay format: case["gold"] with expected_findings
        # 2) labeling queue format: case["labels"] with key_findings
        gold = case.get("gold") or case.get("labels") or {}
        findings = gold.get("expected_findings", gold.get("key_findings", []))
        if isinstance(findings, list):
            findings = ", ".join(str(x).strip() for x in findings if str(x).strip())

        retrieved_results = case.get("retrieved_results", case.get("retrieved_docs", []))
        ex = dspy.Example(
            company_name=str(case.get("company_name", "")).strip(),
            search_context=_build_context(retrieved_results),
            risk_level=str(gold.get("risk_level", "")).strip().upper(),
            should_work_with_company=str(gold.get("should_work_with_company", "")).strip().upper(),
            expected_findings=findings,
        ).with_inputs("company_name", "search_context")
        # Skip unlabeled rows so partially-labeled files still work.
        if ex.risk_level and ex.should_work_with_company:
            examples.append(ex)

    return examples

# The evaluate_replay function takes a replay file and an optional optimized module path, loads the replay examples, evaluates both the baseline and optimized modules on the test set, compares the results, prints summaries, and optionally saves the results to JSON files for further analysis. It returns a dictionary containing the baseline results, optimized results, and comparison for use in downstream analysis or reporting.

def evaluate_replay(replay_file, optimized_module_path=None, output_dir=None):
    setup_dspy_lm()

    testset = _load_replay_examples(replay_file)
    if not testset:
        raise ValueError("No replay cases found.")

    baseline = BaselineModule()
    baseline_results = detailed_evaluation(baseline, testset, "Replay-Baseline")

    student = StudentModule()
    if optimized_module_path:
        student.load(str(optimized_module_path))
    optimized_results = detailed_evaluation(student, testset, "Replay-Optimized")

    comparison = compare_results(baseline_results, optimized_results)

    print_evaluation_summary(baseline_results)
    print_evaluation_summary(optimized_results)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_results_json(baseline_results, out / "replay_baseline_results.json")
        save_results_json(optimized_results, out / "replay_optimized_results.json")
        save_results_json(comparison, out / "replay_comparison_results.json")

    return {
        "baseline_results": baseline_results,
        "optimized_results": optimized_results,
        "comparison": comparison,
    }

# The main function sets up an argument parser for command-line execution, allowing users to specify the replay file, optimized module path, and output directory for results. It then calls the evaluate_replay function with the provided arguments to perform the evaluation and save the results accordingly.

def main():
    parser = argparse.ArgumentParser(description="Evaluate company legal risk agent on retrieval replay data")
    parser.add_argument(
        "--replay-file",
        required=True,
        help="Path to retrieval replay JSON file",
    )
    parser.add_argument(
        "--optimized-module",
        default=str(Path(__file__).parent / "results" / "optimized_module.json"),
        help="Path to optimized DSPy module",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results" / "replay"),
        help="Directory for replay evaluation outputs",
    )
    args = parser.parse_args()

    evaluate_replay(
        replay_file=args.replay_file,
        optimized_module_path=args.optimized_module,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
