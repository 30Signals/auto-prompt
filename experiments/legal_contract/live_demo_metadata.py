"""
Interactive live demo for legal contract metadata extraction.
"""

import argparse
import json
from pathlib import Path

from shared.llm_providers import setup_dspy_lm

from .run_metadata_dspy import (
    BaselineMetadataModule,
    HybridMetadataModule,
    StudentMetadataModule,
    WEAK_OPT_FIELDS,
)


def _read_multiline(prompt: str, end_token: str = "END") -> str:
    print(prompt)
    print(f"Finish input with a line containing only '{end_token}'.")
    lines = []
    while True:
        line = input()
        if line.strip() == end_token:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _load_module(use_baseline: bool, module_path: str):
    baseline = BaselineMetadataModule()
    if use_baseline:
        return baseline

    optimized_student = StudentMetadataModule()
    optimized_student.load(module_path)
    return HybridMetadataModule(baseline, optimized_student, WEAK_OPT_FIELDS)


def _prediction_to_dict(pred) -> dict:
    fields = [
        "agreement_date",
        "effective_date",
        "expiration_date",
        "governing_law",
        "indemnification",
        "limitation_of_liability",
        "non_compete",
        "parties",
        "termination_for_convenience",
    ]
    return {field: getattr(pred, field, "NOT FOUND") for field in fields}


def main():
    parser = argparse.ArgumentParser(description="Legal contract metadata live demo")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use baseline module instead of optimized metadata module",
    )
    parser.add_argument(
        "--module-path",
        default=str(Path(__file__).parent / "results" / "metadata_dspy" / "optimized_module.json"),
        help="Path to optimized DSPy metadata module",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LEGAL CONTRACT METADATA LIVE DEMO")
    print("Input: contract_text")
    print("=" * 70)

    setup_dspy_lm()
    module = _load_module(args.baseline, args.module_path)

    while True:
        contract_text = _read_multiline("\nPaste contract text (blank + END to exit):")
        if not contract_text:
            print("Exiting.")
            break

        try:
            pred = module(contract_text=contract_text)
            print("\nPredicted metadata:")
            print("-" * 70)
            print(json.dumps(_prediction_to_dict(pred), indent=2, ensure_ascii=False))
            print("-" * 70)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
