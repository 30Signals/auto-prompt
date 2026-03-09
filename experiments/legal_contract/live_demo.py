"""
Interactive live demo for legal contract clause extraction.
"""

import argparse
from pathlib import Path

from shared.llm_providers import setup_dspy_lm

from .modules import BaselineModule, StudentModule


def _read_multiline(prompt, end_token="END"):
    print(prompt)
    print(f"Finish input with a line containing only '{end_token}'.")
    lines = []
    while True:
        line = input()
        if line.strip() == end_token:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _load_module(use_baseline, module_path):
    if use_baseline:
        return BaselineModule()
    module = StudentModule()
    module.load(module_path)
    return module


def main():
    parser = argparse.ArgumentParser(description="Legal contract live demo")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use baseline module instead of optimized module",
    )
    parser.add_argument(
        "--module-path",
        default=str(Path(__file__).parent / "results" / "optimized_module.json"),
        help="Path to optimized DSPy module",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LEGAL CONTRACT LIVE DEMO")
    print("Inputs: clause_type + contract_text")
    print("=" * 70)

    setup_dspy_lm()
    module = _load_module(args.baseline, args.module_path)

    while True:
        clause_type = input("\nClause type (blank to exit): ").strip()
        if not clause_type:
            print("Exiting.")
            break

        contract_text = _read_multiline("\nPaste contract text:")
        if not contract_text:
            print("Empty contract text, skipping.")
            continue

        try:
            pred = module(contract_text=contract_text, clause_type=clause_type)
            print("\nPredicted clause_text:")
            print("-" * 70)
            print(pred.clause_text)
            print("-" * 70)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
