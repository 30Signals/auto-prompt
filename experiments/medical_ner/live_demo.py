"""
Interactive live demo for medical NER disease extraction.
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
    parser = argparse.ArgumentParser(description="Medical NER live demo")
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
    print("MEDICAL NER LIVE DEMO")
    print("Input: abstract_text")
    print("=" * 70)

    setup_dspy_lm()
    module = _load_module(args.baseline, args.module_path)

    while True:
        abstract_text = _read_multiline("\nPaste abstract text (blank + END to exit):")
        if not abstract_text:
            print("Exiting.")
            break

        try:
            pred = module(abstract_text=abstract_text)
            print("\nPredicted diseases:")
            print("-" * 70)
            print(pred.diseases)
            print("-" * 70)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
