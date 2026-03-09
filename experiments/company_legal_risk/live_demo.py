"""
Interactive Live Demo for Company Legal Risk Agent
"""

import argparse
import json
import uuid
from pathlib import Path

from .run import run_live_assessment
from .replay_logging import append_jsonl_record


def main():
    parser = argparse.ArgumentParser(description="Company legal risk live assessment demo")

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
    parser.add_argument(
        "--log-file",
        default=str(Path(__file__).parent / "data" / "replay_runs.jsonl"),
        help="JSONL path to append replay records",
    )
    args = parser.parse_args()

    use_optimized = not args.baseline

    print("=" * 70)
    print("COMPANY LEGAL RISK LIVE DEMO")
    print("Type company name, or press Enter to exit")
    print("=" * 70)

    while True:
        company_name = input("\nCompany Name: ").strip()
        if not company_name:
            print("Exiting.")
            break

        try:
            result, replay_record = run_live_assessment(
                company_name=company_name,
                use_optimized=use_optimized,
                module_path=args.module_path,
                return_replay_record=True,
            )

            # Attach a unique run ID so multiple runs of the same company
            # are distinguishable in the JSONL log.
            replay_record["run_id"] = str(uuid.uuid4())

            append_jsonl_record(args.log_file, replay_record)

            print("\n" + "-" * 70)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"Logged replay record to: {args.log_file}")
            print("-" * 70)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
