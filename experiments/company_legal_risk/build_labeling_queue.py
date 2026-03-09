"""
Build analyst labeling queue from replay run logs.
"""

import argparse
from pathlib import Path
from .replay_logging import load_jsonl, build_labeling_queue, save_labeling_queue


def main():
    parser = argparse.ArgumentParser(description="Build labeling queue from replay logs")
    parser.add_argument(
        "--input-log",
        default=str(Path(__file__).parent / "data" / "replay_runs.jsonl"),
        help="Input JSONL log file from live runs",
    )
    parser.add_argument(
        "--output-file",
        default=str(Path(__file__).parent / "data" / "retrieval_replay_labeling_queue.json"),
        help="Output JSON file for analysts",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=500,
        help="Maximum records to export",
    )
    parser.add_argument(
        "--allow-duplicate-companies",
        action="store_true",
        help="Keep multiple records for same company",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input_log)
    queue = build_labeling_queue(
        records,
        max_records=args.max_records,
        dedupe_by_company=not args.allow_duplicate_companies,
    )
    save_labeling_queue(args.output_file, queue)

    print(f"Loaded {len(records)} replay records")
    print(f"Exported {len(queue)} labeling cases")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
