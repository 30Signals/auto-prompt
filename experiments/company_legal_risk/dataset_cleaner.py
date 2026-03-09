"""
Dataset Cleaner for Company Legal Risk JSONL

Fixes three problems in the raw replay_runs labeled dataset:

  1. CRASH DUPLICATES  — same company + same timestamp = a retry after a crash.
                         Keep one, discard the rest.

  2. INTENTIONAL RE-RUNS — same company, different timestamps = the live demo
                            was run again on purpose. Keep all but stamp each
                            with a unique run_id so they're distinguishable.

  3. ZERO-DOC COMPANIES — 21 companies that returned no SearXNG results were
                           auto-labeled LOW risk. That's wrong. Replace their
                           risk_level with UNKNOWN and set requires_review=True
                           so they're not silently used as training data.

Usage:
    python -m company_legal_risk.dataset_cleaner \\
        --input  data/replay_runs_labeled.jsonl \\
        --output data/replay_runs_clean.jsonl
"""

import argparse
import json
import uuid
from collections import defaultdict
from pathlib import Path


ZERO_DOC_RISK_REPLACEMENT = "UNKNOWN"


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(path: str, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_dataset(input_path: str, output_path: str) -> dict:
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records from {input_path}")

    # ------------------------------------------------------------------ #
    # Step 1 — Group by (company_name, timestamp) to find crash dupes     #
    # ------------------------------------------------------------------ #
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        key = (r.get("company_name", "").strip(), r.get("timestamp", "").strip())
        groups[key].append(r)

    crash_dupes_removed = 0
    deduped: list[dict] = []
    for (company, ts), group in groups.items():
        if len(group) > 1:
            # Keep the entry with the most retrieved_docs; tie-break on order
            group.sort(key=lambda r: len(r.get("retrieved_docs", [])), reverse=True)
            crash_dupes_removed += len(group) - 1
            print(f"  [crash dupe] {company} @ {ts} — kept 1, removed {len(group)-1}")
        deduped.append(group[0])

    # ------------------------------------------------------------------ #
    # Step 2 — Intentional re-runs: same company, different timestamps    #
    #           Assign run_id to every record; note which are re-runs     #
    # ------------------------------------------------------------------ #
    by_company: dict[str, list] = defaultdict(list)
    for r in deduped:
        by_company[r.get("company_name", "").strip()].append(r)

    intentional_reruns = 0
    for company, entries in by_company.items():
        for i, entry in enumerate(entries):
            # Always assign a run_id if not already present
            if not entry.get("run_id"):
                entry["run_id"] = str(uuid.uuid4())
            if len(entries) > 1:
                entry["is_rerun"] = True
                entry["rerun_index"] = i
                if i > 0:
                    intentional_reruns += 1
                    print(f"  [re-run] {company} run #{i+1} @ {entry.get('timestamp','?')}")
            else:
                entry["is_rerun"] = False

    # ------------------------------------------------------------------ #
    # Step 3 — Zero-doc companies: flag and replace risk level            #
    # ------------------------------------------------------------------ #
    zero_doc_flagged = 0
    for r in deduped:
        doc_count = len(r.get("retrieved_docs", []))
        if doc_count == 0:
            r["requires_review"] = True
            r["review_reason"] = (
                "No documents were retrieved for this company. "
                "Risk label was assigned without evidence and must be verified."
            )
            # Replace the auto-assigned LOW with UNKNOWN
            if r.get("labels"):
                original_risk = r["labels"].get("risk_level", "")
                r["labels"]["risk_level"] = ZERO_DOC_RISK_REPLACEMENT
                r["labels"]["original_risk_level"] = original_risk
                r["labels"]["key_findings"] = ["No search evidence retrieved — requires manual review"]
            zero_doc_flagged += 1
            print(f"  [zero-doc] {r.get('company_name','?')} — risk set to UNKNOWN, flagged for review")
        else:
            r.setdefault("requires_review", False)

    # ------------------------------------------------------------------ #
    # Save and report                                                      #
    # ------------------------------------------------------------------ #
    save_jsonl(output_path, deduped)

    stats = {
        "input_records": len(records),
        "output_records": len(deduped),
        "crash_duplicates_removed": crash_dupes_removed,
        "intentional_reruns_flagged": intentional_reruns,
        "zero_doc_companies_flagged": zero_doc_flagged,
    }

    print("-" * 60)
    print(f"Input records:               {stats['input_records']}")
    print(f"Output records:              {stats['output_records']}")
    print(f"Crash duplicates removed:    {stats['crash_duplicates_removed']}")
    print(f"Intentional re-runs flagged: {stats['intentional_reruns_flagged']}")
    print(f"Zero-doc companies flagged:  {stats['zero_doc_companies_flagged']}")
    print(f"Saved → {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean and deduplicate labeled replay JSONL")
    parser.add_argument("--input",  required=True, help="Input labeled JSONL (e.g. replay_runs_labeled.jsonl)")
    parser.add_argument("--output", required=True, help="Output cleaned JSONL (e.g. replay_runs_clean.jsonl)")
    args = parser.parse_args()

    clean_dataset(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
