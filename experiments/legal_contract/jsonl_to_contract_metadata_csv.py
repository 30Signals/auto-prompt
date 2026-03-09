"""
Convert clause-level reviewed JSONL into contract-level metadata CSV.

Rules:
- contract_id = id prefix before "__"
- clause_type becomes output column
- value uses reviewed_gold only when review_status == "approved",
  otherwise source_gold
- multiple values for same contract/clause are joined by semicolons
- uses annotation fields only (no contract text extraction)
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _dedupe_keep_order(items):
    out = []
    seen = set()
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def convert(input_path: Path, output_path: Path, min_contracts: int = 100):
    by_contract = defaultdict(lambda: defaultdict(list))
    clause_types = set()

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            row_id = str(row.get("id", "")).strip()
            clause_type = str(row.get("clause_type", "")).strip()
            if not row_id or not clause_type:
                continue

            contract_id = row_id.split("__", 1)[0]
            status = str(row.get("review_status", "")).strip().lower()
            selected = row.get("reviewed_gold") if status == "approved" else row.get("source_gold")
            values = _as_list(selected)
            if not values:
                continue

            by_contract[contract_id][clause_type].extend(values)
            clause_types.add(clause_type)

    contract_ids = sorted(by_contract.keys())
    if len(contract_ids) < min_contracts:
        raise ValueError(
            f"Need at least {min_contracts} contracts, found {len(contract_ids)} in {input_path}"
        )

    ordered_clause_types = sorted(clause_types)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["contract_id"] + ordered_clause_types)
        writer.writeheader()

        for contract_id in contract_ids:
            row = {"contract_id": contract_id}
            for ct in ordered_clause_types:
                joined = "; ".join(_dedupe_keep_order(by_contract[contract_id].get(ct, [])))
                row[ct] = joined
            writer.writerow(row)

    return {
        "contracts": len(contract_ids),
        "clause_types": len(ordered_clause_types),
        "input": str(input_path),
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create one-row-per-contract metadata CSV from clause-level JSONL."
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--min-contracts",
        type=int,
        default=100,
        help="Fail if fewer than this many contracts are present (default: 100)",
    )
    args = parser.parse_args()

    stats = convert(Path(args.input), Path(args.output), min_contracts=args.min_contracts)
    print(f"Contracts: {stats['contracts']}")
    print(f"Clause type columns: {stats['clause_types']}")
    print(f"Input: {stats['input']}")
    print(f"Output: {stats['output']}")


if __name__ == "__main__":
    main()
