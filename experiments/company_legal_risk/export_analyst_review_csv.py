"""
Export an analyst-friendly slim CSV for company legal risk review.
"""

import argparse
import csv
import json
from pathlib import Path

from .replay_logging import RISK_DIMENSIONS


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload.get("cases", [])
    return payload


def _join_list(values, sep=" | "):
    if not values:
        return ""
    if isinstance(values, list):
        return sep.join(str(x).strip() for x in values if str(x).strip())
    return str(values).strip()


def build_row(case):
    labels = case.get("labels", {}) or {}
    scoring = case.get("scoring", {}) or {}
    cleaning = case.get("cleaning", {}) or {}
    dimensions = labels.get("risk_dimensions", {}) or {}

    row = {
        "company_name": str(case.get("company_name", "")).strip(),
        "risk_level": scoring.get("derived_risk_level", labels.get("risk_level", "")),
        "should_work_with_company": scoring.get(
            "derived_should_work_with_company",
            labels.get("should_work_with_company", ""),
        ),
        "risk_score": scoring.get("risk_score", ""),
        "derived_recommendation": scoring.get("derived_recommendation", ""),
        "requires_review": cleaning.get("requires_review", False),
        "cleaning_flags": _join_list(cleaning.get("flags", [])),
    }

    for dimension in RISK_DIMENSIONS:
        row[f"{dimension}_severity"] = (dimensions.get(dimension, {}) or {}).get("severity", "")

    return row


def export_csv(input_json, output_csv):
    cases = load_cases(input_json)
    rows = [build_row(case) for case in cases]
    if not rows:
        raise ValueError("No cases found to export.")

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Export slim analyst review CSV")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    count = export_csv(args.input_json, args.output_csv)
    print(f"Exported rows: {count}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
