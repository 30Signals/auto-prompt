"""
Export cleaned company legal risk dataset JSON to a flat CSV.
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


def _retrieval_titles(docs, limit=5):
    return " || ".join(
        str(doc.get("title", "")).strip()
        for doc in (docs or [])[:limit]
        if str(doc.get("title", "")).strip()
    )


def _query_summary(query_log):
    parts = []
    for row in query_log or []:
        q = str(row.get("query", "")).strip()
        hits = row.get("hits", "")
        if q:
            parts.append(f"{q} [{hits}]")
    return " || ".join(parts)


def build_row(case):
    labels = case.get("labels", {}) or {}
    scoring = case.get("scoring", {}) or {}
    cleaning = case.get("cleaning", {}) or {}
    docs = case.get("retrieved_results", case.get("retrieved_docs", [])) or []
    dimensions = labels.get("risk_dimensions", {}) or {}
    suspicious_docs = cleaning.get("suspicious_docs", []) or []

    row = {
        "company_name": str(case.get("company_name", "")).strip(),
        "timestamp": case.get("timestamp", ""),
        "annotation_status": case.get("annotation_status", ""),
        "risk_level": labels.get("risk_level", ""),
        "should_work_with_company": labels.get("should_work_with_company", ""),
        "key_findings": _join_list(labels.get("key_findings", [])),
        "retrieved_result_count": len(docs),
        "query_count": len(case.get("query_log", []) or []),
        "query_summary": _query_summary(case.get("query_log", [])),
        "top_retrieved_titles": _retrieval_titles(docs),
        "requires_review": cleaning.get("requires_review", False),
        "cleaning_flags": _join_list(cleaning.get("flags", [])),
        "suspicious_doc_indexes": _join_list([d.get("index", "") for d in suspicious_docs], sep=","),
        "suspicious_doc_titles": " || ".join(
            str(d.get("title", "")).strip() for d in suspicious_docs if str(d.get("title", "")).strip()
        ),
        "risk_score": scoring.get("risk_score", ""),
        "assessment_confidence": scoring.get("assessment_confidence", ""),
        "retrieval_quality_score": scoring.get("retrieval_quality_score", ""),
        "derived_risk_level": scoring.get("derived_risk_level", ""),
        "derived_recommendation": scoring.get("derived_recommendation", ""),
    }

    for dimension in RISK_DIMENSIONS:
        dim = dimensions.get(dimension, {}) or {}
        row[f"{dimension}_severity"] = dim.get("severity", "")
        row[f"{dimension}_confidence"] = dim.get("confidence", "")
        row[f"{dimension}_evidence_refs"] = _join_list(dim.get("evidence_refs", []), sep=",")
        row[f"{dimension}_notes"] = dim.get("notes", "")

    return row


def export_csv(input_json, output_csv):
    cases = load_cases(input_json)
    rows = [build_row(case) for case in cases]
    if not rows:
        raise ValueError("No cases found to export.")

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Export cleaned company legal risk dataset to CSV")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    count = export_csv(args.input_json, args.output_csv)
    print(f"Exported rows: {count}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
