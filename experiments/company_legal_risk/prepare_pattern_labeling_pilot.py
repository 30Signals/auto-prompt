"""
Create a small pilot labeling set for pattern-based company risk annotation.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .replay_logging import build_default_labels


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload.get("cases", [])
    return payload


def save_json(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _normalize_risk(value):
    text = str(value or "").strip().upper()
    return text if text in {"LOW", "MEDIUM", "HIGH", "CRITICAL"} else ""


def _normalize_findings(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _pilot_case(case):
    labels = case.get("labels", {}) or {}
    merged_labels = build_default_labels()
    for key, value in labels.items():
        merged_labels[key] = value
    merged_labels["key_findings"] = _normalize_findings(merged_labels.get("key_findings", []))

    return {
        "company_name": str(case.get("company_name", "")).strip(),
        "timestamp": case.get("timestamp", ""),
        "query_log": case.get("query_log", []) or [],
        "retrieved_results": case.get("retrieved_results", case.get("retrieved_docs", [])) or [],
        "model_output": case.get("model_output", {}) or {},
        "labels": merged_labels,
        "annotation_status": "pending",
    }


def sample_pilot(cases, pilot_size, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    fallback = []

    for case in cases:
        risk = _normalize_risk((case.get("labels", {}) or {}).get("risk_level"))
        prepared = _pilot_case(case)
        if risk:
            grouped[risk].append(prepared)
        else:
            fallback.append(prepared)

    for items in grouped.values():
        rng.shuffle(items)
    rng.shuffle(fallback)

    risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    selected = []
    per_bucket_target = max(1, pilot_size // max(1, len(risk_order)))

    for risk in risk_order:
        selected.extend(grouped[risk][:per_bucket_target])

    seen = {c["company_name"].lower() for c in selected if c["company_name"]}
    remaining = []
    for risk in risk_order:
        remaining.extend(grouped[risk][per_bucket_target:])
    remaining.extend(fallback)
    rng.shuffle(remaining)

    for case in remaining:
        key = case["company_name"].lower()
        if key and key in seen:
            continue
        selected.append(case)
        if key:
            seen.add(key)
        if len(selected) >= pilot_size:
            break

    return selected[:pilot_size]


def build_report(cases, pilot_size, seed):
    risk_counts = defaultdict(int)
    evidence_counts = []

    for case in cases:
        risk = _normalize_risk((case.get("labels", {}) or {}).get("risk_level"))
        if risk:
            risk_counts[risk] += 1
        evidence_counts.append(len(case.get("retrieved_results", [])))

    avg_evidence = round(sum(evidence_counts) / len(evidence_counts), 2) if evidence_counts else 0.0
    return {
        "pilot_size": len(cases),
        "requested_pilot_size": pilot_size,
        "seed": seed,
        "risk_distribution": dict(sorted(risk_counts.items())),
        "average_retrieved_results": avg_evidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare a pilot queue for pattern-based company risk labeling")
    parser.add_argument("--input-json", required=True, help="Input replay queue/dataset JSON")
    parser.add_argument("--output-json", required=True, help="Output pilot JSON")
    parser.add_argument("--report-json", required=True, help="Output pilot report JSON")
    parser.add_argument("--pilot-size", type=int, default=40, help="Number of cases to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cases = load_cases(args.input_json)
    pilot = sample_pilot(cases, pilot_size=args.pilot_size, seed=args.seed)
    report = build_report(pilot, pilot_size=args.pilot_size, seed=args.seed)

    save_json(args.output_json, {"cases": pilot})
    save_json(args.report_json, report)

    print(f"Pilot cases: {report['pilot_size']}")
    print(f"Risk distribution: {report['risk_distribution']}")
    print(f"Output: {args.output_json}")
    print(f"Report: {args.report_json}")


if __name__ == "__main__":
    main()
