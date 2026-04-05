"""
Evaluate pattern-based risk-dimension labels and annotation completeness.

Supports:
- completeness analysis for human-labeled pilot files
- agreement analysis when predicted dimensions are present
"""

import argparse
import json
from pathlib import Path

from .pattern_metrics import (
    confidence_alignment_score,
    iter_dimension_pairs,
    normalize_severity,
    severity_distance_score,
    severity_exact_score,
)
from .replay_logging import RISK_DIMENSIONS


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


def _predicted_dimensions(case, predicted_field):
    if predicted_field == "labels":
        return ((case.get("labels", {}) or {}).get("risk_dimensions", {}) or {})
    predicted = case.get(predicted_field, {}) or {}
    if isinstance(predicted, dict) and "risk_dimensions" in predicted:
        return predicted.get("risk_dimensions", {}) or {}
    return predicted if isinstance(predicted, dict) else {}


def completeness_report(cases):
    dimension_presence = {name: 0 for name in RISK_DIMENSIONS}
    fully_labeled_cases = 0
    severity_histograms = {name: {"0": 0, "1": 0, "2": 0, "3": 0} for name in RISK_DIMENSIONS}
    incomplete_cases = []

    for idx, case in enumerate(cases):
        labels = case.get("labels", {}) or {}
        dimensions = labels.get("risk_dimensions", {}) or {}
        present_count = 0

        for name in RISK_DIMENSIONS:
            raw = dimensions.get(name, {}) or {}
            severity = normalize_severity(raw.get("severity"))
            if severity is not None:
                dimension_presence[name] += 1
                severity_histograms[name][str(severity)] += 1
                present_count += 1

        if present_count == len(RISK_DIMENSIONS):
            fully_labeled_cases += 1
        else:
            incomplete_cases.append(
                {
                    "index": idx,
                    "company_name": case.get("company_name", ""),
                    "labeled_dimension_count": present_count,
                }
            )

    total = len(cases) or 1
    coverage = {
        name: {
            "count": dimension_presence[name],
            "coverage": round(dimension_presence[name] / total, 4),
            "severity_histogram": severity_histograms[name],
        }
        for name in RISK_DIMENSIONS
    }

    return {
        "case_count": len(cases),
        "fully_labeled_cases": fully_labeled_cases,
        "full_label_coverage": round(fully_labeled_cases / total, 4),
        "per_dimension_coverage": coverage,
        "incomplete_case_examples": incomplete_cases[:20],
    }


def agreement_report(cases, predicted_field):
    per_dimension = {
        name: {
            "count": 0,
            "severity_exact": [],
            "severity_distance": [],
            "confidence_alignment": [],
        }
        for name in RISK_DIMENSIONS
    }
    rows = []

    for idx, case in enumerate(cases):
        gold_dimensions = ((case.get("labels", {}) or {}).get("risk_dimensions", {}) or {})
        pred_dimensions = _predicted_dimensions(case, predicted_field)

        row = {
            "index": idx,
            "company_name": case.get("company_name", ""),
            "dimension_scores": {},
        }

        for name, pred_dim, gold_dim in iter_dimension_pairs(pred_dimensions, gold_dimensions):
            exact = severity_exact_score(pred_dim.get("severity"), gold_dim.get("severity"))
            distance = severity_distance_score(pred_dim.get("severity"), gold_dim.get("severity"))
            conf = confidence_alignment_score(pred_dim.get("confidence"), gold_dim.get("confidence"))

            row["dimension_scores"][name] = {
                "severity_exact": exact,
                "severity_distance": distance,
                "confidence_alignment": conf,
            }

            if distance is None:
                continue

            per_dimension[name]["count"] += 1
            per_dimension[name]["severity_exact"].append(exact if exact is not None else 0.0)
            per_dimension[name]["severity_distance"].append(distance)
            if conf is not None:
                per_dimension[name]["confidence_alignment"].append(conf)

        rows.append(row)

    summary = {}
    weighted_scores = []
    for name, stats in per_dimension.items():
        count = stats["count"]
        exact_avg = sum(stats["severity_exact"]) / count if count else None
        distance_avg = sum(stats["severity_distance"]) / count if count else None
        conf_vals = stats["confidence_alignment"]
        conf_avg = (sum(conf_vals) / len(conf_vals)) if conf_vals else None
        if distance_avg is not None:
            weighted_scores.append(distance_avg)

        summary[name] = {
            "count": count,
            "severity_exact_accuracy": round(exact_avg, 4) if exact_avg is not None else None,
            "severity_distance_score": round(distance_avg, 4) if distance_avg is not None else None,
            "confidence_alignment_score": round(conf_avg, 4) if conf_avg is not None else None,
        }

    overall = round(sum(weighted_scores) / len(weighted_scores), 4) if weighted_scores else None
    return {
        "predicted_field": predicted_field,
        "overall_dimension_distance_score": overall,
        "per_dimension": summary,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pattern-based company risk labels")
    parser.add_argument("--input-json", required=True, help="Input JSON with labeled cases")
    parser.add_argument("--output-json", required=True, help="Output JSON report")
    parser.add_argument(
        "--predicted-field",
        default="",
        help="Optional field containing predicted risk dimensions for agreement analysis",
    )
    args = parser.parse_args()

    cases = load_cases(args.input_json)
    report = {
        "completeness": completeness_report(cases),
    }
    if args.predicted_field:
        report["agreement"] = agreement_report(cases, args.predicted_field)

    save_json(args.output_json, report)

    print(f"Cases analyzed: {report['completeness']['case_count']}")
    print(f"Full label coverage: {report['completeness']['full_label_coverage']:.2%}")
    if args.predicted_field and report.get("agreement"):
        print(
            "Overall dimension distance score: "
            f"{report['agreement'].get('overall_dimension_distance_score')}"
        )
    print(f"Output: {args.output_json}")


if __name__ == "__main__":
    main()
