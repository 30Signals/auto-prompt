"""
Heuristically prefill pattern-based risk dimensions from retrieved evidence.

This is a draft assistant for human reviewers, not a replacement for annotation.
"""

import argparse
import json
from pathlib import Path

from .replay_logging import RISK_DIMENSIONS, build_default_labels


DIMENSION_RULES = {
    "regulatory_enforcement": {
        "keywords": [
            "sec",
            "ftc",
            "doj",
            "department of justice",
            "regulator",
            "regulatory",
            "consent order",
            "enforcement",
            "probe",
            "investigation",
            "fine",
            "penalty",
            "sanctioned",
            "settle sec charges",
            "antitrust",
        ],
        "high_keywords": [
            "record fine",
            "billions in penalties",
            "major fine",
            "criminal probe",
            "ongoing government investigation",
        ],
    },
    "litigation_exposure": {
        "keywords": [
            "lawsuit",
            "litigation",
            "class action",
            "sued",
            "legal action",
            "settlement",
            "settled",
            "claim",
            "complaint",
        ],
        "high_keywords": [
            "mass tort",
            "multiple lawsuits",
            "class action lawsuit",
            "billions in penalties",
            "wrongful death",
        ],
    },
    "fraud_corruption": {
        "keywords": [
            "fraud",
            "fcpa",
            "bribery",
            "corruption",
            "kickback",
            "embezzlement",
            "money laundering",
            "ponzi",
            "misconduct",
            "false claims",
        ],
        "high_keywords": [
            "criminal scheme",
            "indicted",
            "guilty plea",
            "fraud investigation",
            "systemic misconduct",
        ],
    },
    "data_privacy_cybersecurity": {
        "keywords": [
            "data breach",
            "privacy",
            "cyber",
            "cybersecurity",
            "security incident",
            "hack",
            "hiring ai",
            "spy on users",
            "consumer data",
        ],
        "high_keywords": [
            "major breach",
            "national security concerns",
            "surveillance",
            "privacy settlement",
        ],
    },
    "labor_employment": {
        "keywords": [
            "labor law",
            "employment",
            "worker",
            "disabled passengers",
            "warehouse quotas",
            "workplace",
            "wage",
            "discrimination",
            "osha",
            "union",
            "employee",
        ],
        "high_keywords": [
            "repeat safety violations",
            "systemic labor violations",
            "collective action",
        ],
    },
    "environmental_safety": {
        "keywords": [
            "hazardous waste",
            "environment",
            "emissions",
            "pollution",
            "safety violation",
            "epa",
            "toxic",
            "dam disaster",
            "maintenance failures",
        ],
        "high_keywords": [
            "environmental negligence",
            "billions in penalties",
            "fatal",
            "systemic misconduct involving emissions",
        ],
    },
    "sanctions_trade": {
        "keywords": [
            "sanctions",
            "export control",
            "trade restriction",
            "customs fraud",
            "tariff",
            "cross-border compliance",
        ],
        "high_keywords": [
            "sanctions violation",
            "trade enforcement",
        ],
    },
    "governance_ethics": {
        "keywords": [
            "governance",
            "ethics",
            "board",
            "compliance weaknesses",
            "oversight",
            "misleading",
            "deceptive practices",
            "price label violations",
            "investor trust",
        ],
        "high_keywords": [
            "systemic governance issues",
            "recklessly indifferent",
            "systemic compliance weaknesses",
        ],
    },
}


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


def _normalize_findings(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _merge_labels(labels):
    merged = build_default_labels()
    labels = labels or {}
    for key, value in labels.items():
        if key == "risk_dimensions" and isinstance(value, dict):
            merged["risk_dimensions"].update(value)
        elif key == "retrieval_quality" and isinstance(value, dict):
            merged["retrieval_quality"].update(value)
        else:
            merged[key] = value
    merged["key_findings"] = _normalize_findings(merged.get("key_findings", []))
    return merged


def _doc_text(doc):
    return " ".join(
        [
            str(doc.get("title", "")),
            str(doc.get("content", "")),
            str(doc.get("source", "")),
        ]
    ).lower()


def _severity_from_hits(hit_count, high_hit_count):
    if high_hit_count >= 1 or hit_count >= 4:
        return 3
    if hit_count >= 2:
        return 2
    if hit_count >= 1:
        return 1
    return 0


def _confidence_from_refs(ref_count, hit_count):
    base = 0.45 + min(0.35, 0.12 * ref_count) + min(0.15, 0.03 * hit_count)
    return round(min(0.95, base), 2)


def _build_notes(dimension, refs, severity):
    if severity == 0:
        return "No strong signal found in retrieved evidence; reviewer should confirm."
    ref_list = ", ".join(str(x) for x in refs[:4])
    return f"Heuristic prefill from evidence items {ref_list}."


def prefill_case(case, overwrite=False):
    labels = _merge_labels(case.get("labels"))
    docs = case.get("retrieved_results", case.get("retrieved_docs", [])) or []
    findings_text = " ".join(labels.get("key_findings", [])).lower()
    model_text = " ".join(
        [
            str((case.get("model_output") or {}).get("summary", "")),
            str((case.get("model_output") or {}).get("key_findings", "")),
        ]
    ).lower()

    combined_case_text = " ".join([findings_text, model_text])

    for dimension in RISK_DIMENSIONS:
        current = labels["risk_dimensions"].get(dimension, {}) or {}
        if not overwrite and current.get("severity") is not None:
            continue

        rule = DIMENSION_RULES[dimension]
        refs = []
        hit_count = 0
        high_hit_count = 0

        for idx, doc in enumerate(docs, start=1):
            text = _doc_text(doc)
            doc_hits = sum(1 for kw in rule["keywords"] if kw in text)
            doc_high_hits = sum(1 for kw in rule["high_keywords"] if kw in text)
            if doc_hits or doc_high_hits:
                refs.append(idx)
                hit_count += doc_hits
                high_hit_count += doc_high_hits

        hit_count += sum(1 for kw in rule["keywords"] if kw in combined_case_text)
        high_hit_count += sum(1 for kw in rule["high_keywords"] if kw in combined_case_text)

        severity = _severity_from_hits(hit_count, high_hit_count)
        confidence = _confidence_from_refs(len(refs), hit_count)

        labels["risk_dimensions"][dimension] = {
            "severity": severity,
            "confidence": confidence,
            "evidence_refs": refs[:5],
            "notes": _build_notes(dimension, refs, severity),
        }

    return {
        **case,
        "labels": labels,
        "annotation_status": "prefilled",
    }


def build_report(cases):
    severity_counts = {
        name: {"0": 0, "1": 0, "2": 0, "3": 0}
        for name in RISK_DIMENSIONS
    }
    for case in cases:
        dimensions = ((case.get("labels") or {}).get("risk_dimensions") or {})
        for name in RISK_DIMENSIONS:
            severity = dimensions.get(name, {}).get("severity")
            if severity in {0, 1, 2, 3}:
                severity_counts[name][str(severity)] += 1

    return {
        "case_count": len(cases),
        "per_dimension_severity_counts": severity_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Prefill company legal risk pattern dimensions")
    parser.add_argument("--input-json", required=True, help="Input JSON")
    parser.add_argument("--output-json", required=True, help="Output JSON")
    parser.add_argument("--report-json", required=True, help="Output report JSON")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing severities")
    args = parser.parse_args()

    cases = load_cases(args.input_json)
    enriched = [prefill_case(case, overwrite=args.overwrite) for case in cases]
    report = build_report(enriched)

    save_json(args.output_json, {"cases": enriched})
    save_json(args.report_json, report)

    print(f"Prefilled cases: {report['case_count']}")
    print(f"Output: {args.output_json}")
    print(f"Report: {args.report_json}")


if __name__ == "__main__":
    main()
