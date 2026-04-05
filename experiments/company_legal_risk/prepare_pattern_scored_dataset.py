"""
Prepare a pattern-based scored company legal risk dataset.

This script keeps the legacy labels but enriches each case with:
- risk dimension scores
- a normalized risk score out of 1.0
- an assessment confidence score
- a derived recommendation
"""

import argparse
import json
from pathlib import Path


DIMENSION_WEIGHTS = {
    "regulatory_enforcement": 1.2,
    "litigation_exposure": 1.0,
    "fraud_corruption": 1.3,
    "data_privacy_cybersecurity": 1.0,
    "labor_employment": 0.7,
    "environmental_safety": 1.0,
    "sanctions_trade": 1.2,
    "governance_ethics": 0.9,
}

RISK_LEVEL_TO_SCORE = {
    "LOW": 0.15,
    "MEDIUM": 0.45,
    "HIGH": 0.75,
    "CRITICAL": 0.95,
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


def _normalize_risk(value):
    text = str(value or "").strip().upper()
    return text if text in RISK_LEVEL_TO_SCORE else ""


def _normalize_decision(value):
    text = str(value or "").strip().upper()
    return text if text in {"YES", "NO"} else ""


def _normalize_findings(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _normalize_severity(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip().upper()
        if text in {"0", "NONE"}:
            return 0
        if text in {"1", "LOW"}:
            return 1
        if text in {"2", "MEDIUM"}:
            return 2
        if text in {"3", "HIGH", "SEVERE", "CRITICAL"}:
            return 3
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= 3 else None


def _normalize_unit_score(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, value))


def _compute_dimension_scores(labels):
    dimensions = labels.get("risk_dimensions", {}) or {}
    scored = {}
    weighted_sum = 0.0
    weight_total = 0.0
    confidence_values = []
    labeled_dimensions = 0

    for name, weight in DIMENSION_WEIGHTS.items():
        raw = dimensions.get(name, {}) or {}
        severity = _normalize_severity(raw.get("severity"))
        confidence = _normalize_unit_score(raw.get("confidence"))
        evidence_refs = raw.get("evidence_refs", []) or []

        if severity is None:
            scored[name] = {
                "severity": None,
                "confidence": confidence,
                "weight": weight,
                "score": None,
                "evidence_refs": evidence_refs,
            }
            continue

        if confidence is None:
            confidence = 1.0

        score = (severity / 3.0) * confidence
        weighted_sum += weight * score
        weight_total += weight
        confidence_values.append(confidence)
        labeled_dimensions += 1

        scored[name] = {
            "severity": severity,
            "confidence": confidence,
            "weight": weight,
            "score": round(score, 4),
            "evidence_refs": evidence_refs,
        }

    pattern_score = (weighted_sum / weight_total) if weight_total else None
    avg_dimension_confidence = (
        sum(confidence_values) / len(confidence_values) if confidence_values else None
    )

    return {
        "dimensions": scored,
        "pattern_risk_score": round(pattern_score, 4) if pattern_score is not None else None,
        "avg_dimension_confidence": round(avg_dimension_confidence, 4)
        if avg_dimension_confidence is not None
        else None,
        "labeled_dimensions": labeled_dimensions,
        "labeled_weight_total": round(weight_total, 4),
    }


def _compute_retrieval_quality(labels, retrieved_results):
    retrieval_quality = labels.get("retrieval_quality", {}) or {}
    explicit_scores = [
        _normalize_unit_score(retrieval_quality.get("source_credibility")),
        _normalize_unit_score(retrieval_quality.get("relevance")),
        _normalize_unit_score(retrieval_quality.get("recency")),
    ]
    explicit_scores = [x for x in explicit_scores if x is not None]

    if explicit_scores:
        return round(sum(explicit_scores) / len(explicit_scores), 4)

    count_heuristic = min(1.0, len(retrieved_results) / 8.0) if retrieved_results else 0.0
    return round(count_heuristic, 4)


def _derive_risk_level(score):
    if score is None:
        return ""
    if score >= 0.85:
        return "CRITICAL"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.30:
        return "MEDIUM"
    return "LOW"


def _dimension_severity(dimension_scores, name):
    dim = (dimension_scores or {}).get(name, {}) or {}
    value = dim.get("severity")
    return value if isinstance(value, int) else 0


def _severe_dimension_count(dimension_scores):
    return sum(
        1
        for dim in (dimension_scores or {}).values()
        if isinstance((dim or {}).get("severity"), int) and (dim or {}).get("severity") >= 3
    )


def _derive_recommendation(score):
    if score is None:
        return ""
    if score >= 0.75:
        return "DO_NOT_WORK"
    if score >= 0.50:
        return "ESCALATE"
    if score >= 0.25:
        return "WORK_WITH_GUARDRAILS"
    return "WORK"


def _fallback_legacy_score(labels):
    risk = _normalize_risk(labels.get("risk_level"))
    decision = _normalize_decision(labels.get("should_work_with_company"))
    base = RISK_LEVEL_TO_SCORE.get(risk)
    if base is None:
        return None

    if decision == "NO":
        base = min(1.0, base + 0.05)
    elif decision == "YES":
        base = max(0.0, base - 0.05)
    return round(base, 4)


def _derive_should_work_with_company(score, dimension_scores):
    if score is None:
        return ""

    fraud_severity = _dimension_severity(dimension_scores, "fraud_corruption")
    regulatory_severity = _dimension_severity(dimension_scores, "regulatory_enforcement")
    sanctions_severity = _dimension_severity(dimension_scores, "sanctions_trade")
    severe_count = _severe_dimension_count(dimension_scores)

    # Hard-stop business rules for severe compliance risk.
    if fraud_severity >= 3:
        return "NO"
    if regulatory_severity >= 3 and sanctions_severity >= 2:
        return "NO"
    if severe_count >= 2:
        return "NO"

    # Score-based default decision.
    if score >= 0.50:
        return "NO"
    return "YES"


def enrich_case(case):
    labels = case.get("labels", {}) or {}
    findings = _normalize_findings(labels.get("key_findings", []))
    retrieved_results = case.get("retrieved_results", case.get("retrieved_docs", [])) or []

    dimension_scores = _compute_dimension_scores(labels)
    retrieval_quality_score = _compute_retrieval_quality(labels, retrieved_results)

    risk_score = dimension_scores["pattern_risk_score"]
    score_source = "pattern_dimensions"
    if risk_score is None:
        risk_score = _fallback_legacy_score(labels)
        score_source = "legacy_labels"

    if risk_score is not None:
        risk_score = round(risk_score, 4)

    avg_dimension_confidence = dimension_scores["avg_dimension_confidence"]
    if avg_dimension_confidence is None:
        assessment_confidence = retrieval_quality_score
    else:
        assessment_confidence = round(
            (0.65 * avg_dimension_confidence) + (0.35 * retrieval_quality_score),
            4,
        )

    derived_risk_level = _derive_risk_level(risk_score)
    derived_recommendation = _derive_recommendation(risk_score)
    derived_should_work = _derive_should_work_with_company(
        risk_score,
        dimension_scores["dimensions"],
    )

    return {
        **case,
        "labels": {
            **labels,
            "key_findings": findings,
        },
        "scoring": {
            "score_source": score_source,
            "risk_score": risk_score,
            "assessment_confidence": assessment_confidence,
            "retrieval_quality_score": retrieval_quality_score,
            "derived_risk_level": derived_risk_level,
            "derived_should_work_with_company": derived_should_work,
            "derived_recommendation": derived_recommendation,
            "dimension_scores": dimension_scores["dimensions"],
            "labeled_dimensions": dimension_scores["labeled_dimensions"],
        },
    }


def build_report(cases):
    score_values = [
        c.get("scoring", {}).get("risk_score")
        for c in cases
        if c.get("scoring", {}).get("risk_score") is not None
    ]
    source_counts = {}
    recommendation_counts = {}
    for case in cases:
        scoring = case.get("scoring", {}) or {}
        source = scoring.get("score_source", "")
        recommendation = scoring.get("derived_recommendation", "")
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
        if recommendation:
            recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1

    average_score = round(sum(score_values) / len(score_values), 4) if score_values else None

    return {
        "case_count": len(cases),
        "scored_case_count": len(score_values),
        "average_risk_score": average_score,
        "score_source_distribution": dict(sorted(source_counts.items())),
        "recommendation_distribution": dict(sorted(recommendation_counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare a pattern-scored company legal risk dataset")
    parser.add_argument("--input-json", required=True, help="Input queue/dataset JSON")
    parser.add_argument("--output-json", required=True, help="Output enriched dataset JSON")
    parser.add_argument("--report-json", required=True, help="Output summary report JSON")
    args = parser.parse_args()

    cases = load_cases(args.input_json)
    enriched = [enrich_case(case) for case in cases]
    report = build_report(enriched)

    save_json(args.output_json, {"cases": enriched})
    save_json(args.report_json, report)

    print(f"Scored cases: {report['scored_case_count']} / {report['case_count']}")
    print(f"Average risk score: {report['average_risk_score']}")
    print(f"Output: {args.output_json}")
    print(f"Report: {args.report_json}")


if __name__ == "__main__":
    main()
