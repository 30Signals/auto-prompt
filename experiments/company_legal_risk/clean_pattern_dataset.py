"""
Clean and triage pattern-based company legal risk datasets.

Outputs:
- cleaned dataset with case-level review flags
- review queue containing only suspicious/noisy cases
- report with aggregate flag counts
"""

import argparse
import json
import re
from pathlib import Path

from .replay_logging import RISK_DIMENSIONS


GENERIC_NOISE_MARKERS = [
    "morning minute",
    "audio briefing",
    "opinion |",
    "listen to",
]

NEGATIVE_CONTEXT_MARKERS = [
    "no evidence of internal compliance or legal issues",
    "actively protecting its intellectual property",
    "proactive protection of intellectual property",
    "suitable partner",
    "low risk",
]

PLAINTIFF_SIDE_MARKERS = [
    "takes legal action against",
    "filed a patent infringement lawsuit against",
    "sued",
    "protect patients",
    "to protect",
]

REGULATORY_AGGRESSIVE_MARKERS = [
    "regulator",
    "regulatory",
    "sec",
    "ftc",
    "doj",
    "fine",
    "penalty",
    "consent order",
    "enforcement",
]

HIGH_SEVERITY_DIMENSIONS = {
    "regulatory_enforcement",
    "litigation_exposure",
    "fraud_corruption",
    "environmental_safety",
    "sanctions_trade",
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


def _norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _doc_text(doc):
    return " ".join(
        [
            str(doc.get("title", "")),
            str(doc.get("content", "")),
            str(doc.get("source", "")),
        ]
    )


def _company_tokens(company_name):
    text = _norm(company_name)
    return [tok for tok in re.split(r"[^a-z0-9]+", text) if len(tok) >= 3]


def _is_doc_relevant(company_name, doc):
    text = _norm(_doc_text(doc))
    tokens = _company_tokens(company_name)
    if not tokens:
        return True
    return any(tok in text for tok in tokens)


def _doc_noise_flags(company_name, doc):
    text = _norm(_doc_text(doc))
    flags = []
    if not _is_doc_relevant(company_name, doc):
        flags.append("doc_company_mismatch")
    if any(marker in text for marker in GENERIC_NOISE_MARKERS):
        flags.append("doc_generic_news_noise")
    return flags


def _case_text(case):
    model_output = case.get("model_output", {}) or {}
    labels = case.get("labels", {}) or {}
    findings = labels.get("key_findings", []) or []
    if isinstance(findings, list):
        findings = " ".join(str(x) for x in findings)
    return _norm(
        " ".join(
            [
                str(model_output.get("summary", "")),
                str(model_output.get("key_findings", "")),
                str(findings),
            ]
        )
    )


def _flag_retrieval_issues(case):
    company_name = case.get("company_name", "")
    docs = case.get("retrieved_results", case.get("retrieved_docs", [])) or []
    flags = []
    suspicious_docs = []

    for idx, doc in enumerate(docs, start=1):
        doc_flags = _doc_noise_flags(company_name, doc)
        if doc_flags:
            suspicious_docs.append(
                {
                    "index": idx,
                    "title": doc.get("title", ""),
                    "flags": doc_flags,
                }
            )
            flags.extend(doc_flags)

    if docs and (len(suspicious_docs) / len(docs)) >= 0.4:
        flags.append("high_noise_retrieval_mix")

    return sorted(set(flags)), suspicious_docs


def _flag_label_issues(case):
    labels = case.get("labels", {}) or {}
    dimensions = labels.get("risk_dimensions", {}) or {}
    case_text = _case_text(case)
    flags = []

    if any(marker in case_text for marker in NEGATIVE_CONTEXT_MARKERS):
        high_dims = [
            name
            for name, dim in dimensions.items()
            if isinstance(dim, dict) and dim.get("severity") in {2, 3}
        ]
        if high_dims:
            flags.append("high_dimensions_despite_low_risk_narrative")

    litigation = dimensions.get("litigation_exposure", {}) or {}
    if litigation.get("severity") in {2, 3} and any(marker in case_text for marker in PLAINTIFF_SIDE_MARKERS):
        flags.append("litigation_may_be_plaintiff_side_not_risk")

    reg = dimensions.get("regulatory_enforcement", {}) or {}
    reg_refs = reg.get("evidence_refs", []) or []
    if reg.get("severity") == 3:
        docs = case.get("retrieved_results", case.get("retrieved_docs", [])) or []
        reg_support = 0
        for idx in reg_refs:
            if 1 <= idx <= len(docs):
                text = _norm(_doc_text(docs[idx - 1]))
                if any(marker in text for marker in REGULATORY_AGGRESSIVE_MARKERS):
                    reg_support += 1
        if reg_support == 0:
            flags.append("regulatory_high_without_strong_support")

    for name in HIGH_SEVERITY_DIMENSIONS:
        dim = dimensions.get(name, {}) or {}
        severity = dim.get("severity")
        refs = dim.get("evidence_refs", []) or []
        if severity == 3 and not refs:
            flags.append(f"{name}_high_without_evidence_refs")

    return sorted(set(flags))


def _mark_case(case):
    retrieval_flags, suspicious_docs = _flag_retrieval_issues(case)
    label_flags = _flag_label_issues(case)
    all_flags = sorted(set(retrieval_flags + label_flags))

    cleaned = dict(case)
    cleaned["cleaning"] = {
        "requires_review": bool(all_flags),
        "flags": all_flags,
        "suspicious_docs": suspicious_docs,
    }
    return cleaned


def _flag_counts(cases):
    counts = {}
    for case in cases:
        for flag in ((case.get("cleaning") or {}).get("flags") or []):
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


def clean_pattern_dataset(input_json, cleaned_output_json, review_queue_json, report_json):
    cases = load_cases(input_json)
    cleaned = [_mark_case(case) for case in cases]
    review_cases = [case for case in cleaned if (case.get("cleaning") or {}).get("requires_review")]

    report = {
        "input_case_count": len(cases),
        "cleaned_case_count": len(cleaned),
        "review_queue_count": len(review_cases),
        "flag_counts": _flag_counts(cleaned),
    }

    save_json(cleaned_output_json, {"cases": cleaned})
    save_json(review_queue_json, {"cases": review_cases})
    save_json(report_json, report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Clean a pattern-based company legal risk dataset")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--cleaned-output-json", required=True)
    parser.add_argument("--review-queue-json", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    report = clean_pattern_dataset(
        input_json=args.input_json,
        cleaned_output_json=args.cleaned_output_json,
        review_queue_json=args.review_queue_json,
        report_json=args.report_json,
    )

    print(f"Input cases: {report['input_case_count']}")
    print(f"Review queue cases: {report['review_queue_count']}")
    print(f"Report: {args.report_json}")


if __name__ == "__main__":
    main()
