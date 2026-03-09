"""
Dataset cleaning and quality reporting for company legal risk replay data.
"""

import argparse
import json
from pathlib import Path


RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
DECISIONS = ["YES", "NO"]


def _load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload.get("cases", [])
    return payload


def _normalize_risk(value):
    text = str(value or "").strip().upper()
    if text in RISK_LEVELS:
        return text
    return ""


def _normalize_decision(value):
    text = str(value or "").strip().upper()
    if text in DECISIONS:
        return text
    return ""


def _normalize_findings(value):
    if isinstance(value, list):
        findings = [str(x).strip() for x in value if str(x).strip()]
    else:
        findings = [x.strip() for x in str(value or "").split(",") if x.strip()]
    dedup = []
    seen = set()
    for finding in findings:
        key = finding.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(finding)
    return dedup


def _normalize_case(case):
    labels = case.get("labels", {}) or {}
    normalized = dict(case)
    normalized["company_name"] = str(case.get("company_name", "")).strip()
    normalized["query_log"] = case.get("query_log", []) or []
    normalized["retrieved_results"] = case.get("retrieved_results", []) or []
    normalized["labels"] = {
        "risk_level": _normalize_risk(labels.get("risk_level")),
        "should_work_with_company": _normalize_decision(labels.get("should_work_with_company")),
        "key_findings": _normalize_findings(labels.get("key_findings", [])),
    }
    return normalized


def _is_label_complete(case):
    labels = case["labels"]
    return bool(labels["risk_level"] and labels["should_work_with_company"] and labels["key_findings"])


def _quality_score(case):
    return (
        len(case.get("retrieved_results", [])),
        len(case["labels"]["key_findings"]),
        len(case.get("query_log", [])),
    )


def _dedupe_by_company(cases):
    best = {}
    for case in cases:
        company = case.get("company_name", "").strip()
        if not company:
            continue
        key = company.lower()
        if key not in best or _quality_score(case) > _quality_score(best[key]):
            best[key] = case
    return list(best.values())


def _count_by(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def _balance_recommendation(cases):
    risks = [c["labels"]["risk_level"] for c in cases if c["labels"]["risk_level"]]
    counts = _count_by(risks)
    if not counts:
        return {"target_per_class": 0, "needed_by_class": {}}
    target = max(counts.values())
    needed = {level: max(0, target - counts.get(level, 0)) for level in RISK_LEVELS}
    return {"target_per_class": target, "needed_by_class": needed}


def _flag_relabel_cases(case, min_retrieved_results):
    labels = case["labels"]
    flags = []
    if len(case.get("retrieved_results", [])) < min_retrieved_results:
        flags.append("low_retrieval_coverage")
    if len(labels["key_findings"]) < 3 or len(labels["key_findings"]) > 5:
        flags.append("findings_count_outside_3_to_5")
    if any(len(f.strip()) < 12 for f in labels["key_findings"]):
        flags.append("finding_too_short")
    if labels["risk_level"] in {"HIGH", "CRITICAL"} and labels["should_work_with_company"] == "YES":
        flags.append("label_inconsistency_high_risk_yes")
    if labels["risk_level"] == "LOW" and labels["should_work_with_company"] == "NO":
        flags.append("label_inconsistency_low_risk_no")
    return flags


def improve_dataset(
    input_file,
    cleaned_output_file,
    report_output_file,
    relabel_output_file,
    min_retrieved_results=3,
):
    raw_cases = _load_cases(input_file)
    normalized = [_normalize_case(c) for c in raw_cases]

    without_empty_company = [c for c in normalized if c.get("company_name")]
    complete_labels = [c for c in without_empty_company if _is_label_complete(c)]
    deduped = _dedupe_by_company(complete_labels)

    zero_retrieval = [c for c in deduped if len(c.get("retrieved_results", [])) == 0]
    cleaned = [c for c in deduped if len(c.get("retrieved_results", [])) > 0]

    relabel_cases = []
    for case in deduped:
        flags = _flag_relabel_cases(case, min_retrieved_results=min_retrieved_results)
        if flags:
            relabel_cases.append(
                {
                    "company_name": case["company_name"],
                    "timestamp": case.get("timestamp", ""),
                    "query_log": case.get("query_log", []),
                    "retrieved_results": case.get("retrieved_results", []),
                    "model_output": case.get("model_output", {}),
                    "labels": case.get("labels", {}),
                    "relabel_flags": flags,
                }
            )

    cleaned_risks = [c["labels"]["risk_level"] for c in cleaned if c["labels"]["risk_level"]]
    cleaned_decisions = [c["labels"]["should_work_with_company"] for c in cleaned if c["labels"]["should_work_with_company"]]

    report = {
        "input_file": str(input_file),
        "raw_case_count": len(raw_cases),
        "normalized_case_count": len(normalized),
        "removed_empty_company": len(normalized) - len(without_empty_company),
        "removed_incomplete_labels": len(without_empty_company) - len(complete_labels),
        "deduped_case_count": len(deduped),
        "removed_duplicates": len(complete_labels) - len(deduped),
        "removed_zero_retrieval_cases": len(zero_retrieval),
        "cleaned_case_count": len(cleaned),
        "risk_distribution_cleaned": _count_by(cleaned_risks),
        "decision_distribution_cleaned": _count_by(cleaned_decisions),
        "balance_recommendation": _balance_recommendation(cleaned),
        "relabel_queue_count": len(relabel_cases),
        "relabel_flag_counts": _count_by([f for row in relabel_cases for f in row["relabel_flags"]]),
    }

    cleaned_output = {"cases": cleaned}
    relabel_output = {"cases": relabel_cases}

    Path(cleaned_output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(report_output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(relabel_output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(cleaned_output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_output, f, indent=2, ensure_ascii=False)
    with open(report_output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(relabel_output_file, "w", encoding="utf-8") as f:
        json.dump(relabel_output, f, indent=2, ensure_ascii=False)

    return report


def main():
    parser = argparse.ArgumentParser(description="Clean and quality-check company legal risk dataset")
    parser.add_argument(
        "--input-file",
        default="experiments/company_legal_risk/data/retrieval_replay_labeling_queue.json",
    )
    parser.add_argument(
        "--cleaned-output-file",
        default="experiments/company_legal_risk/data/v2/retrieval_replay_labeling_queue_v2.json",
    )
    parser.add_argument(
        "--report-output-file",
        default="experiments/company_legal_risk/data/v2/dataset_quality_report_v2.json",
    )
    parser.add_argument(
        "--relabel-output-file",
        default="experiments/company_legal_risk/data/v2/relabeling_queue_v2.json",
    )
    parser.add_argument(
        "--min-retrieved-results",
        type=int,
        default=3,
        help="Minimum retrieved results required to avoid relabel flag",
    )
    args = parser.parse_args()

    report = improve_dataset(
        input_file=args.input_file,
        cleaned_output_file=args.cleaned_output_file,
        report_output_file=args.report_output_file,
        relabel_output_file=args.relabel_output_file,
        min_retrieved_results=args.min_retrieved_results,
    )

    print("Dataset improvement completed.")
    print(f"Cleaned cases: {report['cleaned_case_count']}")
    print(f"Removed zero retrieval cases: {report['removed_zero_retrieval_cases']}")
    print(f"Relabel queue size: {report['relabel_queue_count']}")
    print(f"Report: {args.report_output_file}")


if __name__ == "__main__":
    main()
