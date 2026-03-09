"""
Convert labeled replay JSONL into training-ready JSON dataset for run.py.
"""

import argparse
import json
from pathlib import Path


RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
DECISIONS = {"YES", "NO"}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_labels(labels):
    labels = labels or {}
    risk = str(labels.get("risk_level", "")).strip().upper()
    decision = str(labels.get("should_work_with_company", "")).strip().upper()

    findings = labels.get("key_findings", [])
    if isinstance(findings, list):
        findings = [str(x).strip() for x in findings if str(x).strip()]
    else:
        findings = [x.strip() for x in str(findings).split(",") if x.strip()]

    if risk not in RISK_LEVELS:
        risk = ""
    if decision not in DECISIONS:
        decision = ""

    return {
        "risk_level": risk,
        "should_work_with_company": decision,
        "key_findings": findings,
    }


def quality_score(case):
    return (
        len(case.get("retrieved_results", [])),
        len(case.get("labels", {}).get("key_findings", [])),
        len(case.get("query_log", [])),
    )


def convert_rows(rows, dedupe_by_company=True, drop_zero_retrieval=False):
    converted = []
    for row in rows:
        case = {
            "company_name": str(row.get("company_name", "")).strip(),
            "timestamp": row.get("timestamp", ""),
            "query_log": row.get("query_log", []) or [],
            "retrieved_results": row.get("retrieved_results", row.get("retrieved_docs", [])) or [],
            "model_output": row.get("model_output", {}) or {},
            "labels": normalize_labels(row.get("labels", {})),
        }
        converted.append(case)

    converted = [c for c in converted if c["company_name"]]
    converted = [
        c
        for c in converted
        if c["labels"]["risk_level"] and c["labels"]["should_work_with_company"] and c["labels"]["key_findings"]
    ]

    if drop_zero_retrieval:
        converted = [c for c in converted if len(c.get("retrieved_results", [])) > 0]

    if dedupe_by_company:
        best = {}
        for case in converted:
            key = case["company_name"].lower()
            if key not in best or quality_score(case) > quality_score(best[key]):
                best[key] = case
        converted = list(best.values())

    converted.sort(key=lambda x: x["company_name"].lower())
    return converted


def build_report(raw_rows, cases):
    risks = {}
    decisions = {}
    zero_retrieval = 0
    for c in cases:
        risk = c["labels"]["risk_level"]
        decision = c["labels"]["should_work_with_company"]
        risks[risk] = risks.get(risk, 0) + 1
        decisions[decision] = decisions.get(decision, 0) + 1
        if len(c.get("retrieved_results", [])) == 0:
            zero_retrieval += 1

    return {
        "raw_rows": len(raw_rows),
        "prepared_cases": len(cases),
        "risk_distribution": dict(sorted(risks.items())),
        "decision_distribution": dict(sorted(decisions.items())),
        "zero_retrieval_cases": zero_retrieval,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare labeled replay JSONL for company legal risk training")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--drop-zero-retrieval", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    cases = convert_rows(
        rows,
        dedupe_by_company=not args.no_dedupe,
        drop_zero_retrieval=args.drop_zero_retrieval,
    )
    report = build_report(rows, cases)

    out_path = Path(args.output_json)
    report_path = Path(args.report_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, indent=2, ensure_ascii=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Prepared cases: {report['prepared_cases']}")
    print(f"Output: {out_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
