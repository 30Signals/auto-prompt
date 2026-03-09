"""
Auto-prefill draft labels in retrieval replay labeling queue from model output.

This is meant to accelerate human annotation. Analysts must review and correct.
"""

import argparse
import json
from pathlib import Path


def _normalize_risk(value):
    text = str(value or "").strip().upper()
    if text in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        return text
    if "CRIT" in text:
        return "CRITICAL"
    if "HIGH" in text:
        return "HIGH"
    if "MED" in text:
        return "MEDIUM"
    if text:
        return "LOW"
    return ""


def _normalize_decision(value):
    text = str(value or "").strip().upper()
    if text in {"YES", "Y", "WORK", "APPROVE"}:
        return "YES"
    if text in {"NO", "N", "REJECT", "DO NOT WORK"}:
        return "NO"
    return ""


def _extract_findings_from_model_output(model_output):
    raw = model_output.get("key_findings", "") if isinstance(model_output, dict) else ""

    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
    else:
        text = str(raw or "").replace(";", ",").replace("\n", ",")
        items = [x.strip(" -") for x in text.split(",") if x.strip()]

    # Keep at most 5 concise findings
    return items[:5]


def prefill_labels(filepath, force=False):
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cases = payload.get("cases", []) if isinstance(payload, dict) else payload

    updated = 0
    skipped = 0

    for case in cases:
        labels = case.setdefault("labels", {})
        model_output = case.get("model_output", {})

        has_risk = bool(str(labels.get("risk_level", "")).strip())
        has_decision = bool(str(labels.get("should_work_with_company", "")).strip())
        has_findings = bool(labels.get("key_findings"))

        if not force and has_risk and has_decision and has_findings:
            skipped += 1
            continue

        if force or not has_risk:
            labels["risk_level"] = _normalize_risk(model_output.get("risk_level", ""))

        if force or not has_decision:
            labels["should_work_with_company"] = _normalize_decision(
                model_output.get("should_work_with_company", "")
            )

        if force or not has_findings:
            labels["key_findings"] = _extract_findings_from_model_output(model_output)

        updated += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, indent=2, ensure_ascii=False)

    return {"updated": updated, "skipped": skipped, "total": len(cases)}


def main():
    parser = argparse.ArgumentParser(description="Prefill draft labels from model output")
    parser.add_argument(
        "--file",
        default="experiments/company_legal_risk/data/retrieval_replay_labeling_queue.json",
        help="Labeling queue JSON path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing labels too",
    )
    args = parser.parse_args()

    stats = prefill_labels(args.file, force=args.force)
    print(f"Total cases: {stats['total']}")
    print(f"Updated: {stats['updated']}")
    print(f"Skipped: {stats['skipped']}")
    print("Note: These are draft labels; analysts should review.")


if __name__ == "__main__":
    main()
