"""
Replay logging utilities for company legal risk runs.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


RISK_DIMENSIONS = [
    "regulatory_enforcement",
    "litigation_exposure",
    "fraud_corruption",
    "data_privacy_cybersecurity",
    "labor_employment",
    "environmental_safety",
    "sanctions_trade",
    "governance_ethics",
]


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_jsonl_record(filepath, record):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(filepath):
    path = Path(filepath)
    if not path.exists():
        return []

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


def build_empty_risk_dimensions():
    return {
        name: {
            "severity": None,
            "confidence": None,
            "evidence_refs": [],
            "notes": "",
        }
        for name in RISK_DIMENSIONS
    }


def build_default_labels():
    return {
        "risk_level": "",
        "should_work_with_company": "",
        "key_findings": [],
        "risk_dimensions": build_empty_risk_dimensions(),
        "retrieval_quality": {
            "source_credibility": None,
            "relevance": None,
            "recency": None,
        },
        "reviewer_notes": "",
    }


def build_labeling_queue(records, max_records=None, dedupe_by_company=True):
    seen = set()
    queue = []

    for rec in records:
        company = str(rec.get("company_name", "")).strip()
        if not company:
            continue

        key = company.lower()
        if dedupe_by_company and key in seen:
            continue
        seen.add(key)

        queue.append(
            {
                "company_name": company,
                "timestamp": rec.get("timestamp", ""),
                "query_log": rec.get("query_log", []),
                "retrieved_results": rec.get("retrieved_docs", []),
                "model_output": rec.get("model_output", {}),
                "labels": build_default_labels(),
            }
        )

        if max_records is not None and len(queue) >= max_records:
            break

    return queue


def save_labeling_queue(filepath, queue):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cases": queue}, f, indent=2, ensure_ascii=False)
