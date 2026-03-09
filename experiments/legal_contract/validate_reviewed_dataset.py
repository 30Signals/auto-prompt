"""Validate reviewed legal_contract JSONL before training/evaluation."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from .loader import normalize_clause_type
except ImportError:  # pragma: no cover - allows running as standalone script
    from loader import normalize_clause_type


REQUIRED_KEYS = [
    "id",
    "clause_type",
    "question",
    "contract_text",
    "source_gold",
    "reviewed_gold",
    "review_status",
]

GENERIC_PARTY_LABELS = {
    "company",
    "distributor",
    "seller",
    "buyer",
    "licensor",
    "licensee",
    "customer",
    "vendor",
    "party",
    "parties",
}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _is_short_generic_party_label(text):
    words = [w for w in str(text).strip().lower().replace(".", " ").split() if w]
    if not words or len(words) > 2:
        return False
    return any(w in GENERIC_PARTY_LABELS for w in words)


def validate(path: Path, strict: bool = False, clause_types: Iterable[str] = None):
    rows = []
    parse_errors = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                parse_errors.append((line_no, str(exc)))

    status = Counter()
    clause = Counter()
    missing_required = Counter()

    empty_reviewed_approved = []
    question_clause_mismatches = []
    mixed_not_found = []
    short_generic_parties = []
    weak_governing_law = []
    effective_expiration_overlap = []

    allowed_clause_types = set(clause_types or [])

    by_contract = {}

    for row in rows:
        for key in REQUIRED_KEYS:
            if key not in row:
                missing_required[key] += 1

        row_id = str(row.get("id", "")).strip()
        row_clause = str(row.get("clause_type", "")).strip()
        row_status = str(row.get("review_status", "")).strip().lower()
        question = str(row.get("question", "")).strip()
        reviewed = _as_list(row.get("reviewed_gold"))

        status[row_status] += 1
        clause[row_clause] += 1

        if row_status == "approved" and not reviewed:
            empty_reviewed_approved.append((row_id, row_clause))

        inferred_clause = normalize_clause_type(question)
        if inferred_clause and inferred_clause != row_clause:
            question_clause_mismatches.append((row_id, row_clause, inferred_clause))

        lower_reviewed = [x.lower() for x in reviewed]
        if "not found" in lower_reviewed and len(lower_reviewed) > 1:
            mixed_not_found.append((row_id, row_clause))

        if row_clause == "Parties":
            for span in reviewed:
                if span.lower() == "not found":
                    continue
                if _is_short_generic_party_label(span):
                    short_generic_parties.append((row_id, span))
                    break

        if row_clause == "Governing Law":
            for span in reviewed:
                low = span.lower()
                low_norm = " ".join(low.split())
                if low == "not found":
                    continue
                law_signal = any(
                    t in low_norm
                    for t in ("governing law", "governed by", "laws of", "construed in accordance")
                )
                if not law_signal:
                    weak_governing_law.append((row_id, span[:120]))
                    break

        contract_key = row_id.split("__", 1)[0]
        by_contract.setdefault(contract_key, {})
        by_contract[contract_key][row_clause] = reviewed

    # Effective vs Expiration overlap checks within same contract
    for contract_key, clause_map in by_contract.items():
        eff = [x.strip().lower() for x in clause_map.get("Effective Date", []) if x.strip()]
        exp = [x.strip().lower() for x in clause_map.get("Expiration Date", []) if x.strip()]
        eff_non_nf = [x for x in eff if x != "not found"]
        exp_non_nf = [x for x in exp if x != "not found"]
        if not eff_non_nf or not exp_non_nf:
            continue
        if set(eff_non_nf) & set(exp_non_nf):
            effective_expiration_overlap.append((contract_key, "exact_span_overlap"))

    problems = []
    if parse_errors:
        problems.append("parse_errors")
    if sum(missing_required.values()) > 0:
        problems.append("missing_required_keys")
    if empty_reviewed_approved:
        problems.append("empty_reviewed_gold_on_approved_rows")
    if question_clause_mismatches:
        problems.append("question_clause_type_mismatch")
    if mixed_not_found:
        problems.append("mixed_not_found_and_spans")
    if strict and short_generic_parties:
        problems.append("short_generic_parties_labels")
    if strict and weak_governing_law:
        problems.append("weak_governing_law_labels")
    if strict and effective_expiration_overlap:
        problems.append("effective_expiration_overlap")
    if strict and allowed_clause_types:
        extra = sorted(set(clause.keys()) - allowed_clause_types)
        if extra:
            problems.append("unexpected_clause_types")
    else:
        extra = []

    all_approved = status.get("approved", 0) == len(rows)
    ok = (len(problems) == 0) and all_approved

    return {
        "ok": ok,
        "strict_mode": strict,
        "total_rows": len(rows),
        "all_approved": all_approved,
        "status_counts": dict(status),
        "clause_counts": dict(clause),
        "parse_error_count": len(parse_errors),
        "missing_required_keys": dict(missing_required),
        "empty_reviewed_approved_count": len(empty_reviewed_approved),
        "question_clause_mismatch_count": len(question_clause_mismatches),
        "mixed_not_found_count": len(mixed_not_found),
        "short_generic_parties_count": len(short_generic_parties),
        "weak_governing_law_count": len(weak_governing_law),
        "effective_expiration_overlap_count": len(effective_expiration_overlap),
        "unexpected_clause_types": extra,
        "problems": problems,
        "examples": {
            "parse_errors": parse_errors[:10],
            "empty_reviewed_approved": empty_reviewed_approved[:20],
            "question_clause_mismatch": question_clause_mismatches[:20],
            "mixed_not_found": mixed_not_found[:20],
            "short_generic_parties": short_generic_parties[:20],
            "weak_governing_law": weak_governing_law[:20],
            "effective_expiration_overlap": effective_expiration_overlap[:20],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Validate reviewed legal_contract JSONL")
    parser.add_argument("--file", required=True, help="Reviewed JSONL path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict quality checks (including generic Parties labels)",
    )
    parser.add_argument(
        "--clause-types",
        nargs="+",
        default=None,
        help="Optional allowed clause types list for this run",
    )
    args = parser.parse_args()

    report = validate(
        Path(args.file),
        strict=args.strict,
        clause_types=args.clause_types,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
