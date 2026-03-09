"""
Evaluate contract-level metadata CSV predictions against gold CSV.
Uses an LLM judge per field so date/string format differences are handled semantically.
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from shared.llm_providers import setup_dspy_lm
from experiments.legal_contract.metrics import llm_semantic_match_score, normalize_text
from experiments.legal_contract.metadata_normalization import (
    governing_law_match_score,
    indemnification_match_score,
    expiration_match_score,
)


DEFAULT_FIELDS = [
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Governing Law",
    "Indemnification",
    "Limitation Of Liability",
    "Non-Compete",
    "Parties",
    "Termination For Convenience",
]


def lexical_match(pred_value: str, gold_value: str) -> float:
    return 1.0 if normalize_text(pred_value) == normalize_text(gold_value) else 0.0


_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _expand_two_digit_year(year: int) -> int:
    return 1900 + year if year >= 40 else 2000 + year


def _extract_year(text: str):
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    if years:
        return int(years[-1])
    return None


def _parse_date_like(text: str):
    t = normalize_text(text)
    if not t or t in {"not found", "[*]", "[●]", "[ ]"}:
        return None

    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", t)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = _expand_two_digit_year(c) if c < 100 else c
        for month, day in ((a, b), (b, a)):
            try:
                return datetime(year, month, day).date()
            except ValueError:
                continue

    month_regex = r"\b(" + "|".join(sorted(_MONTHS.keys(), key=len, reverse=True)) + r")\b"
    month_match = re.search(month_regex, t)
    year = _extract_year(t)
    if month_match and year:
        month = _MONTHS[month_match.group(1)]
        day_match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", t)
        day = int(day_match.group(1)) if day_match else 1
        day = max(1, min(28, day))
        try:
            return datetime(year, month, day).date()
        except ValueError:
            return datetime(year, month, 1).date()

    return None


def _contains_any(text: str, phrases):
    t = normalize_text(text)
    return any(p in t for p in phrases)


def _classify_indemnification(text: str):
    t = normalize_text(text)
    if not t or t == "not found":
        return "not_found"
    if _contains_any(t, ["ip non-challenge clause", "not indemnification", "no indemn", "not indemn", "without indemn", "does not indemn"]):
        return "negative"
    if _contains_any(t, ["indemn", "hold harmless", "defend", "defense obligation", "reimburse"]):
        return "positive"
    return "other"


def _classify_termination_for_convenience(text: str):
    t = normalize_text(text)
    if not t or t == "not found":
        return "not_found"
    convenience_markers = [
        "for convenience",
        "without cause",
        "at any time",
        "yes | notice",
    ]
    if _contains_any(t, convenience_markers):
        return "positive"
    if ("either party" in t and "written notice" in t and not _contains_any(t, ["for cause", "material breach", "default"])):
        return "positive"
    if _contains_any(t, ["may not terminate for convenience", "no termination for convenience"]):
        return "negative"
    return "other"




_GOVERNING_LAW_ALIASES = {
    "state of new york": "new york",
    "new york state": "new york",
    "laws of new york": "new york",
    "commonwealth of massachusetts": "massachusetts",
    "state of delaware": "delaware",
    "laws of delaware": "delaware",
    "england and wales": "england and wales",
    "people's republic of china": "china",
    "prc": "china",
    "hong kong sar": "hong kong",
}


_GOVERNING_LAW_JURISDICTIONS = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware",
    "district of columbia", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah",
    "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming",
    "united states", "usa", "u.s.", "us",
    "england", "wales", "england and wales", "scotland", "ireland", "uk", "united kingdom",
    "germany", "france", "spain", "italy", "switzerland", "netherlands", "belgium", "austria",
    "sweden", "norway", "denmark", "finland", "poland", "portugal", "greece", "ireland",
    "canada", "ontario", "quebec", "british columbia", "alberta",
    "australia", "new zealand", "singapore", "hong kong", "china", "japan", "india",
    "israel", "cayman islands", "bermuda", "luxembourg", "brazil", "mexico"
]


def _normalize_governing_law_text(text: str):
    t = normalize_text(text)
    if not t:
        return ""
    # Remove common non-law context that may appear in predictions.
    t = t.replace("governed by", " ")
    t = t.replace("governing law", " ")
    t = t.replace("laws of", " ")
    t = t.replace("law of", " ")
    t = t.replace("venue", " ")
    t = t.replace("jurisdiction", " ")
    t = t.replace("arbitration", " ")
    t = " ".join(t.split())

    for k, v in _GOVERNING_LAW_ALIASES.items():
        if k in t:
            return v

    hits = [j for j in _GOVERNING_LAW_JURISDICTIONS if re.search(rf"\b{re.escape(j)}\b", t)]
    if not hits:
        return t
    # Prefer longest jurisdiction phrase if multiple hits (e.g., "england and wales" over "england").
    hits.sort(key=len, reverse=True)
    return hits[0]


def _governing_law_match(pred_value: str, gold_value: str):
    p = _normalize_governing_law_text(pred_value)
    g = _normalize_governing_law_text(gold_value)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    if p == g:
        return 1.0
    if p in g or g in p:
        return 1.0
    return 0.0

def _field_specific_match(field: str, pred_value: str, gold_value: str):
    pred_norm = normalize_text(pred_value)
    gold_norm = normalize_text(gold_value)
    if pred_norm == gold_norm:
        return 1.0

    if field == "Expiration Date":
        return expiration_match_score(pred_value, gold_value)

    if field == "Governing Law":
        return governing_law_match_score(pred_value, gold_value)

    if field == "Indemnification":
        return indemnification_match_score(pred_value, gold_value)

    if field == "Termination For Convenience":
        p_cls = _classify_termination_for_convenience(pred_value)
        g_cls = _classify_termination_for_convenience(gold_value)
        if p_cls == g_cls and p_cls in {"positive", "negative", "not_found"}:
            return 1.0
        return lexical_match(pred_value, gold_value)

    return lexical_match(pred_value, gold_value)


def read_csv_by_id(path: Path, id_col: str):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        key = str(row.get(id_col, "")).strip()
        if key:
            out[key] = row
    return out


def evaluate_metadata(gold_csv: Path, pred_csv: Path, id_col: str, fields, use_llm: bool = True):
    gold = read_csv_by_id(gold_csv, id_col)
    pred = read_csv_by_id(pred_csv, id_col)

    keys = sorted(set(gold.keys()) & set(pred.keys()))
    missing_pred = sorted(set(gold.keys()) - set(pred.keys()))

    per_field_scores = {f: [] for f in fields}
    rows = []

    for cid in keys:
        g = gold[cid]
        p = pred[cid]
        row_result = {"contract_id": cid, "fields": {}}

        for field in fields:
            gval = str(g.get(field, "")).strip()
            pval = str(p.get(field, "")).strip()

            if field in {"Indemnification", "Expiration Date", "Termination For Convenience", "Governing Law"}:
                score = _field_specific_match(field, pval, gval)
                method = "field_specific_rules"
                # For Governing Law/Expiration Date, use LLM fallback when rule-based match is inconclusive.
                if field in {"Governing Law", "Expiration Date"} and score < 1.0 and use_llm:
                    llm_score = llm_semantic_match_score(
                        pred_text=pval,
                        gold_candidates=[gval],
                        clause_type=field,
                    )
                    if llm_score is not None:
                        score = max(score, llm_score)
                        method = "field_rules_plus_llm"
            elif use_llm:
                score = llm_semantic_match_score(
                    pred_text=pval,
                    gold_candidates=[gval],
                    clause_type=field,
                )
                if score is None:
                    score = lexical_match(pval, gval)
                    method = "lexical_fallback"
                else:
                    method = "llm_judge"
            else:
                score = lexical_match(pval, gval)
                method = "lexical_only"

            per_field_scores[field].append(score)
            row_result["fields"][field] = {
                "score": score,
                "method": method,
                "pred": pval,
                "gold": gval,
            }

        rows.append(row_result)

    per_field_avg = {
        f: (sum(vals) / len(vals) if vals else 0.0)
        for f, vals in per_field_scores.items()
    }

    flat_scores = [s for vals in per_field_scores.values() for s in vals]
    overall = sum(flat_scores) / len(flat_scores) if flat_scores else 0.0

    return {
        "gold_csv": str(gold_csv),
        "pred_csv": str(pred_csv),
        "id_col": id_col,
        "fields": list(fields),
        "contracts_compared": len(keys),
        "missing_predictions": len(missing_pred),
        "missing_prediction_contract_ids": missing_pred[:50],
        "overall_accuracy": overall,
        "per_field_accuracy": per_field_avg,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate metadata CSV with LLM semantic matching.")
    parser.add_argument("--gold-csv", required=True, help="Gold metadata CSV path")
    parser.add_argument("--pred-csv", required=True, help="Predicted metadata CSV path")
    parser.add_argument("--id-col", default="contract_id", help="ID column name")
    parser.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS, help="Metadata fields to evaluate")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM judge and use lexical-only matching")
    parser.add_argument("--output-json", default="experiments/legal_contract/results/metadata_eval.json", help="Output report JSON")
    args = parser.parse_args()

    if not args.no_llm:
        setup_dspy_lm()

    report = evaluate_metadata(
        gold_csv=Path(args.gold_csv),
        pred_csv=Path(args.pred_csv),
        id_col=args.id_col,
        fields=args.fields,
        use_llm=not args.no_llm,
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Compared contracts: {report['contracts_compared']}")
    print(f"Overall accuracy: {report['overall_accuracy']:.2%}")
    print("Per-field accuracy:")
    for f, s in sorted(report["per_field_accuracy"].items()):
        print(f"  {f:28} {s:.2%}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
