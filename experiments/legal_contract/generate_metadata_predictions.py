"""
Generate contract-level metadata predictions CSV from contract IDs.

Reads a metadata test CSV (for contract_id list), fetches contract text from CUAD,
and uses the configured DSPy LM to predict metadata fields.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import dspy
import pandas as pd
from datasets import load_dataset

from shared.llm_providers import setup_dspy_lm


FIELDS = [
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


_GOV_LAW_ALIASES = {
    "state of new york": "New York",
    "new york state": "New York",
    "laws of new york": "New York",
    "state of delaware": "Delaware",
    "laws of delaware": "Delaware",
    "commonwealth of massachusetts": "Massachusetts",
    "people's republic of china": "China",
    "prc": "China",
    "hong kong sar": "Hong Kong",
}

_GOV_LAW_CANDIDATES = [
    "england and wales", "new york", "delaware", "california", "texas", "massachusetts", "florida",
    "illinois", "new jersey", "pennsylvania", "washington", "virginia", "ohio", "michigan",
    "united states", "england", "wales", "scotland", "ireland", "united kingdom", "uk",
    "germany", "france", "spain", "italy", "switzerland", "netherlands", "canada", "australia",
    "singapore", "hong kong", "china", "japan", "india", "israel", "bermuda", "cayman islands"
]


def _canonical_title(text: str) -> str:
    return " ".join(w.capitalize() for w in str(text).split())


def _normalize_governing_law(value: str) -> str:
    v = str(value or "").strip()
    if not v:
        return "NOT FOUND"
    low = " ".join(v.lower().split())
    if low in {"not found", "none", "n/a", "na"}:
        return "NOT FOUND"

    for k, canon in _GOV_LAW_ALIASES.items():
        if k in low:
            return canon

    # Remove common noise.
    noise = [
        "governed by", "governing law", "laws of", "law of", "in accordance with",
        "venue", "jurisdiction", "arbitration", "exclusive", "courts of"
    ]
    cleaned = low
    for n in noise:
        cleaned = cleaned.replace(n, " ")
    cleaned = " ".join(cleaned.split())

    hits = [c for c in _GOV_LAW_CANDIDATES if re.search(rf"\b{re.escape(c)}\b", cleaned)]
    if hits:
        hits.sort(key=len, reverse=True)
        return _canonical_title(hits[0])

    # Fallback: if short, keep as-is; else treat as not found to avoid verbose noise.
    if len(cleaned.split()) <= 3:
        return _canonical_title(cleaned)
    return "NOT FOUND"


def _extract_first_json_object(text: str) -> str:
    s = str(text or "").strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    start = s.find("{")
    if start < 0:
        return ""
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return ""


def _build_contract_text_map(cache_dir: str):
    ds = load_dataset(
        "theatticusproject/cuad-qa",
        revision="refs/convert/parquet",
        cache_dir=cache_dir,
    )
    out = {}
    for split_name in ("train", "test"):
        for item in ds[split_name]:
            rid = str(item.get("id", ""))
            cid = rid.split("__", 1)[0]
            ctx = str(item.get("context", "") or "")
            if cid and ctx and cid not in out:
                out[cid] = ctx
    return out


def _predict_one(lm, contract_text: str, max_chars: int = 12000):
    text = contract_text if len(contract_text) <= max_chars else contract_text[:max_chars]
    prompt = f"""
You are extracting contract metadata fields.
Return ONLY JSON object with exactly these keys:
{json.dumps(FIELDS, ensure_ascii=False)}

Rules:
- Use concise values.
- If not found, return "NOT FOUND".
- Never output section labels alone (e.g., "Section 19").
- For Parties, return primary legal entity names separated by " | ".
- For Agreement/Effective/Expiration Date, return date/term text exactly as stated.
- For Expiration Date, valid outputs include explicit dates or term phrases (e.g., "5-Year Initial Term", "Perpetual", "Until Terminated", "Auto-Renewal").
- For Indemnification, return indemnity obligation text if present; else "NOT FOUND".
- For Governing Law, return only the jurisdiction name (e.g., "Delaware", "New York", "England and Wales").
- For Termination For Convenience, return only no-cause/convenience termination text and notice details.

Contract:
{text}
""".strip()

    raw = lm(prompt)
    response = raw[0] if isinstance(raw, list) and raw else raw
    payload = _extract_first_json_object(response)
    if not payload:
        return {k: "NOT FOUND" for k in FIELDS}

    try:
        obj = json.loads(payload)
    except Exception:
        return {k: "NOT FOUND" for k in FIELDS}

    row = {}
    for k in FIELDS:
        v = obj.get(k, "NOT FOUND")
        if isinstance(v, list):
            v = " | ".join(str(x).strip() for x in v if str(x).strip()) or "NOT FOUND"
        v = str(v).strip() or "NOT FOUND"
        if k == "Governing Law":
            v = _normalize_governing_law(v)
        row[k] = v
    return row


def main():
    parser = argparse.ArgumentParser(description="Generate metadata predictions CSV from contract IDs")
    parser.add_argument("--input-csv", required=True, help="CSV with contract_id column")
    parser.add_argument("--output-csv", required=True, help="Output prediction CSV")
    parser.add_argument(
        "--cache-dir",
        default="experiments/legal_contract/data/cache",
        help="HF datasets cache dir",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Max chars of contract text sent to model",
    )
    args = parser.parse_args()

    setup_dspy_lm()
    lm = dspy.settings.lm

    df = pd.read_csv(args.input_csv)
    id_col = "contract_id" if "contract_id" in df.columns else df.columns[0]
    contract_ids = [str(x).strip() for x in df[id_col].tolist() if str(x).strip()]

    text_map = _build_contract_text_map(args.cache_dir)

    rows = []
    total = len(contract_ids)
    for idx, cid in enumerate(contract_ids, start=1):
        ctx = text_map.get(cid, "")
        if not ctx:
            pred = {k: "NOT FOUND" for k in FIELDS}
        else:
            pred = _predict_one(lm, ctx, max_chars=args.max_chars)
        row = {"contract_id": cid}
        row.update(pred)
        rows.append(row)
        if idx % 5 == 0 or idx == total:
            print(f"Processed {idx}/{total}")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["contract_id"] + FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
