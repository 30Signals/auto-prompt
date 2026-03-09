"""Shared normalization and matching helpers for legal contract metadata eval."""

import re
from typing import Dict


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


_GOV_LAW_ALIASES = {
    "state of new york": "new york",
    "new york state": "new york",
    "laws of new york": "new york",
    "state of delaware": "delaware",
    "laws of delaware": "delaware",
    "commonwealth of massachusetts": "massachusetts",
    "people's republic of china": "china",
    "prc": "china",
    "hong kong sar": "hong kong",
}

_GOV_LAW_CANDIDATES = [
    "england and wales", "new york", "delaware", "california", "texas", "massachusetts", "florida",
    "illinois", "new jersey", "pennsylvania", "washington", "virginia", "ohio", "michigan",
    "united states", "england", "wales", "scotland", "ireland", "united kingdom", "uk",
    "germany", "france", "spain", "italy", "switzerland", "netherlands", "canada", "australia",
    "singapore", "hong kong", "china", "japan", "india", "israel", "bermuda", "cayman islands",
]


def normalize_governing_law(value: str) -> str:
    v = _norm(value)
    if not v:
        return "not found"
    if v in {"not found", "none", "n/a", "na"}:
        return "not found"

    for k, canon in _GOV_LAW_ALIASES.items():
        if k in v:
            return canon

    noise = [
        "governed by", "governing law", "laws of", "law of", "in accordance with",
        "venue", "jurisdiction", "arbitration", "exclusive", "courts of",
    ]
    cleaned = v
    for n in noise:
        cleaned = cleaned.replace(n, " ")
    cleaned = " ".join(cleaned.split())

    hits = [c for c in _GOV_LAW_CANDIDATES if re.search(rf"\b{re.escape(c)}\b", cleaned)]
    if hits:
        hits.sort(key=len, reverse=True)
        return hits[0]

    if len(cleaned.split()) <= 3:
        return cleaned
    return "not found"


def governing_law_match_score(pred_value: str, gold_value: str) -> float:
    p = normalize_governing_law(pred_value)
    g = normalize_governing_law(gold_value)
    if p == g:
        return 1.0
    if p in g or g in p:
        return 1.0
    return 0.0


def _indemn_features(text: str) -> Dict[str, int]:
    t = _norm(text)
    return {
        "indemn": int("indemn" in t),
        "hold_harmless": int("hold harmless" in t),
        "defend": int("defend" in t or "defense" in t),
        "third_party": int("third party" in t),
        "exceptions": int("willful misconduct" in t or "gross negligence" in t),
        "ip": int("intellectual property" in t or "ip " in t),
    }


def classify_indemnification(text: str) -> str:
    t = _norm(text)
    if not t or t == "not found":
        return "not_found"
    if any(k in t for k in ["ip non-challenge clause", "not indemnification"]):
        return "non_indemn_clause"
    if any(k in t for k in ["no indemn", "without indemn", "does not indemn", "not indemn"]):
        return "no_indemn"
    if any(k in t for k in ["indemn", "hold harmless", "defend", "reimburse"]):
        return "indemn_present"
    return "other"


def indemnification_match_score(pred_value: str, gold_value: str) -> float:
    p_cls = classify_indemnification(pred_value)
    g_cls = classify_indemnification(gold_value)

    if p_cls == g_cls and p_cls in {"indemn_present", "no_indemn", "non_indemn_clause", "not_found"}:
        return 1.0

    if {p_cls, g_cls} <= {"no_indemn", "non_indemn_clause", "not_found"}:
        return 1.0

    if p_cls == "indemn_present" and g_cls == "indemn_present":
        pf = _indemn_features(pred_value)
        gf = _indemn_features(gold_value)
        keys = ["indemn", "hold_harmless", "defend", "third_party", "exceptions", "ip"]
        inter = sum(1 for k in keys if pf[k] and gf[k])
        union = sum(1 for k in keys if pf[k] or gf[k])
        if union == 0:
            return 0.5
        j = inter / union
        if j >= 0.5:
            return 1.0
        if j >= 0.2:
            return 0.7
        return 0.4

    return 0.0


_MONTH_TERM_RE = re.compile(r"(\d{1,2})\s*[- ]?year|(?:\()\s*(\d{1,3})\s*months?\s*(?:\))|([0-9]{1,3})\s*[- ]?month", re.I)
_DATE_RE = re.compile(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", re.I)


def normalize_expiration(value: str) -> str:
    t = _norm(value)
    if not t:
        return "not found"
    if t in {"not found", "none", "n/a", "na"}:
        return "not found"

    if "perpetual" in t:
        return "perpetual"
    if "until terminated" in t:
        return "until terminated"
    if "co-terminous" in t or "coterminous" in t:
        return "coterminous"

    y = re.search(r"(\d{1,2})\s*[- ]?year", t)
    m = re.search(r"(\d{1,3})\s*[- ]?month", t)
    paren_m = re.search(r"\((\d{1,3})\s*months?\)", t)

    years = int(y.group(1)) if y else None
    months = int(m.group(1)) if m else (int(paren_m.group(1)) if paren_m else None)

    if years is None and months is not None and months % 12 == 0:
        years = months // 12

    auto = "auto-renewal" if ("auto-renew" in t or "auto renewal" in t) else ""

    if years is not None:
        base = f"{years}-year"
        return f"{base}|{auto}" if auto else base
    if months is not None:
        base = f"{months}-month"
        return f"{base}|{auto}" if auto else base

    dm = _DATE_RE.search(t)
    if dm:
        return dm.group(0).lower().strip()

    return t


def expiration_match_score(pred_value: str, gold_value: str) -> float:
    p = normalize_expiration(pred_value)
    g = normalize_expiration(gold_value)

    if p == g:
        return 1.0
    if "not found" in {p, g}:
        return 0.0

    # Match if same term duration and renewal signal aligns (or gold omits renewal detail).
    if re.search(r"\d+-year|\d+-month", p) and re.search(r"\d+-year|\d+-month", g):
        p_term = re.search(r"\d+-year|\d+-month", p).group(0)
        g_term = re.search(r"\d+-year|\d+-month", g).group(0)
        if p_term == g_term:
            p_auto = "auto-renewal" in p
            g_auto = "auto-renewal" in g
            if p_auto == g_auto or not g_auto:
                return 1.0
            return 0.8

    # If both are explicit dates, compare year first then exact fallback.
    py = re.search(r"(19\d{2}|20\d{2})", p)
    gy = re.search(r"(19\d{2}|20\d{2})", g)
    if py and gy:
        return 1.0 if py.group(1) == gy.group(1) else 0.0

    if p in g or g in p:
        return 0.8
    return 0.0
