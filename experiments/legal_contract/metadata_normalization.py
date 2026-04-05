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

_US_STATE_LAWS = {
    "new york", "delaware", "california", "texas", "massachusetts", "florida",
    "illinois", "new jersey", "pennsylvania", "washington", "virginia", "ohio", "michigan",
    "kansas", "nevada", "minnesota", "arkansas", "missouri", "georgia", "maryland",
    "connecticut", "colorado", "utah", "tennessee", "north carolina", "south carolina",
    "louisiana", "alabama", "indiana", "kentucky", "wisconsin", "arizona", "oregon",
}


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

    # Treat a US state as a full match when the cleaned gold is normalized to United States.
    if g == "united states" and p in _US_STATE_LAWS:
        return 1.0
    if p == "united states" and g in _US_STATE_LAWS:
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
    d = re.search(r"(\d{1,4})\s*[- ]?day", t)
    paren_m = re.search(r"\((\d{1,3})\s*months?\)", t)

    years = int(y.group(1)) if y else None
    months = int(m.group(1)) if m else (int(paren_m.group(1)) if paren_m else None)
    days = int(d.group(1)) if d else None

    if years is not None:
        expected_months = years * 12
        if months is None or months != expected_months:
            months = expected_months
    if years is None and months is not None and months % 12 == 0:
        years = months // 12

    auto = "auto-renewal" if ("auto-renew" in t or "auto renewal" in t) else ""

    if years is not None:
        base = f"{years}-year"
        return f"{base}|{auto}" if auto else base
    if months is not None:
        base = f"{months}-month"
        return f"{base}|{auto}" if auto else base
    if days is not None:
        base = f"{days}-day"
        return f"{base}|{auto}" if auto else base

    dm = _DATE_RE.search(t)
    if dm:
        return dm.group(0).lower().strip()

    return t


def expiration_match_score(pred_value: str, gold_value: str, pred_agreement_date: str | None = None, pred_effective_date: str | None = None, gold_agreement_date: str | None = None, gold_effective_date: str | None = None) -> float:
    raw_p = _norm(pred_value)
    raw_g = _norm(gold_value)

    # Step 1: if date is present on both sides, score full match only on date overlap.
    p_dates = _extract_all_dates_like(pred_value)
    g_dates = _extract_all_dates_like(gold_value)
    if p_dates and g_dates:
        return 1.0 if set(p_dates) & set(g_dates) else 0.0

    p_derived = derive_expiration_date(pred_value, agreement_date=pred_agreement_date, effective_date=pred_effective_date)
    g_derived = derive_expiration_date(gold_value, agreement_date=gold_agreement_date, effective_date=gold_effective_date)
    if p_derived and g_derived:
        return 1.0 if p_derived == g_derived else 0.0
    if p_dates and g_derived:
        return 1.0 if g_derived in p_dates else 0.0
    if g_dates and p_derived:
        return 1.0 if p_derived in g_dates else 0.0

    p = normalize_expiration(pred_value)
    g = normalize_expiration(gold_value)

    if p == g:
        return 1.0
    if "not found" in {p, g}:
        return 0.0

    # Step 2: if no usable date, compare period in months/years.
    p_term = _extract_term_period(pred_value)
    g_term = _extract_term_period(gold_value)
    if p_term and g_term:
        if p_term == g_term:
            p_auto = "auto-renewal" in p or "auto renewal" in raw_p or "auto-renew" in raw_p
            g_auto = "auto-renewal" in g or "auto renewal" in raw_g or "auto-renew" in raw_g
            if p_auto == g_auto or not g_auto:
                return 1.0
            return 0.8
        return 0.0

    # Step 3: special normalized phrases when neither side has date or period.
    if ("coterminous" in p and ("co-terminous" in g or "coterminous" in g)) or ("coterminous" in g and ("co-terminous" in p or "coterminous" in p)):
        return 1.0
    if ("until completion" in p and "until completion" in g) or ("milestones" in p and "milestones" in g):
        return 1.0
    if ("event-based termination" in p and "event-based termination" in g):
        return 1.0
    if ("until terminated" in p and "until terminated" in g):
        return 1.0

    if p in g or g in p:
        return 0.8
    return 0.0


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _expand_two_digit_year(year: int) -> int:
    return 1900 + year if year >= 40 else 2000 + year


def _extract_year(text: str):
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    return int(years[-1]) if years else None


def _parse_date_like(text: str):
    t = _norm(text)
    if not t or t in {"not found", "[*]", "[?]", "[ ]"}:
        return None
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t)

    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", t)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = _expand_two_digit_year(c) if c < 100 else c
        for month, day in ((a, b), (b, a)):
            try:
                from datetime import datetime
                return datetime(year, month, day).date()
            except ValueError:
                continue

    month_regex = r"\b(" + "|".join(sorted(_MONTHS.keys(), key=len, reverse=True)) + r")\b"
    month_match = re.search(month_regex, t)
    year = _extract_year(t)
    if month_match and year:
        from datetime import datetime
        month = _MONTHS[month_match.group(1)]
        day_match = re.search(r"\b(\d{1,2})\b", t)
        day = int(day_match.group(1)) if day_match else 1
        day = max(1, min(28, day))
        try:
            return datetime(year, month, day).date()
        except ValueError:
            return datetime(year, month, 1).date()
    return None


def _extract_all_dates_like(text: str):
    t = _norm(text)
    if not t or t in {"not found", "[*]", "[?]", "[ ]"}:
        return []
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t)
    found = []
    from datetime import datetime

    for m in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", t):
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = _expand_two_digit_year(c) if c < 100 else c
        for month, day in ((a, b), (b, a)):
            try:
                d = datetime(year, month, day).date()
                if d not in found:
                    found.append(d)
            except ValueError:
                continue

    month_regex = r"\b(" + "|".join(sorted(_MONTHS.keys(), key=len, reverse=True)) + r")\b"
    for m in re.finditer(rf"{month_regex}\s+(\d{{1,2}})?[,]?\s*(19\d{{2}}|20\d{{2}})", t):
        month = _MONTHS[m.group(1)]
        day = int(m.group(2)) if m.group(2) else 1
        year = int(m.group(3))
        day = max(1, min(28, day))
        try:
            d = datetime(year, month, day).date()
            if d not in found:
                found.append(d)
        except ValueError:
            continue

    single = _parse_date_like(text)
    if single and single not in found:
        found.append(single)
    return found


def _extract_term_period(text: str):
    t = _norm(text)
    if not t or t == "not found":
        return None
    word_pattern = r'one hundred eighty|forty[- ]five|thirty[- ]six|twenty[- ]five|twenty[- ]four|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|eighteen|twenty|thirty|sixty|ninety|fifth'

    def _pick(match):
        if not match:
            return None
        if match.group(1):
            return int(match.group(1))
        if match.group(2):
            return int(match.group(2))
        word = match.group(3)
        if not word:
            return None
        word_map = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'twelve': 12, 'fifteen': 15, 'eighteen': 18, 'twenty': 20, 'twenty four': 24,
            'twenty-five': 25, 'thirty': 30, 'thirty-six': 36, 'forty-five': 45, 'sixty': 60, 'ninety': 90,
            'one hundred eighty': 180, 'fifth': 5,
        }
        return word_map.get(word.replace('  ', ' '))

    y = re.search(rf"(?:([0-9]{{1,2}})(?:st|nd|rd|th)?|\(([0-9]{{1,3}})(?:st|nd|rd|th)?\)|({word_pattern}))\s*(?:[- ]?year|anniversary)", t)
    m = re.search(rf"(?:([0-9]{{1,3}})|\(([0-9]{{1,3}})\)|({word_pattern}))\s*[- ]?month", t)
    d = re.search(rf"(?:([0-9]{{1,4}})|\(([0-9]{{1,3}})\)|({word_pattern}))\s*[- ]?day", t)
    years = _pick(y)
    months = _pick(m)
    days = _pick(d)
    if months is None and years is not None:
        months = years * 12
    if years is None and months is not None and months % 12 == 0:
        years = months // 12
    if years is None and months is None and days is None:
        return None
    if days is not None and years is None and months is None:
        return ("day", days)
    return ("term", years, months)



def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        from datetime import date
        return (date(year + 1, 1, 1) - date.resolution).day
    from datetime import date
    return (date(year, month + 1, 1) - date.resolution).day


def _add_months(base_date, months: int):
    year = base_date.year + (base_date.month - 1 + months) // 12
    month = (base_date.month - 1 + months) % 12 + 1
    day = min(base_date.day, _last_day_of_month(year, month))
    from datetime import date
    return date(year, month, day)


def _select_expiration_base_date(value: str, agreement_date: str | None = None, effective_date: str | None = None):
    t = _norm(value)
    dates = _extract_all_dates_like(value)
    if len(dates) == 1:
        return dates[0]
    if 'agreement date' in t and agreement_date:
        return _parse_date_like(agreement_date)
    if any(k in t for k in ['effective date', 'commencement date', 'commencing on', 'commence on', 'beginning on', 'begin on', 'start date']) and effective_date:
        return _parse_date_like(effective_date)
    if agreement_date:
        parsed = _parse_date_like(agreement_date)
        if parsed:
            return parsed
    if effective_date:
        parsed = _parse_date_like(effective_date)
        if parsed:
            return parsed
    return dates[0] if dates else None


def derive_expiration_date(value: str, agreement_date: str | None = None, effective_date: str | None = None):
    t = _norm(value)
    if not t or t == 'not found':
        return None
    if any(k in t for k in ['terminate on', 'shall terminate on', 'shall expire on', 'expires on', 'effective through', 'through and including', 'ending on', 'end date', 'scheduled expiration date', 'continue until']):
        dates = _extract_all_dates_like(value)
        if dates:
            return dates[-1]
    period = _extract_term_period(value)
    if not period:
        return None
    base = _select_expiration_base_date(value, agreement_date=agreement_date, effective_date=effective_date)
    if not base:
        return None
    from datetime import timedelta
    anniversary_mode = 'anniversary' in t
    if period[0] == 'day':
        end = base + timedelta(days=period[1])
    else:
        _, years, months = period
        total_months = months if months is not None else (years * 12 if years is not None else 0)
        end = _add_months(base, total_months)
    if not anniversary_mode:
        end = end - timedelta(days=1)
    return end


def date_match_score(pred_value: str, gold_value: str) -> float:
    p = _parse_date_like(pred_value)
    g = _parse_date_like(gold_value)
    if p and g:
        return 1.0 if p == g else 0.0
    return 1.0 if _norm(pred_value) == _norm(gold_value) else 0.0


def _normalize_party_name(text: str) -> str:
    t = _norm(text)
    t = re.sub(r"\([^)]*\)", " ", t)
    t = t.replace(" by: ", " ")
    t = t.replace(" and ", " | ")
    t = t.replace(" between ", " ")
    t = re.sub(r"\b(the|a|an|party|parties|provider|recipient|customer|distributor|consultant)\b", " ", t)
    t = re.sub(r"\b(incorporated|corporation|corp|corp\.|inc|inc\.|limited|ltd|ltd\.|llc|l\.l\.c\.|lp|l\.p\.|plc|company|co\.|co)\b", " ", t)
    t = re.sub(r"[^a-z0-9|& ]+", " ", t)
    t = " ".join(t.split())
    return t.strip(" |")


def _split_parties(text: str):
    t = _normalize_party_name(text)
    if not t or t == "not found":
        return []
    parts = [p.strip() for p in re.split(r"\s*\|\s*", t) if p.strip()]
    return parts if parts else [t]


def parties_match_score(pred_value: str, gold_value: str) -> float:
    pred_parts = _split_parties(pred_value)
    gold_parts = _split_parties(gold_value)
    if not pred_parts and not gold_parts:
        return 1.0
    if not pred_parts or not gold_parts:
        return 0.0
    matched = 0
    used = set()
    for g in gold_parts:
        for i, p in enumerate(pred_parts):
            if i in used:
                continue
            if g == p or g in p or p in g:
                matched += 1
                used.add(i)
                break
    coverage = matched / max(1, len(gold_parts))
    precision = matched / max(1, len(pred_parts))
    if matched == len(gold_parts) and len(pred_parts) == len(gold_parts):
        return 1.0
    if matched == len(pred_parts) and len(gold_parts) == len(pred_parts) + 1 and precision == 1.0:
        return 0.95
    if coverage >= 0.95:
        return 0.95
    if coverage >= 0.66 and precision >= 0.66:
        return 0.8
    if coverage >= 0.5:
        return 0.6
    if matched > 0:
        return 0.4
    return 0.0


def _extract_notice_period(text: str):
    t = _norm(text)
    if 'notice: immediate' in t or 'immediate effect' in t or 'forthwith' in t:
        return ('immediate', 0)
    m = re.search(r"(\d{1,3})\s*(business\s+days|calendar\s+days|day|days|month|months|year|years)", t)
    if m:
        return (int(m.group(1)), m.group(2))
    word_map = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
        'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'twelve': 12,
        'fifteen': 15, 'thirty': 30, 'forty-five': 45, 'forty': 40,
        'sixty': 60, 'ninety': 90, 'one hundred twenty': 120,
        'one hundred eighty': 180, 'twenty four': 24,
    }
    m2 = re.search(r"([a-z][a-z -]+)\s*(business\s+days|calendar\s+days|day|days|month|months|year|years)", t)
    if m2:
        key = ' '.join(m2.group(1).split())
        if key in word_map:
            return (word_map[key], m2.group(2))
    if 'written notice' in t or 'notice:' in t or 'upon notice' in t:
        return ('unspecified', None)
    return None



def _canonicalize_duration_phrase(text: str) -> str:
    t = _norm(text)
    if not t:
        return ""
    t = re.sub(r"\bduring the term of (?:this|the) agreement\b", "during the term", t)
    t = re.sub(r"\bfor the term of (?:this|the) agreement\b", "during the term", t)
    t = re.sub(r"\bcalendar days?\b", "days", t)
    t = re.sub(r"\bbusiness days?\b", "business days", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _notice_periods_equivalent(left, right) -> bool:
    if left == right:
        return True
    if left is None or right is None:
        return False
    if left[0] == 'unspecified' or right[0] == 'unspecified':
        return left[0] == right[0]
    if left[0] == 'immediate' or right[0] == 'immediate':
        return left[0] == right[0]
    if isinstance(left[0], int) and isinstance(right[0], int):
        if left[0] != right[0]:
            return False
        l_unit = str(left[1] or "").replace("calendar ", "").strip()
        r_unit = str(right[1] or "").replace("calendar ", "").strip()
        if l_unit.endswith('s'):
            l_unit = l_unit[:-1]
        if r_unit.endswith('s'):
            r_unit = r_unit[:-1]
        return l_unit == r_unit
    return False


def termination_for_convenience_match_score(pred_value: str, gold_value: str) -> float:
    p = _norm(pred_value)
    g = _norm(gold_value)
    if p == g:
        return 1.0
    if g == "not found":
        return 1.0 if p == "not found" else 0.0
    positive_markers = ["for convenience", "without cause", "either party", "notice:", "terminate"]
    pred_positive = any(m in p for m in positive_markers)
    gold_positive = any(m in g for m in positive_markers)
    if pred_positive and gold_positive:
        gp = _extract_notice_period(g)
        pp = _extract_notice_period(p)
        if gp is not None and pp is not None and not _notice_periods_equivalent(gp, pp):
            return 0.4
        gold_needs_either = "either party" in g
        gold_needs_without = "without cause" in g or "for convenience" in g
        if (not gold_needs_either or "either party" in p) and (not gold_needs_without or ("without cause" in p or "for convenience" in p)):
            return 1.0 if (gp is None or pp is None or _notice_periods_equivalent(gp, pp)) else 0.8
        return 0.7
    if pred_positive or gold_positive:
        return 0.3
    return 0.0


def limitation_of_liability_match_score(pred_value: str, gold_value: str) -> float:
    p = _norm(pred_value)
    g = _norm(gold_value)
    if p == g:
        return 1.0
    if g == "not found":
        return 1.0 if p == "not found" else 0.0

    term_groups = {
        "cap": ["cap:", "shall not exceed", "will not exceed", "liability cap", "maximum liability", "aggregate liability"],
        "consequential": ["consequential"],
        "incidental": ["incidental"],
        "indirect": ["indirect"],
        "special": ["special"],
        "punitive": ["punitive"],
        "exemplary": ["exemplary"],
        "lost_profits": ["lost profits", "lost revenue", "loss of revenue", "loss of profits"],
        "confidentiality": ["confidentiality", "confidential"],
        "infringement": ["infringement", "intellectual property"],
        "misconduct": ["misconduct", "gross negligence", "fraud"],
        "indemnification": ["indemnification", "indemnity", "third party claim"],
        "remedy": ["sole remedy", "exclusive remedy", "sole obligation", "direct damages"],
    }
    pred_hits = {name for name, terms in term_groups.items() if any(term in p for term in terms)}
    gold_hits = {name for name, terms in term_groups.items() if any(term in g for term in terms)}
    overlap = len(pred_hits & gold_hits)

    if gold_hits and gold_hits.issubset(pred_hits):
        return 1.0
    if overlap >= max(4, min(len(gold_hits), 5)):
        return 1.0
    if overlap >= 3:
        return 0.85
    if overlap >= 2:
        return 0.7
    if overlap >= 1:
        return 0.5
    return 0.0


def _extract_duration_phrase(text: str) -> str:
    t = _norm(text)
    m = re.search(r"(during the term(?: of this agreement| of the agreement)?|during the restricted period|during the royalty term|during the post-term period|for a period of [^.;]+|for [0-9]+ years?|for [0-9]+ months?|for [0-9]+ days?|two-year period|\d+-year|\d+-month|within [^.;]+ miles?)", t)
    return _canonicalize_duration_phrase(m.group(1)) if m else ""


def non_compete_match_score(pred_value: str, gold_value: str) -> float:
    p = _norm(pred_value)
    g = _norm(gold_value)
    if p == g:
        return 1.0
    if g == "not found":
        return 1.0 if p == "not found" else 0.0

    if p == 'yes' and g.startswith('yes'):
        return 0.4

    term_groups = {
        "exclusivity": ["exclusive", "exclusivity", "exclusive right", "exclusive rights", "grant any rights", "granting any rights"],
        "competition": ["non-competition", "non competition", "noncompetition", "competing", "competitor", "competitive business", "competitive products", "not engage", "not participate", "prohibited from", "restricted territory"],
        "endorsement": ["endorse", "endorsement"],
        "solicit": ["solicit", "entice away", "divert", "direct solicitation", "interfere with", "discontinue using"],
        "hire": ["hire", "recruit", "employ", "employment", "leave the employment", "offer of employment", "solicit for employment"],
        "territory": ["outside the territory", "restricted territory", "within three (3) miles", "within three miles"],
    }
    g_hits = {name for name, terms in term_groups.items() if any(term in g for term in terms)}
    p_hits = {name for name, terms in term_groups.items() if any(term in p for term in terms)}

    overlap = len(g_hits & p_hits)
    if g_hits and g_hits.issubset(p_hits):
        g_dur = _extract_duration_phrase(g)
        p_dur = _extract_duration_phrase(p)
        if not g_dur or not p_dur or g_dur == p_dur or g_dur in p_dur or p_dur in g_dur:
            return 1.0
    duration_match = False
    g_dur = _extract_duration_phrase(g)
    p_dur = _extract_duration_phrase(p)
    if g_dur and p_dur and (g_dur == p_dur or g_dur in p_dur or p_dur in g_dur):
        duration_match = True

    if overlap >= 3 and duration_match:
        return 1.0
    if overlap >= 2 and duration_match:
        return 0.9
    if overlap >= 2:
        return 0.8
    if overlap >= 1:
        return 0.6
    return 0.0
