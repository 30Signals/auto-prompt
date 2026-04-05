"""
DSPy metadata pipeline for legal_contract CSV task.

Trains/evaluates directly on contract-level metadata columns using contract text
fetched from CUAD by contract_id.
"""

import argparse
import json
import random
import re
from pathlib import Path

import dspy
import pandas as pd
from datasets import load_dataset

from shared.llm_providers import get_llm_config, setup_dspy_lm
from shared.llm_providers.provider import save_lm_history_artifacts, enforce_usage_budget
from shared.evaluation.prompt_utils import count_examples
from shared.optimization import run_two_stage_optimization
from shared.evaluation import compare_results, save_results_json, EvaluationResult
from experiments.legal_contract.metrics import llm_semantic_match_score, normalize_text
from experiments.legal_contract import config
from experiments.legal_contract.metadata_normalization import (
    _parse_date_like,
    date_match_score,
    expiration_match_score,
    governing_law_match_score,
    indemnification_match_score,
    limitation_of_liability_match_score,
    normalize_governing_law,
    non_compete_match_score,
    parties_match_score,
    termination_for_convenience_match_score,
)
from experiments.legal_contract.clean_contract_metadata_from_original import (
    clean_expiration,
    clean_indemnification,
    clean_limitation,
    clean_non_compete,
    clean_parties,
    clean_termination,
    extract_dates,
    format_date,
)


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

FIELD_TO_ATTR = {
    "Agreement Date": "agreement_date",
    "Effective Date": "effective_date",
    "Expiration Date": "expiration_date",
    "Governing Law": "governing_law",
    "Indemnification": "indemnification",
    "Limitation Of Liability": "limitation_of_liability",
    "Non-Compete": "non_compete",
    "Parties": "parties",
    "Termination For Convenience": "termination_for_convenience",
}

# Higher weight for weak fields so optimization prioritizes them.
WEAK_OPT_FIELDS = [
    "Effective Date",
    "Indemnification",
    "Limitation Of Liability",
    "Non-Compete",
]

FIELD_WEIGHTS = {
    "Effective Date": 1.35,
    "Indemnification": 1.45,
    "Limitation Of Liability": 1.35,
    "Non-Compete": 1.35,
    "Expiration Date": 1.15,
    "Termination For Convenience": 1.10,
}


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

    if len(cleaned.split()) <= 3:
        return _canonical_title(cleaned)
    return "NOT FOUND"


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


def _safe_text(v):
    s = str(v).strip()
    return s if s else "NOT FOUND"


def _train_val_split(examples, val_ratio: float, seed: int):
    if not examples:
        return [], []
    if val_ratio <= 0:
        return examples, []

    items = list(examples)
    rng = random.Random(seed)
    rng.shuffle(items)

    val_size = max(1, int(round(len(items) * val_ratio)))
    # Keep at least 1 sample in train.
    if val_size >= len(items):
        val_size = len(items) - 1

    valset = items[:val_size]
    trainset = items[val_size:]
    return trainset, valset


def load_metadata_examples(csv_path: Path, text_map: dict):
    df = pd.read_csv(csv_path)
    id_col = "contract_id" if "contract_id" in df.columns else df.columns[0]
    examples = []

    for _, row in df.iterrows():
        cid = str(row[id_col]).strip()
        if not cid:
            continue
        contract_text = text_map.get(cid, "")
        if not contract_text:
            continue

        attrs = {
            "contract_id": cid,
            "contract_text": contract_text,
            "gold_metadata": {},
        }
        for field in FIELDS:
            val = _safe_text(row.get(field, "NOT FOUND"))
            attrs[FIELD_TO_ATTR[field]] = val
            attrs["gold_metadata"][field] = val

        ex = dspy.Example(**attrs).with_inputs("contract_text")
        examples.append(ex)

    return examples


def _split_examples_train_test(examples, test_ratio: float, seed: int):
    if not examples:
        return [], []

    items = list(examples)
    rng = random.Random(seed)
    rng.shuffle(items)

    test_size = max(1, int(round(len(items) * test_ratio)))
    if test_size >= len(items):
        test_size = len(items) - 1

    testset = items[:test_size]
    trainset = items[test_size:]
    return trainset, testset


_METADATA_FIELD_PATTERNS = {
    "agreement_dates": [r"dated as of", r"agreement date", r"made and entered", r"entered into as of", r"this agreement is made", r"date first written above", r"as of [a-z]"],
    "effective_dates": [r"effective date", r"effective as of", r"commence", r"commencement", r"date of commencement", r"become effective", r"first set forth above", r"first above written", r"later of the dates"],
    "expiration_dates": [r"expire", r"expiration", r"term of this agreement", r"term of the agreement", r"initial term", r"renewal term", r"auto renew", r"auto-renew", r"shall continue until", r"co-terminous", r"coterminous", r"terminate on", r"term shall", r"continue for a period", r"remain in effect", r"renew automatically", r"notice of non-renewal", r"until completion", r"royalty term", r"service period", r"duration of the lease", r"perpetual", r"indefinite period", r"earlier of", r"later of", r"effective through", r"end date", r"for the term of", r"earliest to occur", r"until the expiration or earlier termination of", r"continue until the termination of"],
    "governing_law": [r"governing law", r"governed by", r"laws of", r"law of", r"construed according to the laws", r"construed in accordance with", r"interpreted under the laws", r"federal republic", r"state of", r"commonwealth of"],
    "indemnification": [r"indemn", r"indemnif", r"hold harmless", r"save harmless", r"defend", r"defense", r"reimburse", r"reimbursement", r"claims against", r"third party claims", r"losses damages liabilities", r"not challenge", r"contest the validity", r"petition to cancel", r"moral rights", r"not to sue", r"bankruptcy", r"non-petition"],
    "limitation": [r"limitation of liability", r"liable for", r"consequential", r"incidental", r"lost profits", r"cap on liability", r"exclusive remedy", r"shall not exceed", r"indirect", r"punitive", r"exemplary", r"special damages"],
    "non_compete": [r"non-compete", r"non compete", r"exclusive appointment", r"exclusive right", r"exclusive rights", r"exclusive purchase", r"competing", r"competition", r"competitor", r"endorse", r"not sell any competing", r"not directly or indirectly sell", r"handle no products competitive", r"products competitive with", r"competitive product", r"competitive products", r"non-solicit", r"solicit", r"no-hire", r"recruit", r"exclusive rights to exploit"],
    "parties": [r"between", r"by and between", r"party", r"parties", r"entered into by", r"among", r"executed by", r"made by and between", r"this agreement .* between"],
    "termination": [r"termination for convenience", r"without cause", r"for convenience", r"for any reason", r"with or without cause", r"either party may terminate", r"may be terminated by either party", r"may terminate upon", r"terminate the appointment"],
}


def _excerpt_for_patterns(contract_text: str, patterns: list[str], window: int = 500, max_chars: int = 900) -> str:
    text = str(contract_text or "")
    if not text:
        return ""

    hits = []
    low = text.lower()
    for pattern in patterns:
        for match in re.finditer(pattern, low, flags=re.IGNORECASE):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            hits.append((start, end))
            if len(hits) >= 4:
                break
        if len(hits) >= 4:
            break

    if not hits:
        return ""

    hits.sort()
    merged = []
    for start, end in hits:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    chunks = []
    total = 0
    for start, end in merged:
        chunk = text[start:end].strip()
        if not chunk:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        total += len(chunk)

    return "\n\n...\n\n".join(chunks)


_EXPIRATION_FOCUS_PATTERNS = [
    r"\bterm\s*:",
    r"duration of agreement",
    r"shall be in effect until",
    r"shall terminate on",
    r"shall expire on",
    r"expires on",
    r"effective through(?: and including)?",
    r"through and including",
    r"contract end",
    r"termination date",
    r"scheduled expiration date",
    r"shall continue for a period",
    r"shall have an initial term",
    r"shall have a term of",
    r"initial period of",
    r"initial term of",
    r"remain in effect for",
    r"continue in operation for",
    r"continue indefinitely",
    r"indefinite period",
    r"perpetual",
    r"perpetually thereafter",
    r"as long as fees are paid",
    r"until terminated",
    r"for the term of the referenced",
    r"for the term of the lease",
    r"until the expiration or earlier termination of",
    r"continue until the termination of",
    r"until completion of",
    r"successful remarketing",
    r"earlier of the occurrence",
    r"earliest to occur",
    r"if any public authority cancels",
    r"terminate automatically one year after",
]

_WEAK_FIELD_CONTEXT_CONFIG = {
    "effective_date": {
        "patterns": _METADATA_FIELD_PATTERNS["effective_dates"] + [r"date first set forth above", r"date of commencement", r"first above written"],
        "window": 700,
        "max_chars": 1400,
        "label": "Weak Field Focus: Effective Date",
    },
    "governing_law": {
        "patterns": _METADATA_FIELD_PATTERNS["governing_law"] + [r"jurisdiction", r"venue", r"conflict of law"],
        "window": 700,
        "max_chars": 1400,
        "label": "Weak Field Focus: Governing Law",
    },
    "indemnification": {
        "patterns": _METADATA_FIELD_PATTERNS["indemnification"] + [r"liabilit(?:y|ies)", r"claims", r"damages", r"losses"],
        "window": 750,
        "max_chars": 1500,
        "label": "Weak Field Focus: Indemnification",
    },
    "expiration_date": {
        "patterns": _EXPIRATION_FOCUS_PATTERNS + [r"termination election", r"until terminated", r"notice of non-renewal"],
        "window": 900,
        "max_chars": 2000,
        "label": "Weak Field Focus: Expiration Date",
    },
    "non_compete": {
        "patterns": _METADATA_FIELD_PATTERNS["non_compete"] + [r"restricted party", r"scope", r"duration", r"during the term", r"competitive products", r"non-solicit", r"no-hire"],
        "window": 850,
        "max_chars": 1800,
        "label": "Weak Field Focus: Non-Compete",
    },
    "limitation_of_liability": {
        "patterns": _METADATA_FIELD_PATTERNS["limitation"] + [r"exclusive remedy", r"shall not exceed", r"aggregate liability", r"special damages"],
        "window": 850,
        "max_chars": 1800,
        "label": "Weak Field Focus: Limitation Of Liability",
    },
    "parties": {
        "patterns": _METADATA_FIELD_PATTERNS["parties"] + [r"between", r"by and between", r"executed by", r"signed by", r"in witness whereof"],
        "window": 850,
        "max_chars": 1800,
        "label": "Weak Field Focus: Parties",
    },
}


def _append_weak_field_sections(text: str, sections: list[str]) -> None:
    for cfg in _WEAK_FIELD_CONTEXT_CONFIG.values():
        excerpt = _excerpt_for_patterns(
            text,
            cfg["patterns"],
            window=cfg["window"],
            max_chars=cfg["max_chars"],
        )
        if excerpt:
            sections.append(f"[{cfg['label']}]\n" + excerpt)


def _build_metadata_context(contract_text: str) -> str:
    text = str(contract_text or "")
    if not text:
        return ""

    sections = []
    intro = text[:2200].strip()
    if intro:
        sections.append("[Intro and Parties]\n" + intro)

    ordered = [
        ("Dates", _METADATA_FIELD_PATTERNS["agreement_dates"] + _METADATA_FIELD_PATTERNS["effective_dates"] + _METADATA_FIELD_PATTERNS["expiration_dates"], 2200),
        ("Governing Law", _METADATA_FIELD_PATTERNS["governing_law"], 900),
        ("Indemnification", _METADATA_FIELD_PATTERNS["indemnification"], 1000),
        ("Liability", _METADATA_FIELD_PATTERNS["limitation"], 1500),
        ("Termination", _METADATA_FIELD_PATTERNS["termination"], 800),
        ("Non-Compete", _METADATA_FIELD_PATTERNS["non_compete"], 1400),
    ]
    for label, patterns, budget in ordered:
        excerpt = _excerpt_for_patterns(text, patterns, window=450, max_chars=budget)
        if excerpt:
            sections.append(f"[{label}]\n" + excerpt)

    _append_weak_field_sections(text, sections)

    combined = "\n\n==========\n\n".join(sections).strip()
    return combined[:8200] if len(combined) > 8200 else combined


def _title_case_jurisdiction(value: str) -> str:
    if not value or value == "not found":
        return "NOT FOUND"
    return " ".join(part.capitalize() for part in str(value).split())


def _extract_intro_block(contract_text: str, max_chars: int = 2500) -> str:
    text = str(contract_text or "")
    if not text:
        return ""
    stop_markers = [
        "recitals", "whereas", "witnesseth", "the parties agree as follows",
        "now, therefore", "now therefore", "1.", "section 1", "article 1"
    ]
    lower = text.lower()
    end = min(len(text), max_chars)
    for marker in stop_markers:
        idx = lower.find(marker)
        if 0 < idx < end:
            end = idx
    return text[:end].strip()


def _extract_signature_block(contract_text: str, max_chars: int = 2500) -> str:
    text = str(contract_text or "")
    if not text:
        return ""
    lower = text.lower()
    markers = ["in witness whereof", "signatories", "signature page", "executed by", "signed by", "signed and effective"]
    starts = [lower.rfind(m) for m in markers if lower.rfind(m) >= 0]
    if not starts:
        return text[-max_chars:].strip()
    start = max(starts)
    return text[start:start + max_chars].strip()



def _contract_has_strong_effective_support(contract_text: str) -> bool:
    low = str(contract_text or '').lower()
    support_markers = [
        'effective date means', '"effective date" means', "'effective date' means",
        'effective as of', 'effective as from', 'effective on', 'shall become effective',
        'signed and effective', 'effective this', 'made effective this', 'executed as of',
        'entered into as of', 'entered into on', 'later of the dates that it is executed',
        'later of the two signature dates', 'last date of signature', 'present agreement is effective as from',
        'shall commence on', 'date of commencement', 'on the date of this agreement'
    ]
    return any(marker in low for marker in support_markers)

def _extract_agreement_date_from_contract(contract_text: str) -> str:
    intro = _extract_intro_block(contract_text, max_chars=2400)
    signature = _extract_signature_block(contract_text, max_chars=2000)
    focused = _excerpt_for_patterns(
        contract_text,
        _METADATA_FIELD_PATTERNS["agreement_dates"] + [r"signed and effective", r"executed as of", r"as of the date first set forth above", r"made effective this"],
        window=450,
        max_chars=2200,
    )
    candidates = []
    patterns = [
        r"dated as of\s+([^.;\n]{5,80})",
        r"entered into as of\s+([^.;\n]{5,80})",
        r"this agreement (?:is )?made(?: and entered into)?(?: as of| on)?\s+([^.;\n]{5,80})",
        r"agreement dated\s+([^.;\n]{5,80})",
        r"signed and effective(?: as of)?\s*([^.;\n]{5,80})",
        r"executed as of\s+([^.;\n]{5,80})",
        r"made effective this\s+([^.;\n]{5,80})",
    ]
    for source in [focused, intro, signature]:
        for pattern in patterns:
            for match in re.finditer(pattern, source, flags=re.I):
                nearby = source[max(0, match.start()-120): min(len(source), match.end()+120)].lower()
                if any(k in nearby for k in ['amends that certain', 'as previously amended', 'previously amended', 'prior agreement', 'reseller agreement between']) and 'this agreement' not in nearby:
                    continue
                candidates.append(match.group(1))
        for match in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+[_*\d]{1,8},?\s*\d{4}", source, flags=re.I):
            nearby = source[max(0, match.start()-100): min(len(source), match.end()+100)].lower()
            if any(k in nearby for k in ['amends that certain', 'as previously amended', 'previously amended', 'prior agreement']):
                continue
            candidates.append(match.group(0))
    for candidate in candidates:
        normalized = _normalize_date_prediction(candidate)
        if normalized != "NOT FOUND":
            return normalized
    return "NOT FOUND"

def _looks_like_governing_law_clause(text: str) -> bool:
    low = " ".join(str(text or "").lower().split())
    if not low:
        return False
    strong = ["governed by", "governing law", "laws of", "law of", "construed in accordance with", "interpreted under the laws"]
    return any(token in low for token in strong)


def _looks_like_expiration_term_text(low: str) -> bool:
    strong_markers = [
        'term of this agreement', 'term of the agreement', 'initial term', 'renewal term', 'expiration date',
        'shall continue for', 'continue for a period', 'remain in effect for', 'shall remain in effect for',
        'shall continue in force', 'will be in effect for', 'shall have a term of', 'shall be valid for',
        'effective through', 'through and including', 'contract end', 'scheduled expiration date',
        'co-termin', 'cotermin', 'until terminated', 'indefinite period', 'perpetual', 'perpetually thereafter',
        'annual renewal', 'automatically renew', 'auto-renew', 'notice of non-renewal', 'expire on', 'expires on',
        'shall expire on', 'terminate on', 'shall terminate on', 'royalty term', 'service period', 'duration of the lease',
        'until completion of', 'last addendum to expire', 'for the term of', 'continue indefinitely',
        'until the expiration or earlier termination of', 'continue until the termination of'
    ]
    return any(marker in low for marker in strong_markers)


def _normalize_expiration_prediction(value: str, agreement_date: str = "NOT FOUND", effective_date: str = "NOT FOUND") -> str:
    raw = str(value or "").strip()
    if not raw:
        return "NOT FOUND"
    low = " ".join(raw.lower().split())
    if low in {"not found", "none", "n/a", "na"}:
        return "NOT FOUND"

    cleaned = clean_expiration(raw, agreement_date=agreement_date, effective_date=effective_date)
    if cleaned and cleaned != raw and cleaned != 'NOT FOUND':
        return cleaned

    parsed_date = _parse_date_like(raw)

    if not _looks_like_expiration_term_text(low) and not any(k in low for k in ["earlier of", "earliest of", "termination election"]):
        return "NOT FOUND"

    if "co-termin" in low or "cotermin" in low:
        return "Co-terminous with Related Agreement"
    if "until terminated" in low:
        return "Until Terminated"
    if 'annual meeting' in low and not any(k in low for k in ['initial term', 'term of this agreement', 'term of the agreement', 'shall continue for', 'shall have a term of']):
        return "NOT FOUND"
    if 'expiry of the cooperation period' in low or 'expiration of the cooperation period' in low:
        return "NOT FOUND"
    if any(k in low for k in ["earlier of", "earliest of", "termination election", "earlier of the occurrence", "earliest to occur", "successful remarketing"]):
        return "Event-Based Termination"

    if 'notice' in low and not any(k in low for k in ['initial term', 'term of this agreement', 'term of the agreement', 'shall have a term of', 'shall continue for', 'shall be valid for', 'remain in effect for']):
        return "NOT FOUND"

    years = re.search(r"(\d{1,2})\s*(?:-| )?year", low)
    months = re.search(r"(\d{1,3})\s*(?:-| )?month", low)
    days = re.search(r"(\d{1,4})\s*(?:-| )?day", low)
    paren_months = re.search(r"\((\d{1,3})\s*months?\)", low)
    auto = any(k in low for k in ["auto-renew", "auto renew", "automatic renewal", "renew automatically"])

    explicit_end_markers = any(k in low for k in [
        "terminate on", "shall terminate on", "shall expire on", "expires on", "effective through",
        "through and including", "ending on", "end date", "scheduled expiration date", "continue until",
        "expiration date", "term shall end", "shall end on", "continue for a period", "initial term",
        "renewal term", "commence on", "commencing on"
    ])

    year_count = int(years.group(1)) if years else None
    month_count = int(months.group(1)) if months else (int(paren_months.group(1)) if paren_months else None)
    day_count = int(days.group(1)) if days else None
    if year_count is None and month_count and month_count % 12 == 0:
        year_count = month_count // 12
    if month_count is None and year_count is not None:
        month_count = year_count * 12

    if year_count is not None:
        label = f"{year_count}-Year ({month_count} months) Initial Term"
        if auto:
            label += ", Auto-Renewal"
        return label
    if month_count is not None:
        return f"{month_count}-Month Initial Term"
    if day_count is not None:
        if 'notice' in low and not any(k in low for k in ['initial term', 'term of this agreement', 'term of the agreement', 'shall have a term of']):
            return "NOT FOUND"
        return f"{day_count}-Day Initial Term"

    if parsed_date and explicit_end_markers:
        return parsed_date.strftime("%B %d, %Y")

    return "NOT FOUND"


def _normalize_date_prediction(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "NOT FOUND"
    low = " ".join(raw.lower().split())
    if any(k in low for k in ["not specified", "tbd", "to be determined", "unknown"]):
        return "NOT FOUND"
    if len(raw) < 6 and not re.search(r'\d{4}', raw):
        return "NOT FOUND"
    if any(k in low for k in ['effective date means', '"effective date" means', "'effective date' means", '12:01 a.m.', '12:01 a.m', 'eastern standard time on']) :
        date_candidates = extract_dates(raw)
        if date_candidates:
            return date_candidates[-1]
    normalized = format_date(raw)
    if normalized == raw and len(raw) > 45 and ',' not in raw:
        return "NOT FOUND"
    if normalized == raw:
        date_candidates = extract_dates(raw)
        if date_candidates:
            return date_candidates[-1]
    if normalized == raw and any(k in low for k in ["date of commencement", "later of the dates", "first above written", "first set forth above", "effective date", "effective as of", "effective on"]):
        parsed = _parse_date_like(raw)
        return parsed.strftime("%B %d, %Y") if parsed else "NOT FOUND"
    if normalized == raw:
        if re.search(r'\b(?:inc\.?|corp\.?|corporation|llc|ltd\.?|limited|company|agreement|affiliate|vendor|licensee|licensor)\b', low):
            return "NOT FOUND"
        if re.fullmatch(r'\d{4}[^\n]*[A-Za-z][^\n]*', raw):
            return "NOT FOUND"
    if not re.search(r'\d{4}|__|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', normalized):
        return "NOT FOUND"
    return normalized


def _normalize_parties_prediction(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "NOT FOUND"
    normalized = clean_parties(raw)
    return normalized if normalized else "NOT FOUND"


def _extract_effective_date_from_contract(contract_text: str) -> str:
    text = str(contract_text or "")
    if not text:
        return "NOT FOUND"
    intro = _extract_intro_block(text, max_chars=2600)
    signature = _extract_signature_block(text, max_chars=2200)
    focused = _excerpt_for_patterns(
        text,
        _METADATA_FIELD_PATTERNS["effective_dates"] + [r"commencement of appointment", r"begin on", r"shall become effective", r"signed and effective", r"entered into force", r"date first set forth above", r"date first above written", r"effective on", r"effective this", r"present agreement is effective as from", r"executed as of", r"made effective this", r"date of this agreement", r"date hereof"],
        window=500,
        max_chars=2600,
    )
    patterns = [
        r"signed and effective(?: as of)?\s*([^.;\n]{5,80})",
        r"effective date[^\n:]{0,40}[:\-]?\s*([^.;\n]{5,80})",
        r"effective as of\s+([^.;\n]{5,80})",
        r"effective as from\s+([^.;\n]{5,80})",
        r"shall become effective(?: as of)?\s+([^.;\n]{5,80})",
        r"effective on\s+([^.;\n]{5,80})",
        r"begin(?:ning)? on\s+([^.;\n]{5,80})",
        r"commence(?:ment)?(?: date)?(?: on| as of)?\s+([^.;\n]{5,80})",
        r"date of commencement[^\n:]{0,30}[:\-]?\s*([^.;\n]{5,80})",
        r"enters? into force on\s+([^.;\n]{5,80})",
        r"effective through and including\s+([^.;\n]{5,80})",
        r"[\"']effective date[\"']?\s+means\s+([^.;\n]{5,80})",
        r"is effective\s+([^.;\n]{5,80})\s*\(the [\"']effective date[\"']\)",
        r"effective this\s+([^.;\n]{5,80})",
        r"made effective this\s+([^.;\n]{5,80})",
        r"executed as of\s+([^.;\n]{5,80})",
        r"entered into(?: in [^.;\n]{0,40})? on\s+([^.;\n]{5,80})",
        r"entered into as of\s+([^.;\n]{5,80})",
        r"present agreement is effective as from\s+([^.;\n]{5,80})",
        r"commence on\s+([^.;\n]{5,80})",
        r"shall commence on\s+([^.;\n]{5,80})",
    ]
    for source in [focused, intro, signature]:
        for pattern in patterns:
            for match in re.finditer(pattern, source, flags=re.I):
                candidate = match.group(1)
                nearby = source[max(0, match.start()-120): min(len(source), match.end()+120)].lower()
                if ("board" in nearby or "meeting" in nearby) and "agreement" not in nearby:
                    continue
                if any(k in nearby for k in ['amends that certain', 'as previously amended', 'previously amended', 'prior agreement']) and 'effective date' not in nearby:
                    continue
                if pattern.startswith("effective date") and not extract_dates(candidate) and not re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', candidate, flags=re.I):
                    continue
                normalized = _normalize_date_prediction(candidate)
                if normalized != "NOT FOUND":
                    return normalized
        low_source = source.lower()
        if any(k in low_source for k in ["later of the two signature dates", "last date of signature", "last date of the signatures", "later of the dates"]):
            date_candidates = extract_dates(source)
            if len(date_candidates) >= 2:
                return date_candidates[-1]
        if agreement_date := _extract_agreement_date_from_contract(source):
            if agreement_date != "NOT FOUND" and any(k in low_source for k in ["date of this agreement", "on the date of this agreement", "date hereof", "made and entered into this", "entered into this", "entered into on", "executed as of", "made as of", "effective on the later of the dates that it is executed", "later of the dates that it is executed"]):
                return agreement_date
    return "NOT FOUND"

def _extract_limitation_from_contract(contract_text: str) -> str:
    text = str(contract_text or "")
    if not text:
        return "NOT FOUND"
    focused = _excerpt_for_patterns(
        text,
        [r"limitation of liability", r"liable for", r"consequential", r"incidental", r"lost profits", r"cap on liability", r"exclusive remedy", r"shall not exceed", r"aggregate liability", r"in no event shall", r"not be liable"],
        window=500,
        max_chars=2600,
    )
    normalized = _normalize_limitation_prediction(focused or text)
    return normalized if normalized else "NOT FOUND"


def _extract_non_compete_from_contract(contract_text: str) -> str:
    text = str(contract_text or "")
    if not text:
        return "NOT FOUND"
    focused = _excerpt_for_patterns(
        text,
        _METADATA_FIELD_PATTERNS["non_compete"] + [r"non-solicit", r"no-hire", r"not sell any competing", r"competitive product", r"products competitive with", r"competitive business", r"direct or indirect interest", r"own, manage, engage in, be employed by", r"divert or attempt to divert", r"exclusive appointment", r"exclusive purchase"],
        window=550,
        max_chars=2600,
    )
    normalized = _normalize_non_compete_prediction(focused or text)
    return normalized if normalized else "NOT FOUND"


def _count_party_items(value: str) -> int:
    v = str(value or "").strip()
    if not v or v == "NOT FOUND":
        return 0
    return len([p for p in v.split("|") if p.strip()])


def _merge_party_values(*values: str) -> str:
    merged = []
    seen = set()
    for value in values:
        normalized = clean_parties(value)
        if not normalized or normalized == "NOT FOUND":
            continue
        for part in [p.strip() for p in normalized.split("|") if p.strip()]:
            key = part.lower().rstrip('.')
            if key not in seen:
                seen.add(key)
                merged.append(part)
    return " | ".join(merged) if merged else "NOT FOUND"


def _pick_best_party_value(*values: str) -> str:
    candidates = []
    for value in values:
        normalized = clean_parties(value)
        if not normalized or normalized == "NOT FOUND":
            continue
        count = _count_party_items(normalized)
        density = len(normalized) / max(count, 1)
        candidates.append((count, -density, normalized))
    if not candidates:
        return "NOT FOUND"
    candidates.sort(reverse=True)
    return candidates[0][2]


def _extract_parties_from_contract(contract_text: str) -> str:
    text = str(contract_text or "")
    if not text:
        return "NOT FOUND"

    intro = _extract_intro_block(text, max_chars=2600)
    signature = _extract_signature_block(text, max_chars=2200)
    between_block = ""
    m = re.search(r"(?:between|by and between|among)\s+(.+?)(?:whereas|recitals|witnesseth|the parties agree|now, therefore|now therefore)", intro, flags=re.I | re.S)
    if m:
        between_block = m.group(1)
    parties_excerpt = _excerpt_for_patterns(
        text,
        [r"between", r"by and between", r"among", r"in witness whereof", r"executed by", r"signed by"],
        window=500,
        max_chars=2400,
    )
    return _pick_best_party_value(between_block, intro, signature, parties_excerpt, _merge_party_values(between_block, intro, signature, parties_excerpt))


def _normalize_indemnification_prediction(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "NOT FOUND"
    return clean_indemnification(raw)


def _normalize_limitation_prediction(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "NOT FOUND"
    return clean_limitation(raw)


def _normalize_non_compete_prediction(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "NOT FOUND"
    return clean_non_compete(raw)


def _extract_governing_law_from_contract(contract_text: str) -> str:
    text = str(contract_text or "")
    if not text:
        return "NOT FOUND"

    focused = _excerpt_for_patterns(
        text,
        _METADATA_FIELD_PATTERNS["governing_law"] + [r"miscellaneous", r"general provisions", r"section\s+\d+[.:]?\s+governing law"],
        window=500,
        max_chars=3200,
    )
    signature = _extract_signature_block(text, max_chars=2200)
    tail = text[-3200:]
    candidates = [focused, tail, signature]
    for candidate in candidates:
        if not _looks_like_governing_law_clause(candidate):
            continue
        normalized = _title_case_jurisdiction(normalize_governing_law(candidate))
        if normalized != "NOT FOUND":
            return normalized
    return "NOT FOUND"


def _extract_expiration_from_contract(contract_text: str, agreement_date: str = "NOT FOUND", effective_date: str = "NOT FOUND") -> str:
    text = str(contract_text or "")
    if not text:
        return "NOT FOUND"

    focused = _excerpt_for_patterns(
        text,
        _EXPIRATION_FOCUS_PATTERNS + [r"termination election", r"notice of non-renewal", r"through the date"],
        window=700,
        max_chars=4200,
    )
    term_section = _excerpt_for_patterns(
        text,
        [r"\bterm\s*:", r"duration of agreement", r"shall continue for", r"shall have an initial term", r"shall have a term of", r"remain in effect for", r"continue in operation for", r"for an initial period", r"continue indefinitely", r"until terminated", r"shall be in effect until", r"shall terminate on", r"shall expire on", r"effective through(?: and including)?", r"for the term of the referenced", r"for the term of the lease", r"until the expiration or earlier termination of", r"continue until the termination of", r"until completion of", r"as long as fees are paid"],
        window=550,
        max_chars=2600,
    )
    candidates = [term_section, focused]
    normalized_candidates = []
    for candidate in candidates:
        normalized = _normalize_expiration_prediction(candidate, agreement_date=agreement_date, effective_date=effective_date)
        if normalized != "NOT FOUND":
            normalized_candidates.append(normalized)
    if not normalized_candidates:
        return "NOT FOUND"

    low_text = text.lower()
    if "Perpetual" in normalized_candidates and "continue indefinitely" in low_text and "terminate automatically one year after" not in low_text and "terminate automatically 1 year after" not in low_text:
        return "Perpetual"

    def _expiration_rank(value: str) -> int:
        if re.fullmatch(r"[A-Z][a-z]+ \d{2}, \d{4}", value or ""):
            return 5
        if "Initial Term" in (value or ""):
            return 4
        if value in {"Co-terminous with Related Agreement", "Until Completion of Program/Milestones"}:
            return 3
        if value in {"Perpetual", "Until Terminated"}:
            return 2
        if value == "Event-Based Termination":
            return 1
        return 0

    return max(normalized_candidates, key=_expiration_rank)


def _extract_indemnification_from_contract(contract_text: str) -> str:
    text = str(contract_text or "")
    if not text:
        return "NOT FOUND"

    focused = _excerpt_for_patterns(
        text,
        [r"indemnif(?:y|ies|ication)", r"hold harmless", r"save harmless", r"defend .* against", r"release,? defend,? indemnif", r"reimburse .* (?:claims|losses|damages|liabilities|costs)", r"not challenge", r"contest the validity", r"moral rights", r"not to sue", r"non-petition"],
        window=500,
        max_chars=2600,
    )
    normalized = _normalize_indemnification_prediction(focused or text)
    return normalized if normalized else "NOT FOUND"


def _extract_termination_for_convenience_from_contract(contract_text: str) -> str:
    text = str(contract_text or "")
    if not text:
        return "NOT FOUND"

    tfc_patterns = [
        r"termination for convenience",
        r"with or without cause",
        r"terminate[^\n.;]{0,40}without cause",
        r"without cause[^\n.;]{0,40}terminate",
        r"for any reason or no reason",
        r"for any reason upon",
        r"reserves the right to terminate",
        r"any party may terminate its participation",
        r"(?:either party|each party|both parties)[^\n.;]{0,40}?(?:may|shall have the right to|can)[^\n.;]{0,20}?terminate(?: this agreement| the agreement)?[^\n.;]{0,160}?(?:written notice|prior written notice|notice)",
        r"may be terminated by either party[^\n.;]{0,160}?(?:written notice|prior written notice|notice)",
        r"(?:customer|company|consultant|distributor|licensor|licensee|provider|recipient|contractor|vendor|reseller|manufacturer|buyer|seller|agency|affiliate|sparkling|principal|distributor)[^\n.;]{0,80}?may[^\n.;]{0,20}?terminate(?: this agreement| the agreement)?[^\n.;]{0,160}?(?:written notice|prior written notice|notice)",
        r"is terminating this agreement",
        r"terminate this agreement immediately upon written notice for any reason",
        r"upon one hundred eighty \(180\) days' written notice",
        r"upon \[\*\*\*\] written notice",
    ]
    focused = _excerpt_for_patterns(
        text,
        tfc_patterns,
        window=700,
        max_chars=3200,
    )
    normalized = _normalize_termination_prediction(focused or "")
    return normalized if normalized else "NOT FOUND"


def _contract_has_strong_tfc_support(contract_text: str) -> bool:
    low = str(contract_text or '').lower()
    strong_markers = [
        'termination for convenience', 'with or without cause',
        'for any reason or no reason', 'for any reason upon', 'for any reason upon written notice',
        'terminate this agreement immediately upon written notice for any reason',
        'any party may terminate its participation', 'upon 180 days written notice',
        "one hundred eighty (180) days' written notice", 'is terminating this agreement',
        'may be terminated by either party', 'either party may terminate', 'unless sooner terminated by either party'
    ]
    if 'for convenience of reference only' in low or 'headings are for convenience' in low:
        return False
    if re.search(r'terminat(?:e|ed|ion)[^.;]{0,40}without cause|without cause[^.;]{0,40}terminat', low):
        return True
    return any(marker in low for marker in strong_markers)

def _normalize_termination_prediction(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "NOT FOUND"
    return clean_termination(raw)


_LIMITATION_EXCLUDE_ORDER = [
    (r"consequential", "Consequential"),
    (r"incidental", "Incidental"),
    (r"indirect", "Indirect"),
    (r"special", "Special"),
    (r"punitive", "Punitive"),
    (r"exemplary", "Exemplary"),
    (r"lost\s+profit", "Lost Profits"),
    (r"lost\s+revenue", "Lost Revenue"),
    (r"loss\s+of\s+data|lost\s+data|data\s+loss", "Loss of Data"),
    (r"business\s+interrupt", "Business Interruption"),
]

_LIMITATION_EXCEPTION_ORDER = [
    (r"willful\s+misconduct|wilful\s+misconduct", "Willful Misconduct"),
    (r"gross\s+negligence", "Gross Negligence"),
    (r"fraud", "Fraud"),
    (r"confidential", "Confidentiality Breach"),
    (r"ip\s+infring|intellectual\s+property|infringement", "IP Infringement"),
    (r"indemn", "Indemnification"),
    (r"third\s+party", "Third Party Claims"),
    (r"mandatory\s+applicable\s+law|mandatory\s+law|applicable\s+law", "Mandatory Law"),
]

_LIMITATION_REMEDY_ORDER = [
    (r"direct\s+damages\s+only", "Direct Damages Only"),
    (r"sole\s+remedy", "Sole Remedy"),
    (r"exclusive\s+remedy", "Exclusive Remedy"),
    (r"sole\s+obligation", "Sole Obligation"),
]

_NON_COMPETE_SCOPE_ORDER = [
    (r"exclusiv", "Exclusivity"),
    (r"non[- ]?competition|competitive\s+business|competing\s+business", "Non-Competition"),
    (r"competitive\s+products?|competing\s+products?", "Competitive Products Restriction"),
    (r"endorse", "Non-Endorsement"),
    (r"non[- ]?solicit|solicit", "Non-Solicitation"),
    (r"no[- ]?hire|recruit|hire|employ", "No-Hire"),
    (r"territor|radius|mile", "Territorial Restriction"),
]


def _ordered_hits(text: str, ordered_patterns: list[tuple[str, str]]) -> list[str]:
    low = " ".join(str(text or "").lower().split())
    hits = []
    for pattern, label in ordered_patterns:
        if re.search(pattern, low) and label not in hits:
            hits.append(label)
    return hits


def _canonicalize_limitation_label(value: str) -> str:
    raw = str(value or "").strip()
    if not raw or raw.upper() == "NOT FOUND":
        return "NOT FOUND"
    if raw == "Yes":
        return "Yes"
    if not raw.lower().startswith("yes"):
        return raw

    excludes = []
    exceptions = []
    remedies = []
    cap = ""
    for part in [p.strip() for p in raw.split("|") if p.strip()]:
        low = part.lower()
        if low.startswith("excludes:"):
            excludes = _ordered_hits(part, _LIMITATION_EXCLUDE_ORDER)
        elif low.startswith("exceptions:"):
            exceptions = _ordered_hits(part, _LIMITATION_EXCEPTION_ORDER)
        elif low.startswith("remedy:"):
            remedies = _ordered_hits(part, _LIMITATION_REMEDY_ORDER)
        elif low.startswith("cap:"):
            cap = " ".join(part.split(":", 1)[1].split()).strip()

    out = ["Yes"]
    if excludes:
        out.append(f"Excludes: {', '.join(excludes)}")
    if cap:
        out.append(f"Cap: {cap}")
    if exceptions:
        out.append(f"Exceptions: {', '.join(exceptions)}")
    if remedies:
        out.append(f"Remedy: {', '.join(remedies)}")
    return " | ".join(out)


def _canonicalize_non_compete_label(value: str) -> str:
    raw = str(value or "").strip()
    if not raw or raw.upper() == "NOT FOUND":
        return "NOT FOUND"
    if not raw.lower().startswith("yes"):
        return raw

    restricted_party = ""
    duration = ""
    scope = []
    for part in [p.strip() for p in raw.split("|") if p.strip()]:
        low = part.lower()
        if low.startswith("restricted party:"):
            restricted_party = " ".join(part.split(":", 1)[1].split()).strip()
        elif low.startswith("duration:"):
            duration = " ".join(part.split(":", 1)[1].split()).strip()
            if duration:
                duration = duration[:1].lower() + duration[1:]
        elif low.startswith("scope:"):
            scope = _ordered_hits(part, _NON_COMPETE_SCOPE_ORDER)

    out = ["Yes"]
    if restricted_party:
        out.append(f"Restricted Party: {restricted_party}")
    if duration:
        out.append(f"Duration: {duration}")
    if scope:
        out.append(f"Scope: {', '.join(scope)}")
    return " | ".join(out)


def _canonicalize_indemnification_label(value: str) -> str:
    raw = str(value or "").strip()
    if not raw or raw.upper() == "NOT FOUND":
        return "NOT FOUND"
    low = " ".join(raw.lower().split())
    if raw.startswith("Yes") or any(k in low for k in ["indemnify", "hold harmless", "defend"]):
        return "Yes | Explicit indemnification / defense / hold harmless obligations"
    if "moral rights" in low:
        return "Moral Rights Waiver (not indemnification)"
    if "non-petition" in low or "bankruptcy" in low:
        return "Bankruptcy Non-Petition Covenant (not indemnification)"
    if "not to sue" in low or "covenant not to sue" in low:
        return "Covenant Not to Sue (not indemnification)"
    if any(k in low for k in ["tarnish", "goodwill", "disparage"]):
        return "Trademark Non-Disparagement Clause (not indemnification)"
    if any(k in low for k in ["challenge", "contest", "petition to cancel", "validity"]):
        return "IP Non-Challenge Clause (not indemnification)"
    return raw


def _canonicalize_termination_label(value: str) -> str:
    raw = str(value or "").strip()
    if not raw or raw.upper() == "NOT FOUND":
        return "NOT FOUND"
    if "|" not in raw:
        normalized = clean_termination(raw)
        return normalized if normalized else "NOT FOUND"

    low = " ".join(raw.lower().split())
    notice = ""
    for part in [p.strip() for p in raw.split("|") if p.strip()]:
        if part.lower().startswith("notice:"):
            notice = " ".join(part.split(":", 1)[1].split()).strip()
            break
    if not notice:
        m = re.search(r"notice:\s*([^|]+)", raw, flags=re.I)
        if m:
            notice = " ".join(m.group(1).split()).strip()

    party = "Either Party" if "either party" in low else ""
    without_cause = "Without Cause" if any(k in low for k in ["without cause", "for convenience", "for any reason"]) else ""

    out = ["Yes"]
    if notice:
        out.append(f"Notice: {notice}")
    if party:
        out.append(party)
    if without_cause:
        out.append(without_cause)
    return " | ".join(out)


def _canonicalize_governing_law_label(value: str) -> str:
    raw = _safe_text(value)
    if raw == "NOT FOUND":
        return raw
    if re.fullmatch(r"[A-Za-z .'-]+,\s*[A-Za-z .'-]+", raw):
        parts = [p.strip() for p in raw.split(",")]
        return ", ".join(_canonical_title(p) for p in parts if p)
    if re.fullmatch(r"[A-Za-z .'-]+", raw):
        return _canonical_title(raw)
    normalized = _normalize_governing_law(raw)
    return normalized if normalized != "NOT FOUND" else raw


def _canonicalize_expiration_label(value: str) -> str:
    raw = _safe_text(value)
    if raw == "NOT FOUND":
        return raw
    if re.fullmatch(r"[A-Z][a-z]+ \d{2}, \d{4}", raw):
        return raw
    if any(k in raw for k in [
        "Initial Term", "Auto-Renewal", "Co-terminous with Related Agreement",
        "Event-Based Termination", "Until Terminated", "Until Completion of Program/Milestones", "Perpetual"
    ]):
        return raw
    normalized = _normalize_expiration_prediction(raw)
    return normalized if normalized != "NOT FOUND" else raw


def _contract_has_strong_expiration_support(contract_text: str) -> bool:
    low = " ".join(str(contract_text or "").lower().split())
    markers = [
        'term of this agreement', 'term of the agreement', 'initial term', 'renewal term', 'expiration date',
        'shall continue for', 'continue for a period', 'remain in effect for', 'shall remain in effect for',
        'shall continue in force', 'will be in effect for', 'shall have a term of', 'shall be valid for',
        'co-termin', 'cotermin', 'until terminated', 'perpetual', 'perpetually thereafter', 'expire on',
        'expires on', 'shall expire on', 'terminate on', 'shall terminate on', 'scheduled expiration date',
        'contract end', 'end date', 'until completion of', 'royalty term', 'service period', 'duration of the lease'
    ]
    return any(m in low for m in markers)


def _normalize_gold_value(field: str, value: str) -> str:
    raw = _safe_text(value)
    if field in {"Agreement Date", "Effective Date"}:
        return _normalize_date_prediction(raw)
    if field == "Expiration Date":
        return _canonicalize_expiration_label(raw)
    if field == "Governing Law":
        return _canonicalize_governing_law_label(raw)
    if field == "Parties":
        return _normalize_parties_prediction(raw)
    if field == "Indemnification":
        return _canonicalize_indemnification_label(raw)
    if field == "Limitation Of Liability":
        return _canonicalize_limitation_label(raw)
    if field == "Non-Compete":
        return _canonicalize_non_compete_label(raw)
    if field == "Termination For Convenience":
        return _canonicalize_termination_label(raw)
    return raw


def _postprocess_metadata_prediction(pred, contract_text: str = ""):
    model_agreement_date = _normalize_date_prediction(getattr(pred, "agreement_date", "NOT FOUND"))
    heuristic_agreement_date = _extract_agreement_date_from_contract(contract_text)
    pred.agreement_date = heuristic_agreement_date if (heuristic_agreement_date != "NOT FOUND" and model_agreement_date == "NOT FOUND") else model_agreement_date

    model_effective_date = _normalize_date_prediction(getattr(pred, "effective_date", "NOT FOUND"))
    heuristic_effective_date = _extract_effective_date_from_contract(contract_text)
    contract_dates = set(extract_dates(contract_text)) if contract_text else set()
    low_contract = contract_text.lower() if contract_text else ""
    if heuristic_effective_date != "NOT FOUND" and (model_effective_date == "NOT FOUND" or model_effective_date not in contract_dates):
        pred.effective_date = heuristic_effective_date
    elif heuristic_effective_date == "NOT FOUND" and model_effective_date != "NOT FOUND" and (model_effective_date not in contract_dates or not _contract_has_strong_effective_support(contract_text)):
        pred.effective_date = "NOT FOUND"
    elif heuristic_effective_date == "NOT FOUND" and model_effective_date == "NOT FOUND" and pred.agreement_date != "NOT FOUND" and any(k in low_contract for k in ["on the date of this agreement", "signed and effective this", "entered into on", "entered into as of", "later of the dates that it is executed", "effective on the later of the dates that it is executed"]):
        pred.effective_date = pred.agreement_date
    else:
        pred.effective_date = model_effective_date

    model_gov = _title_case_jurisdiction(normalize_governing_law(getattr(pred, "governing_law", "NOT FOUND")))
    heuristic_gov = _extract_governing_law_from_contract(contract_text)
    pred.governing_law = heuristic_gov if (heuristic_gov != "NOT FOUND" and model_gov == "NOT FOUND") else model_gov

    model_exp = _normalize_expiration_prediction(
        getattr(pred, "expiration_date", "NOT FOUND"),
        agreement_date=pred.agreement_date,
        effective_date=pred.effective_date,
    )
    heuristic_exp = _extract_expiration_from_contract(
        contract_text,
        agreement_date=pred.agreement_date,
        effective_date=pred.effective_date,
    )
    if model_exp == "NOT FOUND" and heuristic_exp != "NOT FOUND":
        pred.expiration_date = heuristic_exp
    elif heuristic_exp != "NOT FOUND" and model_exp == "Event-Based Termination" and heuristic_exp != "Event-Based Termination":
        pred.expiration_date = heuristic_exp
    elif heuristic_exp != "NOT FOUND" and model_exp in {pred.agreement_date, pred.effective_date} and heuristic_exp != model_exp:
        pred.expiration_date = heuristic_exp
    elif heuristic_exp != "NOT FOUND" and 'Initial Term' in heuristic_exp and model_exp in {pred.agreement_date, pred.effective_date}:
        pred.expiration_date = heuristic_exp
    elif heuristic_exp != "NOT FOUND" and re.fullmatch(r"[A-Z][a-z]+ \d{2}, \d{4}", model_exp or "") and any(tag in heuristic_exp for tag in ["Initial Term", "Co-terminous with Related Agreement", "Until Completion of Program/Milestones", "Until Terminated", "Perpetual"]):
        pred.expiration_date = heuristic_exp
    elif heuristic_exp == "NOT FOUND" and model_exp != "NOT FOUND":
        if model_exp in {pred.agreement_date, pred.effective_date}:
            pred.expiration_date = "NOT FOUND"
        elif not _contract_has_strong_expiration_support(contract_text):
            pred.expiration_date = "NOT FOUND"
        else:
            pred.expiration_date = model_exp
    else:
        pred.expiration_date = model_exp

    model_ind = _normalize_indemnification_prediction(getattr(pred, "indemnification", "NOT FOUND"))
    heuristic_ind = _extract_indemnification_from_contract(contract_text)
    if model_ind == "NOT FOUND" and heuristic_ind != "NOT FOUND":
        pred.indemnification = heuristic_ind
    elif "not indemnification" in model_ind.lower() and heuristic_ind.startswith("Yes |"):
        pred.indemnification = heuristic_ind
    elif model_ind.startswith("Yes |") and heuristic_ind == "NOT FOUND":
        pred.indemnification = "NOT FOUND"
    else:
        pred.indemnification = model_ind

    model_lol = _normalize_limitation_prediction(getattr(pred, "limitation_of_liability", "NOT FOUND"))
    heuristic_lol = _extract_limitation_from_contract(contract_text)
    if heuristic_lol != "NOT FOUND":
        heuristic_lol_parts = heuristic_lol.count("|")
        model_lol_parts = model_lol.count("|") if model_lol != "NOT FOUND" else -1
        pred.limitation_of_liability = heuristic_lol if (model_lol == "NOT FOUND" or heuristic_lol_parts > model_lol_parts) else model_lol
    else:
        pred.limitation_of_liability = model_lol

    model_nc = _normalize_non_compete_prediction(getattr(pred, "non_compete", "NOT FOUND"))
    heuristic_nc = _extract_non_compete_from_contract(contract_text)
    if heuristic_nc != "NOT FOUND":
        heuristic_nc_parts = heuristic_nc.count("|")
        model_nc_parts = model_nc.count("|") if model_nc != "NOT FOUND" else -1
        pred.non_compete = heuristic_nc if (model_nc == "NOT FOUND" or heuristic_nc_parts > model_nc_parts) else model_nc
    else:
        pred.non_compete = model_nc

    model_parties = _normalize_parties_prediction(getattr(pred, "parties", "NOT FOUND"))
    heuristic_parties = _extract_parties_from_contract(contract_text)
    pred.parties = heuristic_parties if _count_party_items(heuristic_parties) > _count_party_items(model_parties) else model_parties

    model_tfc = _normalize_termination_prediction(getattr(pred, "termination_for_convenience", "NOT FOUND"))
    heuristic_tfc = _extract_termination_for_convenience_from_contract(contract_text)
    if heuristic_tfc != "NOT FOUND" and (model_tfc == "NOT FOUND" or heuristic_tfc.count('|') > model_tfc.count('|') or 'Notice: Immediate' in heuristic_tfc and 'Notice: Immediate' not in model_tfc):
        pred.termination_for_convenience = heuristic_tfc
    elif heuristic_tfc == "NOT FOUND" and model_tfc != "NOT FOUND" and not _contract_has_strong_tfc_support(contract_text):
        pred.termination_for_convenience = "NOT FOUND"
    else:
        pred.termination_for_convenience = model_tfc

    return pred


class ContractMetadataSignature(dspy.Signature):
    """Extract contract metadata fields from contract text.

    Rules:
    - Return concise normalized values only, never section numbers like "Section 19" and never raw clause paragraphs.
    - Use "NOT FOUND" only when the contract truly lacks the clause/value.
    - Agreement Date / Effective Date: return an exact normalized date like "June 08, 2010" when present; if the contract only says the date is not specified or references an undefined commencement date, return "NOT FOUND".
    - Expiration Date: return ONLY one of these normalized forms: exact date like "March 18, 2021"; "X-Year (Y months) Initial Term"; that same phrase with ", Auto-Renewal" appended if present; "Co-terminous with Related Agreement"; "Event-Based Termination"; or "NOT FOUND".
    - Governing Law: return only jurisdiction name (e.g., "Delaware", "Texas, United States", "England and Wales"), not full sentence; do not return venue, arbitration forum, or court names.
    - Indemnification: return a normalized indemnification result only if explicit indemnify / hold harmless / defend language exists; otherwise return "NOT FOUND" or a narrow "not indemnification" label when the clause is clearly one of those categories.
    - Limitation Of Liability: return a concise normalized summary like "Yes | Excludes: ... | Cap: ... | Exceptions: ..." or "NOT FOUND".
    - Non-Compete: return a concise normalized summary like "Yes | Restricted Party: ... | Duration: ... | Scope: ..." or "NOT FOUND".
    - Parties: this field is always expected; return the principal signatory/legal-entity names only, separated by " | ". Exclude role labels like Provider, Recipient, Customer, Distributor, Party, and exclude descriptive prose.
    - Termination For Convenience: return only a normalized summary like "Yes | Notice: 30 days | Either Party | Without Cause" or "NOT FOUND".
    """

    contract_text = dspy.InputField(desc="Full contract text")

    agreement_date = dspy.OutputField(desc="Agreement Date as normalized exact date like June 08, 2010, or NOT FOUND")
    effective_date = dspy.OutputField(desc="Effective Date as normalized exact date like June 08, 2010, or NOT FOUND")
    expiration_date = dspy.OutputField(desc="Expiration Date only in normalized schema: exact date, X-Year (Y months) Initial Term, optional Auto-Renewal, Co-terminous with Related Agreement, Event-Based Termination, or NOT FOUND")
    governing_law = dspy.OutputField(desc="Governing Law jurisdiction only in normalized form, or NOT FOUND")
    indemnification = dspy.OutputField(desc="Normalized indemnification result only, else NOT FOUND")
    limitation_of_liability = dspy.OutputField(desc="Concise normalized limitation-of-liability summary, or NOT FOUND")
    non_compete = dspy.OutputField(desc="Concise normalized non-compete summary, or NOT FOUND")
    parties = dspy.OutputField(desc="Principal party names only, separated by |, or NOT FOUND")
    termination_for_convenience = dspy.OutputField(desc="Normalized termination-for-convenience summary with notice/party if present, else NOT FOUND")


class BaselineMetadataModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(ContractMetadataSignature)

    def forward(self, contract_text):
        focused_context = _build_metadata_context(contract_text)
        pred = self.predictor(contract_text=focused_context)
        return _postprocess_metadata_prediction(pred, contract_text)


class StudentMetadataModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(ContractMetadataSignature)

    def forward(self, contract_text):
        focused_context = _build_metadata_context(contract_text)
        pred = self.predictor(contract_text=focused_context)
        return _postprocess_metadata_prediction(pred, contract_text)


class HybridMetadataModule(dspy.Module):
    def __init__(self, baseline_module: dspy.Module, optimized_module: dspy.Module, weak_fields: list[str]):
        super().__init__()
        self.baseline_module = baseline_module
        self.optimized_module = optimized_module
        self.weak_fields = list(weak_fields)

    def forward(self, contract_text):
        base_pred = self.baseline_module(contract_text=contract_text)
        opt_pred = self.optimized_module(contract_text=contract_text)
        for field in self.weak_fields:
            attr = FIELD_TO_ATTR[field]
            setattr(base_pred, attr, getattr(opt_pred, attr, getattr(base_pred, attr, "NOT FOUND")))
        return base_pred


def _field_score(gold: str, pred: str, field: str, use_llm=True, gold_agreement_date: str | None = None, gold_effective_date: str | None = None, pred_agreement_date: str | None = None, pred_effective_date: str | None = None):
    g = _safe_text(gold)
    p = _safe_text(pred)

    if field == "Governing Law":
        rule_score = governing_law_match_score(p, g)
        if rule_score >= 1.0 or not use_llm:
            return rule_score
        llm_score = llm_semantic_match_score(p, [g], field) if use_llm else None
        return max(rule_score, llm_score if llm_score is not None else 0.0)

    if field == "Indemnification":
        rule_score = indemnification_match_score(p, g)
        if rule_score >= 1.0 or not use_llm:
            return rule_score
        llm_score = llm_semantic_match_score(p, [g], field) if use_llm else None
        return max(rule_score, llm_score if llm_score is not None else 0.0)

    if field in {"Agreement Date", "Effective Date"}:
        return date_match_score(p, g)

    if field == "Expiration Date":
        rule_score = expiration_match_score(p, g, pred_agreement_date=pred_agreement_date, pred_effective_date=pred_effective_date, gold_agreement_date=gold_agreement_date, gold_effective_date=gold_effective_date)
        if rule_score >= 1.0 or not use_llm:
            return rule_score
        llm_score = llm_semantic_match_score(p, [g], field) if use_llm else None
        return max(rule_score, llm_score if llm_score is not None else 0.0)

    if field == "Non-Compete":
        rule_score = non_compete_match_score(p, g)
        if rule_score >= 1.0 or not use_llm:
            return rule_score
        llm_score = llm_semantic_match_score(p, [g], field) if use_llm else None
        return max(rule_score, llm_score if llm_score is not None else 0.0)

    if field == "Parties":
        rule_score = parties_match_score(p, g)
        if rule_score >= 1.0 or not use_llm:
            return rule_score
        llm_score = llm_semantic_match_score(p, [g], field) if use_llm else None
        return max(rule_score, llm_score if llm_score is not None else 0.0)

    if field == "Termination For Convenience":
        rule_score = termination_for_convenience_match_score(p, g)
        if rule_score >= 1.0 or not use_llm:
            return rule_score
        llm_score = llm_semantic_match_score(p, [g], field) if use_llm else None
        return max(rule_score, llm_score if llm_score is not None else 0.0)

    if field == "Limitation Of Liability":
        rule_score = limitation_of_liability_match_score(p, g)
        if rule_score >= 1.0 or not use_llm:
            return rule_score
        llm_score = llm_semantic_match_score(p, [g], field) if use_llm else None
        return max(rule_score, llm_score if llm_score is not None else 0.0)

    if use_llm:
        llm_score = llm_semantic_match_score(p, [g], field)
        if llm_score is not None:
            return llm_score

    return 1.0 if normalize_text(g) == normalize_text(p) else 0.0


def validate_metadata_prediction(example, pred, trace=None):
    # Weighted lexical metric for optimization loop using the same field-aware rules as evaluation.
    weighted_sum = 0.0
    total_weight = 0.0
    for field in WEAK_OPT_FIELDS:
        attr = FIELD_TO_ATTR[field]
        g = getattr(example, attr, "NOT FOUND")
        p = getattr(pred, attr, "NOT FOUND")
        if field == "Governing Law":
            gold_eval = normalize_governing_law(g)
            pred_eval = normalize_governing_law(p)
        else:
            gold_eval = g
            pred_eval = p
        s = _field_score(gold_eval, pred_eval, field, use_llm=False, gold_agreement_date=getattr(example, "agreement_date", "NOT FOUND"), gold_effective_date=getattr(example, "effective_date", "NOT FOUND"), pred_agreement_date=getattr(pred, "agreement_date", "NOT FOUND"), pred_effective_date=getattr(pred, "effective_date", "NOT FOUND"))
        w = FIELD_WEIGHTS.get(field, 1.0)
        weighted_sum += s * w
        total_weight += w
    return (weighted_sum / total_weight) if total_weight else 0.0


def evaluate_module(module, testset, name: str, use_llm=True):
    results = []
    per_field = {f: [] for f in FIELDS}

    for i, ex in enumerate(testset):
        try:
            pred = module(contract_text=ex.contract_text)
            row_score = []
            row_detail = {}
            for field in FIELDS:
                attr = FIELD_TO_ATTR[field]
                gold = getattr(ex, attr, "NOT FOUND")
                pv = getattr(pred, attr, "NOT FOUND")
                if field == "Governing Law":
                    gold_eval = normalize_governing_law(gold)
                    pred_eval = normalize_governing_law(pv)
                else:
                    gold_eval = gold
                    pred_eval = pv
                s = _field_score(gold_eval, pred_eval, field, use_llm=use_llm, gold_agreement_date=getattr(ex, "agreement_date", "NOT FOUND"), gold_effective_date=getattr(ex, "effective_date", "NOT FOUND"), pred_agreement_date=getattr(pred, "agreement_date", "NOT FOUND"), pred_effective_date=getattr(pred, "effective_date", "NOT FOUND"))
                per_field[field].append(s)
                row_score.append(s)
                row_detail[field] = {
                    "gold": _safe_text(gold_eval),
                    "pred": _safe_text(pred_eval),
                    "score": s,
                }

            overall = sum(row_score) / len(row_score) if row_score else 0.0
            results.append({
                "index": i,
                "contract_id": getattr(ex, "contract_id", ""),
                "details": row_detail,
                "overall_score": overall,
            })
        except Exception as e:
            results.append({
                "index": i,
                "contract_id": getattr(ex, "contract_id", ""),
                "error": str(e),
                "overall_score": 0.0,
            })
            for field in FIELDS:
                per_field[field].append(0.0)

    total = len(testset)
    per_field_avg = {
        f: (sum(vals) / len(vals) if vals else 0.0)
        for f, vals in per_field.items()
    }
    overall = sum(r["overall_score"] for r in results) / total if total else 0.0

    return EvaluationResult(
        name=name,
        results=results,
        field_accuracies={f: per_field_avg[f] for f in FIELDS},
        overall_accuracy=overall,
        total_samples=total,
        metadata={
            "per_field_accuracy": per_field_avg,
            "llm_eval_used": bool(use_llm),
        },
    )


def print_summary(result: EvaluationResult):
    print("\n" + "=" * 50)
    print(f"Evaluation Summary: {result.name}")
    print("=" * 50)
    print(f"Overall Accuracy: {result.overall_accuracy:.2%}")
    print("Per-field accuracy:")
    per = result.metadata.get("per_field_accuracy", {})
    for field, score in sorted(per.items()):
        print(f"  {field:28} {score:.2%}")


def _render_signature_prompt_text() -> str:
    prompt_path = Path(__file__).resolve().parent / "prompts" / "baseline.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    sig = ContractMetadataSignature
    lines = [
        "# Baseline Metadata Prompt (Signature)",
        "",
        (sig.__doc__ or "").strip(),
        "",
        "## Fields",
    ]
    for field in FIELDS:
        lines.append(f"- {field}")

    model_fields = getattr(sig, "model_fields", {}) or {}
    if model_fields:
        lines.append("")
        lines.append("## Field Metadata")
        for fname, fobj in model_fields.items():
            desc = getattr(fobj, "description", "") or ""
            if desc:
                lines.append(f"- {fname}: {desc}")

    return "\n".join(lines).strip() + "\n"


def _extract_optimized_prompt_text(optimized_module: dspy.Module) -> str:
    predictor = getattr(optimized_module, "predictor", None)
    parts = ["# Optimized Metadata Prompt", ""]

    baseline_prompt = _render_signature_prompt_text().strip()
    parts.append("## Base Prompt")
    parts.append("")
    parts.append("```text")
    parts.append(baseline_prompt)
    parts.append("```")
    parts.append("")

    signature = getattr(predictor, "signature", None)
    if signature is not None:
        sig_doc = (getattr(signature, "__doc__", "") or "").strip()
        if sig_doc and sig_doc not in baseline_prompt:
            parts.append("## Optimized Instructions")
            parts.append("")
            parts.append("```text")
            parts.append(sig_doc)
            parts.append("```")
            parts.append("")

    found_instruction = False
    for attr in ("instructions", "_instructions", "optimized_prompt"):
        val = getattr(predictor, attr, None)
        if isinstance(val, str) and val.strip():
            found_instruction = True
            parts.append(f"## {attr}")
            parts.append("")
            parts.append("```text")
            parts.append(val.strip())
            parts.append("```")
            parts.append("")

    demos = getattr(predictor, "demos", None) or []
    if demos:
        parts.append(f"## Few-Shot Examples ({len(demos)})")
        parts.append("")
        for idx, demo in enumerate(demos, start=1):
            parts.append(f"### Example {idx}")
            parts.append("")
            if hasattr(demo, "inputs"):
                demo_inputs = demo.inputs()
                parts.append("**Inputs**")
                parts.append("```json")
                parts.append(json.dumps(demo_inputs, ensure_ascii=False, indent=2, default=str))
                parts.append("```")
                parts.append("")
            if hasattr(demo, "labels"):
                demo_labels = demo.labels()
                parts.append("**Outputs**")
                parts.append("```json")
                parts.append(json.dumps(demo_labels, ensure_ascii=False, indent=2, default=str))
                parts.append("```")
                parts.append("")

    if not found_instruction and not demos:
        parts.append("No explicit optimized instruction string exposed by DSPy object.")
        parts.append("The optimized artifact currently differs mainly through selected demonstrations and DSPy-internal state.")
        parts.append("")

    return "\n".join(parts).strip() + "\n"


def _save_prompt_artifacts(output_dir: Path, baseline_module: dspy.Module, optimized_module: dspy.Module):
    baseline_prompt = _render_signature_prompt_text()
    optimized_prompt = _extract_optimized_prompt_text(optimized_module)

    baseline_path = output_dir / "baseline_prompt.txt"
    optimized_path = output_dir / "optimized_prompt.txt"
    compare_path = output_dir / "prompt_comparison.md"

    baseline_path.write_text(baseline_prompt, encoding="utf-8")
    optimized_path.write_text(optimized_prompt, encoding="utf-8")

    comparison = (
        "# Prompt Comparison\n\n"
        "## Baseline Prompt\n\n"
        f"```text\n{baseline_prompt}\n```\n\n"
        "## Optimized Prompt\n\n"
        f"```text\n{optimized_prompt}\n```\n"
    )
    compare_path.write_text(comparison, encoding="utf-8")

def _save_runtime_prompt_history(output_dir: Path):
    """Save actual LM prompt history captured during the run."""
    lm = dspy.settings.lm
    history = getattr(lm, "history", None)
    if not isinstance(history, list) or not history:
        return

    raw_path = output_dir / "dspy_runtime_history.json"
    raw_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Best-effort plain text extraction of prompt/messages per call.
    prompts_txt = []
    for i, item in enumerate(history, start=1):
        prompts_txt.append(f"===== CALL {i} =====")
        if isinstance(item, dict):
            if item.get("prompt"):
                prompts_txt.append("[prompt]")
                prompts_txt.append(str(item.get("prompt")))
            if item.get("messages"):
                prompts_txt.append("[messages]")
                prompts_txt.append(json.dumps(item.get("messages"), ensure_ascii=False, indent=2, default=str))
            if item.get("response"):
                prompts_txt.append("[response]")
                prompts_txt.append(str(item.get("response")))
        else:
            prompts_txt.append(str(item))
        prompts_txt.append("")

    txt_path = output_dir / "dspy_runtime_prompts.txt"
    txt_path.write_text("\n".join(prompts_txt), encoding="utf-8")

def run_metadata_dspy(
    train_csv: Path,
    test_csv: Path,
    metadata_csv: Path | None,
    cache_dir: str,
    output_dir: Path,
    use_llm_eval: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    baseline_only: bool,
    smoke_run: bool,
    medium_run: bool,
    use_copro: bool,
    eval_all_rows: bool,
    report_all_rows: bool,
):
    print("=" * 70)
    print("LEGAL CONTRACT METADATA DSPY PIPELINE")
    print("=" * 70)

    print("\n[1/5] Setting up LLM...")
    lm_config = get_llm_config()
    lm_config['max_tokens'] = max(int(lm_config.get('max_tokens') or 800), 1400)
    setup_dspy_lm(lm_config)

    print("[2/5] Loading metadata datasets + CUAD contract text...")
    text_map = _build_contract_text_map(cache_dir)
    all_examples = None
    if metadata_csv is not None:
        all_examples = load_metadata_examples(metadata_csv, text_map)
        print(f"      Source metadata CSV: {metadata_csv}")
        if eval_all_rows:
            testset = list(all_examples)
            full_trainset = list(all_examples)
            baseline_only = True
            print("      Eval-all-rows mode: evaluating on all loaded metadata rows")
            print("      Optimization disabled in eval-all-rows mode to avoid train/test leakage")
        else:
            full_trainset, testset = _split_examples_train_test(
                all_examples,
                test_ratio=test_ratio,
                seed=seed,
            )
    else:
        full_trainset = load_metadata_examples(train_csv, text_map)
        testset = load_metadata_examples(test_csv, text_map)
        if eval_all_rows:
            raise ValueError("--eval-all-rows requires --metadata-csv so the full reviewed dataset can be loaded directly")
    if not full_trainset or not testset:
        raise ValueError("Empty train/test set after loading metadata + contract text map")

    if not eval_all_rows:
        if smoke_run:
            full_trainset = full_trainset[:16]
            testset = testset[:8]
            print("      Smoke run: using train=16, test=8 before optimization split")
        elif medium_run:
            full_trainset = full_trainset[:48]
            testset = testset[:16]
            print("      Medium run: using train=48, test=16 before optimization split")

    opt_trainset, valset = _train_val_split(full_trainset, val_ratio=val_ratio, seed=seed)
    if not eval_all_rows:
        if smoke_run:
            opt_trainset = opt_trainset[:8]
            valset = valset[:4]
        elif medium_run:
            opt_trainset = opt_trainset[:32]
            valset = valset[:8]
    print(f"      Train (input): {len(full_trainset)} contracts, Test: {len(testset)} contracts")
    print(f"      Optimization split: train={len(opt_trainset)}, validation={len(valset)}")

    print("[3/5] Evaluating baseline...")
    baseline = BaselineMetadataModule()
    baseline_results = evaluate_module(baseline, testset, "BaselineMetadata", use_llm=use_llm_eval)
    print_summary(baseline_results)

    # Validation anchor for safer model selection.
    baseline_val = evaluate_module(baseline, valset, "BaselineValidation", use_llm=False) if valset else None

    if baseline_only:
        print("[4/5] Skipping DSPy optimization (--baseline-only)...")
        final_module = baseline
        final_name = "BaselineMetadataOnly"
    else:
        print("[4/5] Optimizing with DSPy...")
        student = StudentMetadataModule()
        run_profile = "smoke" if smoke_run else ("medium" if medium_run else "full")
        bootstrap_config = dict(config.METADATA_BOOTSTRAP_CONFIGS[run_profile])
        effective_use_copro = use_copro or (config.METADATA_USE_COPRO and run_profile == "full")
        copro_config = dict(config.METADATA_COPRO_CONFIGS[run_profile]) if effective_use_copro else None
        print(
            f"      DSPy config: profile={run_profile}, COPRO={'on' if effective_use_copro else 'off'}, "
            f"bootstrap={bootstrap_config}, copro={copro_config if copro_config is not None else 'None'}"
        )
        optimized_student = run_two_stage_optimization(
            student_module=student,
            trainset=opt_trainset,
            metric=validate_metadata_prediction,
            bootstrap_config=bootstrap_config,
            copro_config=copro_config,
        )

        # Keep optimized model only if it beats baseline on lexical validation.
        hybrid_optimized = HybridMetadataModule(baseline, optimized_student, WEAK_OPT_FIELDS)
        if valset:
            optimized_val = evaluate_module(hybrid_optimized, valset, "DSPyValidation", use_llm=False)
            print(f"      Validation baseline: {baseline_val.overall_accuracy:.2%}")
            print(f"      Validation DSPy:     {optimized_val.overall_accuracy:.2%}")
            if optimized_val.overall_accuracy < baseline_val.overall_accuracy:
                print("      DSPy underperformed on validation. Falling back to baseline for final test evaluation.")
                final_module = baseline
                final_name = "DSPyMetadata(FallbackBaseline)"
            else:
                final_module = hybrid_optimized
                final_name = "DSPyMetadata(HybridWeakFields)"
        else:
            final_module = hybrid_optimized
            final_name = "DSPyMetadata(HybridWeakFields)"

    print("[5/5] Evaluating optimized...")
    optimized_results = evaluate_module(final_module, testset, final_name, use_llm=use_llm_eval)
    print_summary(optimized_results)

    comparison = compare_results(baseline_results, optimized_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_results_json(baseline_results, output_dir / "baseline_metadata_results.json")
    save_results_json(optimized_results, output_dir / "dspy_metadata_results.json")
    save_results_json(comparison, output_dir / "comparison_metadata_results.json")

    all_rows_results = None
    if report_all_rows:
        if all_examples is None:
            raise ValueError("--report-all-rows requires --metadata-csv so the full reviewed dataset can be loaded directly")
        print("      Running final module on all loaded metadata rows for reporting...")
        all_rows_results = evaluate_module(final_module, all_examples, f"{final_name}(AllRows)", use_llm=use_llm_eval)
        print_summary(all_rows_results)
        save_results_json(all_rows_results, output_dir / "all_rows_metadata_results.json")

    _save_prompt_artifacts(output_dir, baseline, final_module if not baseline_only else baseline)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Baseline overall:  {baseline_results.overall_accuracy:.2%}")
    print(f"DSPy overall:      {optimized_results.overall_accuracy:.2%}")
    print(f"Improvement:       {(optimized_results.overall_accuracy - baseline_results.overall_accuracy):+.2%}")
    if all_rows_results is not None:
        print(f"All-rows overall:  {all_rows_results.overall_accuracy:.2%}")
    print(f"Saved to: {output_dir}")

    summary = save_lm_history_artifacts(output_dir)
    if summary:
        print(f"      Token summary saved to: {output_dir / 'dspy_runtime_usage_summary.json'}")
        enforce_usage_budget(summary)


def main():
    parser = argparse.ArgumentParser(description="Run DSPy metadata pipeline on contract metadata CSV files")
    parser.add_argument(
        "--metadata-csv",
        default="experiments/legal_contract/data/reviewed/contract_metadata_cleaned_3_resolved.csv",
        help="Single reviewed metadata CSV to split into train/test internally. Set empty string to use explicit --train-csv/--test-csv instead.",
    )
    parser.add_argument(
        "--train-csv",
        default="experiments/legal_contract/data/reviewed/contract_metadata_cleaned_train.csv",
        help="Train metadata CSV when not using --metadata-csv",
    )
    parser.add_argument(
        "--test-csv",
        default="experiments/legal_contract/data/reviewed/contract_metadata_cleaned_test.csv",
        help="Test metadata CSV when not using --metadata-csv",
    )
    parser.add_argument(
        "--cache-dir",
        default="experiments/legal_contract/data/cache",
        help="HF datasets cache dir",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/legal_contract/results/metadata_dspy",
        help="Output directory",
    )
    parser.add_argument(
        "--no-llm-eval",
        action="store_true",
        help="Use lexical-only evaluation (faster)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio from training set for safer model selection",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio when using --metadata-csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation split",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip DSPy optimization and run baseline metadata extraction only.",
    )
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        help="Use a much smaller metadata split and cheaper optimization settings.",
    )
    parser.add_argument(
        "--medium-run",
        action="store_true",
        help="Use a medium-sized metadata split for a more realistic DSPy comparison without going full scale.",
    )
    parser.add_argument(
        "--use-copro",
        action="store_true",
        help="Enable COPRO prompt optimization. By default metadata uses BootstrapFewShot only.",
    )
    parser.add_argument(
        "--eval-all-rows",
        action="store_true",
        help="Evaluate on all rows from --metadata-csv. This disables optimization automatically to avoid train/test leakage.",
    )
    parser.add_argument(
        "--report-all-rows",
        action="store_true",
        help="After split-based baseline/DSPy evaluation, also run the chosen final module on all rows from --metadata-csv and save a separate all-rows report.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    try:
        run_metadata_dspy(
            train_csv=Path(args.train_csv),
            test_csv=Path(args.test_csv),
            metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
            cache_dir=args.cache_dir,
            output_dir=output_dir,
            use_llm_eval=not args.no_llm_eval,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            baseline_only=args.baseline_only,
            smoke_run=args.smoke_run,
            medium_run=args.medium_run,
            use_copro=args.use_copro,
            eval_all_rows=args.eval_all_rows,
            report_all_rows=args.report_all_rows,
        )
    finally:
        summary = save_lm_history_artifacts(output_dir)
        if summary:
            print(f"      Token summary saved to: {output_dir / 'dspy_runtime_usage_summary.json'}")
            enforce_usage_budget(summary)


if __name__ == "__main__":
    main()
