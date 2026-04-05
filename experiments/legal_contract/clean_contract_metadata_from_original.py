from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

MONTHS = {
    'january': 'January', 'february': 'February', 'march': 'March', 'april': 'April',
    'may': 'May', 'june': 'June', 'july': 'July', 'august': 'August',
    'september': 'September', 'october': 'October', 'november': 'November', 'december': 'December',
}
US_STATES = {
    'alabama','alaska','arizona','arkansas','california','colorado','connecticut','delaware','florida','georgia',
    'hawaii','idaho','illinois','indiana','iowa','kansas','kentucky','louisiana','maine','maryland','massachusetts',
    'michigan','minnesota','mississippi','missouri','montana','nebraska','nevada','new hampshire','new jersey',
    'new mexico','new york','north carolina','north dakota','ohio','oklahoma','oregon','pennsylvania','rhode island',
    'south carolina','south dakota','tennessee','texas','utah','vermont','virginia','washington','west virginia',
    'wisconsin','wyoming','district of columbia'
}
ENTITY_SUFFIXES = (
    'inc', 'inc.', 'incorporated', 'corp', 'corp.', 'corporation', 'llc', 'l.l.c.', 'ltd', 'ltd.', 'limited',
    'lp', 'l.p.', 'llp', 'plc', 'gmbh', 'ag', 'sa', 'nv', 'bv', 'pte', 'pty', 'company', 'co.', 'co', 's.a.',
)
ROLE_WORDS = {
    'party', 'parties', 'provider', 'recipient', 'consultant', 'owner', 'servicer', 'licensor', 'licensee',
    'seller', 'buyer', 'distributor', 'company', 'customer', 'supplier'
}
NOT_FOUND_LABELS = {
    'ip non-challenge clause (not indemnification)': 'IP Non-Challenge Clause (not indemnification)',
    'trademark non-disparagement clause (not indemnification)': 'Trademark Non-Disparagement Clause (not indemnification)',
    'moral rights waiver (not indemnification)': 'Moral Rights Waiver (not indemnification)',
    'covenant not to sue (not indemnification)': 'Covenant Not to Sue (not indemnification)',
    'bankruptcy non-petition covenant (not indemnification)': 'Bankruptcy Non-Petition Covenant (not indemnification)',
}


def collapse_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


NUMBER_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
    'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'twenty four': 24
}


def _number_word_to_int(text: str):
    t = collapse_ws(text).lower().replace('-', ' ')
    if t.isdigit():
        return int(t)
    if t in NUMBER_WORDS:
        return NUMBER_WORDS[t]
    parts = [p for p in t.split() if p]
    total = 0
    current = 0
    for part in parts:
        if part not in NUMBER_WORDS:
            continue
        val = NUMBER_WORDS[part]
        if val == 100:
            current = max(1, current) * val
        else:
            current += val
    total += current
    return total or None


def _extract_notice_phrase(lower: str):
    patterns = [
        r'at least\s+([a-z -]+)\s*\((\d{1,3})\)\s*(business\s+days|calendar\s+days|days|months|years)',
        r'not less than\s+([a-z -]+)\s*\((\d{1,3})\)\s*(business\s+days|calendar\s+days|days|months|years)',
        r'([a-z -]+)\s*\((\d{1,3})\)\s*(business\s+days|calendar\s+days|days|months|years)',
        r'(\d{1,3})\s*\((?:[a-z -]+)\)\s*(business\s+days|calendar\s+days|days|months|years)',
        r'(\d{1,3})\s*(business\s+days|calendar\s+days|days|months|years)',
        r'([a-z -]+)\s*(business\s+days|calendar\s+days|days|months|years)',
    ]
    for pat in patterns:
        m = re.search(pat, lower)
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        unit = groups[-1]
        value = None
        numeric_groups = [g for g in groups[:-1] if str(g).strip().isdigit()]
        if numeric_groups:
            value = int(numeric_groups[-1])
        else:
            for g in groups[:-1]:
                value = _number_word_to_int(g)
                if value is not None:
                    break
        if value is not None:
            return f'Notice: {value} {unit}'
    if 'immediately' in lower or 'immediate effect' in lower or 'forthwith' in lower:
        return 'Notice: Immediate'
    if 'written or e-mail notice' in lower or 'written notice' in lower or 'upon notice' in lower or 'by notice' in lower:
        return 'Notice: Not specified'
    return None


def _extract_tfc_initiator(lower: str):
    either_markers = ['either party', 'each party', 'both parties', 'any party']
    if any(m in lower for m in either_markers):
        return 'Either Party'
    named_patterns = [
        (r'bank of america may terminate', 'Initiating Party: Bank of America'),
        (r'customer may terminate', 'Initiating Party: Customer'),
        (r'company may terminate', 'Initiating Party: Company'),
        (r'consultant may terminate', 'Initiating Party: Consultant'),
        (r'distributor may terminate', 'Initiating Party: Distributor'),
        (r'licensor may terminate', 'Initiating Party: Licensor'),
        (r'licensee may terminate', 'Initiating Party: Licensee'),
        (r'provider may terminate', 'Initiating Party: Provider'),
        (r'recipient may terminate', 'Initiating Party: Recipient'),
        (r'contractor may terminate', 'Initiating Party: Contractor'),
        (r'vendor may terminate', 'Initiating Party: Vendor'),
        (r'reseller may terminate', 'Initiating Party: Reseller'),
        (r'manufacturer may terminate', 'Initiating Party: Manufacturer'),
        (r'buyer may terminate', 'Initiating Party: Buyer'),
        (r'seller may terminate', 'Initiating Party: Seller'),
        (r'affiliate may terminate', 'Initiating Party: Affiliate'),
        (r'board of trustees', 'Initiating Party: Board of Trustees'),
    ]
    for pat, label in named_patterns:
        if re.search(pat, lower):
            return label
    m = re.search(r'([a-z][a-z0-9& .,-]{2,40}) may terminate', lower)
    if not m:
        return None
    actor = collapse_ws(m.group(1)).strip(' ,.;:-').title()
    bad_tokens = ['section', 'provisions', 'non-defaulting', 'defaulting', 'notice', 'termination', 'agreement', 'agent shall', 'micoa', 'efforts to cure']
    if len(actor) > 28 or any(tok in actor.lower() for tok in bad_tokens):
        return None
    return f'Initiating Party: {actor}'


def is_not_found(value) -> bool:
    text = collapse_ws('' if value is None else str(value))
    return not text or text.upper() == 'NOT FOUND' or text.lower() == 'nan'


def format_date(value) -> str:
    if value is None:
        return 'NOT FOUND'
    if hasattr(value, 'strftime'):
        return value.strftime('%B %d, %Y')
    text = collapse_ws(str(value)).strip(' ,.;()')
    if is_not_found(text):
        return 'NOT FOUND'

    # Normalize common OCR / legal drafting variants before parsing.
    text = re.sub(r'(\d{1,2})\s*t\s*h\b', r'\1th', text, flags=re.I)
    text = re.sub(r'([A-Za-z]+)\.\s+(\d{1,2})(?:st|nd|rd|th)?\b', r'\1 \2', text)
    text = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\s*[.,]\s*(\d{4})\b', r'\1 \2, \3', text, flags=re.I)
    text = re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+(?:[A-Za-z]+,\s+)?(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{3,4}|\d{3}_)\b', r'\2 \1, \3', text, flags=re.I)
    text = re.sub(r'\b(\d{1,2})\s+day\s+of\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{3,4}|\d{3}_)\b', r'\2 \1, \3', text, flags=re.I)
    text = re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+day\s+of\s+(\d{3,4}|\d{3}_)\b', r'\2 \1, \3', text, flags=re.I)
    text = re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b$', r'\2 \1', text, flags=re.I)

    if re.fullmatch(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b', text, flags=re.I):
        return 'NOT FOUND'
    masked_month_day_year = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{3}_)\b', text, flags=re.I)
    if masked_month_day_year:
        return f"{masked_month_day_year.group(1).capitalize()} {int(masked_month_day_year.group(2)):02d}, {masked_month_day_year.group(3)}"

    parsed = pd.to_datetime(text, errors='coerce')
    if pd.notna(parsed):
        return parsed.strftime('%B %d, %Y')

    patterns = [
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s+\d{4}',
        r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{1,2}\.\d{1,2}\.\d{2,4}',
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if not m:
            continue
        candidate = m.group(0)
        if '.' in candidate and re.fullmatch(r'\d{1,2}\.\d{1,2}\.\d{2,4}', candidate):
            day, month, year = candidate.split('.')
            year = int(year) + (2000 if len(year) == 2 and int(year) < 40 else 1900 if len(year) == 2 else 0)
            year = int(year) if len(str(year)) == 4 else year
            try:
                parsed = pd.Timestamp(year=int(year), month=int(month), day=int(day))
            except Exception:
                parsed = pd.to_datetime(candidate, errors='coerce', dayfirst=True)
        else:
            parsed = pd.to_datetime(candidate, errors='coerce', dayfirst=False)
        if pd.isna(parsed):
            parsed = pd.to_datetime(candidate, errors='coerce', dayfirst=True)
        if pd.notna(parsed):
            return parsed.strftime('%B %d, %Y')

    full_partial = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+([_*\[\]??]+|__+)\s*,?\s*(\d{4})\b', text, flags=re.I)
    if full_partial:
        return f'{full_partial.group(1).capitalize()} ____, {full_partial.group(3)}'
    day_masked = re.search(r'\b(\d{1,2})\s+day\s+of\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{3}_)\b', text, flags=re.I)
    if day_masked:
        return f'{day_masked.group(2).capitalize()} {int(day_masked.group(1)):02d}, {day_masked.group(3)}'
    month_year = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s+(\d{4})\b', text, flags=re.I)
    if month_year and not re.search(r'\b\d{1,2}\b', text):
        return f'{month_year.group(1).capitalize()} ____, {month_year.group(2)}'
    three_digit_year = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{3})\b', text, flags=re.I)
    if three_digit_year:
        return f'{three_digit_year.group(1).capitalize()} {int(three_digit_year.group(2)):02d}, {three_digit_year.group(3)}_'

    if any(k in text.lower() for k in ['later of the two signature dates', 'date first set forth above', 'first above written', 'first set forth above', 'date of commencement', 'effective upon the completion', 'closing date', 'date of last signing']) and not re.search(r'\d{4}', text):
        return 'NOT FOUND'
    if re.fullmatch(r'(?:this\s+day\s+of\s*,?\s*\d{2,4}|\[.?\]\s*day\s*of\s*\[.?\],?\s*\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Za-z]+|____\s+day\s+of\s+[A-Za-z]+,\s+\d{4}|this\s+_+\s+day\s+of\s+_+,?\s*\d{2,4})', text, flags=re.I):
        return 'NOT FOUND'
    return text
def _parse_date_candidate(candidate: str):
    candidate = collapse_ws(candidate)
    try:
        if re.fullmatch(r'\d{1,2}\.\d{1,2}\.\d{2,4}', candidate):
            day, month, year = candidate.split('.')
            if len(year) == 2:
                year = str(int(year) + (2000 if int(year) < 40 else 1900))
            return pd.Timestamp(year=int(year), month=int(month), day=int(day))
        if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{2,4}', candidate):
            left, middle, year = candidate.split('/')
            if len(year) == 2:
                year = str(int(year) + (2000 if int(year) < 40 else 1900))
            if int(left) > 12:
                day, month = int(left), int(middle)
            else:
                month, day = int(left), int(middle)
            return pd.Timestamp(year=int(year), month=month, day=day)
        if re.fullmatch(r'\d{1,2}-\d{1,2}-\d{2,4}', candidate):
            left, middle, year = candidate.split('-')
            if len(year) == 2:
                year = str(int(year) + (2000 if int(year) < 40 else 1900))
            if int(left) > 12:
                day, month = int(left), int(middle)
            else:
                month, day = int(left), int(middle)
            return pd.Timestamp(year=int(year), month=month, day=day)
    except Exception:
        pass
    return pd.to_datetime(candidate, errors='coerce')


def extract_dates(text: str) -> list[str]:
    if is_not_found(text):
        return []
    text = collapse_ws(text)
    text = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\s*[.,]\s*(\d{4})\b', r'\1 \2, \3', text, flags=re.I)
    candidates = []
    patterns = [
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s+\d{4}',
        r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{1,2}-\d{1,2}-\d{2,4}',
        r'\d{1,2}\.\d{1,2}\.\d{2,4}',
    ]
    for pat in patterns:
        for match in re.finditer(pat, text, flags=re.I):
            candidate = match.group(0)
            parsed = _parse_date_candidate(candidate)
            if pd.notna(parsed):
                candidates.append(parsed.strftime('%B %d, %Y'))
    out = []
    for c in candidates:
        if c not in out:
            out.append(c)
    return out


def _extract_duration_numbers(lower: str):
    word_pattern = r'one hundred eighty|forty[- ]five|thirty[- ]six|twenty[- ]five|twenty[- ]four|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|eighteen|twenty|thirty|sixty|ninety|fifth'

    def _pick_value(match):
        if not match:
            return None
        numeric = match.group(2) or match.group(3)
        if numeric:
            return int(numeric)
        word = match.group(1) or match.group(4)
        return _number_word_to_int(word) if word else None

    years = re.search(rf'(?:([a-z -]+)\s*\((\d{{1,3}})(?:st|nd|rd|th)?\)|([\d]+)|({word_pattern}))\s*(?:[- ]?years?|\s+anniversary)', lower)
    months = re.search(rf'(?:([a-z -]+)\s*\((\d{{1,3}})(?:st|nd|rd|th)?\)|([\d]+)|({word_pattern}))\s*[- ]?months?', lower)
    days = re.search(rf'(?:([a-z -]+)\s*\((\d{{1,3}})(?:st|nd|rd|th)?\)|([\d]+)|({word_pattern}))\s*[- ]?days?', lower)
    y = _pick_value(years)
    m = _pick_value(months)
    d = _pick_value(days)
    if m is None and y is not None:
        m = y * 12
    if y is None and m is not None and m % 12 == 0:
        y = m // 12
    return y, m, d


def _extract_date_range(text: str):
    dates = extract_dates(text)
    if len(dates) >= 2:
        return dates[0], dates[-1]
    if len(dates) == 1:
        return dates[0], dates[0]
    return None, None


def _parse_timestamp(text: str):
    if is_not_found(text):
        return None
    normalized = format_date(text)
    if is_not_found(normalized):
        dates = extract_dates(str(text))
        if not dates:
            return None
        normalized = dates[0]
    parsed = pd.to_datetime(normalized, errors='coerce')
    return parsed if pd.notna(parsed) else None


def _compute_expiration_from_base(base_ts, years: int | None, months: int | None, days: int | None, anniversary_mode: bool = False):
    if base_ts is None:
        return None
    end_ts = pd.Timestamp(base_ts)
    if months is not None:
        end_ts = end_ts + pd.DateOffset(months=months)
    elif years is not None:
        end_ts = end_ts + pd.DateOffset(years=years)
    if days is not None:
        end_ts = end_ts + pd.Timedelta(days=days)
    if not anniversary_mode:
        end_ts = end_ts - pd.Timedelta(days=1)
    return end_ts.strftime('%B %d, %Y')


def _select_expiration_base_date(text: str, agreement_date=None, effective_date=None):
    lower = collapse_ws(text).lower()
    dates = extract_dates(text)
    if len(dates) == 1:
        return dates[0]
    if 'agreement date' in lower and not is_not_found(agreement_date):
        return agreement_date
    if any(k in lower for k in ['effective date', 'commencement date', 'commencing on', 'commence on', 'beginning on', 'begin on', 'start date']) and not is_not_found(effective_date):
        return effective_date
    if not is_not_found(agreement_date):
        return agreement_date
    if not is_not_found(effective_date):
        return effective_date
    return dates[0] if dates else None


def _looks_like_term_clause(lower: str) -> bool:
    strong_markers = [
        'term of the agreement', 'term of agreement', 'initial term of', 'renewal term of', 'expiration date',
        'shall continue for', 'continue for a period', 'remain in effect for', 'shall remain in effect for',
        'shall continue in force', 'continue in operation for', 'will be in effect for', 'shall have a term of',
        'shall have an initial term', 'shall be valid for', 'effective through', 'through and including',
        'contract end', 'termination date', 'scheduled expiration date', 'co-termin', 'cotermin',
        'until terminated', 'indefinite period', 'continue indefinitely', 'perpetual', 'perpetually thereafter',
        'annual renewal', 'automatically renew', 'auto-renew', 'notice of non-renewal', 'expire on', 'expires on',
        'shall expire on', 'terminate on', 'shall terminate on', 'commence on the effective date and continue',
        'commencing on the effective date and continue', 'initial period of', 'royalty term', 'service period',
        'duration of the lease', 'duration of agreement', 'until completion of', 'last addendum to expire', 'term:', 'shall be in effect until'
    ]
    return any(marker in lower for marker in strong_markers)


def _looks_like_expiration_noise(lower: str) -> bool:
    noise_markers = [
        'annual meeting', 'board meeting', 'special meeting', 'business days prior', 'calendar days prior',
        'days prior to', 'support day', 'support hours', 'response time', 'severity level', 'problem report',
        'ownership threshold', 'minimum ownership threshold', 'expirations for', 'ownership interest in the expirations',
        'purchase from agency', 'nominate the jana nominees', 'merchant accounts', 'service levels', 'status reports',
        'prior written notice', 'written notice prior', 'days written notice', 'months written notice',
        'on written notice', 'notice to the other party', 'notice thereof', 'thirty (30) days notice',
        'fifteen (15) days notice', 'one year and one day', 'card expiration', 'card life', 'cooperation period'
    ]
    return any(marker in lower for marker in noise_markers)


def clean_expiration(value, agreement_date=None, effective_date=None) -> str:
    text = collapse_ws('' if value is None else str(value))
    if is_not_found(text):
        return 'NOT FOUND'
    lower = text.lower()
    dates = extract_dates(text)

    if 'expiry of the cooperation period' in lower or 'expiration of the cooperation period' in lower:
        return 'NOT FOUND'
    if any(k in lower for k in ['joint venture agreement', 'cooperation agreement']) and not any(k in lower for k in ['initial term', 'shall continue for', 'shall remain in effect', 'term of this agreement', 'term of the agreement', 'expire on', 'terminate on']):
        if any(k in lower for k in ['perpetual', 'co-termin', 'cotermin']):
            return 'NOT FOUND'
    if _looks_like_expiration_noise(lower) and not _looks_like_term_clause(lower):
        return 'NOT FOUND'

    if any(k in lower for k in ['until completion of the research program', 'until completion of', 'completion of milestone', 'completion of milestones', 'through the completion or termination of developer', 'service period', 'duration of the lease', 'royalty term', 'last addendum to expire', 'last to expire of the patents', 'until all of the intellectual property licensed', 'until the end of the fifteenth']):
        return 'Until Completion of Program/Milestones'
    if any(k in lower for k in ['co-termin', 'cotermin', 'co termin', 'for the term of the referenced', 'for the term of the lease', 'until the expiration or earlier termination of', 'until the termination of the strategic alliance agreement', 'until the expiration or earlier termination of the development and license agreement', 'continue until the termination of']):
        return 'Co-terminous with Related Agreement'

    # Prefer clause-shaped exact end dates and paired date ranges before later notice/termination language.
    m = re.search(
        r'\bterm\s*:\s*('
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+to\s+('
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        text,
        flags=re.I,
    )
    if m:
        return format_date(m.group(2))
    m = re.search(r'(?:shall be in effect until|shall terminate on|shall expire on|effective through(?: and including)?|continue until)\s+((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,|\.)?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, flags=re.I)
    if m:
        candidate = m.group(1)
        normalized = format_date(candidate)
        if normalized in {'NOT FOUND', candidate}:
            date_hits = extract_dates(candidate)
            normalized = date_hits[-1] if date_hits else 'NOT FOUND'
        if normalized != 'NOT FOUND':
            return normalized
    m = re.search(
        r'(?:from|commencing on)\s+('
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+'
        r'(?:through|to|until)\s+('
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        text,
        flags=re.I,
    )
    if m:
        return format_date(m.group(2))
    if dates and any(k in lower for k in ['terminate on', 'shall terminate on', 'shall expire on', 'expires on', 'effective through', 'through and including', 'scheduled expiration date', 'termination date']):
        return dates[-1]

    if 'terminate automatically one year after' in lower or 'terminate automatically 1 year after' in lower:
        return '1-Year (12 months) Initial Term'
    if 'until terminated' in lower or 'when terminated by either party' in lower or 'remain effective until terminated' in lower or 'remain in effect until terminated' in lower:
        return 'Until Terminated'
    if 'perpetual' in lower or 'perpetually thereafter' in lower or 'continue indefinitely' in lower or 'indefinite period' in lower or 'unlimited period of time' in lower or 'remain in effect perpetually' in lower or 'as long as fees are paid' in lower:
        return 'Perpetual'
    if 'if any public authority cancels' in lower or 'automatically if any public authority cancels' in lower:
        return 'Event-Based Termination'

    auto = any(k in lower for k in ['automatically renew', 'automatically renewing', 'successive annual periods', 'renew for additional', 'renewal term', 'auto renew', 'auto-renew'])
    explicit_duration_patterns = [
        r'shall continue for a period of\s+([a-z -]+\s*\(\d{1,3}\)|\d{1,3}|[a-z -]+)\s+years?',
        r'shall have an initial term of\s+([a-z -]+\s*\(\d{1,3}\)|\d{1,3}|[a-z -]+)\s+years?',
        r'shall have a term of\s+([a-z -]+\s*\(\d{1,3}\)|\d{1,3}|[a-z -]+)\s+years?',
        r'initial period of\s+([a-z -]+\s*\(\d{1,3}\)|\d{1,3}|[a-z -]+)\s+years?',
        r'continue in operation for at least an initial period of\s+([a-z -]+\s*\(\d{1,3}\)|\d{1,3}|[a-z -]+)\s+years?',
    ]
    for pat in explicit_duration_patterns:
        dm = re.search(pat, lower, flags=re.I)
        if dm:
            raw_n = dm.group(1)
            num_match = re.search(r'(\d{1,3})', raw_n)
            years_val = int(num_match.group(1)) if num_match else _number_word_to_int(raw_n.strip())
            if years_val:
                label = f'{years_val}-Year ({years_val * 12} months) Initial Term'
                return label + ', Auto-Renewal' if auto else label

    y, m, d = _extract_duration_numbers(lower)
    if y is not None and (m is None or m < y * 12):
        m = y * 12
    if _looks_like_term_clause(lower):
        if y is not None:
            label = f'{y}-Year ({m} months) Initial Term'
            return label + ', Auto-Renewal' if auto else label
        if m is not None:
            label = f'{m}-Month Initial Term'
            return label + ', Auto-Renewal' if auto else label
        if d is not None and 'notice' not in lower and not any(k in lower for k in ['30 day period', '30 days subsequent', 'within 14 days', 'payment to be effective', 'installment', 'cure period']) and any(k in lower for k in ['term shall be', 'initial term of', 'term of the agreement', 'agreement shall continue for', 'remain in effect for', 'for an initial period', 'shall have an initial term', 'shall have a term of']):
            label = f'{d}-Day Initial Term'
            return label + ', Auto-Renewal' if auto else label

    if any(k in lower for k in ['earlier to occur', 'later of', 'upon the earlier of', 'upon the later of', 'earliest to occur', 'the earlier of', 'the later of', 'earlier of the occurrence', 'successful remarketing', 'event that', 'if any public authority cancels', 'effective upon the occurrence', 'terminate if', 'automatically terminate in the event of', 'shall expire if']):
        return 'Event-Based Termination'
    if any(k in lower for k in ['public domain', 'fund no longer owns', 'term of such fund', 'removed as general partner']):
        return 'Event-Based Termination'
    if '[*' in text or '*****' in text or 'redact' in lower:
        if dates:
            return dates[-1]
        if y is not None or m is not None or d is not None:
            if y is not None:
                return f'{y}-Year ({m} months) Initial Term'
            if m is not None:
                return f'{m}-Month Initial Term'
            return f'{d}-Day Initial Term'
        return 'Event-Based Termination'
    return 'NOT FOUND'


def clean_governing_law(value) -> str:
    text = collapse_ws('' if value is None else str(value))
    if is_not_found(text):
        return 'NOT FOUND'
    lower = text.lower()
    if 'england and wales' in lower:
        return 'United Kingdom'
    if 'federal republic of germany' in lower:
        return 'Germany'
    country_patterns = [
        ('state of israel', 'Israel'),
        ('laws of israel', 'Israel'),
        ('australia', 'Australia'),
        ('singapore', 'Singapore'),
        ('india', 'India'),
        ('germany', 'Germany'),
        ('england', 'United Kingdom'),
        ('united kingdom', 'United Kingdom'),
        ('canada', 'Canada'),
        ('japan', 'Japan'),
        ('france', 'France'),
        ('china', 'China'),
        ('hong kong', 'Hong Kong'),
    ]
    for needle, label in country_patterns:
        if needle in lower:
            return label
    for state in sorted(US_STATES, key=len, reverse=True):
        if f'state of {state}' in lower or f'laws of {state}' in lower or re.search(rf'\b{re.escape(state)}\b', lower):
            state_label = ' '.join(w.capitalize() for w in state.split())
            return f'{state_label}, United States'
    m = re.search(r'laws of ([A-Z][A-Za-z .&-]+?)(?:\.|,|;| without| excluding| and)', text)
    if m:
        label = collapse_ws(m.group(1)).strip(',.;')
        return label
    return text[:120]


def is_entity_like(token: str) -> bool:
    t = collapse_ws(token).strip('"\'()[]{}.,;:')
    if not t:
        return False
    lower = t.lower()
    if lower in ROLE_WORDS:
        return False
    if 'referred to' in lower or 'together as' in lower or 'hereinafter' in lower:
        return False
    if len(t) < 3:
        return False
    words = re.findall(r'[A-Za-z][A-Za-z&.-]*', t)
    if len(words) >= 2 and any(w.lower().rstrip('.') in ENTITY_SUFFIXES for w in words):
        return True
    if t.isupper() and len(words) >= 2:
        return True
    if len(words) >= 2 and sum(1 for w in words if w[0].isupper()) >= 2:
        return True
    return False


def clean_parties(value) -> str:
    text = collapse_ws('' if value is None else str(value))
    if is_not_found(text):
        return 'NOT FOUND'

    between_match = re.search(
        r'(?:between|by and between|among)[:\s]+(.+?)(?:recitals|whereas|witnesseth|the parties agree|now therefore|now, therefore|it is agreed as follows|1\.)',
        text,
        flags=re.I,
    )
    working_text = between_match.group(1) if between_match else text
    cleanup_patterns = [
        r'\([^)]*party[^)]*\)',
        r'\([^)]*parties[^)]*\)',
        r'\([^)]*collectively[^)]*\)',
        r'\([^)]*individually[^)]*\)',
        r'\([^)]*hereinafter[^)]*\)',
        r'\b(?:each and both of them|each of the foregoing parties.*?|together as.*?|hereinafter.*?|individually as.*?|collectively as.*?|may be referred to herein.*?|shall be referred to herein.*?|referred to herein.*?|each a "party".*?|collectively,? the "parties".*?|individually,? a "party".*?|the party or parties specified.*?|the party specified.*?|each of which may be referred to.*?|the foregoing parties.*?|each shall be referred to.*?|party a and party b.*?|party a .*? party b.*?)\b',
    ]
    for pat in cleanup_patterns:
        working_text = re.sub(pat, ' ', working_text, flags=re.I)
    working_text = re.sub(r'"[^"]{0,40}"', ' ', working_text)
    working_text = re.sub(r"'[^']{0,40}'", ' ', working_text)

    raw_tokens = re.split(r';|\||\n', working_text)
    role_only_tokens = {
        'provider', 'recipient', 'consultant', 'owner', 'servicer', 'licensor', 'licensee', 'seller', 'buyer',
        'distributor', 'customer', 'supplier', 'party', 'parties', 'company', 'contractor', 'vendor', 'shipper',
        'transporter', 'sponsor', 'adviser', 'advisor', 'repairer', 'member', 'manufacturer', 'client',
        'developer', 'employee', 'executive', 'sub-advisor', 'subcontractor', 'agent', 'issuer', 'depositor',
        'custodian', 'principal', 'franchisee', 'affiliate', 'reseller', 'network affiliate', 'fund', 'trust'
    }
    role_prefixes = (
        'collectively', 'individually', 'hereinafter', 'together', 'sometimes', 'commonly', 'each of',
        'each individually', 'each party', 'the party', 'parties means', 'referred to herein', 'may be referred'
    )
    exact_drop_tokens = {
        'party a', 'party b', 'notes trustee', 'remarketing agent', 'stock purchase contract agent',
        'co-trustee', 'underwriter', 'investment adviser', 'the event', 'fund', 'trust', 'company', 'consultant',
        'contractor', 'customer', 'supplier', 'vendor', 'shipper', 'transporter', 'reseller', 'licensee',
        'licensor', 'recipient', 'provider', 'manufacturer', 'member', 'employee', 'executive', 'franchisee'
    }

    entities: list[str] = []
    seen = set()
    for token in raw_tokens:
        token = collapse_ws(token).strip("\"'()[]{}")
        token = re.sub(r'^(?:each|both)\s+', '', token, flags=re.I)
        token = re.sub(r'^(?:collectively|individually|hereinafter|together|sometimes|commonly|respectively).*$', '', token, flags=re.I)
        token = re.sub(r'^(?:doing business as|d/b/a|also known as|f/k/a|formerly known as)\s+', '', token, flags=re.I)
        token = re.sub(r'^(?:the\s+)?party\s+[ab]', '', token, flags=re.I)
        token = collapse_ws(token).strip(' ,.;:-')
        lower = token.lower()
        if not token or lower in exact_drop_tokens or lower in role_only_tokens:
            continue
        if any(lower.startswith(prefix) for prefix in role_prefixes):
            continue
        if any(phrase in lower for phrase in ['referred to herein', 'collectively as', 'individually as', 'hereinafter', 'together as', 'each a party', 'collectively the parties']):
            continue
        if re.search(r'\band\b.*\bare\b', lower):
            continue
        if 'schedule 1' in lower or 'attached hereto' in lower or 'collectively as the parties' in lower:
            continue
        if token.upper() == 'NOT FOUND' or len(token) > 140:
            continue
        if not is_entity_like(token):
            continue
        key = re.sub(r'[^a-z0-9]+', ' ', lower).strip()
        if key not in seen:
            seen.add(key)
            entities.append(token)

    if not entities:
        return 'NOT FOUND'

    def _norm_entity(s: str) -> str:
        return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()

    filtered = []
    for entity in sorted(entities, key=lambda s: (-len(s.split()), -len(s))):
        norm = _norm_entity(entity)
        drop = False
        for other in entities:
            other_norm = _norm_entity(other)
            if norm == other_norm:
                continue
            if len(norm) < len(other_norm) and (norm in other_norm or other_norm.startswith(norm + ' ') or other_norm.endswith(' ' + norm)):
                drop = True
                break
        if not drop:
            filtered.append(entity)

    filtered = sorted(dict.fromkeys(filtered), key=lambda s: s.lower())[:80]
    return ' | '.join(filtered) if filtered else 'NOT FOUND'


def clean_termination(value) -> str:
    text = collapse_ws('' if value is None else str(value))
    if is_not_found(text):
        return 'NOT FOUND'
    lower = text.lower()

    explicit_convenience = any(k in lower for k in [
        'for convenience', 'with or without cause'
    ]) or bool(re.search(r'terminat(?:e|ed|ion)[^.;]{0,40}without cause|without cause[^.;]{0,40}terminat', lower)) or bool(re.search(r'terminat(?:e|ed|ion)[^.;]{0,60}for any reason|for any reason[^.;]{0,60}terminat', lower))
    bilateral_notice_pattern = bool(re.search(
        r'\b(?:either party|each party|both parties)\b[^.;]{0,40}?\b(?:may|shall have the right to|can)\b[^.;]{0,20}?\bterminate\b[^.;]{0,160}?\b(?:written notice|prior written notice|notice)\b',
        lower,
    ))
    named_party_notice_pattern = bool(re.search(
        r'\b(?:customer|company|consultant|distributor|licensor|licensee|provider|recipient|contractor|vendor|reseller|manufacturer|buyer|seller|agency|affiliate|sparkling|chase|bank of america)\b[^.;]{0,80}?\bmay\b[^.;]{0,20}?\bterminate(?: this agreement| the agreement)?\b[^.;]{0,120}?\b(?:written notice|prior written notice|notice)\b',
        lower,
    ))
    at_will_notice = bilateral_notice_pattern or named_party_notice_pattern

    notice_only_markers = [
        'notice of non-renewal', 'intention not to renew', 'annual meeting', 'cooperation period',
        'subagency', 'sublicense', 'sublicensee', 'successful remarketing', 'defaulting party', 'non-defaulting party',
        'failure of any such conditions', 'remarketing agent', 'conditions with respect to', 'not be renewed',
        'renewal term', 'processing agreement', 'iso sponsorship agreement', 'for convenience of reference only',
        'for convenience of reference', 'headings are for convenience'
    ]
    if any(k in lower for k in notice_only_markers) and not explicit_convenience:
        return 'NOT FOUND'

    negative_markers = [
        'for cause', 'upon breach', 'default', 'material breach', 'insolvency', 'bankruptcy',
        'misconduct', 'misrepresentation', 'failure to cure', 'uncured breach', 'force majeure',
        'in the event of', 'upon the occurrence of', 'admit in writing its inability to pay',
        'general assignment for the benefit of creditors', 'voluntary bankrupt', 'petition of bankruptcy',
        'if the other party materially breaches', 'breaching party', 'successful remarketing',
        'provided, however, that', 'unless (a)', 'unless (b)'
    ]
    if any(k in lower for k in negative_markers) and not explicit_convenience:
        return 'NOT FOUND'
    if not explicit_convenience and re.search(r'\bterminate(?:d|s|ion)?\b', lower) and not at_will_notice:
        return 'NOT FOUND'
    if not explicit_convenience and not at_will_notice:
        return 'NOT FOUND'

    parts = ['Yes']
    notice = _extract_notice_phrase(lower)
    if notice:
        parts.append(notice)

    initiator = _extract_tfc_initiator(lower)
    if initiator:
        parts.append(initiator)

    if explicit_convenience:
        parts.append('Without Cause')

    deduped = []
    seen = set()
    for part in parts:
        if part not in seen:
            seen.add(part)
            deduped.append(part)
    return ' | '.join(deduped)


def _extract_nc_duration(lower: str):
    patterns = [
        r'during the term(?: of this agreement| of the agreement| of this franchise agreement)?',
        r'for the term(?: of this agreement)?',
        r'following the initial term',
        r'after the initial term',
        r'during the restricted period',
        r'during the royalty term',
        r'during the post-term period',
        r'for a period of\s+[^.;]{1,80}?\s*(?:business\s+days|calendar\s+days|days|months|years)',
        r'for a\s+[^.;]{1,40}?period\s+following',
        r'for a\s+[^.;]{1,40}?period\s+after',
        r'for\s+[^.;]{1,60}?\s*(?:business\s+days|calendar\s+days|days|months|years)\s+(?:after|following|thereafter)',
        r'for\s+[a-z0-9() -]{1,40}?\s*(?:business\s+days|calendar\s+days|days|months|years)',
        r'within\s+[^.;]{1,40}?\s*(?:mile|miles|km|kilometers|kilometres)',
        r'until\s+[a-z0-9 ,()/-]{1,80}',
    ]
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            phrase = collapse_ws(m.group(0).strip(' ,.;'))
            phrase = re.sub(r'the term', 'the Term', phrase, flags=re.I)
            return f'Duration: {phrase}'
    return None


def _extract_nc_restricted_party(lower: str):
    labels = [
        ('network affiliate', 'Restricted Party: Network Affiliate'),
        ('franchisee', 'Restricted Party: Franchisee'),
        ('distributor', 'Restricted Party: Distributor'),
        ('consultant', 'Restricted Party: Consultant'),
        ('agency', 'Restricted Party: Agency'),
        ('reseller', 'Restricted Party: Reseller'),
        ('vendor', 'Restricted Party: Vendor'),
        ('licensee', 'Restricted Party: Licensee'),
        ('licensor', 'Restricted Party: Licensor'),
        ('subcontractor', 'Restricted Party: Subcontractor'),
        ('contractor', 'Restricted Party: Contractor'),
        ('executive', 'Restricted Party: Executive'),
        ('member', 'Restricted Party: Member'),
        ('talent', 'Restricted Party: Talent'),
        ('publisher', 'Restricted Party: Publisher'),
        ('party b', 'Restricted Party: Party B'),
        ('party a ', 'Restricted Party: Party A'),
        ('customer', 'Restricted Party: Customer'),
        ('manufacturer', 'Restricted Party: Manufacturer'),
        ('company', 'Restricted Party: Company'),
    ]
    for needle, label in labels:
        if needle in lower:
            return label
    if 'either party' in lower or 'both parties' in lower or 'neither party' in lower or 'each party' in lower:
        return 'Restricted Party: Either Party'
    return None


def clean_non_compete(value) -> str:
    text = collapse_ws('' if value is None else str(value))
    if is_not_found(text):
        return 'NOT FOUND'
    lower = text.lower()

    # Avoid broad false positives from unrelated uses of "exclusive" or generic employment terms.
    exclusivity_markers = [
        'exclusive appointment', 'exclusive right', 'exclusive rights', 'exclusive purchase',
        'exclusive provider', 'exclusive supplier', 'grant any rights', 'granting any rights'
    ]
    noncomp_markers = [
        'non-compete', 'non compete', 'noncompetition', 'compete with', 'competitive business',
        'competitive product', 'competitive products', 'products competitive with', 'handle no products competitive',
        'competing product', 'competitor', 'competitive retailer', 'not compete', 'shall not compete',
        'not engage', 'not participate', 'shall be prohibited', 'prohibited from', 'engage in any business'
    ]
    product_restriction_markers = [
        'not directly or indirectly sell', 'not directly or indirectly handle', 'shall handle no products',
        'not sell any competing', 'not develop', 'not manufacture', 'not market', 'not distribute',
        'not commercialize', 'not promote', 'not supply', 'not exploit', 'develop or commercialize any competing product',
        'discover, research, develop, manufacture or commercialize', 'will discover, research, develop, manufacture or commercialize',
        'not authorize', 'may not sell or license', 'direct or indirect interest in any competitive business',
        'own, manage, engage in, be employed by', 'have any direct or indirect interest', 'divert or attempt to divert'
    ]
    nonsolicit_markers = [
        'non-solicit', 'solicit', 'entice away', 'divert', 'induce any employee', 'contact any customer',
        'direct solicitation', 'discontinue using', 'refer prospective clients', 'approach, rewrite, pursue', 'interfere with'
    ]
    nohire_markers = [
        'no-hire', 'recruit', 'offer of employment', 'leave the employment', 'employment of any current', 'solicit for employment'
    ]
    territory_markers = ['outside the territory', 'outside of the delivery', 'outside the delivery', 'within three (3) miles', 'within three miles', 'restricted territory']

    noncomp_signal = any(k in lower for k in noncomp_markers)
    product_signal = any(k in lower for k in product_restriction_markers + ['competitive product', 'competitive products', 'products competitive with'])
    endorse_signal = 'endorse' in lower and any(k in lower for k in ['compet', 'competitive'])
    nonsolicit_signal = any(k in lower for k in nonsolicit_markers)
    nohire_signal = any(k in lower for k in nohire_markers)
    territory_signal = any(k in lower for k in territory_markers)
    exclusivity_signal = any(k in lower for k in exclusivity_markers)
    ip_license_noise = any(k in lower for k in [
        'intellectual property', 'trademark', 'marks', 'brand', 'domain name', 'copyright', 'patent',
        'license agreement', 'content license', 'hosting agreement', 'maintenance agreement', 'service agreement',
        'non-exclusive', 'non exclusive', 'competitive harm to the company if disclosed', 'competitive harm if disclosed',
        'exclusive artistic and editorial control'
    ])

    # Pure customer/employee solicitation language is too broad to call non-compete unless tied to competitive or exclusive restrictions.
    if nonsolicit_signal and not any([noncomp_signal, product_signal, endorse_signal, territory_signal, exclusivity_signal]):
        return 'NOT FOUND'
    if nohire_signal and not any([noncomp_signal, product_signal, endorse_signal, territory_signal, exclusivity_signal, nonsolicit_signal]):
        return 'NOT FOUND'
    if exclusivity_signal and not any([noncomp_signal, product_signal, endorse_signal, territory_signal]):
        if ip_license_noise:
            return 'NOT FOUND'
    if territory_signal and not any([noncomp_signal, product_signal, exclusivity_signal, endorse_signal]):
        return 'NOT FOUND'
    if 'non-exclusive' in lower or 'non exclusive' in lower:
        return 'NOT FOUND'
    if endorse_signal and 'compet' not in lower and 'competitive' not in lower:
        return 'NOT FOUND'
    if not any([noncomp_signal, product_signal, endorse_signal, nonsolicit_signal, nohire_signal, territory_signal, exclusivity_signal]):
        return 'NOT FOUND'

    scopes = []
    if exclusivity_signal:
        scopes.append('Exclusivity')
    if noncomp_signal:
        scopes.append('Non-Competition')
    if product_signal:
        scopes.append('Competitive Products Restriction')
    if endorse_signal:
        scopes.append('Non-Endorsement')
    if nonsolicit_signal:
        scopes.append('Non-Solicitation')
    if nohire_signal:
        scopes.append('No-Hire')
    if territory_signal:
        scopes.append('Territorial Restriction')

    if not scopes:
        return 'NOT FOUND'

    parts = ['Yes']
    restricted_party = _extract_nc_restricted_party(lower)
    if restricted_party:
        parts.append(restricted_party)
    duration = _extract_nc_duration(lower)
    if duration:
        parts.append(duration)
    parts.append('Scope: ' + ', '.join(dict.fromkeys(scopes)))
    return ' | '.join(parts)


def clean_indemnification(value) -> str:
    text = collapse_ws('' if value is None else str(value))
    if is_not_found(text):
        return 'NOT FOUND'
    lower = text.lower()

    if any(k in lower for k in ['goodwill and reputation', 'impair or tarnish', 'negative light or context', 'non-disparage', 'disparage', 'tarnish', 'goodwill of the supplier', 'goodwill and reputation in the']) and 'indemn' not in lower:
        return 'Trademark Non-Disparagement Clause (not indemnification)'

    if any(k in lower for k in [
        'not challenge', 'contest the validity', 'contest or assist others to contest', 'challenge, dispute',
        'oppose, challenge', 'petition to cancel', 'register or attempt to register', 'claim any interest in',
        'adversely affect the validity', 'impairing the validity', 'contest the ownership', 'challenge the validity',
        'ownership of any intellectual property', 'rights, title or interest in the', 'contest the use of',
        'register any trademarks', 'attempt to register any domain names', 'attack or challenge', 'dispute or deny the validity',
        'do anything inconsistent with such ownership', 'shall not now or in the future contest', 'validity or ownership of the marks'
    ]) and any(k in lower for k in ['trademark', 'mark', 'marks', 'domain name', 'brand', 'intellectual property', 'patent', 'copyright', 'trade name', 'proprietary marks']):
        return 'IP Non-Challenge Clause (not indemnification)'

    if 'moral rights' in lower and any(k in lower for k in ['waive', 'waiver', 'irrevocably and unconditionally waive', 'right of integrity', 'false attribution']):
        return 'Moral Rights Waiver (not indemnification)'

    if any(k in lower for k in ['covenant not to sue', 'shall not, under any circumstances, sue', 'shall not sue the other party', 'not to sue']) and 'indemn' not in lower:
        return 'Covenant Not to Sue (not indemnification)'

    if any(k in lower for k in ['bankruptcy', 'insolvency', 'receiver', 'liquidator', 'custodian', 'sequestrator', 'winding up', 'petition or otherwise invoke']) and any(k in lower for k in ['non-petition', 'petition or otherwise invoke the process', 'commencing or sustaining a case against', 'commencing a case against']):
        return 'Bankruptcy Non-Petition Covenant (not indemnification)'

    obligation_patterns = [
        r'\b(?:shall|will|must|agrees? to)\b[^.;]{0,120}\b(?:indemnif(?:y|ies|ication)?|hold harmless|save harmless)\b[^.;]{0,160}\b(?:against|from and against)\b[^.;]{0,160}\b(?:claims?|actions?|loss(?:es)?|damages|liabilit(?:y|ies)|expenses|costs)\b',
        r'\b(?:release,? )?defend,? indemnif(?:y|ies|ication)? and hold harmless\b[^.;]{0,200}\b(?:against|from and against)\b[^.;]{0,160}\b(?:claims?|actions?|loss(?:es)?|damages|liabilit(?:y|ies)|expenses|costs)\b',
        r'\b(?:shall|will|must|agrees? to)\b[^.;]{0,120}\bdefend\b[^.;]{0,160}\b(?:against|from and against)\b[^.;]{0,160}\b(?:claims?|actions?|loss(?:es)?|damages|liabilit(?:y|ies)|expenses|costs)\b',
    ]
    reimburse_pattern = r'\b(?:shall|will|must|agrees? to)\b[^.;]{0,120}\breimburse\b[^.;]{0,160}\b(?:for|against)\b[^.;]{0,160}\b(?:claims?|actions?|loss(?:es)?|damages|liabilit(?:y|ies)|expenses|costs)\b'

    explicit_obligation = any(re.search(pat, lower) for pat in obligation_patterns)
    explicit_reimburse = re.search(reimburse_pattern, lower) is not None
    warranty_noise = any(k in lower for k in ['warranty', 'disclaimer', 'liable for', 'limitation of liability', 'exclusive remedy'])
    mere_reference = any(k in lower for k in [
        'indemnification obligations', 'rights or obligations under section', 'subject to the indemnification obligations',
        'indemnification claims under section', 'table of contents', 'article 11 indemnification', 'section 11 indemnification'
    ]) and not explicit_obligation and not explicit_reimburse

    if explicit_obligation or explicit_reimburse:
        return 'Yes | Explicit indemnification / defense / hold harmless obligations'
    if warranty_noise or mere_reference:
        return 'NOT FOUND'
    return 'NOT FOUND'


def clean_limitation(value) -> str:
    text = collapse_ws('' if value is None else str(value))
    if is_not_found(text):
        return 'NOT FOUND'
    lower = text.lower()
    limiter_markers = [
        'exclusive remedy', 'sole remedy', 'sole and exclusive remedy', 'limited to', 'shall not exceed',
        'aggregate liability', 'maximum liability', 'liability cap', 'in no event shall', 'not be liable',
        'exclusive obligation', 'entire liability', 'maximum extent permitted by law'
    ]
    liability_core = any(k in lower for k in ['liability', 'liable', 'damages', 'in no event', 'exclusive remedy', 'sole remedy']) and 'limited liability company' not in lower
    if not any(k in lower for k in limiter_markers) or not liability_core:
        return 'NOT FOUND'

    parts = ['Yes']
    excludes = []
    exclude_terms = [
        ('consequential', 'Consequential'),
        ('incidental', 'Incidental'),
        ('punitive', 'Punitive'),
        ('exemplary', 'Exemplary'),
        ('indirect', 'Indirect'),
        ('special', 'Special'),
        ('lost profits', 'Lost Profits'),
        ('loss of profits', 'Lost Profits'),
        ('lost revenue', 'Lost Revenue'),
        ('loss of revenue', 'Lost Revenue'),
        ('loss of data', 'Loss of Data'),
        ('loss of use', 'Loss of Use'),
        ('business interruption', 'Business Interruption'),
        ('goodwill', 'Goodwill'),
        ('cover damages', 'Cover Damages'),
    ]
    for needle, label in exclude_terms:
        if needle in lower:
            excludes.append(label)
    if excludes:
        parts.append('Excludes: ' + ', '.join(dict.fromkeys(excludes)))

    cap_patterns = [
        r'aggregate liability[^.;]{0,120}?(?:shall be limited to|will be limited to|shall not exceed|will not exceed|exceed)\s*([^.;]{5,180})',
        r'max(?:imum)? liability[^.;]{0,120}?(?:shall be limited to|will be limited to|shall not exceed|will not exceed|is limited to)\s*([^.;]{5,180})',
        r'liability[^.;]{0,120}?(?:shall be limited to|will be limited to|shall not exceed|will not exceed|is limited to)\s*([^.;]{5,180})',
        r'cumulative liability[^.;]{0,120}?shall not exceed\s*([^.;]{5,180})',
        r'liability cap[^.;]{0,80}?\(?"?liability cap"?\)?[^.;]{0,80}',
    ]
    for pat in cap_patterns:
        m = re.search(pat, text, flags=re.I)
        if not m:
            continue
        candidate = collapse_ws(m.group(1) if m.lastindex else m.group(0)).strip(' ,.;:')
        if len(candidate) >= 4:
            parts.append('Cap: ' + candidate)
            break

    exceptions = []
    exception_terms = [
        ('willful misconduct', 'Willful Misconduct'),
        ('wilful misconduct', 'Willful Misconduct'),
        ('gross negligence', 'Gross Negligence'),
        ('fraud', 'Fraud'),
        ('confidential', 'Confidentiality Breach'),
        ('intellectual property', 'IP Infringement'),
        ('infringement', 'IP Infringement'),
        ('indemnification', 'Indemnification'),
        ('indemnity', 'Indemnification'),
        ('third party claim', 'Third Party Claims'),
        ('mandatory applicable law', 'Mandatory Law'),
        ('personal injury', 'Personal Injury'),
        ('death', 'Personal Injury/Death'),
    ]
    for needle, label in exception_terms:
        if needle in lower:
            exceptions.append(label)
    if exceptions:
        parts.append('Exceptions: ' + ', '.join(dict.fromkeys(exceptions)))

    remedy_terms = []
    if 'sole remedy' in lower or 'sole and exclusive remedy' in lower:
        remedy_terms.append('Sole Remedy')
    if 'exclusive remedy' in lower:
        remedy_terms.append('Exclusive Remedy')
    if 'sole and exclusive obligation' in lower or 'sole obligation' in lower:
        remedy_terms.append('Sole Obligation')
    if 'direct damages' in lower or 'direct and actual damages' in lower:
        remedy_terms.append('Direct Damages Only')
    if remedy_terms:
        parts.append('Remedy: ' + ', '.join(dict.fromkeys(remedy_terms)))

    has_excludes = any(part.startswith('Excludes:') for part in parts[1:])
    has_cap = any(part.startswith('Cap:') for part in parts[1:])
    has_remedy = any(part.startswith('Remedy:') for part in parts[1:])
    has_exceptions = any(part.startswith('Exceptions:') for part in parts[1:])
    if has_exceptions and not any([has_excludes, has_cap, has_remedy]):
        return 'NOT FOUND'
    if has_remedy and not any([has_excludes, has_cap]) and not any(k in lower for k in ['liability', 'liable', 'damages']):
        return 'NOT FOUND'
    has_real_limit = any(part.startswith(('Excludes:', 'Cap:', 'Remedy:')) for part in parts[1:])
    return ' | '.join(parts) if has_real_limit else 'NOT FOUND'


def build_cleaned_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rows.append({
            'contract_id': collapse_ws(str(row.get('contract_id', ''))),
            'Agreement Date': format_date(row.get('Agreement Date')),
            'Effective Date': format_date(row.get('Effective Date')),
            'Expiration Date': clean_expiration(row.get('Expiration Date')),
            'Governing Law': clean_governing_law(row.get('Governing Law')),
            'Indemnification': clean_indemnification(row.get('Indemnification')),
            'Limitation Of Liability': clean_limitation(row.get('Limitation Of Liability')),
            'Non-Compete': clean_non_compete(row.get('Non-Compete')),
            'Parties': clean_parties(row.get('Parties')),
            'Termination For Convenience': clean_termination(row.get('Termination For Convenience')),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a cleaned metadata CSV from the original contract_metadata_150 Excel file.')
    parser.add_argument('--input-xlsx', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-xlsx', default='')
    args = parser.parse_args()

    df = pd.read_excel(args.input_xlsx)
    cleaned = build_cleaned_df(df)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_csv, index=False, encoding='utf-8-sig')

    if args.output_xlsx:
        out_xlsx = Path(args.output_xlsx)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_excel(out_xlsx, index=False)

    print(f'Saved cleaned CSV: {out_csv}')
    if args.output_xlsx:
        print(f'Saved cleaned XLSX: {args.output_xlsx}')
    print(f'Rows: {len(cleaned)}')
    print(f'Columns: {list(cleaned.columns)}')


if __name__ == '__main__':
    main()
