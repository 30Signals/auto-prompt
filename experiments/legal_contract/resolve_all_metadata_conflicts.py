from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from experiments.legal_contract.clean_contract_metadata_from_original import (
    clean_expiration,
    clean_governing_law,
    clean_indemnification,
    clean_limitation,
    clean_non_compete,
    clean_parties,
    clean_termination,
    format_date,
)

DATE_PAT = re.compile(r'^[A-Z][a-z]+\s+(?:\d{2}|____),\s+\d{4}$')
REDACTED_DATE_PAT = re.compile(r'^[A-Z][a-z]+\s+____,\s+\d{4}$')


FIELD_CLEANERS = {
    'Agreement Date': format_date,
    'Effective Date': format_date,
    'Expiration Date': clean_expiration,
    'Governing Law': clean_governing_law,
    'Indemnification': clean_indemnification,
    'Limitation Of Liability': clean_limitation,
    'Non-Compete': clean_non_compete,
    'Parties': clean_parties,
    'Termination For Convenience': clean_termination,
}


def load_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def clean_field(field: str, raw_value):
    return FIELD_CLEANERS[field](raw_value)


def is_structured_expiration(value: str) -> bool:
    value = (value or '').strip()
    return (
        bool(DATE_PAT.match(value))
        or 'Year (' in value
        or 'Month' in value
        or value.startswith('Co-terminous')
        or value in {'Event-Based Termination', 'Until Terminated', 'See Contract Terms (redacted duration)'}
    )


def is_clean_date(value: str) -> bool:
    value = (value or '').strip()
    return bool(DATE_PAT.match(value) or REDACTED_DATE_PAT.match(value))


def should_update(field: str, gold: str, candidate: str, pred: str) -> tuple[bool, str]:
    gold = (gold or '').strip()
    candidate = (candidate or '').strip()
    pred = (pred or '').strip()

    if not candidate or candidate == gold:
        return False, 'keep_gold_no_better_source_value'

    if field in {'Agreement Date', 'Effective Date'}:
        if gold == 'NOT FOUND' and is_clean_date(candidate):
            return True, 'update_from_original_date'
        return False, 'keep_gold_existing_date_preferred'

    if field == 'Governing Law':
        if gold.lower() == 'united states' and candidate.endswith(', United States'):
            return True, 'update_from_original_more_specific_us_jurisdiction'
        if gold == 'NOT FOUND' and candidate != 'NOT FOUND':
            return True, 'update_from_original_missing_governing_law'
        return False, 'keep_gold_existing_governing_law_preferred'

    if field == 'Expiration Date':
        if gold == 'NOT FOUND' and is_structured_expiration(candidate):
            return True, 'update_from_original_missing_expiration'
        if len(gold) > 120 and is_structured_expiration(candidate):
            return True, 'update_from_original_structured_expiration'
        return False, 'keep_gold_existing_expiration_preferred'

    if field == 'Indemnification':
        if gold == 'NOT FOUND' and candidate != 'NOT FOUND':
            return True, 'update_from_original_missing_indemnification'
        return False, 'keep_gold_existing_indemnification_preferred'

    if field == 'Limitation Of Liability':
        if gold == 'NOT FOUND' and candidate != 'NOT FOUND':
            return True, 'update_from_original_missing_limitation'
        return False, 'keep_gold_existing_limitation_preferred'

    if field == 'Non-Compete':
        if gold == 'NOT FOUND' and candidate != 'NOT FOUND':
            return True, 'update_from_original_missing_non_compete'
        return False, 'keep_gold_existing_non_compete_preferred'

    if field == 'Parties':
        if gold == 'NOT FOUND' and candidate != 'NOT FOUND':
            return True, 'update_from_original_missing_parties'
        return False, 'keep_gold_existing_parties_preferred'

    if field == 'Termination For Convenience':
        if gold == 'NOT FOUND' and candidate != 'NOT FOUND':
            return True, 'update_from_original_missing_termination_for_convenience'
        return False, 'keep_gold_existing_termination_preferred'

    return False, 'keep_gold_default'


def main() -> None:
    parser = argparse.ArgumentParser(description='Resolve all metadata conflicts against cleaned_3 using the original Excel as grounding.')
    parser.add_argument('--base-csv', required=True)
    parser.add_argument('--conflicts-csv', required=True)
    parser.add_argument('--original-xlsx', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--resolution-log-csv', required=True)
    args = parser.parse_args()

    base_rows = load_csv(Path(args.base_csv))
    conflict_rows = load_csv(Path(args.conflicts_csv))
    original_df = pd.read_excel(args.original_xlsx)
    original_by_id = {str(r['contract_id']).strip(): r for _, r in original_df.iterrows()}
    base_by_id = {r['contract_id']: dict(r) for r in base_rows}

    resolution_log = []
    for conflict in conflict_rows:
        cid = conflict['contract_id']
        field = conflict['field']
        gold = base_by_id[cid][field]
        pred = conflict['pred']
        raw = original_by_id[cid][field]
        candidate = clean_field(field, raw)
        update, basis = should_update(field, gold, candidate, pred)
        resolved = candidate if update else gold
        if update:
            base_by_id[cid][field] = resolved
        resolution_log.append({
            'contract_id': cid,
            'field': field,
            'old_gold': gold,
            'pred': pred,
            'original_cleaned_candidate': candidate,
            'resolved_value': resolved,
            'action': 'update_from_original' if update else 'keep_gold',
            'basis': basis,
            'score': conflict['score'],
        })

    out_rows = [base_by_id[r['contract_id']] for r in base_rows]
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(base_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    log_path = Path(args.resolution_log_csv)
    with log_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['contract_id','field','old_gold','pred','original_cleaned_candidate','resolved_value','action','basis','score'],
        )
        writer.writeheader()
        writer.writerows(resolution_log)

    print(f'Saved resolved CSV: {out_path}')
    print(f'Saved resolution log: {log_path}')
    print(f'Conflicts resolved: {len(resolution_log)}')
    print(f'Updates applied: {sum(1 for r in resolution_log if r["action"] == "update_from_original")}')


if __name__ == '__main__':
    main()
