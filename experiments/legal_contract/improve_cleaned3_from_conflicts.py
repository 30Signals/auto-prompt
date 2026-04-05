from __future__ import annotations

import argparse
import csv
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

SAFE_FIELDS = [
    'Agreement Date',
    'Effective Date',
    'Expiration Date',
    'Governing Law',
    'Indemnification',
    'Limitation Of Liability',
    'Non-Compete',
    'Termination For Convenience',
]


def load_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def clean_field(field: str, raw_value):
    if field == 'Agreement Date':
        return format_date(raw_value)
    if field == 'Effective Date':
        return format_date(raw_value)
    if field == 'Expiration Date':
        return clean_expiration(raw_value)
    if field == 'Governing Law':
        return clean_governing_law(raw_value)
    if field == 'Indemnification':
        return clean_indemnification(raw_value)
    if field == 'Limitation Of Liability':
        return clean_limitation(raw_value)
    if field == 'Non-Compete':
        return clean_non_compete(raw_value)
    if field == 'Parties':
        return clean_parties(raw_value)
    if field == 'Termination For Convenience':
        return clean_termination(raw_value)
    return str(raw_value)


def should_apply(field: str, current_gold: str, cleaned_candidate: str, pred: str) -> bool:
    g = (current_gold or '').strip()
    c = (cleaned_candidate or '').strip()
    p = (pred or '').strip()
    g_lower = g.lower()
    c_lower = c.lower()
    p_lower = p.lower()

    if not c or c == g:
        return False

    if field in {'Agreement Date', 'Effective Date'}:
        if g == 'NOT FOUND' and c != 'NOT FOUND' and ',' in c and any(ch.isdigit() for ch in c):
            return True
        return False

    if field == 'Governing Law':
        if g_lower == 'united states' and c_lower.endswith(', united states'):
            return True
        if g == 'NOT FOUND' and c != 'NOT FOUND':
            return True
        return False

    if field == 'Expiration Date':
        structured_markers = ['Year', 'Month', 'Co-terminous', 'Event-Based', 'Until Terminated']
        if g == 'NOT FOUND' and c != 'NOT FOUND':
            return True
        if len(g) > 120 and any(m in c for m in structured_markers) and 'This Agreement' not in c:
            return True
        return False

    if field in {'Indemnification', 'Limitation Of Liability', 'Non-Compete', 'Termination For Convenience'}:
        if g == 'NOT FOUND' and c != 'NOT FOUND':
            return True
        return False

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a conflict-guided improved version of cleaned_3 using the original Excel as grounding.')
    parser.add_argument('--base-csv', required=True)
    parser.add_argument('--conflicts-csv', required=True)
    parser.add_argument('--original-xlsx', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--changes-csv', required=True)
    args = parser.parse_args()

    base_rows = load_csv(Path(args.base_csv))
    conflict_rows = load_csv(Path(args.conflicts_csv))
    original_df = pd.read_excel(args.original_xlsx)
    original_by_id = {str(row['contract_id']).strip(): row for _, row in original_df.iterrows()}

    conflicts_by_key = {(r['contract_id'], r['field']): r for r in conflict_rows}
    changes = []

    out_rows = []
    for row in base_rows:
        cid = row['contract_id']
        original = original_by_id.get(cid)
        if original is None:
            out_rows.append(row)
            continue

        updated = dict(row)
        for field in SAFE_FIELDS:
            conflict = conflicts_by_key.get((cid, field))
            if not conflict:
                continue
            cleaned_candidate = clean_field(field, original.get(field))
            if should_apply(field, row[field], cleaned_candidate, conflict['pred']):
                changes.append({
                    'contract_id': cid,
                    'field': field,
                    'old_value': row[field],
                    'new_value': cleaned_candidate,
                    'pred_value': conflict['pred'],
                    'score': conflict['score'],
                })
                updated[field] = cleaned_candidate
        out_rows.append(updated)

    fieldnames = list(base_rows[0].keys())
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    changes_path = Path(args.changes_csv)
    with changes_path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['contract_id','field','old_value','new_value','pred_value','score'])
        w.writeheader()
        w.writerows(changes)

    print(f'Saved improved CSV: {out_path}')
    print(f'Saved change log: {changes_path}')
    print(f'Applied changes: {len(changes)}')


if __name__ == '__main__':
    main()
