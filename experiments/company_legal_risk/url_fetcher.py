"""
URL Fetcher for Company Legal Risk Dataset

Enriches retrieved_docs in a labeled JSONL by:
  1. Fetching the full article text from each doc's URL
  2. Filtering to sentences that mention the company name
  3. Writing the result into full_content and setting content_fetched = True
  4. Flagging docs where the company name never appears as irrelevant = True

This fixes two problems identified in the dataset:
  - 96% of docs are truncated SearXNG snippets (~151 chars)
  - 40% of docs don't mention the company they were retrieved for

Usage:
    python -m company_legal_risk.url_fetcher \\
        --input  data/replay_runs_clean.jsonl \\
        --output data/replay_runs_enriched.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

DEFAULT_TIMEOUT = 10       # seconds per request
DEFAULT_DELAY   = 0.5      # seconds between requests (be polite)
MAX_CONTENT_LEN = 8_000    # chars — cap so LLM context stays manageable

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CompanyLegalRiskBot/1.0; "
        "+https://github.com/your-org/company-legal-risk)"
    )
}


def fetch_full_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Fetch a URL and extract readable text from the HTML body.
    Returns empty string on any failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        html = resp.text

        # Strip script / style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        # Strip all remaining tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_CONTENT_LEN]
    except Exception:
        return ""


def extract_relevant_sentences(full_text: str, company_name: str) -> str:
    """
    Return only sentences from full_text that mention company_name (case-insensitive).
    Falls back to the full_text excerpt if nothing matches.
    """
    if not full_text:
        return ""

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    name_lower = company_name.lower()
    relevant = [s for s in sentences if name_lower in s.lower()]

    if relevant:
        return " ".join(relevant)[:MAX_CONTENT_LEN]

    # No sentences match — the whole article is off-topic
    return ""


def enrich_record(record: dict, delay: float = DEFAULT_DELAY) -> dict:
    """
    Process a single JSONL record in-place.
    Fetches each doc's URL, extracts relevant sentences, updates fields.
    """
    company = str(record.get("company_name", "")).strip()
    docs = record.get("retrieved_docs", [])
    enriched_count = 0
    irrelevant_count = 0

    for doc in docs:
        if doc.get("content_fetched"):
            continue  # already processed, skip

        url = doc.get("url", "").strip()
        if not url:
            doc["content_fetched"] = False
            doc["full_content"] = ""
            doc["irrelevant"] = False
            continue

        full_text = fetch_full_text(url)
        relevant = extract_relevant_sentences(full_text, company)

        doc["full_content"] = relevant
        doc["content_fetched"] = bool(full_text)

        if full_text and not relevant:
            # Article was fetched but never mentions the company
            doc["irrelevant"] = True
            irrelevant_count += 1
        else:
            doc["irrelevant"] = False

        if relevant:
            enriched_count += 1

        time.sleep(delay)

    record["_enrichment"] = {
        "docs_total": len(docs),
        "docs_enriched": enriched_count,
        "docs_irrelevant": irrelevant_count,
        "docs_failed": len(docs) - enriched_count - irrelevant_count,
    }
    return record


def enrich_dataset(
    input_path: str,
    output_path: str,
    delay: float = DEFAULT_DELAY,
    limit: int = None,
) -> None:
    """
    Read a labeled JSONL, enrich every record, write to output_path.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    if limit:
        records = records[:limit]

    print(f"Enriching {len(records)} records from {input_path}")
    print(f"Output → {output_path}")
    print("-" * 60)

    total_docs = total_enriched = total_irrelevant = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for i, record in enumerate(records, 1):
            company = record.get("company_name", "?")
            doc_count = len(record.get("retrieved_docs", []))
            print(f"[{i}/{len(records)}] {company} ({doc_count} docs)...", end=" ", flush=True)

            record = enrich_record(record, delay=delay)
            e = record["_enrichment"]
            print(f"enriched={e['docs_enriched']} irrelevant={e['docs_irrelevant']} failed={e['docs_failed']}")

            total_docs      += e["docs_total"]
            total_enriched  += e["docs_enriched"]
            total_irrelevant += e["docs_irrelevant"]

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("-" * 60)
    print(f"Done. Total docs: {total_docs} | Enriched: {total_enriched} | "
          f"Irrelevant: {total_irrelevant} | Failed: {total_docs - total_enriched - total_irrelevant}")


def main():
    parser = argparse.ArgumentParser(description="Enrich JSONL dataset with full article content")
    parser.add_argument("--input",  required=True, help="Input labeled JSONL path")
    parser.add_argument("--output", required=True, help="Output enriched JSONL path")
    parser.add_argument("--delay",  type=float, default=DEFAULT_DELAY,
                        help=f"Seconds between requests (default {DEFAULT_DELAY})")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only process first N records (for testing)")
    args = parser.parse_args()

    enrich_dataset(
        input_path=args.input,
        output_path=args.output,
        delay=args.delay,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
