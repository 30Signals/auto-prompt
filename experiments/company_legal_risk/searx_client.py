"""
SearXNG Client for Company Legal Risk Task

Reads credentials from environment variables only.
"""

import os
import requests
from requests.auth import HTTPBasicAuth


class SearXNGClient:
    def __init__(self, timeout=20):
        self.base_url = os.getenv("SEARXNG_BASE_URL", "").rstrip("/")
        self.username = os.getenv("SEARXNG_USERNAME", "")
        self.password = os.getenv("SEARXNG_PASSWORD", "")
        self.timeout = timeout

    def is_configured(self):
        return bool(self.base_url)

    def search(self, query, categories="news", language="en", safesearch=1, limit=5):
        if not self.is_configured():
            raise ValueError("SEARXNG_BASE_URL is not configured.")

        params = {
            "q": query,
            "format": "json",
            "categories": categories,
            "language": language,
            "safesearch": safesearch,
        }

        auth = None
        if self.username and self.password:
            auth = HTTPBasicAuth(self.username, self.password)

        response = requests.get(
            f"{self.base_url}/search",
            params=params,
            auth=auth,
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        return results[:limit]


def build_company_queries(company_name):
    """Primary quoted queries — exact company name match."""
    return [
        f'"{company_name}" lawsuit settlement regulatory action',
        f'"{company_name}" compliance violation fine penalty',
        f'"{company_name}" fraud investigation sanction news',
    ]


def build_fallback_queries(company_name):
    """
    Broader unquoted fallback queries used when primary queries return zero hits.
    Drops strict quoting so SearXNG can match partial/variant name forms.
    """
    return [
        f"{company_name} lawsuit legal action",
        f"{company_name} fine penalty regulatory",
        f"{company_name} investigation misconduct",
        f"{company_name} court settlement",
    ]


def build_search_context(results, max_items=8):
    """Format retrieved docs into a context string for the LLM."""
    if not results:
        return "No search evidence retrieved."

    lines = []
    for i, item in enumerate(results[:max_items], start=1):
        title = str(item.get("title", "")).strip()
        # Prefer full_content if the URL fetcher has already enriched this doc
        content = str(item.get("full_content") or item.get("content", "")).strip()
        url = str(item.get("url", "")).strip()
        source = str(item.get("source", "")).strip() or str(item.get("engine", "")).strip()
        lines.append(
            f"[{i}] Title: {title}\nSource: {source}\nURL: {url}\nSnippet: {content}"
        )
    return "\n\n".join(lines)


def normalize_retrieved_docs(results, max_items=12):
    """
    Normalize raw SearXNG results into a consistent doc format.

    Fields:
      title           - article headline
      content         - snippet from SearXNG (truncated ~150 chars)
      full_content    - full article text populated later by url_fetcher.py
      content_fetched - False until url_fetcher.py has processed this doc
      source          - news source name
      url             - article URL
      irrelevant      - True if url_fetcher finds the doc does not mention the company
    """
    docs = []
    for item in results[:max_items]:
        docs.append(
            {
                "title": str(item.get("title", "")).strip(),
                "content": str(item.get("content", "")).strip(),
                "full_content": "",
                "content_fetched": False,
                "source": str(item.get("source", "")).strip() or str(item.get("engine", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "irrelevant": False,
            }
        )
    return docs


def collect_company_evidence(client, company_name, limit_per_query=4, return_query_log=False):
    """
    Query SearXNG for legal/compliance evidence on a company.

    Strategy:
      1. Run 3 primary quoted queries.
      2. If ALL primary queries return zero hits, run 4 broader fallback queries.
         Fallback results are flagged with used_fallback=True in the query log.
      3. Deduplicate by URL across all results.
    """
    all_results = []
    query_log = []

    # --- Primary queries ---
    for query in build_company_queries(company_name):
        entry = {"query": query, "status": "ok", "hits": 0, "error": "", "used_fallback": False}
        try:
            chunk = client.search(query=query, limit=limit_per_query)
            all_results.extend(chunk)
            entry["hits"] = len(chunk)
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
        query_log.append(entry)

    # --- Fallback queries (only when every primary query returned 0 hits) ---
    primary_total_hits = sum(e["hits"] for e in query_log)
    if primary_total_hits == 0:
        for query in build_fallback_queries(company_name):
            entry = {"query": query, "status": "ok", "hits": 0, "error": "", "used_fallback": True}
            try:
                chunk = client.search(query=query, limit=limit_per_query)
                all_results.extend(chunk)
                entry["hits"] = len(chunk)
            except Exception as exc:
                entry["status"] = "error"
                entry["error"] = str(exc)
            query_log.append(entry)

    # --- Deduplicate by URL ---
    seen_urls = set()
    deduped = []
    for item in all_results:
        url = str(item.get("url", "")).strip()
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        deduped.append(item)

    if return_query_log:
        return deduped, query_log
    return deduped
