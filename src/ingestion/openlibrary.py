"""
Ingestion: fetch OpenLibrary search results and store raw JSON to Bronze layer.
Supports multi-query paging and a global max_docs cap.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, UTC
from typing import Iterable

import requests


def ingest_openlibrary_many(
    queries: Iterable[str],
    max_docs: int = 5000,
    page_size: int = 100,
    sleep_s: float = 0.2,
) -> Path:
    """
    Fetch OpenLibrary docs for multiple queries with pagination, until max_docs reached.
    Writes one JSON file per (query, page) into a date-partitioned Bronze folder.
    Returns today's bronze partition directory.
    """
    base_url = "https://openlibrary.org/search.json"
    today = datetime.now(UTC).date()
    out_base = Path("data/bronze/books_raw") / f"ingestion_date={today}"
    out_base.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    ingested_at = datetime.now(UTC).isoformat()

    session = requests.Session()

    for q in queries:
        page = 1
        while total_docs < max_docs:
            out_dir = out_base / f"query={q}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"page={page:04d}.json"
            if out_file.exists():
                page += 1
                continue

            params = {"q": q, "limit": page_size, "page": page}
            try:
                resp = session.get(base_url, params=params, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"[WARN] request failed q={q} page={page}: {e}")
                time.sleep(1.0)
                try:
                    resp = session.get(base_url, params=params, timeout=30)
                    resp.raise_for_status()
                except requests.RequestException as e2:
                    print(f"[ERROR] giving up q={q} page={page}: {e2}")
                    break

            data = resp.json()
            docs = data.get("docs", [])

           
            if not docs:
                break

            
            for d in docs:
                d["_ingested_at"] = ingested_at

            
            payload = {
                "query": q,
                "page": page,
                "page_size": page_size,
                "ingested_at": ingested_at,
                "numFound": data.get("numFound"),
                "docs": docs,
            }
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            total_docs += len(docs)
            print(f"[OK] q={q} page={page} docs={len(docs)} total_docs={total_docs} -> {out_file}")

            # 达到上限就结束
            if total_docs >= max_docs:
                break

            page += 1
            time.sleep(sleep_s)

        if total_docs >= max_docs:
            break

    print(f"[DONE] bronze partition: {out_base} total_docs={total_docs}")
    return out_base


if __name__ == "__main__":
    seed_queries = [
        "bestsellers", "fiction", "novel", "mystery", "thriller", "romance",
        "fantasy", "science fiction", "history", "biography", "philosophy",
        "psychology", "business", "self help", "technology", "data science",
        "travel", "cooking", "poetry", "children",
    ]
    ingest_openlibrary_many(seed_queries, max_docs=5000, page_size=100)