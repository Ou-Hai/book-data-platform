
import json
import time
from pathlib import Path
from datetime import date

import pandas as pd
import requests


BASE = "https://openlibrary.org"


def normalize_description(desc):
    if desc is None:
        return None
    if isinstance(desc, dict):
        return desc.get("value")
    if isinstance(desc, list):
        return "\n".join([str(x) for x in desc if x])
    return str(desc)


def enrich_descriptions(ingestion_date: str, limit: int = 5000, sleep_s: float = 0.2) -> Path:
    books_path = Path("data/silver") / f"openlibrary_books_{ingestion_date}.parquet"
    if not books_path.exists():
        raise FileNotFoundError(f"{books_path} not found")

    out_dir = Path("data/silver")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"openlibrary_text_enriched_{ingestion_date}.parquet"

    books = pd.read_parquet(books_path).head(limit).copy()
    keys = books["key"].tolist()

    # ---- load existing progress (resume) ----
    existing = {}
    if out_path.exists():
        old = pd.read_parquet(out_path)
        # keep even None -> means we already attempted it
        existing = dict(zip(old["key"].tolist(), old["description"].tolist()))
        print(f"resume: found existing {len(existing)} rows in {out_path}")

    # We'll build results in a dict, then write periodically
    results = dict(existing)

    def save_checkpoint():
        df_out = pd.DataFrame(
            {"key": list(results.keys()), "description": pd.Series(list(results.values()), dtype="string")}
        )
        df_out.to_parquet(out_path, index=False)

    # ---- fetch missing only ----
    max_retries = 5
    for i, k in enumerate(keys, start=1):

        if i % 50 == 0:
            done = len(results)
            print(f"progress: scanned {i}/{len(keys)} | saved {done} descriptions")

        if k in results:
            continue  # already attempted/saved

        url = f"{BASE}{k}.json"
        desc = None

        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(url, timeout=20)

                if r.status_code == 503:
                    wait = 1.0 * attempt
                    print(f"[WARN] 503 for {k} attempt {attempt}/{max_retries}, sleep {wait:.1f}s")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                data = r.json()
                desc = normalize_description(data.get("description"))
                break

            except requests.RequestException as e:
                wait = 1.0 * attempt
                print(f"[WARN] request failed for {k} attempt {attempt}/{max_retries}: {e}")
                time.sleep(wait)

        results[k] = desc
        time.sleep(sleep_s)

        # save checkpoint every 200 new items
        if len(results) % 200 == 0:
            save_checkpoint()
            print(f"checkpoint saved: {out_path} rows={len(results)}")

    # final save
    save_checkpoint()
    print(f"Saved: {out_path} rows={len(results)}")
    return out_path

if __name__ == "__main__":
    enrich_descriptions("2026-02-20", limit=5000, sleep_s=0.2)