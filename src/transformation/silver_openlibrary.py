import json
from pathlib import Path
from datetime import date

import pandas as pd



def bronze_to_silver_today() -> Path:
    """Load today's bronze OpenLibrary JSON files, deduplicate by 'key', and write silver parquet."""
    bronze_dir = Path("data/bronze/books_raw") / f"ingestion_date={date.today()}"
    bronze_files = sorted(bronze_dir.glob("*.json"))

    all_docs = []
    for fp in bronze_files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        all_docs.extend(data.get("docs", []))

    df = pd.DataFrame(all_docs)

    keep_cols = [c for c in ["key", "title", "author_name", "first_publish_year", "language"] if c in df.columns]
    df = df[keep_cols].copy()

    df = df.drop_duplicates(subset=["key"])

    out_dir = Path("data/silver")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"openlibrary_silver_{date.today()}.parquet"

    df.to_parquet(out_path, index=False)
    print(f"Silver saved to {out_path} rows={len(df)} bronze_files={len(bronze_files)}")

    return out_path