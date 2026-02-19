import json
from pathlib import Path
from datetime import date

import pandas as pd


def bronze_to_silver_today() -> Path:
    """Load today's bronze OpenLibrary file, deduplicate by 'key', and write silver parquet."""
    bronze_path = (
        Path("data/bronze/books_raw")
        / f"ingestion_date={date.today()}"
        / "openlibrary_bestsellers.json"
    )

    data = json.loads(bronze_path.read_text(encoding="utf-8"))
    docs = data.get("docs", [])

    df = pd.DataFrame(docs)

    # keep only a few columns for now
    keep_cols = [c for c in ["key", "title", "author_name", "first_publish_year", "language"] if c in df.columns]
    df = df[keep_cols].copy()

    # deduplicate by primary key
    df = df.drop_duplicates(subset=["key"])

    out_dir = Path("data/silver")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"openlibrary_silver_{date.today()}.parquet"

    df.to_parquet(out_path, index=False)
    print(f"Silver saved to {out_path} rows={len(df)}")

    return out_path