"""
Silver transformation for OpenLibrary pipeline.

Creates:
- data/silver/openlibrary_books_<date>.parquet (metadata)
- data/silver/openlibrary_text_<date>.parquet (description text)
"""

import json
from pathlib import Path
from datetime import date
import pandas as pd


def _list_to_str(x, sep="; "):
    """Convert list-like fields to a single string."""
    if isinstance(x, list):
        items = [str(i).strip() for i in x if i is not None and str(i).strip() != ""]
        return sep.join(items) if items else None
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s != "" else None


def _extract_description(x):
    """Normalize OpenLibrary description to plain string."""
    if isinstance(x, dict):
        x = x.get("value") 
    return _list_to_str(x, sep="\n") 


def bronze_to_silver(ingestion_date: str) -> tuple[Path, Path]:
    bronze_dir = Path("data/bronze/books_raw") / f"ingestion_date={ingestion_date}"
    bronze_files = sorted(bronze_dir.rglob("*.json"))
    if not bronze_files:
        raise FileNotFoundError(f"No bronze json files found in {bronze_dir}")

    all_docs = []
    for fp in bronze_files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        all_docs.extend(data.get("docs", []))

    df = pd.DataFrame(all_docs)

    meta_cols = ["key", "title", "author_name", "first_publish_year", "language"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df[meta_cols].copy()

    if "author_name" in meta.columns:
        meta["author"] = meta["author_name"].apply(_list_to_str)
        meta.drop(columns=["author_name"], inplace=True)

    if "language" in meta.columns:
        meta["language"] = meta["language"].apply(_list_to_str)

    if "first_publish_year" in meta.columns:
        meta["first_publish_year"] = pd.to_numeric(meta["first_publish_year"], errors="coerce").astype("Int64")

    if "key" in meta.columns:
        meta = meta.drop_duplicates(subset=["key"])

   # Always output a stable text schema: key + description
    text = df[["key"]].copy() if "key" in df.columns else pd.DataFrame(columns=["key"])
    text["description"] = pd.Series([None] * len(text), dtype="string")

    if "description" in df.columns:
        text["description"] = df["description"].apply(_extract_description).astype("string")

    if "key" in text.columns:
        text = text.drop_duplicates(subset=["key"])

    out_dir = Path("data/silver")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / f"openlibrary_books_{date.today()}.parquet"
    text_path = out_dir / f"openlibrary_text_{date.today()}.parquet"

    meta.to_parquet(meta_path, index=False)
    text.to_parquet(text_path, index=False)

    print(
        f"Silver saved:\n"
        f"- metadata: {meta_path} rows={len(meta)}\n"
        f"- text:     {text_path} rows={len(text)}\n"
        f"bronze_files={len(bronze_files)}"
    )
    return meta_path, text_path


if __name__ == "__main__":
    bronze_to_silver("2026-02-20")


