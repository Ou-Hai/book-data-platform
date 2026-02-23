from pathlib import Path
import pandas as pd


def join_books_text(books_path: str, text_path: str, out_path: str) -> None:
    books = pd.read_parquet(books_path)
    text = pd.read_parquet(text_path)

    if "key" not in books.columns:
        raise ValueError("books file must contain column: key")
    if "key" not in text.columns or "description" not in text.columns:
        raise ValueError("text file must contain columns: key, description")

    out = books.merge(text, on="key", how="left")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"Saved: {out_path} rows={len(out)} desc_non_null={out['description'].notna().sum()}")


if __name__ == "__main__":
    join_books_text(
        books_path="data/silver/openlibrary_books_2026-02-23.parquet",
        text_path="data/silver/openlibrary_text_enriched_2026-02-20.parquet",
        out_path="data/silver/openlibrary_books_joined_2026-02-23.parquet",
    )
