import pandas as pd
from pathlib import Path


def build_book_text(row: pd.Series) -> str:
    """
    Combine title, authors, subjects and description
    into one clean text field for embedding.
    """
    parts = []

    if pd.notna(row.get("title")):
        parts.append(f"Title: {row['title']}")

    if pd.notna(row.get("author")) and str(row.get("author")).strip() != "":
        parts.append(f"Author: {row['author']}")

    if pd.notna(row.get("subjects")):
        parts.append(f"Subjects: {row['subjects']}")

    if pd.notna(row.get("description")):
        parts.append(f"Description: {row['description']}")

    return "\n".join(parts)


def main():

    input_path = Path("data/silver/joined/openlibrary_books_joined_2026-02-26.parquet")
    output_path = Path("data/gold/books_embedding_input.parquet")

    print("Reading joined dataset...")
    df = pd.read_parquet(input_path)

    print(f"Total rows: {len(df)}")
    print("Columns:", sorted(df.columns.tolist()))

    # Define usability based on description length (simple + robust)
    df["description"] = df["description"].fillna("")
    

    print(f"Usable books (desc_len>=30): {len(df)}")


    # Build embedding text
    df["book_text"] = df.apply(build_book_text, axis=1)

    # Keep only necessary columns
    df_final = df[["key", "title", "book_text", "cover_i"]]

    # Create gold directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving embedding input dataset...")
    df_final.to_parquet(output_path, index=False)

    print("Done.")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
