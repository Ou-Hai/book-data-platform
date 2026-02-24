import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import json

_LOCAL_MODEL_CACHE = {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key",
        default="",
        help="Book key to search similar items for",
    )
    parser.add_argument(
        "--title-contains",
        default="",
        help="If provided, find first book whose title contains this text (case-insensitive) and use its key",
    )
    parser.add_argument(
        "--query-text",
        default="",
        help="If provided, embed this free-text query and retrieve similar books (ignores --key/--title-contains)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of similar books to return",
    )
    parser.add_argument(
        "--index-path",
        default="data/gold/faiss_all-MiniLM-L6-v2.index",
    )
    parser.add_argument(
        "--meta-path",
        default="data/gold/faiss_all-MiniLM-L6-v2_meta.parquet",
    )
    parser.add_argument(
        "--embeddings-path",
        default="data/gold/book_embeddings_all-MiniLM-L6-v2.parquet",
    )
    parser.add_argument(
        "--joined-path",
        default="data/silver/joined/openlibrary_books_joined_2026-02-23.parquet",
        help="Joined parquet with description",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="If set, output results as JSON instead of formatted text",
    )
    args = parser.parse_args()

    def snippet(text: str, n: int = 180) -> str:
        text = (text or "").replace("\n", " ").strip()
        return text[:n] + ("..." if len(text) > n else "")

    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss not installed") from e
    

    def embed_query_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        """Embed free-text query using sentence-transformers and return a normalized float32 vector."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers not installed. Install with: uv add sentence-transformers"
            ) from e

        if model_name not in _LOCAL_MODEL_CACHE:
            _LOCAL_MODEL_CACHE[model_name] = SentenceTransformer(model_name)

        st_model = _LOCAL_MODEL_CACHE[model_name]

        vec = st_model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        vec = np.asarray(vec, dtype=np.float32)

        # defensive normalize
        n = np.linalg.norm(vec)
        if n != 0:
            vec = vec / n

        return vec.reshape(1, -1)

        # defensive normalize
        n = np.linalg.norm(vec)
        if n != 0:
            vec = vec / n

        return vec.reshape(1, -1)

    index_path = Path(args.index_path)
    meta_path = Path(args.meta_path)
    emb_path = Path(args.embeddings_path)



    print("Loading FAISS index...")
    index = faiss.read_index(str(index_path))

    print("Loading metadata...")
    meta = pd.read_parquet(meta_path)

    print("Loading embeddings...")
    df_emb = pd.read_parquet(emb_path)

    # If query-text is provided, we will embed it and search directly (no need for query_key)
    query_text = args.query_text.strip()
    # Choose query key
    query_key = args.key.strip()
    
    if not query_key and args.title_contains.strip():
        q = args.title_contains.strip().lower()
        hits = df_emb[df_emb["title"].fillna("").astype(str).str.lower().str.contains(q)]
        if len(hits) == 0:
            raise ValueError(f"No title contains: {args.title_contains}")
        query_key = str(hits.iloc[0]["key"])
        print(f"Selected key by title match: {query_key} | title={hits.iloc[0]['title']}")

    if not query_text and not query_key:
        raise ValueError("Provide --query-text or --key or --title-contains")

    df_joined = pd.read_parquet(Path(args.joined_path))
    df_joined["key"] = df_joined["key"].astype(str)


    df_emb["key"] = df_emb["key"].astype(str)

    if not query_text:
        if query_key not in set(df_emb["key"]):
            raise ValueError(f"Key not found: {query_key}")

    if query_text:
        query_vector = embed_query_text(query_text, model_name="all-MiniLM-L6-v2")
        print("\nQuery text:")
        print(f"  {query_text}")
        query_row = None  # no book row
    else:
        # Get query vector from a book key
        query_row = df_emb[df_emb["key"] == query_key].iloc[0]
        query_vector = np.array(query_row["embedding"], dtype=np.float32)

        # Normalize (defensive, though already normalized)
        norm = np.linalg.norm(query_vector)
        if norm != 0:
            query_vector = query_vector / norm

        query_vector = query_vector.reshape(1, -1)

    print(f"Searching top {args.topk} similar books...")

    scores, indices = index.search(query_vector, args.topk + 1)

    if not query_text:
        print("\nQuery book:")
        if "title" in query_row:
            print(f"  {query_row['title']} (key={query_key})")
        else:
            print(f"  key={query_key}")

    results = []

    for score, idx in zip(scores[0], indices[0]):
        result_key = str(meta.iloc[idx]["key"])

        if not query_text and result_key == query_key:
            continue

        title = meta.iloc[idx]["title"] if "title" in meta.columns else ""

        desc = ""
        match = df_joined[df_joined["key"] == result_key]
        if len(match) > 0:
            desc = match.iloc[0].get("description", "") or ""

        results.append(
            {
                "key": result_key,
                "title": title,
                "score": float(score),
                "snippet": snippet(str(desc)) if desc else "",
            }
        )

    if args.json:
        print(json.dumps(results[: args.topk], indent=2, ensure_ascii=False))
    else:
        print("\nTop similar books:\n")
        for r in results[: args.topk]:
            print(f"Score: {r['score']:.4f} | {r['title']} | key={r['key']}")
            if r["snippet"]:
                print(f"  - {r['snippet']}")
    


if __name__ == "__main__":
    main()