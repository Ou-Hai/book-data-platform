import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        default="data/gold/book_embeddings_all-MiniLM-L6-v2.parquet",
        help="Parquet with columns: key, embedding (list[float])",
    )
    parser.add_argument(
        "--index-out",
        default="data/gold/faiss_all-MiniLM-L6-v2.index",
        help="FAISS index output path",
    )
    parser.add_argument(
        "--meta-out",
        default="data/gold/faiss_all-MiniLM-L6-v2_meta.parquet",
        help="Metadata mapping output (same row order as FAISS vectors)",
    )
    args = parser.parse_args()

    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss not available. Did you run: uv add faiss-cpu ?") from e

    emb_path = Path(args.embeddings)
    index_out = Path(args.index_out)
    meta_out = Path(args.meta_out)

    print(f"Reading embeddings: {emb_path}")
    df = pd.read_parquet(emb_path)

    required = {"key", "embedding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

    df["key"] = df["key"].astype(str)
    df = df.dropna(subset=["embedding"]).copy()

    # Convert to numpy matrix (N, D)
    vectors = np.array(df["embedding"].tolist(), dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (N,D). Got shape: {vectors.shape}")

    n, d = vectors.shape
    print(f"Vectors: N={n}, D={d}")

    # Use Inner Product index. With normalized vectors, IP == cosine similarity.
    # Your local provider used normalize_embeddings=True, but we normalize again defensively.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    index = faiss.IndexFlatIP(d)
    index.add(vectors)

    index_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_out))

    # Save mapping in the same order as vectors in FAISS
    meta_cols = ["key"]
    if "title" in df.columns:
        meta_cols.append("title")
    meta = df[meta_cols].reset_index(drop=True)
    meta.to_parquet(meta_out, index=False)

    print(f"Saved FAISS index: {index_out}")
    print(f"Saved meta mapping: {meta_out}")
    print("Done.")


if __name__ == "__main__":
    main()