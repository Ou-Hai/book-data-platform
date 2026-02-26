import argparse
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

_LOCAL_MODEL_CACHE = {}

# Optional: load .env if you use it (recommended)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def embed_texts_dry_run(texts: List[str], dim: int = 16) -> List[List[float]]:
    """
    Deterministic fake embeddings for pipeline testing (no API calls).
    """
    embs: List[List[float]] = []
    for t in texts:
        # simple deterministic vector based on byte sum
        s = sum(t.encode("utf-8")) % 9973
        vec = [((s + i * 17) % 1000) / 1000.0 for i in range(dim)]
        embs.append(vec)
    return embs


def embed_texts_openai(texts: List[str], model: str) -> List[List[float]]:
    """
    OpenAI embeddings (requires OPENAI_API_KEY in env or .env).
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "openai package not found. Install it with: pip install openai"
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Put it in your environment or a .env file."
        )

    client = OpenAI(api_key=api_key)

    # Simple retry for transient failures / rate limits
    backoff = 2.0
    for attempt in range(1, 6):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            # Keep order as returned; OpenAI returns in same order with index
            data_sorted = sorted(resp.data, key=lambda x: x.index)
            return [d.embedding for d in data_sorted]
        except Exception as e:
            if attempt == 5:
                raise
            print(f"[WARN] embedding call failed (attempt {attempt}/5): {e}")
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("Unreachable")


def load_existing_keys(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    df_out = pd.read_parquet(output_path)
    if "key" not in df_out.columns:
        return set()
    return set(df_out["key"].astype(str).tolist())


def append_parquet(output_path: Path, df_new: pd.DataFrame) -> None:
    """
    Append by reading existing parquet (if any) then writing back.
    OK for 3k~50k scale in early stage. We can optimize later if needed.
    """
    if output_path.exists():
        df_old = pd.read_parquet(output_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(output_path, index=False)

def embed_texts_local(texts: List[str], model: str) -> List[List[float]]:
    """
    Local embeddings using sentence-transformers (free).
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers not found. Install with: uv add sentence-transformers"
        ) from e

    if model not in _LOCAL_MODEL_CACHE:
        _LOCAL_MODEL_CACHE[model] = SentenceTransformer(model)
    st_model = _LOCAL_MODEL_CACHE[model]

    # normalize_embeddings=True gives cosine-sim friendly vectors
    embs = st_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # Convert numpy array -> list[list[float]] for parquet
    return embs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/gold/books_embedding_input.parquet",
        help="Input parquet with columns: key, title, book_text",
    )
    parser.add_argument(
        "--output",
        default="data/gold/book_embeddings.parquet",
        help="Output parquet with embeddings",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="For quick tests, 0 = no limit")
    parser.add_argument(
    "--provider",
    choices=["dry-run", "openai", "local"],
    default="dry-run",
    help="Embedding provider",
)
    parser.add_argument(
    "--model",
    default="all-MiniLM-L6-v2",
    help="Model name (OpenAI model when provider=openai; sentence-transformers model when provider=local)",
)
    parser.add_argument(
        "--dry-dim",
        type=int,
        default=16,
        help="Vector dim for dry-run provider",
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[{_now_ts()}] Reading input: {input_path}")
    df = pd.read_parquet(input_path)

    required = {"key", "title", "book_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {missing}. Found: {list(df.columns)}")

    df["key"] = df["key"].astype(str)

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    #existing = load_existing_keys(output_path)
    #if existing:
        #before = len(df)
        #df = df[~df["key"].isin(existing)].copy()
        #print(f"[{_now_ts()}] Resume: skip {before - len(df)} already-embedded rows")

    if len(df) == 0:
        print(f"[{_now_ts()}] Nothing to do. Output already up to date: {output_path}")
        return

    print(f"[{_now_ts()}] Rows to embed: {len(df)} | provider={args.provider}")

    keys = df["key"].tolist()
    titles = df["title"].fillna("").astype(str).tolist()
    texts = df["book_text"].fillna("").astype(str).tolist()

    total = len(texts)
    done = 0

    for batch_idx, idxs in enumerate(_chunked(list(range(total)), args.batch_size), start=1):
        batch_texts = [texts[i] for i in idxs]
        batch_keys = [keys[i] for i in idxs]
        batch_titles = [titles[i] for i in idxs]

        if args.provider == "dry-run":
            batch_embs = embed_texts_dry_run(batch_texts, dim=args.dry_dim)
            model_used = f"dry-run-dim{args.dry_dim}"
        elif args.provider == "local":
            batch_embs = embed_texts_local(batch_texts, model=args.model)
            model_used = args.model
        else:
            batch_embs = embed_texts_openai(batch_texts, model=args.model)
            model_used = args.model

        cover_vals = df.iloc[idxs]["cover_i"].tolist() if "cover_i" in df.columns else [None]*len(batch_keys)
        df_new = pd.DataFrame(
            {
                "key": batch_keys,
                "title": batch_titles,
                "cover_i": cover_vals,
                "model": [model_used] * len(batch_keys),
                "embedded_at": [_now_ts()] * len(batch_keys),
                "embedding": batch_embs,
            }
        )

        append_parquet(output_path, df_new)

        done += len(batch_texts)
        print(f"[{_now_ts()}] Batch {batch_idx} saved | progress: {done}/{total} -> {output_path}")

    print(f"[{_now_ts()}] Done. Output: {output_path}")


if __name__ == "__main__":
    main()
