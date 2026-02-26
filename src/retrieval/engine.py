from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

_LOCAL_MODEL_CACHE = {}


class SemanticSearchEngine:
    """
    Reusable semantic search engine based on:
    - FAISS index
    - sentence-transformers embeddings
    """

    def __init__(
        self,
        index_path: str,
        meta_path: str,
        joined_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.joined_path = Path(joined_path)
        self.embedding_model = embedding_model

        self._load_resources()

    def _load_resources(self):
        import faiss  # type: ignore

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))

        print("Loading metadata...")
        self.meta = pd.read_parquet(self.meta_path)
        self.meta["key"] = self.meta["key"].astype(str)

        print("Loading joined data...")
        self.joined = pd.read_parquet(self.joined_path)
        self.joined["key"] = self.joined["key"].astype(str)

    def _get_model(self):
        from sentence_transformers import SentenceTransformer  # type: ignore

        if self.embedding_model not in _LOCAL_MODEL_CACHE:
            _LOCAL_MODEL_CACHE[self.embedding_model] = SentenceTransformer(
                self.embedding_model
            )
        return _LOCAL_MODEL_CACHE[self.embedding_model]

    def _embed_text(self, text: str) -> np.ndarray:
        model = self._get_model()
        vec = model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]

        vec = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec = vec / norm

        return vec.reshape(1, -1)

    def _get_snippet(self, key: str, max_len: int = 600) -> str:
        match = self.joined[self.joined["key"] == key]
        if match.empty:
            return ""

        val = match.iloc[0].get("description", "")
        if pd.isna(val):
            val = ""

        text = str(val)
        return text[:max_len] + ("..." if len(text) > max_len else "")
    
    def get_snippet(self, book_id: str, n: int = 180) -> str:
        return self._get_snippet(book_id, max_len=n)
    
    def get_description(self, key: str) -> str:
        match = self.joined[self.joined["key"] == key]
        if match.empty:
            return ""
        val = match.iloc[0].get("description", "")
        if pd.isna(val):
            return ""
        return str(val)

    def search_by_text(self, query_text: str, topk: int = 10) -> List[Dict]:
        query_vector = self._embed_text(query_text)

        scores, indices = self.index.search(query_vector, topk)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            key = str(self.meta.iloc[idx]["key"])
            title = (
                self.meta.iloc[idx]["title"]
                if "title" in self.meta.columns
                else ""
            )
            cover_i = self.meta.iloc[idx].get("cover_i") if "cover_i" in self.meta.columns else None

            results.append(
                {
                    "book_id": key,
                    "title": title,
                    "score": float(score),
                    "snippet": self._get_snippet(key),
                    "cover_i": int(cover_i) if cover_i and not pd.isna(cover_i) else None,
                }
            ) 

        return results

    def search_by_key(self, key: str, topk: int = 10) -> List[Dict]:
        # Find embedding vector by key from meta index position
        if key not in set(self.meta["key"]):
            raise ValueError(f"Key not found: {key}")

        idx = self.meta[self.meta["key"] == key].index[0]

        query_vector = self.index.reconstruct(int(idx)).reshape(1, -1)

        scores, indices = self.index.search(query_vector, topk + 1)

        results = []
        for score, i in zip(scores[0], indices[0]):
            result_key = str(self.meta.iloc[i]["key"])

            if result_key == key:
                continue

            title = (
                self.meta.iloc[i]["title"]
                if "title" in self.meta.columns
                else ""
            )

            results.append(
                {
                    "book_id": result_key,
                    "title": title,
                    "score": float(score),
                    "snippet": self._get_snippet(result_key),
                }
            )

        return results[:topk]
