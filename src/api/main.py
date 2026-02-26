from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
from src.retrieval.engine import SemanticSearchEngine
from typing import Any, Dict

app = FastAPI(title="Book Search API", version="0.1.0")
engine = None

@app.on_event("startup")
def startup():
    global engine
    engine = SemanticSearchEngine(
        index_path="data/gold/faiss_all-MiniLM-L6-v2.index",
        meta_path="data/gold/faiss_all-MiniLM-L6-v2_meta.parquet",
        joined_path="data/silver/joined/openlibrary_books_joined_2026-02-23.parquet",
        embedding_model="all-MiniLM-L6-v2",
    )
    print("✅ SemanticSearchEngine loaded")
    # warm up embedding model (avoid first-query latency)
    engine._get_model()
    engine.search_by_text("warmup", topk=1)
    print("✅ Embedding model warmed up")


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchHit(BaseModel):
    book_id: str
    score: float
    title: str = ""
    snippet: str = ""
    cover_i: int | None = None
    cover_url: str | None = None
    full_description: str = ""


class SearchResponse(BaseModel):
    query: str
    k: int
    results: List[SearchHit]

class SimilarResponse(BaseModel):
    seed: SearchHit
    query: str
    k: int
    results: List[SearchHit]

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    global engine
    assert engine is not None

    raw_results = engine.search_by_text(req.query, topk=req.k)

    results = []
    for r in raw_results:
        results.append(
            {
                "book_id": r["book_id"],
                "score": r["score"],
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "cover_i": r.get("cover_i"),
                "full_description": engine.get_description(r["book_id"]),
            }
        )
    
    for r in results:
        if r.get("cover_i"):
            r["cover_url"] = f"https://covers.openlibrary.org/b/id/{r['cover_i']}-M.jpg?default=false"
        else:
            r["cover_url"] = None

    return {
        "query": req.query,
        "k": req.k,
        "results": results,
    }

@app.get("/similar/{book_id:path}", response_model=SimilarResponse)
def similar(book_id: str, k: int = 5):
    global engine
    assert engine is not None

    raw_results = engine.search_by_key(book_id, topk=k)

    # build seed info
    seed_title = ""
    m = engine.meta[engine.meta["key"].astype(str) == str(book_id)]
    if len(m) > 0 and "title" in engine.meta.columns:
        seed_title = str(m.iloc[0].get("title", "") or "")

    seed = {
    "book_id": str(book_id),
    "score": 1.0,
    "title": seed_title,
    "snippet": engine.get_snippet(str(book_id))
    }

    results = []
    for r in raw_results:
        results.append(
            {
                "book_id": r["book_id"],
                "score": r["score"],
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
            }
        )

    payload = {
    "seed": {
        "book_id": str(book_id),
        "score": 1.0,
        "title": seed_title,
        "snippet": engine.get_snippet(str(book_id))
    },
    "query": f"similar:{book_id}",
    "k": k,
    "results": results,
    }
    print("SIMILAR PAYLOAD KEYS:", payload.keys())
    return payload