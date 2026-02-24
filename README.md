# Book Data Platform

A production-style data pipeline for building a semantic book recommendation dataset using OpenLibrary.

---

## Project Goal

Build a structured, quality-controlled data platform that:
	•	Ingests book metadata from OpenLibrary
	•	Enriches descriptions via Works API
	•	Validates data quality
	•	Generates semantic embeddings
	•	Supports similarity search using FAISS

---

## Pipeline Overview

OpenLibrary Search API
→ Bronze (raw JSON)
→ Silver (cleaned metadata in Parquet)
→ Data Quality Checks
→ Description Enrichment (Works API, resumable)
→ Joined Dataset (metadata + description)
→ Embedding Layer (sentence-transformers)
→ FAISS Vector Index

---

## Key Features

- Bronze / Silver layered architecture  
- DuckDB-based transformations  
- Data quality validation with hard gates  
- Resumable API enrichment with retry & checkpoint  
- Duplicate prevention  
- Parquet-based storage 
- 384-dim semantic embeddings
- FAISS cosine similarity search
- Reusable semantic search engine class 

---

## Data Quality

Latest run:

- 5000 total books  
- 3054 books with description  
- 3014 embedding-ready (description length > 30)  
- 384-dimensional vector index

Quality checks include:

- Null validation (title, key, author)
- Duplicate key detection
- Row count validation
- Description completeness metrics

---

## Embedding & Retrieval

Embedding model:
- sentence-transformers/all-MiniLM-L6-v2
- 384-dimensional normalized vectors

Vector search:
- FAISS IndexFlatIP
- Cosine similarity
- Search by:
	•	book key
	•	title keyword
	•	free-text query

Example:

```bash
uv run python src/retrieval/query_similar.py \
  --query-text "mafia family crime in New York with loyalty and betrayal" \
  --topk 5 \
  --json

---

## Final Output

The system produces:

- Cleaned book dataset (Parquet) 
- Embedding dataset
- FAISS vector index
- SemanticSearchEngine (reusable retrieval component) 

Ready for semantic recommendation and API integration.

---

## Tech Stack

- Python  
- DuckDB  
- Pandas  
- Parquet  
- OpenLibrary API
- sentence-transformers
- FAISS  

---

## How to Run

```bash
python src/ingestion/openlibrary.py
python src/transformation/silver_openlibrary.py
python src/quality/silver_quality.py
python src/ingestion/enrich_descriptions.py
python src/transformation/join_books_text.py
python src/embedding/prepare_embedding_input.py
python src/embedding/embed_books.py --provider local
python src/retrieval/build_faiss_index.py
