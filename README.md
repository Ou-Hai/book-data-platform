# Book Data Platform

A production-style data pipeline for building a semantic book recommendation dataset using OpenLibrary.

---

## Project Goal

Build a structured, quality-controlled data platform that prepares book metadata and descriptions for semantic embedding and recommendation.

---

## Pipeline Overview

OpenLibrary Search API  
→ Bronze (raw JSON)  
→ Silver (cleaned metadata in Parquet)  
→ Data Quality Checks  
→ Description Enrichment (Works API, resumable)  
→ Joined Dataset (metadata + description)  

---

## Key Features

- Bronze / Silver layered architecture  
- DuckDB-based transformations  
- Data quality validation with hard gates  
- Resumable API enrichment with retry & checkpoint  
- Duplicate prevention  
- Parquet-based storage  

---

## Data Quality

Latest run:

- 5000 total books  
- 3054 books with description  
- 3014 embedding-ready (description length > 30)  

Quality checks include:

- Null validation (title, key, author)
- Duplicate key detection
- Row count validation
- Description completeness metrics

---

## Final Output

The final dataset contains:

- key  
- title  
- author  
- first_publish_year  
- language  
- description  

Ready for semantic embedding.

---

## Tech Stack

- Python  
- DuckDB  
- Pandas  
- Parquet  
- OpenLibrary API  

---

## How to Run

```bash
python src/ingestion/openlibrary.py
python src/transformation/silver_openlibrary.py
python src/quality/silver_quality.py
python src/ingestion/enrich_descriptions.py
python src/transformation/join_books_text.py
