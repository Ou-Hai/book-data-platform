"""
OpenLibrary ingestion module.
"""
import json
from datetime import datetime
from pathlib import Path
import requests

def ingest_openlibrary(query: str = "bestsellers", limit: int = 100):
    """
    Fetch data from OpenLibrary API and store raw JSON to Bronze layer.
    """
    url = "https://openlibrary.org/search.json"
    response = requests.get(url, params={"q": query, "limit": limit}, timeout=30)
    response.raise_for_status()

    data = response.json()
    ingested_at = datetime.utcnow().isoformat()
    for doc in data.get("docs", []):
        doc["_ingested_at"] = ingested_at


    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

  
    output_dir = Path("data/bronze/books_raw") / f"ingestion_date={datetime.utcnow().date()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"openlibrary_{query}.json"
    if output_file.exists():
        print(f"Already ingested today: {output_file}")
        return


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    ingest_openlibrary(limit=10)
