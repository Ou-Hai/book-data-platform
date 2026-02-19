import requests

def fetch_openlibrary_count(query: str = "harry potter") -> int:
    """Return how many results OpenLibrary search has for a query."""
    url = "https://openlibrary.org/search.json"
    r = requests.get(url, params={"q": query, "limit": 1}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return int(data.get("numFound", 0))
