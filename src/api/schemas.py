from pydantic import BaseModel, Field
from typing import List, Optional


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=50)


class SearchHit(BaseModel):
    book_id: str
    score: float
    title: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    k: int
    results: List[SearchHit]