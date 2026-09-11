"""
Example FastAPI wiring for content_recommender.py.

Run with:  uvicorn api:app --reload
Interactive docs (auto-generated from the Pydantic models):
  http://127.0.0.1:8000/docs

Since UserProfile is a Pydantic model that already mirrors your quiz's JSON
shape, POSTing that exact JSON to /recommend as the request body just works --
FastAPI validates and parses it for you, no glue code needed.
"""
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from content_recommender2 import Recommender, SentenceTransformerEmbedder, UserProfile, load_catalog

CATALOG_PATH = "all_books.json"
CACHE_PATH = "catalog_embeddings.npy"

# Populated once at startup, reused for every request. Re-embedding ~326k
# books per request would make each call take minutes instead of
# milliseconds -- this is exactly why Recommender separates fit() (slow,
# once) from recommend() (fast, per-request).
state: dict[str, Recommender] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    catalog = load_catalog(CATALOG_PATH)
    embedder = SentenceTransformerEmbedder()
    state["recommender"] = Recommender(catalog, embedder).fit(cache_path=CACHE_PATH)
    yield
    state.clear()


app = FastAPI(title="Book Recommender", lifespan=lifespan)


class BookRecommendation(BaseModel):
    book_id: str | None
    title: str | None
    authors: str | None
    genre: str | None
    review_count: float | None
    average_rating: float | None
    sim_preferences: float
    sim_liked_books: float
    sim_demographics: float
    score: float


@app.post("/recommend", response_model=list[BookRecommendation])
def recommend(profile: UserProfile, top_n: int = 20) -> list[BookRecommendation]:
    recs = state["recommender"].recommend(profile, top_n=top_n)

    # A DataFrame isn't directly JSON-safe: NaN (e.g. a book missing a
    # rating) isn't valid JSON and most clients will choke on it. Swap NaN
    # for None before handing rows to Pydantic.
    recs = recs.replace({np.nan: None})

    return [
        BookRecommendation(
            book_id=row["Book ID"],
            title=row["Title"],
            authors=row["Authors"],
            genre=row["Category/Genre"],
            review_count=row["Review_Count"],
            average_rating=row["Average_Normalized_Rating"],
            sim_preferences=row["sim_preferences"],
            sim_liked_books=row["sim_liked_books"],
            sim_demographics=row["sim_demographics"],
            score=row["score"],
        )
        for _, row in recs.iterrows()
    ]


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ready": "recommender" in state}
