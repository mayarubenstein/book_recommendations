"""
Example FastAPI wiring for content_recommender2.py.

Run with:  uvicorn api:app --reload
Interactive docs (auto-generated from the Pydantic models):
  http://127.0.0.1:8000/docs

Since UserProfile is a Pydantic model that already mirrors your quiz's JSON
shape, POSTing that exact JSON to /recommend as the request body just works --
FastAPI validates and parses it for you, no glue code needed.
"""
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from book_verifier import BookEvaluation, verify_books
from content_recommender2 import (
    Recommender,
    SentenceTransformerEmbedder,
    UserProfile,
    load_catalog,
    load_runtime_catalog,
)

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "all_books.json"
ARTIFACT_PATH = BASE_DIR / "catalog_embeddings.npz"
RUNTIME_CATALOG_PATH = BASE_DIR / "catalog_runtime.pkl"

# Populated once at startup, reused for every request. Re-embedding ~326k
# books per request would make each call take minutes instead of
# milliseconds -- this is exactly why Recommender separates fit() (slow,
# once) from recommend() (fast, per-request).
state: dict[str, Recommender] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if RUNTIME_CATALOG_PATH.exists():
        catalog, fingerprint = load_runtime_catalog(RUNTIME_CATALOG_PATH, CATALOG_PATH)
    else:
        catalog = load_catalog(CATALOG_PATH)
        fingerprint = None
    embedder = SentenceTransformerEmbedder()
    state["recommender"] = Recommender(catalog, embedder).load_embedding_artifact(
        ARTIFACT_PATH, CATALOG_PATH, fingerprint
    )
    yield
    state.clear()


app = FastAPI(title="Book Recommender", lifespan=lifespan)


class BookRecommendation(BaseModel):
    book_id: str | None
    title: str | None
    authors: str | None
    genre: str | None
    description: str | None
    review_count: float | None
    average_rating: float | None
    sim_preferences: float
    sim_liked_books: float
    sim_demographics: float
    score: float
    is_approved: bool
    hard_constraint_check: str
    soft_constraint_check: str
    rejection_reason: str | None
    review_summaries: list[dict] = Field(default_factory=list)


@app.post("/recommend", response_model=list[BookRecommendation])
def recommend(
    profile: UserProfile,
    top_n: int = Query(default=20, ge=1, le=100),
) -> list[BookRecommendation]:
    candidate_count = top_n * 2
    recs = state["recommender"].recommend(profile, top_n=candidate_count)

    # A DataFrame isn't directly JSON-safe: NaN (e.g. a book missing a
    # rating) isn't valid JSON and most clients will choke on it. Swap NaN
    # for None before handing rows to Pydantic.
    recs = recs.replace({np.nan: None})
    candidates = [
        {
            "book_id": row["Book ID"],
            "title": row["Title"],
            "authors": row["Authors"],
            "genre": row["Category/Genre"],
            "description": row["Description"],
            "review_count": row["Review_Count"],
            "average_rating": row["Average_Normalized_Rating"],
            "sim_preferences": row["sim_preferences"],
            "sim_liked_books": row["sim_liked_books"],
            "sim_demographics": row["sim_demographics"],
            "score": row["score"],
            "review_summaries": row.get("_review_summaries", []),
        }
        for _, row in recs.iterrows()
    ]
    try:
        verification = verify_books(profile.model_dump(), candidates)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Recommendation verification failed: {exc}",
        ) from exc

    evaluations_by_id = {
        evaluation.book_id: evaluation
        for evaluation in verification.evaluations
        if evaluation.book_id
    }
    evaluations_by_title = {
        evaluation.title.strip().casefold(): evaluation
        for evaluation in verification.evaluations
    }
    approved: list[tuple[dict, BookEvaluation]] = []
    for candidate in candidates:
        evaluation = evaluations_by_id.get(str(candidate["book_id"]))
        if evaluation is None:
            evaluation = evaluations_by_title.get(str(candidate["title"]).strip().casefold())
        if evaluation is not None and evaluation.is_approved:
            approved.append((candidate, evaluation))
        if len(approved) == top_n:
            break

    return [
        BookRecommendation(
            book_id=row["book_id"],
            title=row["title"],
            authors=row["authors"],
            genre=row["genre"],
            description=row["description"],
            review_count=row["review_count"],
            average_rating=row["average_rating"],
            sim_preferences=row["sim_preferences"],
            sim_liked_books=row["sim_liked_books"],
            sim_demographics=row["sim_demographics"],
            score=row["score"],
            is_approved=evaluation.is_approved,
            hard_constraint_check=evaluation.hard_constraint_check,
            soft_constraint_check=evaluation.soft_constraint_check,
            rejection_reason=evaluation.rejection_reason,
            review_summaries=row["review_summaries"],
        )
        for row, evaluation in approved
    ]


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ready": "recommender" in state}
