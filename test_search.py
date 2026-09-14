"""Tests for search.py and the fuzzy-matching additions to content_recommender2.py.

Uses a small synthetic catalog throughout -- the real all_books.json (~326k
books) is gitignored and not expected to be present in this environment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from content_recommender2 import (
    Layer1,
    LikedBook,
    Recommender,
    UserProfile,
)
from search import build_search_corpus, search_books


def _make_catalog() -> pd.DataFrame:
    rows = [
        {
            "Book ID": "BX_1",
            "Title": "Harry Potter and the Sorcerer's Stone",
            "Authors": "J.K. Rowling",
            "Category/Genre": "Fantasy",
            "Description": "A boy wizard.",
            "Review_Count": 1000,
            "Average_Normalized_Rating": 4.5,
            "content_text": "harry potter fantasy rowling",
        },
        {
            "Book ID": "BX_2",
            "Title": "The Hobbit",
            "Authors": "J.R.R. Tolkien",
            "Category/Genre": "Fantasy",
            "Description": "A hobbit's journey.",
            "Review_Count": 800,
            "Average_Normalized_Rating": 4.7,
            "content_text": "the hobbit fantasy tolkien",
        },
        {
            "Book ID": "BX_3",
            "Title": "Pride and Prejudice",
            "Authors": "Jane Austen",
            "Category/Genre": "Romance",
            "Description": "Society and marriage in Regency England.",
            "Review_Count": 500,
            "Average_Normalized_Rating": 4.6,
            "content_text": "pride and prejudice romance austen",
        },
    ]
    df = pd.DataFrame(rows)
    df["_authors_str"] = df["Authors"]
    df["_clean_title_str"] = df["Title"].str.strip().str.lower()
    df["_is_english"] = True
    return df


@pytest.fixture
def catalog() -> pd.DataFrame:
    return _make_catalog()


@pytest.fixture
def corpus(catalog: pd.DataFrame) -> pd.DataFrame:
    return build_search_corpus(catalog)


class _IdentityEmbedder:
    """Stand-in for SentenceTransformerEmbedder -- tests never need real
    embeddings, just a distinguishable vector per catalog row."""

    def fit(self, texts):
        pass

    def encode(self, texts):
        return np.zeros((len(texts), 1))


def _make_recommender(catalog: pd.DataFrame) -> Recommender:
    embeddings = np.eye(len(catalog))
    return Recommender(catalog, _IdentityEmbedder(), _catalog_embeddings=embeddings)


# ---------------------------------------------------------------------------
# build_search_corpus / search_books
# ---------------------------------------------------------------------------


def test_exact_title_scores_near_100(corpus):
    matches = search_books(corpus, "Harry Potter and the Sorcerer's Stone")
    assert matches
    assert matches[0].book_id == "BX_1"
    assert matches[0].score >= 95.0


def test_one_typo_query_still_finds_correct_book(corpus):
    matches = search_books(corpus, "hary poter", score_cutoff=50.0)
    assert matches
    assert matches[0].book_id == "BX_1"


def test_nonsense_query_returns_empty(corpus):
    assert search_books(corpus, "zzz qqq xyz nonsense", score_cutoff=60.0) == []


def test_blank_query_returns_empty(corpus):
    assert search_books(corpus, "   ") == []


def test_limit_is_respected(corpus):
    matches = search_books(corpus, "book", limit=2, score_cutoff=0.0)
    assert len(matches) <= 2


def test_common_stopwords_do_not_cause_false_positive_matches(corpus):
    """Regression test: WRatio's token_set component treats any shared word
    (e.g. "and") as a full intersection match, which used to score a title
    that isn't in the catalog at all (~85/100) against "Pride and Prejudice"
    purely because both contain "and". Stopword-stripping must prevent this."""
    matches = search_books(corpus, "The Martian and Other Stories", score_cutoff=60.0)
    assert matches == []


# ---------------------------------------------------------------------------
# Recommender._resolve_liked_books
# ---------------------------------------------------------------------------


def _profile_with_liked(liked_books: list[LikedBook]) -> UserProfile:
    return UserProfile(layer1=Layer1(liked_books=liked_books))


def test_book_id_first_resolves_even_with_garbage_title(catalog):
    rec = _make_recommender(catalog)
    profile = _profile_with_liked([LikedBook(book_id="BX_2", title="not a real title at all")])
    embeddings, matched_ids, not_found = rec._resolve_liked_books(profile)
    assert matched_ids == {"BX_2"}
    assert not_found == []
    assert embeddings is not None
    assert embeddings.shape[0] == 1


def test_near_miss_title_resolves_via_fuzzy_fallback(catalog):
    rec = _make_recommender(catalog)
    profile = _profile_with_liked([LikedBook(title="hary poter and the sorcerers stone")])
    embeddings, matched_ids, not_found = rec._resolve_liked_books(profile)
    assert matched_ids == {"BX_1"}
    assert not_found == []


def test_genuinely_unmatched_title_lands_in_not_found(catalog):
    rec = _make_recommender(catalog)
    profile = _profile_with_liked([LikedBook(title="a completely unrelated made up book")])
    embeddings, matched_ids, not_found = rec._resolve_liked_books(profile)
    assert matched_ids == set()
    assert not_found == ["a completely unrelated made up book"]
    assert embeddings is None


def test_duplicate_book_via_book_id_and_raw_typo_counted_once(catalog):
    rec = _make_recommender(catalog)
    profile = _profile_with_liked(
        [
            LikedBook(book_id="BX_1", title="Harry Potter and the Sorcerer's Stone"),
            LikedBook(book_id=None, title="hary poter and the sorcerers stone"),
        ]
    )
    embeddings, matched_ids, not_found = rec._resolve_liked_books(profile)
    assert matched_ids == {"BX_1"}
    assert not_found == []
    assert embeddings.shape[0] == 1


# ---------------------------------------------------------------------------
# GET /books/search via FastAPI TestClient, against an injected fixture catalog
# ---------------------------------------------------------------------------


def test_books_search_endpoint(catalog, monkeypatch):
    import api as api_module
    from fastapi.testclient import TestClient

    rec = _make_recommender(catalog)
    rec.ensure_search_corpus()
    monkeypatch.setitem(api_module.state, "recommender", rec)

    client = TestClient(api_module.app)
    response = client.get("/books/search", params={"q": "hary poter", "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert body
    assert body[0]["book_id"] == "BX_1"
    assert set(body[0].keys()) == {"book_id", "title", "authors", "score"}


def test_books_search_endpoint_blank_query(catalog, monkeypatch):
    import api as api_module
    from fastapi.testclient import TestClient

    rec = _make_recommender(catalog)
    rec.ensure_search_corpus()
    monkeypatch.setitem(api_module.state, "recommender", rec)

    client = TestClient(api_module.app)
    response = client.get("/books/search", params={"q": ""})
    assert response.status_code == 200
    assert response.json() == []


def test_recommend_accepts_liked_book_score_field_without_rejecting(catalog, monkeypatch):
    """A frontend BookRef carries an extra `score` field the backend LikedBook
    model doesn't declare -- confirm Pydantic ignores it rather than 422ing,
    since /recommend calls verify_books() which needs a real OpenAI key and
    isn't otherwise exercised here."""
    import api as api_module
    from content_recommender2 import LikedBook

    payload = {"book_id": "BX_1", "title": "Harry Potter and the Sorcerer's Stone", "score": 97.5}
    liked_book = LikedBook.model_validate(payload)
    assert liked_book.book_id == "BX_1"
    assert not hasattr(liked_book, "score")
