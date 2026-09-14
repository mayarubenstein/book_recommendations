"""Fuzzy free-text search over the book catalog (title + authors).

Local string matching via rapidfuzz, not semantic embeddings or an LLM call --
this needs to run synchronously on every /books/search request against the
full ~326k-book catalog, and rapidfuzz's C implementation keeps that cheap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz, process

# WRatio's token_set/partial_token_set components treat ANY shared token as a
# full intersection match -- so a query like "the martian" (not in the
# catalog at all) scores ~85/100 against "The Hobbit", "The Da Vinci Code",
# etc. purely because they all contain "the". Stripping common stopwords
# before scoring removes that false-positive inflation while leaving genuine
# typo/reorder/partial-author matches (which don't rely on stopwords) intact.
_STOPWORDS = frozenset({"the", "a", "an", "of", "and"})
_WORD_RE = re.compile(r"\S+")


def _strip_stopwords(text: str) -> str:
    return " ".join(w for w in _WORD_RE.findall(text) if w not in _STOPWORDS)


@dataclass(frozen=True)
class BookMatch:
    book_id: str
    title: str
    authors: str
    score: float


def build_search_corpus(catalog: pd.DataFrame) -> pd.DataFrame:
    """Derive a lightweight book_id/title/authors/_search_key table from an
    already-loaded catalog DataFrame -- no second load of all_books.json."""
    corpus = pd.DataFrame(
        {
            "book_id": catalog["Book ID"].fillna("").astype(str),
            "title": catalog["Title"].fillna("").astype(str),
            "authors": catalog["_authors_str"].fillna("").astype(str),
        }
    ).reset_index(drop=True)
    clean_titles = catalog["_clean_title_str"].fillna("").astype(str).reset_index(drop=True)
    combined = (clean_titles + " " + corpus["authors"].str.lower()).str.strip()
    corpus["_search_key"] = combined.map(_strip_stopwords)
    return corpus


def search_books(
    corpus: pd.DataFrame,
    query: str,
    limit: int = 10,
    score_cutoff: float = 68.0,
) -> list[BookMatch]:
    """Rank corpus rows by fuzzy similarity to query. Empty query -> []."""
    query = (query or "").strip()
    if not query:
        return []

    results = process.extract(
        _strip_stopwords(query.lower()),
        corpus["_search_key"].tolist(),
        scorer=fuzz.WRatio,
        limit=limit,
        score_cutoff=score_cutoff,
    )
    matches = []
    for _, score, position in results:
        row = corpus.iloc[position]
        matches.append(
            BookMatch(
                book_id=row["book_id"],
                title=row["title"],
                authors=row["authors"],
                score=float(score),
            )
        )
    return matches
