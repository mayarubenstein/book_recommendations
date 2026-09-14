from __future__ import annotations

import requests

from src.data.book_provider import BackendUnavailableError, BookProvider, BookRef
from src.data.mock_provider import MockBookProvider

__all__ = ["BackendUnavailableError", "CatalogBookProvider"]


class CatalogBookProvider:
    """Real BookProvider backed by the FastAPI /books/search endpoint. The
    real catalog (~326k books) only lives in that backend process's memory,
    so search is a live HTTP call, not a local lookup.

    get_popular_books/get_available_genres are out of scope for this change
    (no such endpoints exist yet) - they delegate to MockBookProvider so
    Section 2's genre pills keep working, rather than going silently blank."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 3.0,
        fallback: BookProvider | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._fallback = fallback or MockBookProvider()

    def search_books(self, query: str, limit: int = 25) -> list[BookRef]:
        query = query.strip()
        if not query:
            return []
        try:
            response = requests.get(
                f"{self._base_url}/books/search",
                params={"q": query, "limit": limit},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise BackendUnavailableError(str(exc)) from exc
        return [
            BookRef(
                book_id=item["book_id"],
                title=item["title"],
                authors=item["authors"],
                score=item["score"],
            )
            for item in response.json()
        ]

    def get_popular_books(self, limit: int = 25) -> list[BookRef]:
        return self._fallback.get_popular_books(limit)

    def get_available_genres(self) -> list[str]:
        return self._fallback.get_available_genres()
