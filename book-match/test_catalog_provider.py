"""Tests for CatalogBookProvider and the liked-books draft-keying helper."""
from __future__ import annotations

import requests

from src.data.book_provider import BackendUnavailableError, BookRef, liked_book_key
from src.data.catalog_provider import CatalogBookProvider


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def test_search_books_success(monkeypatch):
    provider = CatalogBookProvider(base_url="http://example.test")
    payload = [{"book_id": "BX_1", "title": "The Hobbit", "authors": "J.R.R. Tolkien", "score": 92.0}]

    def fake_get(url, params=None, timeout=None):
        assert url == "http://example.test/books/search"
        assert params == {"q": "tolkein", "limit": 10}
        return _FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)
    results = provider.search_books("tolkein", limit=10)
    assert results == [BookRef(book_id="BX_1", title="The Hobbit", authors="J.R.R. Tolkien", score=92.0)]


def test_search_books_blank_query_skips_request(monkeypatch):
    provider = CatalogBookProvider(base_url="http://example.test")

    def fail_get(*args, **kwargs):
        raise AssertionError("requests.get should not be called for a blank query")

    monkeypatch.setattr(requests, "get", fail_get)
    assert provider.search_books("   ") == []


def test_search_books_http_error_raises_backend_unavailable(monkeypatch):
    provider = CatalogBookProvider(base_url="http://example.test")

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(None, status_code=500)

    monkeypatch.setattr(requests, "get", fake_get)
    try:
        provider.search_books("dune")
        assert False, "expected BackendUnavailableError"
    except BackendUnavailableError:
        pass


def test_search_books_timeout_raises_backend_unavailable(monkeypatch):
    provider = CatalogBookProvider(base_url="http://example.test")

    def fake_get(url, params=None, timeout=None):
        raise requests.Timeout("too slow")

    monkeypatch.setattr(requests, "get", fake_get)
    try:
        provider.search_books("dune")
        assert False, "expected BackendUnavailableError"
    except BackendUnavailableError:
        pass


def test_get_popular_books_and_genres_delegate_to_fallback(monkeypatch):
    provider = CatalogBookProvider(base_url="http://example.test")
    assert provider.get_popular_books(limit=5)
    assert provider.get_available_genres()


# ---------------------------------------------------------------------------
# liked_book_key
# ---------------------------------------------------------------------------


def test_liked_book_key_uses_book_id_when_present():
    book = BookRef(book_id="BX_1", title="Dune", authors="Frank Herbert")
    assert liked_book_key(book) == "BX_1"


def test_liked_book_key_dedupes_identical_raw_entries():
    a = BookRef(book_id=None, title="  Hary Poter  ", authors="")
    b = BookRef(book_id=None, title="hary poter", authors="")
    assert liked_book_key(a) == liked_book_key(b)


def test_liked_book_key_keeps_distinct_raw_entries_separate():
    a = BookRef(book_id=None, title="Hary Poter", authors="")
    b = BookRef(book_id=None, title="The Hobbit but misspelled", authors="")
    assert liked_book_key(a) != liked_book_key(b)


def test_liked_book_key_matched_and_raw_of_same_title_differ():
    matched = BookRef(book_id="BX_1", title="Dune", authors="Frank Herbert")
    raw = BookRef(book_id=None, title="Dune", authors="")
    assert liked_book_key(matched) != liked_book_key(raw)
