from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BookRef:
    """A dataset-agnostic reference to a book. The real dataset (built separately)
    decides what book_id actually is (ISBN, internal ID, etc.) - this UI only
    needs enough to display and store the reference.

    book_id is None for a book the user typed themselves that wasn't matched
    (with confidence) to any catalog entry - kept as free text rather than
    dropped. score is the fuzzy-match confidence (0-100) for a suggested
    match, or None for a popular-books listing or a raw/typed entry."""

    book_id: str | None
    title: str
    authors: str = ""
    score: float | None = None


def liked_book_key(book: BookRef) -> str:
    """Dict key for draft["l1_liked_books"]: matched books key by book_id;
    raw/typed entries (book_id=None) key by normalized title, so retyping the
    same raw text dedupes but distinct raw entries can coexist."""
    if book.book_id:
        return book.book_id
    return f"raw::{book.title.strip().lower()}"


class BackendUnavailableError(Exception):
    """Raised by a BookProvider when its backend can't be reached or errors out."""


class BookProvider(Protocol):
    """Abstract seam between this UI and the (separately built) book dataset."""

    def search_books(self, query: str, limit: int = 25) -> list[BookRef]: ...

    def get_popular_books(self, limit: int = 25) -> list[BookRef]: ...

    def get_available_genres(self) -> list[str]: ...
