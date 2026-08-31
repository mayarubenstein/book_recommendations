from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BookRef:
    """A dataset-agnostic reference to a book. The real dataset (built separately)
    decides what book_id actually is (ISBN, internal ID, etc.) - this UI only
    needs enough to display and store the reference."""

    book_id: str
    title: str
    authors: str


class BookProvider(Protocol):
    """Abstract seam between this UI and the (separately built) book dataset."""

    def search_books(self, query: str, limit: int = 25) -> list[BookRef]: ...

    def get_popular_books(self, limit: int = 25) -> list[BookRef]: ...

    def get_available_genres(self) -> list[str]: ...
