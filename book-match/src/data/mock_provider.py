from __future__ import annotations

from src.data.book_provider import BookRef

_MOCK_BOOKS: list[BookRef] = [
    BookRef("mock-001", "Harry Potter and the Sorcerer's Stone", "J.K. Rowling"),
    BookRef("mock-002", "The Hobbit", "J.R.R. Tolkien"),
    BookRef("mock-003", "Dune", "Frank Herbert"),
    BookRef("mock-004", "The Hunger Games", "Suzanne Collins"),
    BookRef("mock-005", "Pride and Prejudice", "Jane Austen"),
    BookRef("mock-006", "1984", "George Orwell"),
    BookRef("mock-007", "The Da Vinci Code", "Dan Brown"),
    BookRef("mock-008", "Gone Girl", "Gillian Flynn"),
    BookRef("mock-009", "The Fault in Our Stars", "John Green"),
    BookRef("mock-010", "A Game of Thrones", "George R.R. Martin"),
    BookRef("mock-011", "The Silent Patient", "Alex Michaelides"),
    BookRef("mock-012", "Educated", "Tara Westover"),
    BookRef("mock-013", "The Night Circus", "Erin Morgenstern"),
    BookRef("mock-014", "Where the Crawdads Sing", "Delia Owens"),
    BookRef("mock-015", "The Martian", "Andy Weir"),
    BookRef("mock-016", "Circe", "Madeline Miller"),
    BookRef("mock-017", "It", "Stephen King"),
    BookRef("mock-018", "The Alchemist", "Paulo Coelho"),
    BookRef("mock-019", "Project Hail Mary", "Andy Weir"),
    BookRef("mock-020", "The Seven Husbands of Evelyn Hugo", "Taylor Jenkins Reid"),
]

_MOCK_GENRES: list[str] = [
    "Fantasy",
    "Science Fiction",
    "Mystery",
    "Thriller",
    "Romance",
    "Historical Fiction",
    "Horror",
    "Young Adult",
    "Classics",
    "Non-Fiction",
]


class MockBookProvider:
    """Hardcoded stand-in for the real BookProvider, used for local dev/testing
    of this UI before the separately-built dataset part is ready. Satisfies the
    BookProvider protocol structurally (no explicit inheritance required)."""

    def search_books(self, query: str, limit: int = 25) -> list[BookRef]:
        q = query.strip().lower()
        if not q:
            return self.get_popular_books(limit)
        matches = [b for b in _MOCK_BOOKS if q in b.title.lower() or q in b.authors.lower()]
        return matches[:limit]

    def get_popular_books(self, limit: int = 25) -> list[BookRef]:
        return _MOCK_BOOKS[:limit]

    def get_available_genres(self) -> list[str]:
        return list(_MOCK_GENRES)
