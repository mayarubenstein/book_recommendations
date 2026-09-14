"""Extract unique book titles and authors from all_books.json."""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

import ijson


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "all_books.json"
DEFAULT_TITLES_PATH = Path(__file__).resolve().parent / "book_titles.json"
DEFAULT_AUTHORS_PATH = Path(__file__).resolve().parent / "book_authors.json"


def _iter_catalog_records(catalog_path: Path) -> Iterable[dict[str, Any]]:
    with catalog_path.open("rb") as catalog_file:
        yield from ijson.items(catalog_file, "item")


def _add_unique(value: Any, values: list[str], seen: set[str]) -> None:
    if not isinstance(value, str):
        return
    cleaned = value.strip()
    key = cleaned.casefold()
    if cleaned and key not in seen:
        seen.add(key)
        values.append(cleaned)


def _author_parts(value: Any) -> Iterable[str]:
    if isinstance(value, list):
        for author in value:
            if isinstance(author, str):
                yield author
        return

    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = ast.literal_eval(cleaned)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                for author in parsed:
                    if isinstance(author, str):
                        yield author
                return
        yield from re.split(r"\s*&\s*|\s+and\s+", cleaned, flags=re.IGNORECASE)


def extract_catalog_lists(catalog_path: Path) -> tuple[list[str], list[str]]:
    titles: list[str] = []
    authors: list[str] = []
    seen_titles: set[str] = set()
    seen_authors: set[str] = set()

    for record in _iter_catalog_records(catalog_path):
        _add_unique(record.get("Title"), titles, seen_titles)
        for author in _author_parts(record.get("Authors")):
            _add_unique(author, authors, seen_authors)

    return titles, authors


def _write_json(path: Path, values: list[str]) -> None:
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--titles-output", type=Path, default=DEFAULT_TITLES_PATH)
    parser.add_argument("--authors-output", type=Path, default=DEFAULT_AUTHORS_PATH)
    args = parser.parse_args()

    titles, authors = extract_catalog_lists(args.catalog)
    _write_json(args.titles_output, titles)
    _write_json(args.authors_output, authors)
    print(f"Wrote {len(titles)} unique titles to {args.titles_output}")
    print(f"Wrote {len(authors)} unique authors to {args.authors_output}")


if __name__ == "__main__":
    main()