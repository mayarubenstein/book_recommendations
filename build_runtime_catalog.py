"""Preprocess the catalog fields needed by the API at runtime."""
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "all_books.json"
RUNTIME_CATALOG_PATH = BASE_DIR / "catalog_runtime.pkl"
load_dotenv(BASE_DIR / ".env")

from content_recommender2 import load_catalog, save_runtime_catalog


def main() -> None:
    catalog = load_catalog(CATALOG_PATH)
    save_runtime_catalog(catalog, RUNTIME_CATALOG_PATH, CATALOG_PATH)
    print(f"Saved runtime catalog with {len(catalog)} books to {RUNTIME_CATALOG_PATH}")


if __name__ == "__main__":
    main()
