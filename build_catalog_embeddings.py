"""Build the local, persisted catalog embedding artifact."""
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "all_books.json"
ARTIFACT_PATH = BASE_DIR / "catalog_embeddings.npz"
load_dotenv(BASE_DIR / ".env")

from content_recommender2 import Recommender, SentenceTransformerEmbedder, load_catalog


def main() -> None:
    catalog = load_catalog(CATALOG_PATH)
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    recommender = Recommender(catalog, embedder).fit()
    recommender.save_embedding_artifact(ARTIFACT_PATH, CATALOG_PATH)
    print(f"Saved {len(catalog)} catalog embeddings to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()