"""Download private catalog artifacts for a Hugging Face Space when needed."""
from __future__ import annotations

import os
from pathlib import Path


def _runtime_catalog_needs_refresh(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        import pandas as pd

        payload = pd.read_pickle(path)
        catalog = payload.get("catalog")
        required = {
            "Book ID",
            "Title",
            "Authors",
            "Category/Genre",
            "Description",
            "Review_Count",
            "Average_Normalized_Rating",
            "content_text",
            "_clean_title_str",
            "_authors_str",
            "_demographic_age_buckets",
            "_demographic_location_sets",
            "_review_summaries",
            "_is_english",
        }
        return not required.issubset(catalog.columns)
    except Exception as exc:
        print(f"Could not validate {path}: {exc}; downloading a fresh copy")
        return True


def main() -> None:
    data_dir = Path(os.getenv("SPACE_DATA_DIR", "/data"))
    dataset_id = os.getenv("HF_DATASET_ID")
    if not dataset_id:
        print("HF_DATASET_ID is not set; using files already present in the Space.")
        return

    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "all_books.json",
        "book_titles.json",
        "catalog_embeddings.npz",
        "catalog_runtime.pkl",
    ):
        target = data_dir / filename
        is_runtime_catalog = filename == "catalog_runtime.pkl"
        needs_refresh = is_runtime_catalog and _runtime_catalog_needs_refresh(target)
        if target.exists() and not is_runtime_catalog:
            print(f"Using existing {target}")
            continue
        if needs_refresh:
            print(f"Refreshing stale {target}")
        downloaded = hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=filename,
            local_dir=data_dir,
            force_download=needs_refresh,
            token=os.getenv("HF_TOKEN"),
        )
        print(f"Downloaded {filename} to {downloaded}")


if __name__ == "__main__":
    main()