"""Download private catalog artifacts for a Hugging Face Space when needed."""
from __future__ import annotations

import os
from pathlib import Path


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
        if target.exists():
            print(f"Using existing {target}")
            continue
        downloaded = hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=filename,
            local_dir=data_dir,
            token=os.getenv("HF_TOKEN"),
        )
        print(f"Downloaded {filename} to {downloaded}")


if __name__ == "__main__":
    main()