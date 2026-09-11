# Book Recommendation System

Local FastAPI backend for the book recommendation system. The Streamlit frontend
lives in the sibling `book_recommendations_frontend/book-match` worktree.

## Local setup

From this directory, create or activate a Python environment and install the
backend dependencies:

```powershell
python -m pip install -r requirements.txt
```

The catalog reader processes the JSON as a stream and keeps review payloads out
of the working table, because the local file can be several gigabytes while
the recommender only needs book metadata, IDs, and aggregate ratings.

Place the locally generated `all_books.json` in this directory. It is intentionally
ignored by Git because the catalog is large. The exported catalog is a flat JSON
array whose records contain these fields:

```text
Title, Clean_Title, Description, Authors, Category/Genre,
ISBN/ID, Book ID, Review_Count, Reviews_List,
Average_Normalized_Rating
```

The sources do not provide identical data. Some records have descriptions, some
have reviewer information, and review text is only available for some sources.
The embedding uses title, category/genre, authors, description, and a bounded
sample of each book's review summaries and review text. `Review_Count` and
`Average_Normalized_Rating` remain numeric ranking signals. Reviewer IDs and
demographics are not embedded as book content.

## Build catalog embeddings once

The initial model is Hugging Face's `all-MiniLM-L6-v2` (384 dimensions). Run the
offline indexing command once after placing or regenerating `all_books.json`:

```powershell
python build_catalog_embeddings.py
```

This creates the ignored `catalog_embeddings.npz` artifact. It contains one
normalized vector per catalog book, the book IDs, model metadata, text-format
version, and a fingerprint of the exact catalog file. FastAPI loads this artifact
at startup and does not re-embed books for each request. If the catalog or model
changes, rerun the indexing command.

Do not run `content_recommender2.py` to start the application. It is the core
library module and does not run an embedding job when imported. Use `api.py`
through Uvicorn for the backend, or use `main.py` to send a profile to an
already-running backend from the command line:

```powershell
python main.py --profile sample_preference_profile1.json --count 5
```

`all-mpnet-base-v2` (768 dimensions) may be evaluated later. Switching models
requires changing the configured model and regenerating the artifact.

## Run the backend

Start the API from any working directory after the catalog artifact exists:

```powershell
uvicorn api:app --reload --app-dir "C:\path\to\book_recommendations_repo"
```

The first model initialization may download model weights from Hugging Face. User
profiles are embedded on demand when `POST /recommend` is called.

Endpoints:

- `GET /health`
- `POST /recommend?top_n=10`

The request body is the JSON profile exported by the frontend. `top_n` must be
between 1 and 100.

## Run the frontend

In the sibling `book_recommendations_frontend/book-match` directory:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The frontend sends profiles to `http://127.0.0.1:8000` by default. Set
`BOOK_RECOMMENDER_API_URL` to use another local API URL:

```powershell
$env:BOOK_RECOMMENDER_API_URL = "http://127.0.0.1:8000"
```
