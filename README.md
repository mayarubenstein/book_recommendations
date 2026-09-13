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

Artifacts created with the earlier `metadata-and-reviews-v2` text format remain
loadable. The API prints a warning and uses its separate reviewer-demographic
similarity signal; rerun indexing later if you want reviewer demographics folded
into the main content embeddings as well.

Run the runtime preprocessing command once to create `catalog_runtime.pkl`. This is a compact
runtime catalog containing the fields needed for scoring and filtering, including
precomputed reviewer-demographic fields. FastAPI loads this cache at startup and
does not parse the multi-gigabyte `all_books.json` or rebuild demographic data on
every restart. Regenerate it whenever `all_books.json` changes:

```powershell
python build_runtime_catalog.py
```

Do not run `content_recommender2.py` to start the application. It is the core
library module and does not run an embedding job when imported. Use `api.py`
through Uvicorn for the backend, or use `main.py` to send a profile to an
already-running backend from the command line:

```powershell
python main.py --profile sample_preference_profile1.json --count 5
```

`all-mpnet-base-v2` (768 dimensions) may be evaluated later. Switching models
requires changing the configured model and regenerating the artifact.

## Recommendation verification

The API first generates `2 * n` candidates for a request of `n` books. It sends
those candidates, including catalog metadata and recommender scores, to the
OpenAI verifier. The frontend receives only approved books, capped at `n`.
If verification fails, the API returns an error instead of showing unverified
recommendations.

Each approved book also includes up to five short review excerpts and any
available reviewer age/location details. These are precomputed into
`catalog_runtime.pkl`; raw review lists are not sent to the frontend.

The runtime catalog also marks obvious non-English records and the recommender
filters them before ranking. The verifier applies a second English-language
check. Regenerate the runtime catalog after this language-filter change.

Source quality is also included as a small ranking prior: Amazon (`AMZ`) and
Book-Crossing (`BX`) are treated equally because they provide stronger review
or reviewer evidence, while Goodreads (`GR`) is a fallback source.

Create a local `.env` file in this directory and add the key without quotes:

```text
OPENAI_API_KEY=your-openai-api-key
```

Never commit `.env` or expose the key to the frontend. The key is used only by
the FastAPI backend.

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
