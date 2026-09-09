"""
Cold-start book recommender.

Fits the data you actually have:
  - `all_books.json` (the output of creating_one_file.ipynb): ~326k books with
    Title / Clean_Title / Description / Authors / Category-Genre / Review_Count /
    Reviews_List / Average_Normalized_Rating.
  - a user preference *profile*, matching the exact JSON shape your quiz already
    produces (user_id/locale/schema_version + layer1/layer2/layer3). It's an
    onboarding profile, not a rating history -- the profile's user_id never
    appears in Reviews_List, so this is a pure cold-start problem for every
    single user, every time. Every field in it is optional; UserProfile below
    reflects that (defaults everywhere, nothing required to construct one).

Approach: content-based filtering, blending THREE signals per book:
  1. similarity to the user's stated preferences (genres/moods/sliders/free text)
  2. similarity to books the user says they already loved
  3. a Bayesian-adjusted popularity prior (so a book with 2 five-star reviews
     doesn't outrank one with 5,000 reviews averaging 4.3)
...then hard-filters on avoid_list / already-liked books.

Embeddings come from a real semantic model (sentence-transformers), not
TF-IDF -- TF-IDF only matches literal words, so "a really long doorstop of a
novel" and "an 800-page epic" would look unrelated to it despite meaning the
same thing. A pretrained sentence embedder places paraphrases near each other
in vector space, which is what fixes that.

Requires network access to Hugging Face Hub the first time you run this (to
download the model weights, a few hundred MB). After that it's cached
locally and runs offline. Encoding all ~326k books is the expensive one-time
step -- pass cache_path to Recommender.fit() so you only pay it once; see the
docstring there for how staleness is handled if the catalog changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import ijson
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

CATALOG_COLUMNS = [
    "Title",
    "Clean_Title",
    "Description",
    "Authors",
    "Category/Genre",
    "ISBN/ID",
    "Book ID",
    "Review_Count",
    "Reviews_List",
    "Average_Normalized_Rating",
]
EMBEDDING_TEXT_VERSION = "metadata-and-reviews-v2"

# ---------------------------------------------------------------------------
# 1. Loading the book catalog produced by creating_one_file.ipynb
# ---------------------------------------------------------------------------

def _iter_catalog_records(json_path: str | Path):
    with open(json_path, "rb") as catalog_file:
        yield from ijson.items(catalog_file, "item")


def _reviews_to_text(reviews: Any, max_reviews: int = 8, max_chars: int = 2400) -> str:
    """Keep a bounded, source-tolerant text sample from each book's reviews."""
    if not isinstance(reviews, list):
        return ""

    parts: list[str] = []
    for review in reviews[:max_reviews]:
        if not isinstance(review, dict):
            continue
        summary = str(review.get("summary") or "").strip()
        text = str(review.get("text") or "").strip()
        if summary:
            parts.append(f"review summary: {summary}")
        if text:
            parts.append(f"reader review: {text}")
        if sum(len(part) for part in parts) >= max_chars:
            break
    return " ".join(parts)[:max_chars]


def load_catalog(json_path: str | Path) -> pd.DataFrame:
    """Load all_books.json into a DataFrame and add a single text field per
    book that we'll embed. Missing text fields degrade gracefully."""
    records = []
    required_fields = set(CATALOG_COLUMNS) - {"Reviews_List"}
    for record in _iter_catalog_records(json_path):
        if not isinstance(record, dict):
            raise ValueError(f"Catalog at {json_path} must contain book objects.")
        row = {field: record.get(field) for field in required_fields}
        row["_reviews_text"] = _reviews_to_text(record.get("Reviews_List"))
        records.append(row)
    if not records:
        raise ValueError(f"Catalog at {json_path} must contain at least one book.")
    df = pd.DataFrame.from_records(records)

    missing = [
        column for column in CATALOG_COLUMNS
        if column != "Reviews_List" and column not in df.columns
    ]
    if missing:
        raise ValueError(f"Catalog is missing required fields: {', '.join(missing)}")

    for col in ["Title", "Clean_Title", "Description", "Authors", "Category/Genre"]:
        if col not in df.columns:
            df[col] = None

    def _authors_to_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str) and value.startswith("["):
            # Amazon authors arrive as a stringified list, e.g. "['Philip Nel']"
            return " ".join(re.findall(r"'([^']+)'", value))
        return str(value)

    df["_authors_str"] = df["Authors"].apply(_authors_to_str)
    df["_genre_str"] = df["Category/Genre"].fillna("").astype(str).str.replace(",", " ")
    df["_desc_str"] = df["Description"].fillna("").astype(str)
    df["_title_str"] = df["Title"].fillna("").astype(str)
    df["_clean_title_str"] = (
        df["Clean_Title"].fillna(df["_title_str"]).astype(str).str.strip().str.lower()
    )

    # Genre/author repeated so they weigh more than one-off description words
    # once embedded.
    df["content_text"] = (
        (df["_title_str"] + " ") * 2
        + (df["_genre_str"] + " ") * 3
        + (df["_authors_str"] + " ") * 2
        + df["_desc_str"]
            + " reviews "
            + df["_reviews_text"].fillna("").astype(str)
    ).str.lower()

    df["Review_Count"] = pd.to_numeric(df["Review_Count"], errors="coerce").fillna(0)
    df["Average_Normalized_Rating"] = pd.to_numeric(
        df["Average_Normalized_Rating"], errors="coerce"
    )
    return df


def catalog_fingerprint(json_path: str | Path) -> str:
    """Return a stable fingerprint for the exact local catalog file."""
    digest = hashlib.sha256()
    with open(json_path, "rb") as catalog_file:
        for chunk in iter(lambda: catalog_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 2. The preference-profile schema, and turning one into a query string
# ---------------------------------------------------------------------------
# These mirror your quiz's actual JSON shape (layer1/layer2/layer3) field for
# field, so a request body posted straight from the front end validates
# against this directly -- no separate "parse the JSON" step needed. Every
# field defaults to empty/None: a profile with nothing in it at all is valid
# and recommend() degrades gracefully down to a pure popularity ranking (see
# Recommender.recommend below).


class LikedBook(BaseModel):
    book_id: str | None = None
    title: str | None = None
    authors: str | None = None


class Layer1(BaseModel):
    """Quiz basics: books/genres/moods the user explicitly picked."""
    liked_books: list[LikedBook] = Field(default_factory=list)
    selected_genres: list[str] = Field(default_factory=list)
    selected_moods: list[str] = Field(default_factory=list)


class Layer2(BaseModel):
    """Slider preferences (0-100) plus hard constraints."""
    plot_vs_character: float | None = None
    pace: float | None = None
    complexity: float | None = None
    length_preference: str | None = None
    tone: float | None = None
    avoid_list: list[str] = Field(default_factory=list)
    avoid_other: str | None = None


class Layer3(BaseModel):
    """Free-form taste description in the user's own words."""
    free_text: str = ""


class PersonalInfo(BaseModel):
    age: int | None = None
    country: str | None = None
    city: str | None = None


def _describe_slider(value: float | None, low: str, mid: str, high: str) -> str | None:
    """Turn a 0-100 slider into a graded phrase instead of one hard cutoff.
    A semantic embedder can tell "somewhat slow-paced" apart from "very
    slow-paced". This still hand-picks 5 buckets,
    but the buckets are just there to turn a number into words the model can
    read -- the actual similarity judgment (is "very slow-paced" close to
    "leisurely, unhurried"?) is the embedder's job now, not a keyword table's.
    """
    if value is None:
        return None
    if value <= 15:
        return f"very {low}"
    if value <= 40:
        return f"somewhat {low}"
    if value <= 60:
        return mid
    if value <= 85:
        return f"somewhat {high}"
    return f"very {high}"


class UserProfile(BaseModel):
    user_id: str | None = None
    locale: str | None = None
    schema_version: str | None = None
    created_at: str | None = None
    layer1: Layer1 = Field(default_factory=Layer1)
    layer2: Layer2 = Field(default_factory=Layer2)
    layer3: Layer3 = Field(default_factory=Layer3)
    personal_info: PersonalInfo | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "UserProfile":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_query_text(self) -> str:
        """Text for the 'stated preferences' signal. Deliberately excludes
        liked_books -- those get their own similarity signal in Recommender
        (see _resolve_liked_books) so they're not double-counted.

        No more hand-written mood-keyword dictionary: a mood tag like
        "quiet_introspective" is just turned into the words "quiet
        introspective" and handed to the embedder as-is. A pretrained
        sentence embedder already knows what that phrase is semantically
        close to; a synonym table hand-authored by me was never going to be
        as good a source of truth as the model's own training data.
        """
        l1, l2, l3 = self.layer1, self.layer2, self.layer3
        parts: list[str | None] = []
        parts += l1.selected_genres * 3  # explicit genre picks matter most
        parts += [m.replace("_", " ") for m in l1.selected_moods]
        parts.append(_describe_slider(
            l2.plot_vs_character, "character-driven", "balanced between plot and character", "plot-driven"
        ))
        parts.append(_describe_slider(l2.pace, "slow-paced", "moderately paced", "fast-paced"))
        parts.append(_describe_slider(
            l2.complexity, "simple and straightforward", "moderately complex", "intricate and complex"
        ))
        parts.append(_describe_slider(l2.tone, "dark and serious", "a mix of light and dark", "light and hopeful"))
        if l2.length_preference:
            parts.append(f"{l2.length_preference}-length book")
        if l2.avoid_other:
            parts.append(f"avoid {l2.avoid_other}")
        if l3.free_text:
            parts.append(l3.free_text)
        if self.personal_info:
            if self.personal_info.age is not None:
                parts.append(f"reader age {self.personal_info.age}")
            if self.personal_info.country:
                parts.append(f"reader country {self.personal_info.country}")
            if self.personal_info.city:
                parts.append(f"reader city {self.personal_info.city}")
        return " ".join(p for p in parts if p).lower()


AVOID_KEYWORDS = {
    "graphic_violence": ["graphic violence", "gore", "torture"],
    "child_harm": ["child abuse", "child harm", "abuse of children"],
    "sexual_content": ["explicit sexual content", "erotic"],
}


# ---------------------------------------------------------------------------
# 3. Embedding backend
# ---------------------------------------------------------------------------

class Embedder(ABC):
    """Common interface between the catalog side and the query side of
    Recommender, so the scoring logic doesn't care what produced the
    vectors."""

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> None:
        """Learn anything backend-specific from the catalog text. No-op for
        a pretrained model."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a dense array with one row per text."""


class SentenceTransformerEmbedder(Embedder):
    """A pretrained semantic embedding model. Needs
    `pip install sentence-transformers` and, the first time it runs, network
    access to download model weights from Hugging Face Hub (a few hundred MB,
    cached locally afterwards -- fully offline after that first call).

    'all-MiniLM-L6-v2' (384-dim) is the usual default: fast on CPU, good
    quality for this kind of similarity search. 'all-mpnet-base-v2' (768-dim)
    is noticeably better and noticeably slower -- worth trying once you've
    validated the pipeline end to end with the smaller model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_chars: int = 800,
                 batch_size: int = 128):
        from sentence_transformers import SentenceTransformer  # deferred import

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, token=os.getenv("HF_TOKEN"))
        self.max_chars = max_chars
        self.batch_size = batch_size

    def fit(self, texts: Sequence[str]) -> None:
        pass  # pretrained -- nothing to fit

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        truncated = [t[: self.max_chars] for t in texts]
        return self.model.encode(
            truncated,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # so cosine similarity == dot product
            convert_to_numpy=True,
            show_progress_bar=len(truncated) > 1000,
        )


# ---------------------------------------------------------------------------
# 4. Scoring: stated-preference similarity + liked-books similarity +
#    popularity prior, then hard filters
# ---------------------------------------------------------------------------

@dataclass
class Recommender:
    catalog: pd.DataFrame
    embedder: Embedder
    _catalog_embeddings: Any = field(default=None, repr=False)

    def fit(self, cache_path: str | Path | None = None) -> "Recommender":
        """Embed every book once. This is the expensive step -- encoding
        ~326k books through a neural network, not a lookup -- so pass
        cache_path to save the result to a .npy file and skip re-encoding on
        every restart.

        There's no automatic way to tell "the cache is still valid" from the
        file alone, so this does the one cheap check it can: if the cached
        row count doesn't match the current catalog's row count, the catalog
        clearly changed since the cache was built, and it re-embeds instead
        of silently serving recommendations built from a stale/mismatched
        catalog. That check does NOT catch every kind of staleness (e.g. you
        edited descriptions but kept the same row count) -- delete the cache
        file yourself whenever you regenerate all_books.json to be safe.
        """
        texts = self.catalog["content_text"]

        if cache_path is not None and Path(cache_path).exists():
            cached = np.load(cache_path)
            if len(cached) == len(self.catalog):
                self._catalog_embeddings = cached
                return self
            print(
                f"[fit] cache at {cache_path} has {len(cached)} rows but the catalog "
                f"has {len(self.catalog)} -- re-embedding instead of using a stale cache."
            )

        self.embedder.fit(texts)
        self._catalog_embeddings = self.embedder.encode(list(texts))

        if cache_path is not None and isinstance(self._catalog_embeddings, np.ndarray):
            np.save(cache_path, self._catalog_embeddings)
        return self

    def save_embedding_artifact(
        self,
        artifact_path: str | Path,
        catalog_path: str | Path,
    ) -> None:
        """Persist catalog vectors and the metadata needed to validate them."""
        if not isinstance(self._catalog_embeddings, np.ndarray):
            raise RuntimeError("Call fit() before saving an embedding artifact.")

        book_ids = self.catalog["Book ID"].fillna("").astype(str).to_numpy()
        metadata = {
            "catalog_fingerprint": catalog_fingerprint(catalog_path),
            "embedding_model": getattr(self.embedder, "model_name", None),
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "row_count": len(self.catalog),
        }
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            embeddings=self._catalog_embeddings,
            book_ids=book_ids,
            metadata=json.dumps(metadata),
        )

    def load_embedding_artifact(
        self,
        artifact_path: str | Path,
        catalog_path: str | Path,
    ) -> "Recommender":
        """Load vectors created by the offline indexing step after validation."""
        artifact = Path(artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(
                f"Missing catalog embedding artifact: {artifact}. "
                "Run `python build_catalog_embeddings.py` first."
            )

        with np.load(artifact, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            embeddings = saved["embeddings"]
            saved_book_ids = saved["book_ids"].astype(str)

        expected_model = getattr(self.embedder, "model_name", None)
        if metadata.get("catalog_fingerprint") != catalog_fingerprint(catalog_path):
            raise ValueError(
                "Catalog embedding artifact is stale because all_books.json changed. "
                "Run `python build_catalog_embeddings.py` again."
            )
        if metadata.get("embedding_model") != expected_model:
            raise ValueError(
                f"Artifact model {metadata.get('embedding_model')!r} does not match "
                f"the configured model {expected_model!r}. Rebuild the artifact."
            )
        if metadata.get("embedding_text_version") != EMBEDDING_TEXT_VERSION:
            raise ValueError("Embedding text format changed. Rebuild the artifact.")

        current_book_ids = self.catalog["Book ID"].fillna("").astype(str).to_numpy()
        if len(embeddings) != len(self.catalog) or not np.array_equal(saved_book_ids, current_book_ids):
            raise ValueError("Catalog embedding artifact does not match the current catalog order.")

        self._catalog_embeddings = embeddings
        return self

    def _bayesian_rating(self, min_reviews: int = 20) -> pd.Series:
        """IMDB-style shrinkage so a book with 2 five-star reviews doesn't
        outrank one with 5,000 reviews averaging 4.3."""
        r = self.catalog["Average_Normalized_Rating"]
        v = self.catalog["Review_Count"]
        c = r.mean(skipna=True)
        m = min_reviews
        return (v / (v + m)) * r.fillna(c) + (m / (v + m)) * c

    def _resolve_liked_books(self, profile: UserProfile) -> tuple[Any, list[str]]:
        """Match liked_books against the catalog by cleaned title so we can
        reuse each liked book's *own* embedding (built from its real
        description/genre) rather than just its bare title string. Entries
        with no title (allowed -- everything in a liked book entry is
        optional) can't be matched to anything and are skipped. Returns
        (embeddings-for-matched-books-or-None, titles-not-found)."""
        wanted = {b.title.strip().lower() for b in profile.layer1.liked_books if b.title}
        if not wanted:
            return None, []

        mask = self.catalog["_clean_title_str"].isin(wanted)
        matched_positions = np.flatnonzero(mask.to_numpy())
        found_titles = set(self.catalog.loc[mask, "_clean_title_str"])
        not_found = sorted(wanted - found_titles)

        if len(matched_positions) == 0:
            return None, sorted(wanted)

        return self._catalog_embeddings[matched_positions], not_found

    def recommend(
        self,
        profile: UserProfile,
        top_n: int = 20,
        weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
        liked_agg: str = "mean",
    ) -> pd.DataFrame:
        """weights = (stated_preferences, liked_books, popularity). If the
        profile has no liked books we can match -- including a profile with
        nothing filled in at all -- liked_books weight is redistributed to
        stated_preferences; if stated preferences are ALSO empty, sim_pref is
        uniformly ~0 too, and score collapses to just the popularity prior.
        That's intentional: a maximally empty profile falls back to "show the
        most reliably good books," which is a reasonable cold-start default.

        liked_agg: "mean" blends the vibe of everything they liked into one
        target; "max" instead surfaces books close to any ONE loved book --
        better when someone's liked list spans very different genres/moods.
        """
        w_pref, w_liked, w_pop = weights

        pref_vec = self.embedder.encode([profile.to_query_text()])
        sim_pref = cosine_similarity(pref_vec, self._catalog_embeddings).ravel()

        liked_embeddings, not_found = self._resolve_liked_books(profile)
        if not_found:
            print(f"[recommend] liked books not found in catalog, skipped: {not_found}")

        if liked_embeddings is not None:
            sim_liked_matrix = cosine_similarity(liked_embeddings, self._catalog_embeddings)
            sim_liked = sim_liked_matrix.max(axis=0) if liked_agg == "max" \
                else sim_liked_matrix.mean(axis=0)
        else:
            sim_liked = np.zeros(len(self.catalog))
            w_pref, w_liked = w_pref + w_liked, 0.0  # nothing to blend in, reallocate

        pop = self._bayesian_rating()
        pop_norm = ((pop - pop.min()) / (pop.max() - pop.min() + 1e-9)).to_numpy()

        score = w_pref * sim_pref + w_liked * sim_liked + w_pop * pop_norm

        result = self.catalog.copy()
        result["sim_preferences"] = sim_pref
        result["sim_liked_books"] = sim_liked
        result["score"] = score

        # Hard filters -- deterministic constraints, not signals to blend in,
        # so they're applied after scoring rather than as features.
        mask = pd.Series(True, index=result.index)

        liked_lower = {b.title.strip().lower() for b in profile.layer1.liked_books if b.title}
        if liked_lower:
            mask &= ~result["_clean_title_str"].isin(liked_lower)

        for tag in profile.layer2.avoid_list:
            for kw in AVOID_KEYWORDS.get(tag, [tag.replace("_", " ")]):
                mask &= ~result["content_text"].str.contains(re.escape(kw), na=False)

        result = result[mask]
        return result.sort_values("score", ascending=False).head(top_n)[
            ["Book ID", "Title", "Authors", "Category/Genre", "Review_Count",
             "Average_Normalized_Rating", "sim_preferences", "sim_liked_books", "score"]
        ]


# ---------------------------------------------------------------------------
# 5. Example wiring (see api.py for a FastAPI-served version of this)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    catalog = load_catalog("all_books.json")
    embedder = SentenceTransformerEmbedder()
    recommender = Recommender(catalog, embedder).fit(cache_path="catalog_embeddings.npy")

    profile = UserProfile.from_json("sample_preference_profile1.json")
    recs = recommender.recommend(profile, top_n=15)
    print(recs.to_string(index=False))
