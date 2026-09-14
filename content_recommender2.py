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

Approach: content-based filtering, blending FOUR signals per book:
  1. similarity to the user's stated preferences (genres/moods/sliders/free text)
  2. similarity to books the user says they already loved
    3. similarity between the user's demographics and reviewer demographics
    4. a Bayesian-adjusted popularity prior (so a book with 2 five-star reviews
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import ijson
from pydantic import BaseModel, Field, model_validator
from sklearn.metrics.pairwise import cosine_similarity

from search import build_search_corpus, search_books

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
# Bumped from "metadata-and-reviews-v2": content_text now also folds in
# reviewer age/location text. Legacy artifacts remain usable because reviewer
# demographics are also scored separately by _demographic_similarity().
EMBEDDING_TEXT_VERSION = "metadata-reviews-demographics-v3"
LEGACY_EMBEDDING_TEXT_VERSIONS = {"metadata-and-reviews-v2"}
RUNTIME_CATALOG_COLUMNS = [
    "Book ID", "Title", "Authors", "Category/Genre", "Description",
    "Review_Count", "Average_Normalized_Rating", "content_text",
    "_clean_title_str", "_authors_str", "_demographic_age_buckets",
    "_demographic_location_sets", "_review_summaries",
    "_is_english",
]

NON_ENGLISH_MARKERS = {
    " polish": {" oraz ", " dla ", " jest ", " nie ", " książka ", " książki "},
    "spanish": {" el ", " la ", " los ", " las ", " una ", " que ", " del "},
    "french": {" le ", " les ", " des ", " une ", " est ", " dans "},
    "german": {" der ", " die ", " das ", " und ", " ein ", " eine "},
    "italian": {" il ", " gli ", " una ", " che ", " della "},
}


# Input: a review record (dict or JSON string). Output: it as a dict, or None if unusable.
def _as_review_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


# Input: a raw age value. Output: it as a float, if it's a plausible human age (0-120), else None.
def _valid_age(age: Any) -> float | None:
    try:
        numeric_age = float(age)
    except (TypeError, ValueError):
        return None
    if not 0 <= numeric_age <= 120:
        return None
    return numeric_age


# Input: a raw age value. Output: its decade bucket (e.g. "age 30-39"), or None if invalid.
def _age_bucket(age: Any) -> str | None:
    numeric_age = _valid_age(age)
    if numeric_age is None:
        return None
    lower = int(numeric_age // 10 * 10)
    return f"age {lower}-{lower + 9}"


# Input: a decade-bucket label like "age 30-39". Output: its numeric midpoint (34.5), or None if unparseable.
def _bucket_midpoint(bucket: str) -> float | None:
    match = re.fullmatch(r"age (\d+)-(\d+)", bucket)
    if not match:
        return None
    low, high = int(match.group(1)), int(match.group(2))
    return (low + high) / 2


# Input: a location string. Output: a set of normalized lowercase place tokens.
def _location_keys(location: Any) -> set[str]:
    if not isinstance(location, str) or not location.strip():
        return set()
    parts = set()
    for component in location.split(","):
        value = re.sub(r"[^a-zA-Z ]", " ", component.lower())
        value = re.sub(r"\s+", " ", value).strip()
        if value:
            parts.add(value)
    return parts


def _is_probably_english(title: Any, description: Any) -> bool:
    """Reject obvious non-English records when the source has no language field."""
    text = f" {title or ''} {description or ''} ".casefold()
    for character in text:
        codepoint = ord(character)
        if (
            0x0590 <= codepoint <= 0x08FF  # Hebrew/Arabic
            or 0x0370 <= codepoint <= 0x052F  # Greek/Cyrillic
            or 0x1100 <= codepoint <= 0x11FF  # Hangul
            or 0x3040 <= codepoint <= 0x30FF  # Japanese
            or 0x3400 <= codepoint <= 0x9FFF  # Chinese
        ):
            return False
    normalized = re.sub(r"[^\w]+", " ", text)
    marker_hits = [
        marker
        for markers in NON_ENGLISH_MARKERS.values()
        for marker in markers
        if marker in normalized
    ]
    distinctive_markers = {" książ", " del ", " dans ", " und ", " della "}
    if len(marker_hits) >= 2 or any(marker in normalized for marker in distinctive_markers):
        return False
    polish_letters = set("ąćęłńóśźż")
    return not any(character in polish_letters for character in text)


def _recommendation_key(title: Any, authors: Any) -> str:
    """Normalize edition/series variants so one work is shown once."""
    normalized_title = str(title or "").casefold().strip()
    normalized_title = re.sub(r"\s*\([^)]*(?:#|series|book|volume|vol\.)[^)]*\)\s*$", "", normalized_title)
    normalized_title = re.sub(r"[^a-z0-9]+", " ", normalized_title).strip()
    normalized_authors = re.sub(r"[^a-z0-9]+", " ", str(authors or "").casefold()).strip()
    return f"{normalized_title}|{normalized_authors}" if normalized_authors else normalized_title


def _source_quality_prior(book_id: Any) -> float:
    """Prefer Amazon and Book-Crossing equally over Goodreads."""
    source = str(book_id or "").upper().split("_", 1)[0]
    return {"AMZ": 1.0, "BX": 1.0, "GR": 0.35}.get(source, 0.5)


# Input: a book's review list. Output: (age_buckets, locations) sets aggregated across all reviewers.
def _reviewer_demographics(reviews: Any) -> tuple[set[str], set[str]]:
    age_buckets: set[str] = set()
    locations: set[str] = set()
    if not isinstance(reviews, list):
        return age_buckets, locations
    for review in reviews:
        review_mapping = _as_review_mapping(review)
        demographics = review_mapping.get("user_demographics") if review_mapping else None
        if not isinstance(demographics, Mapping):
            continue
        age = _age_bucket(demographics.get("age"))
        location = _location_keys(demographics.get("location"))
        if age:
            age_buckets.add(age)
        locations.update(location)
    return age_buckets, locations


# Input: a book's review list. Output: one location-token set per review that reported a location.
def _reviewer_location_sets(reviews: Any) -> list[set[str]]:
    sets: list[set[str]] = []
    if not isinstance(reviews, list):
        return sets
    for review in reviews:
        review_mapping = _as_review_mapping(review)
        demographics = review_mapping.get("user_demographics") if review_mapping else None
        if not isinstance(demographics, Mapping):
            continue
        location = _location_keys(demographics.get("location"))
        if location:
            sets.append(location)
    return sets


# Input: a book's review list. Output: a text string naming reviewer age/location, for embedding.
def _reviewer_demographic_text(reviews: Any) -> str:
    age_buckets, locations = _reviewer_demographics(reviews)
    parts = [f"reviewers {age}" for age in sorted(age_buckets)]
    parts.extend(f"reviewers from {location}" for location in sorted(locations))
    return " ".join(parts)


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


def _review_summaries(reviews: Any, max_reviews: int = 5, max_chars: int = 500) -> list[dict[str, Any]]:
    """Keep short review excerpts and non-identifying demographics for display."""
    if not isinstance(reviews, list):
        return []

    summaries: list[dict[str, Any]] = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        summary = str(review.get("summary") or "").strip()
        text = str(review.get("text") or "").strip()
        excerpt = (summary or text)[:max_chars]
        if not excerpt:
            continue
        demographics = review.get("user_demographics")
        if not isinstance(demographics, Mapping):
            demographics = {}
        reviewer: dict[str, Any] = {}
        age = _valid_age(demographics.get("age"))
        location = demographics.get("location")
        if age is not None:
            reviewer["age"] = age
        if isinstance(location, str) and location.strip():
            reviewer["location"] = location.strip()
        summaries.append({"text": excerpt, "reviewer": reviewer})
        if len(summaries) >= max_reviews:
            break
    return summaries


def load_catalog(json_path: str | Path) -> pd.DataFrame:
    """Load all_books.json into a DataFrame and add a single text field per
    book that we'll embed. Missing text fields degrade gracefully."""
    records = []
    # Reviews_List is kept (not excluded like the rest of CATALOG_COLUMNS
    # would suggest) because the reviewer-demographics pipeline below needs
    # each review's raw user_demographics block, not just the review text.
    required_fields = set(CATALOG_COLUMNS)
    for record in _iter_catalog_records(json_path):
        if not isinstance(record, dict):
            raise ValueError(f"Catalog at {json_path} must contain book objects.")
        row = {field: record.get(field) for field in required_fields}
        row["_reviews_text"] = _reviews_to_text(record.get("Reviews_List"))
        records.append(row)
    if not records:
        raise ValueError(f"Catalog at {json_path} must contain at least one book.")
    df = pd.DataFrame.from_records(records)

    missing = [column for column in CATALOG_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Catalog is missing required fields: {', '.join(missing)}")

    for col in ["Title", "Clean_Title", "Description", "Authors", "Category/Genre"]:
        if col not in df.columns:
            df[col] = None

    # Input: a raw Authors field value. Output: it as a plain string.
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

    reviews = df.get("Reviews_List", pd.Series(index=df.index))
    df["_demographic_age_buckets"] = reviews.apply(
        lambda value: _reviewer_demographics(value)[0]
    )
    df["_demographic_location_sets"] = reviews.apply(_reviewer_location_sets)
    df["_demographic_text"] = reviews.apply(_reviewer_demographic_text)
    df["_review_summaries"] = reviews.apply(_review_summaries)
    df["_is_english"] = [
        _is_probably_english(title, description)
        for title, description in zip(df["Title"], df["Description"])
    ]
    # Title is mentioned once, not repeated: repeating it would pull books
    # with superficially similar titles (shared words, unrelated meaning)
    # closer together in embedding space, which isn't a signal we want.
    # Genre/author repetition is intentional -- clustering books that share a
    # genre or author is exactly the desired effect there.
    df["content_text"] = (
        df["_title_str"] + " "
        + (df["_genre_str"] + " ") * 3
        + (df["_authors_str"] + " ") * 2
        + " reviews "
        + df["_reviews_text"].fillna("").astype(str)
        + " "
        + df["_desc_str"]
        + " "
        + df["_demographic_text"]
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


def save_runtime_catalog(
    catalog: pd.DataFrame,
    runtime_path: str | Path,
    catalog_path: str | Path,
) -> str:
    """Save the preprocessed fields needed by the API without raw reviews."""
    source = Path(catalog_path)
    fingerprint = catalog_fingerprint(source)
    payload = {
        "catalog_fingerprint": fingerprint,
        "source_size": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "catalog": catalog[RUNTIME_CATALOG_COLUMNS],
    }
    runtime = Path(runtime_path)
    runtime.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, runtime)
    return fingerprint


def load_runtime_catalog(
    runtime_path: str | Path,
    catalog_path: str | Path,
) -> tuple[pd.DataFrame, str]:
    """Load the preprocessed API catalog and reject an obviously stale copy."""
    runtime = Path(runtime_path)
    source = Path(catalog_path)
    if not runtime.exists():
        raise FileNotFoundError(
            f"Missing runtime catalog: {runtime}. Run build_runtime_catalog.py."
        )
    payload = pd.read_pickle(runtime)
    stat = source.stat()
    expected_fingerprint = payload.get("catalog_fingerprint")
    actual_fingerprint = catalog_fingerprint(source)
    if expected_fingerprint:
        is_stale = expected_fingerprint != actual_fingerprint
    else:
        is_stale = (
            payload.get("source_size") != stat.st_size
            or payload.get("source_mtime_ns") != stat.st_mtime_ns
        )
    if is_stale:
        raise ValueError(
            "Runtime catalog is stale because all_books.json changed. "
            "Run build_runtime_catalog.py again."
        )
    catalog = payload["catalog"]
    missing = [column for column in RUNTIME_CATALOG_COLUMNS if column not in catalog.columns]
    if missing:
        raise ValueError(
            "Runtime catalog is missing review fields. Run build_runtime_catalog.py again."
        )
    return catalog, payload["catalog_fingerprint"]


# ---------------------------------------------------------------------------
# 2. The preference-profile schema, and turning one into a query string
# ---------------------------------------------------------------------------
# These mirror your quiz's actual JSON shape (layer1/layer2/layer3) field for
# field, so a request body posted straight from the front end validates
# against this directly -- no separate "parse the JSON" step needed. Every
# field defaults to empty/None: a profile with nothing in it at all is valid
# and recommend() degrades gracefully down to a pure popularity ranking (see
# Recommender.recommend below).


# Schema: one liked book (book_id/title/authors); all fields optional.
class LikedBook(BaseModel):
    book_id: str | None = None
    title: str | None = None
    authors: str | None = None


# Schema: liked books, selected genres, and selected moods.
class Layer1(BaseModel):
    liked_books: list[LikedBook] = Field(default_factory=list)
    selected_genres: list[str] = Field(default_factory=list)
    selected_moods: list[str] = Field(default_factory=list)


# Schema: preference sliders (0-100) plus the hard avoid-list.
class Layer2(BaseModel):
    plot_vs_character: float | None = None
    pace: float | None = None
    complexity: float | None = None
    length_preference: str | None = None
    tone: float | None = None
    avoid_list: list[str] = Field(default_factory=list)
    avoid_other: str | None = None


# Schema: user age plus location, as a combined string or as separate country/city.
class Demographics(BaseModel):
    age: float | None = None
    location: str | None = None
    country: str | None = None
    city: str | None = None


# Schema: the user's free-text taste description.
class Layer3(BaseModel):
    free_text: str = ""


# Schema: a second, independent carrier of reader age/country/city, used only
# in to_query_text()'s raw (unbucketed) text -- kept alongside Demographics
# since it's not yet settled which shape the quiz frontend actually sends.
class PersonalInfo(BaseModel):
    age: int | None = None
    country: str | None = None
    city: str | None = None


# Input: a 0-100 slider value and its low/mid/high labels. Output: a graded phrase, or None.
def _describe_slider(value: float | None, low: str, mid: str, high: str) -> str | None:
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


# Schema: the full onboarding profile (identifiers, demographics, layer1-3); all fields optional.
class UserProfile(BaseModel):
    user_id: str | None = None
    locale: str | None = None
    schema_version: str | None = None
    created_at: str | None = None
    age: float | None = None
    location: str | None = None
    demographics: Demographics = Field(default_factory=Demographics)
    layer1: Layer1 = Field(default_factory=Layer1)
    layer2: Layer2 = Field(default_factory=Layer2)
    layer3: Layer3 = Field(default_factory=Layer3)
    personal_info: PersonalInfo | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_optional_layers(cls, values: Any) -> Any:
        if isinstance(values, dict):
            for field_name, model_type in (
                ("layer1", Layer1),
                ("layer2", Layer2),
                ("layer3", Layer3),
            ):
                if values.get(field_name) is None:
                    values[field_name] = model_type()
        return values

    # Input: a JSON file path. Output: a validated UserProfile.
    @classmethod
    def from_json(cls, path: str | Path) -> "UserProfile":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    # Input: none (uses self). Output: one text string encoding all stated preferences, for embedding.
    def to_query_text(self) -> str:
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
        age_text = _age_bucket(_profile_age(self))
        location_parts = _profile_location_keys(self)
        if age_text:
            parts.append(f"user {age_text}")
        parts.extend(f"user from {part}" for part in sorted(location_parts))
        if self.personal_info:
            if self.personal_info.age is not None:
                parts.append(f"reader age {self.personal_info.age}")
            if self.personal_info.country:
                parts.append(f"reader country {self.personal_info.country}")
            if self.personal_info.city:
                parts.append(f"reader city {self.personal_info.city}")
        return " ".join(p for p in parts if p).lower()


# Input: a UserProfile. Output: its age from whichever field actually carries
# it -- demographics, the legacy top-level field, or personal_info -- so a
# profile shaped like the real frontend's (personal_info-only) still gets a
# usable age instead of silently falling back to "no data".
def _profile_age(profile: "UserProfile") -> float | None:
    if profile.demographics.age is not None:
        return profile.demographics.age
    if profile.age is not None:
        return profile.age
    if profile.personal_info and profile.personal_info.age is not None:
        return profile.personal_info.age
    return None


# Input: a UserProfile. Output: the union of its location tokens from
# location/country/city, checking both demographics and personal_info.
def _profile_location_keys(profile: "UserProfile") -> set[str]:
    keys: set[str] = set()
    keys |= _location_keys(profile.demographics.location or profile.location)
    keys |= _location_keys(profile.demographics.country)
    keys |= _location_keys(profile.demographics.city)
    if profile.personal_info:
        keys |= _location_keys(profile.personal_info.country)
        keys |= _location_keys(profile.personal_info.city)
    return keys


AVOID_KEYWORDS = {
    "graphic_violence": ["graphic violence", "gore", "torture"],
    "child_harm": ["child abuse", "child harm", "abuse of children"],
    "sexual_content": ["explicit sexual content", "erotic"],
}


# ---------------------------------------------------------------------------
# 3. Embedding backend
# ---------------------------------------------------------------------------

# Interface: any embedding backend must implement fit() and encode().
class Embedder(ABC):

    # Input: catalog texts. Output: none; learns backend-specific state (no-op for a pretrained model).
    @abstractmethod
    def fit(self, texts: Sequence[str]) -> None:
        ...

    # Input: a list of texts. Output: one embedding vector per text.
    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


# Embedding backend: a pretrained sentence-transformer model (e.g. all-MiniLM-L6-v2).
class SentenceTransformerEmbedder(Embedder):

    # Input: model name, max chars, batch size. Output: none; loads the pretrained model.
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_chars: int = 800,
                 batch_size: int = 128):
        from sentence_transformers import SentenceTransformer  # deferred import

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, token=os.getenv("HF_TOKEN"))
        self.max_chars = max_chars
        self.batch_size = batch_size

    # Input: catalog texts. Output: none; a pretrained model needs no fitting.
    def fit(self, texts: Sequence[str]) -> None:
        pass

    # Input: a list of texts. Output: their normalized embedding vectors.
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

# Holds a book catalog and an embedder: fit() embeds the catalog once, recommend() scores it per user.
@dataclass
class Recommender:
    catalog: pd.DataFrame
    embedder: Embedder
    _catalog_embeddings: Any = field(default=None, repr=False)
    search_corpus: pd.DataFrame | None = field(default=None, repr=False)

    # Input: none. Output: the fuzzy-search corpus, building it from self.catalog
    # on first use if the caller (e.g. api.py's lifespan) hasn't already.
    def ensure_search_corpus(self) -> pd.DataFrame:
        if self.search_corpus is None:
            self.search_corpus = build_search_corpus(self.catalog)
        return self.search_corpus

    # Input: an optional cache path. Output: self, with every catalog book embedded.
    def fit(self, cache_path: str | Path | None = None) -> "Recommender":
        """Embed every book once. This is the expensive step -- encoding
        ~326k books through a neural network, not a lookup -- so pass
        cache_path to save the result to a .npy file and resume from the last
        completed batch after a restart.

        There's no automatic way to tell "the cache is still valid" from the
        file alone, so this does the one cheap check it can: if the cached
        row count doesn't match the current catalog's row count, the catalog
        clearly changed since the cache was built, and it re-embeds instead
        of silently serving recommendations built from a stale/mismatched
        catalog. That check does NOT catch every kind of staleness (e.g. you
        edited descriptions but kept the same row count) -- delete the cache
        file yourself whenever you regenerate all_books.json to be safe.
        """
        texts = list(self.catalog["content_text"])

        if cache_path is not None and Path(cache_path).exists():
            cached = np.load(cache_path, mmap_mode="r+")
            progress_path = Path(f"{cache_path}.progress")
            progress = None
            if progress_path.exists():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if (
                len(cached) == len(self.catalog)
                and progress is None
            ):
                self._catalog_embeddings = cached
                return self
            if (
                progress is not None
                and progress.get("row_count") == len(self.catalog)
                and progress.get("completed_rows", 0) < len(self.catalog)
            ):
                self.embedder.fit(texts)
                start = int(progress["completed_rows"])
                print(f"[fit] resuming embedding at row {start}/{len(texts)}")
            else:
                print(f"[fit] discarding stale or incomplete cache at {cache_path}")
                Path(cache_path).unlink()
                progress_path.unlink(missing_ok=True)
                cached = None
                start = 0
        else:
            cached = None
            start = 0

        self.embedder.fit(texts)
        batch_size = 4096
        progress_path = Path(f"{cache_path}.progress") if cache_path is not None else None
        for batch_start in range(start, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch = self.embedder.encode(texts[batch_start:batch_end])
            if cache_path is None:
                if self._catalog_embeddings is None:
                    self._catalog_embeddings = np.empty((len(texts), batch.shape[1]), dtype=batch.dtype)
                self._catalog_embeddings[batch_start:batch_end] = batch
                continue
            if cached is None:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                cached = np.lib.format.open_memmap(
                    cache_path, mode="w+", dtype=batch.dtype,
                    shape=(len(texts), batch.shape[1]),
                )
            cached[batch_start:batch_end] = batch
            cached.flush()
            progress_path.write_text(
                json.dumps({"row_count": len(texts), "completed_rows": batch_end}),
                encoding="utf-8",
            )

        self._catalog_embeddings = cached if cache_path is not None else self._catalog_embeddings
        if progress_path is not None:
            progress_path.unlink(missing_ok=True)
        return self

    def save_embedding_artifact(
        self,
        artifact_path: str | Path,
        catalog_path: str | Path,
        catalog_fingerprint_value: str | None = None,
    ) -> None:
        """Persist catalog vectors and the metadata needed to validate them."""
        if not isinstance(self._catalog_embeddings, np.ndarray):
            raise RuntimeError("Call fit() before saving an embedding artifact.")

        book_ids = self.catalog["Book ID"].fillna("").astype(str).to_numpy(dtype=str)
        metadata = {
            "catalog_fingerprint": catalog_fingerprint_value or catalog_fingerprint(catalog_path),
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
        catalog_fingerprint_value: str | None = None,
    ) -> "Recommender":
        """Load vectors created by the offline indexing step after validation.

        The local artifact is trusted and may contain legacy object-typed IDs.
        New artifacts store IDs as plain strings.
        """
        artifact = Path(artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(
                f"Missing catalog embedding artifact: {artifact}. "
                "Run `python build_catalog_embeddings.py` first."
            )

        with np.load(artifact, allow_pickle=True) as saved:
            metadata = json.loads(str(saved["metadata"]))
            embeddings = saved["embeddings"]
            saved_book_ids = saved["book_ids"].astype(str)

        expected_model = getattr(self.embedder, "model_name", None)
        expected_fingerprint = catalog_fingerprint_value or catalog_fingerprint(catalog_path)
        if metadata.get("catalog_fingerprint") != expected_fingerprint:
            raise ValueError(
                "Catalog embedding artifact is stale because all_books.json changed. "
                "Run `python build_catalog_embeddings.py` again."
            )
        if metadata.get("embedding_model") != expected_model:
            raise ValueError(
                f"Artifact model {metadata.get('embedding_model')!r} does not match "
                f"the configured model {expected_model!r}. Rebuild the artifact."
            )
        artifact_text_version = metadata.get("embedding_text_version")
        if artifact_text_version != EMBEDDING_TEXT_VERSION:
            if artifact_text_version not in LEGACY_EMBEDDING_TEXT_VERSIONS:
                raise ValueError("Embedding text format changed. Rebuild the artifact.")
            print(
                "[artifact] loading legacy embedding text version "
                f"{artifact_text_version}; rebuild to include reviewer demographics "
                "in content embeddings."
            )

        current_book_ids = self.catalog["Book ID"].fillna("").astype(str).to_numpy()
        if len(embeddings) != len(self.catalog) or not np.array_equal(saved_book_ids, current_book_ids):
            raise ValueError("Catalog embedding artifact does not match the current catalog order.")

        self._catalog_embeddings = embeddings
        return self

    # Input: a minimum-reviews threshold. Output: a shrinkage-adjusted rating per book.
    def _bayesian_rating(self, min_reviews: int = 20) -> pd.Series:
        r = self.catalog["Average_Normalized_Rating"]
        v = self.catalog["Review_Count"]
        c = r.mean(skipna=True)
        m = min_reviews
        return (v / (v + m)) * r.fillna(c) + (m / (v + m)) * c

    # Input: a UserProfile. Output: a per-book demographic-overlap score (neutral 0.5 if data is missing).
    def _demographic_similarity(self, profile: UserProfile) -> np.ndarray:
        user_age = _valid_age(_profile_age(profile))
        user_locations = _profile_location_keys(profile)
        if user_age is None and not user_locations:
            return np.full(len(self.catalog), 0.5)

        similarities: list[float] = []
        for age_buckets, location_sets in zip(
            self.catalog["_demographic_age_buckets"],
            self.catalog["_demographic_location_sets"],
        ):
            component_scores: list[float] = []
            if user_age is not None and age_buckets:
                # Smooth decay on the gap to the closest reviewer age-bucket,
                # instead of an all-or-nothing same-bucket check -- so a user
                # aged 51 scores close to a book whose reviewers are "40-49",
                # rather than as unrelated as a user aged 20.
                midpoints = [m for b in age_buckets if (m := _bucket_midpoint(b)) is not None]
                if midpoints:
                    closest_gap = min(abs(user_age - m) for m in midpoints)
                    component_scores.append(1 / (1 + closest_gap / 10))
            if user_locations and location_sets:
                # What fraction of this book's location-tagged reviewers share
                # a location with the user, not just whether any single one
                # happens to -- so a book overwhelmingly read by people from
                # the user's country/city scores higher than one where a
                # single reviewer coincidentally matches.
                matching = sum(1 for loc_set in location_sets if user_locations & loc_set)
                component_scores.append(matching / len(location_sets))
            similarities.append(float(np.mean(component_scores)) if component_scores else 0.5)
        return np.asarray(similarities)

    # Input: a UserProfile. Output: matched liked-book embeddings (or None),
    # the set of matched catalog Book IDs (so callers can hard-filter them
    # out of recommendations regardless of *how* each was resolved), and any
    # titles not found by book_id, exact title, or fuzzy title match.
    def _resolve_liked_books(self, profile: UserProfile) -> tuple[Any, set[str], list[str]]:
        liked = [b for b in profile.layer1.liked_books if b.book_id or b.title]
        if not liked:
            return None, set(), []

        book_id_col = self.catalog["Book ID"].fillna("").astype(str)
        matched_positions: list[int] = []
        matched_book_ids: set[str] = set()
        seen_positions: set[int] = set()

        def _claim(position: int) -> None:
            if position not in seen_positions:
                seen_positions.add(position)
                matched_positions.append(position)
                matched_book_ids.add(str(book_id_col.iat[position]))

        # 1. book_id-first: trust a caller-confirmed match over any text.
        remaining: list[LikedBook] = []
        for liked_book in liked:
            if liked_book.book_id:
                positions = np.flatnonzero((book_id_col == liked_book.book_id).to_numpy())
                if len(positions):
                    _claim(int(positions[0]))
                    continue
            remaining.append(liked_book)

        # 2. exact title match against the normalized _clean_title_str column.
        wanted_titles = {b.title.strip().lower() for b in remaining if b.title}
        mask = self.catalog["_clean_title_str"].isin(wanted_titles) if wanted_titles else None
        if mask is not None:
            for position in np.flatnonzero(mask.to_numpy()):
                _claim(int(position))
            unmatched_titles = wanted_titles - set(self.catalog.loc[mask, "_clean_title_str"])
        else:
            unmatched_titles = set()

        # 3. fuzzy fallback for titles that survived both exact passes above --
        # a higher cutoff than the live search endpoint since this runs on
        # already-frontend-resolved input, not open-ended search.
        not_found: list[str] = []
        corpus = self.ensure_search_corpus()
        for title in sorted(unmatched_titles):
            hits = search_books(corpus, title, limit=1, score_cutoff=80.0)
            if hits:
                positions = np.flatnonzero((book_id_col == hits[0].book_id).to_numpy())
                if len(positions):
                    _claim(int(positions[0]))
                    continue
            not_found.append(title)

        if not matched_positions:
            return None, matched_book_ids, sorted(not_found)

        return (
            self._catalog_embeddings[np.array(matched_positions, dtype=int)],
            matched_book_ids,
            sorted(not_found),
        )

    # Input: a UserProfile and scoring options (weights, top_n, liked_agg). Output: the top_n ranked books.
    def recommend(
        self,
        profile: UserProfile,
        top_n: int = 20,
        weights: tuple[float, ...] = (0.40, 0.30, 0.10, 0.20),
        liked_agg: str = "mean",
    ) -> pd.DataFrame:
        if len(weights) == 3:
            w_pref, w_liked, w_pop = weights
            w_demo = 0.0
        elif len(weights) == 4:
            w_pref, w_liked, w_demo, w_pop = weights
        else:
            raise ValueError("weights must contain 3 or 4 values")

        pref_vec = self.embedder.encode([profile.to_query_text()])
        sim_pref = cosine_similarity(pref_vec, self._catalog_embeddings).ravel()

        liked_embeddings, matched_liked_ids, not_found = self._resolve_liked_books(profile)
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
        sim_demo = self._demographic_similarity(profile)
        source_prior = self.catalog["Book ID"].map(_source_quality_prior).to_numpy()

        score = (
            w_pref * sim_pref
            + w_liked * sim_liked
            + w_demo * sim_demo
            + w_pop * pop_norm
            + 0.08 * source_prior
        )

        result = self.catalog.copy()
        result["sim_preferences"] = sim_pref
        result["sim_liked_books"] = sim_liked
        result["sim_demographics"] = sim_demo
        result["source_prior"] = source_prior
        result["score"] = score

        # Hard filters -- deterministic constraints, not signals to blend in,
        # so they're applied after scoring rather than as features.
        mask = pd.Series(True, index=result.index)
        mask &= result["_is_english"]

        if matched_liked_ids:
            mask &= ~result["Book ID"].fillna("").astype(str).isin(matched_liked_ids)

        for tag in profile.layer2.avoid_list:
            for kw in AVOID_KEYWORDS.get(tag, [tag.replace("_", " ")]):
                mask &= ~result["content_text"].str.contains(re.escape(kw), na=False)

        result = result[mask].sort_values("score", ascending=False)
        result["_recommendation_key"] = [
            _recommendation_key(title, authors)
            for title, authors in zip(result["Title"], result["Authors"])
        ]
        result = result.drop_duplicates("_recommendation_key", keep="first")
        return result.head(top_n)[
            ["Book ID", "Title", "Authors", "Category/Genre", "Description", "Review_Count",
               "Average_Normalized_Rating", "sim_preferences", "sim_liked_books",
                    "sim_demographics", "score", "_review_summaries"]
        ]

