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

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity


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


# Input: a book's review list and a character cap. Output: concatenated review excerpt text.
def _review_text(reviews: Any, max_chars: int = 1600) -> str:
    excerpts: list[str] = []
    used_chars = 0
    if not isinstance(reviews, list):
        return ""
    for review in reviews:
        review_mapping = _as_review_mapping(review)
        if not review_mapping:
            continue
        parts = [
            value.strip()
            for key in ("summary", "text")
            if isinstance(value := review_mapping.get(key), str) and value.strip()
        ]
        excerpt = " ".join(parts)
        if not excerpt:
            continue
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        excerpts.append(excerpt[:remaining])
        used_chars += min(len(excerpt), remaining)
    return " ".join(excerpts)

# ---------------------------------------------------------------------------
# 1. Loading the book catalog produced by creating_one_file.ipynb
# ---------------------------------------------------------------------------

# Input: path to all_books.json. Output: a DataFrame with a combined content_text column per book.
def load_catalog(json_path: str | Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)

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
    df["_review_text"] = reviews.apply(_review_text)
    df["_demographic_text"] = reviews.apply(_reviewer_demographic_text)
    # Keep review excerpts before the description and demographic tail because
    # SentenceTransformerEmbedder truncates long book texts.
    # Title is mentioned once, not repeated: repeating it would pull books
    # with superficially similar titles (shared words, unrelated meaning)
    # closer together in embedding space, which isn't a signal we want.
    # Genre/author repetition is intentional -- clustering books that share a
    # genre or author is exactly the desired effect there.
    df["content_text"] = (
        df["_title_str"] + " "
        + (df["_genre_str"] + " ") * 3
        + (df["_authors_str"] + " ") * 2
        + " review excerpts "
        + df["_review_text"]
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


# Schema: user age plus location, as a combined string or as separate country/city.
class Demographics(BaseModel):
    age: float | None = None
    location: str | None = None
    country: str | None = None
    city: str | None = None


# Schema: the user's free-text taste description.
class Layer3(BaseModel):
    free_text: str = ""


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
        if l3.free_text:
            parts.append(l3.free_text)
        age = self.demographics.age if self.demographics.age is not None else self.age
        age_text = _age_bucket(age)
        location_parts = _profile_location_keys(self)
        if age_text:
            parts.append(f"user {age_text}")
        parts.extend(f"user from {part}" for part in sorted(location_parts))
        return " ".join(p for p in parts if p).lower()


# Input: a UserProfile. Output: the union of its location tokens from location/country/city.
def _profile_location_keys(profile: "UserProfile") -> set[str]:
    keys: set[str] = set()
    keys |= _location_keys(profile.demographics.location or profile.location)
    keys |= _location_keys(profile.demographics.country)
    keys |= _location_keys(profile.demographics.city)
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

        self.model = SentenceTransformer(model_name)
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

    # Input: an optional cache path. Output: self, with every catalog book embedded.
    def fit(self, cache_path: str | Path | None = None) -> "Recommender":
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

    # Input: a minimum-reviews threshold. Output: a shrinkage-adjusted rating per book.
    def _bayesian_rating(self, min_reviews: int = 20) -> pd.Series:
        r = self.catalog["Average_Normalized_Rating"]
        v = self.catalog["Review_Count"]
        c = r.mean(skipna=True)
        m = min_reviews
        return (v / (v + m)) * r.fillna(c) + (m / (v + m)) * c

    # Input: a UserProfile. Output: a per-book demographic-overlap score (neutral 0.5 if data is missing).
    def _demographic_similarity(self, profile: UserProfile) -> np.ndarray:
        age = profile.demographics.age if profile.demographics.age is not None else profile.age
        user_age = _valid_age(age)
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

    # Input: a UserProfile. Output: matched liked-book embeddings (or None) and any titles not found.
    def _resolve_liked_books(self, profile: UserProfile) -> tuple[Any, list[str]]:
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
        sim_demo = self._demographic_similarity(profile)

        score = (
            w_pref * sim_pref
            + w_liked * sim_liked
            + w_demo * sim_demo
            + w_pop * pop_norm
        )

        result = self.catalog.copy()
        result["sim_preferences"] = sim_pref
        result["sim_liked_books"] = sim_liked
        result["sim_demographics"] = sim_demo
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
               "Average_Normalized_Rating", "sim_preferences", "sim_liked_books",
               "sim_demographics", "score"]
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
