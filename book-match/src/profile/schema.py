from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from src.data.book_provider import BookRef

SCHEMA_VERSION = "1.1"


@dataclass
class Layer1Profile:
    liked_books: list[BookRef] = field(default_factory=list)
    selected_genres: list[str] = field(default_factory=list)
    selected_moods: list[str] = field(default_factory=list)


@dataclass
class Layer2Profile:
    plot_vs_character: int | None
    pace: int | None
    complexity: int | None
    length_preference: str | None
    tone: int | None
    avoid_list: list[str] = field(default_factory=list)


@dataclass
class Layer3Profile:
    free_text: str


@dataclass
class PersonalInfoProfile:
    """About-the-user fields, independent of book preferences. Only age and
    location today - add fields here as more personal info is collected;
    each one should stay individually optional, same as these two."""

    age: int | None = None
    country: str | None = None
    city: str | None = None


@dataclass
class UserPreferenceProfile:
    user_id: str
    locale: str = "en"
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    layer1: Layer1Profile | None = None
    layer2: Layer2Profile | None = None
    layer3: Layer3Profile | None = None
    personal_info: PersonalInfoProfile | None = None
    num_recommendations: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
