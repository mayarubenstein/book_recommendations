from __future__ import annotations

from src.data.book_provider import BookRef
from src.profile.schema import UserPreferenceProfile


def recommend(profile: UserPreferenceProfile, top_n: int = 10) -> list[BookRef]:
    raise NotImplementedError("Recommendation engine arrives in a future phase")
