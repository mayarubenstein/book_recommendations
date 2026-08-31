from __future__ import annotations

from typing import Protocol

from src.profile.schema import UserPreferenceProfile


class ProfileStore(Protocol):
    """Abstract seam for persisting profiles. A future login phase swaps
    InMemoryProfileStore for a real (e.g. database-backed) implementation
    without changing the wizard or schema."""

    def save_profile(self, profile: UserPreferenceProfile) -> None: ...

    def load_latest_profile(self, user_id: str) -> UserPreferenceProfile | None: ...


class InMemoryProfileStore:
    def __init__(self) -> None:
        self._profiles: dict[str, UserPreferenceProfile] = {}

    def save_profile(self, profile: UserPreferenceProfile) -> None:
        self._profiles[profile.user_id] = profile

    def load_latest_profile(self, user_id: str) -> UserPreferenceProfile | None:
        return self._profiles.get(user_id)
