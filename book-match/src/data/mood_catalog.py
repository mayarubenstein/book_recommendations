from __future__ import annotations

# Editorial mood chip keys, independent of any dataset. Display labels are
# resolved via i18n (src/i18n/strings.py, keys "mood.<key>") so new languages
# only require new translation entries, not new keys here.
MOOD_CHIPS: list[str] = [
    "cozy",
    "dark_gritty",
    "fast_thrilling",
    "funny_light",
    "emotional",
    "mind_bending",
    "epic_sweeping",
    "quiet_introspective",
]
