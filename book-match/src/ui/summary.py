from __future__ import annotations

import os

import requests
import streamlit as st

from src.data.profile_store import ProfileStore
from src.i18n.strings import t, translate_genre
from src.profile.schema import UserPreferenceProfile
from src.ui.html_utils import render_html
from src.ui.state import MAX_NUM_RECOMMENDATIONS, MIN_NUM_RECOMMENDATIONS, reset_wizard

_TEAL = "#2E6F6E"
_CHIP_BG = "#EEF3F0"
_BORDER = "#D7E1DC"
_TEXT = "#1B211E"
_API_URL = os.getenv("BOOK_RECOMMENDER_API_URL", "http://127.0.0.1:8000").rstrip("/")

# (dataclass field name, low-end label key, high-end label key) - drives the
# 0-100 slider values into a plain-language phrase instead of a raw number.
_AXES = [
    ("plot_vs_character", "layer2.plot_character.low", "layer2.plot_character.high"),
    ("pace", "layer2.pace.low", "layer2.pace.high"),
    ("complexity", "layer2.complexity.low", "layer2.complexity.high"),
    ("tone", "layer2.tone.low", "layer2.tone.high"),
]


def _describe_axis(value: int, low_key: str, high_key: str) -> str:
    if value <= 33:
        return t("summary.axis.leans").format(direction=t(low_key))
    if value >= 67:
        return t("summary.axis.leans").format(direction=t(high_key))
    return t("summary.axis.balanced").format(low=t(low_key), high=t(high_key))


def _chip_row(labels: list[str]) -> None:
    chips = "".join(
        f'<span style="display:inline-block; background:{_CHIP_BG}; color:{_TEAL}; '
        f'border:1px solid {_BORDER}; border-radius:999px; padding:0.25rem 0.75rem; '
        f"margin:0 0.35rem 0.35rem 0; font-family:'Outfit',sans-serif; font-size:0.85rem;\">{label}</span>"
        for label in labels
    )
    render_html(f'<div style="margin:0.25rem 0 0.75rem 0;">{chips}</div>')


def render(profile_store: ProfileStore) -> None:
    raw = st.session_state.profile

    render_html(f"""<h2 style="font-family:'Lora',serif; color:{_TEAL}; margin-bottom:0.75rem;">{t("summary.title")}</h2>""")

    num_recommendations = st.slider(
        t("summary.num_recommendations.label"),
        min_value=MIN_NUM_RECOMMENDATIONS,
        max_value=MAX_NUM_RECOMMENDATIONS,
        value=st.session_state.num_recommendations,
    )
    st.session_state.num_recommendations = num_recommendations

    profile = UserPreferenceProfile(
        user_id=st.session_state.user_id,
        locale=st.session_state.locale,
        layer1=raw.get("layer1"),
        layer2=raw.get("layer2"),
        layer3=raw.get("layer3"),
        personal_info=raw.get("personal_info"),
        num_recommendations=num_recommendations,
    )
    profile_store.save_profile(profile)

    layer1 = profile.layer1
    has_books = bool(layer1 and layer1.liked_books)
    has_vibe = bool(layer1 and (layer1.selected_genres or layer1.selected_moods))

    if has_books:
        st.markdown(f"**{t('summary.section.books')}**")
        for book in layer1.liked_books:
            st.markdown(f"- {book.title} — {book.authors}")
    if has_vibe:
        st.markdown(f"**{t('summary.section.vibe')}**")
        mood_labels = [t(f"mood.{m}") for m in layer1.selected_moods]
        genre_labels = [translate_genre(g) for g in layer1.selected_genres]
        _chip_row(genre_labels + mood_labels)
    if not has_books and not has_vibe:
        st.caption(t("summary.section.books_empty"))

    if profile.layer2:
        l2 = profile.layer2
        lines = [
            f"- {_describe_axis(value, low_key, high_key)}"
            for field_name, low_key, high_key in _AXES
            if (value := getattr(l2, field_name)) is not None
        ]
        if l2.length_preference:
            lines.append(f"- {t('summary.length_prefix')} {t(f'layer2.length.{l2.length_preference}')}")
        avoid_parts = [t(f"layer2.avoid.{a}") for a in l2.avoid_list]
        if avoid_parts:
            lines.append(f"- {t('summary.avoid_prefix')} {', '.join(avoid_parts)}")
        if lines:
            st.markdown(f"**{t('summary.section.finetune')}**")
            st.markdown("\n".join(lines))

    if profile.layer3 and profile.layer3.free_text.strip():
        st.markdown(f"**{t('summary.section.freetext')}**")
        render_html(
            f"""<div style="font-family:'Lora',serif; font-style:italic; color:{_TEXT};
                        border-left:3px solid {_TEAL}; padding-left:0.85rem; margin:0.25rem 0 1rem 0;">
                        "{profile.layer3.free_text}"</div>"""
        )

    if profile.personal_info and (profile.personal_info.age is not None or profile.personal_info.country):
        pi = profile.personal_info
        lines = []
        if pi.age is not None:
            lines.append(f"- {t('summary.age_prefix')} {pi.age}")
        if pi.country:
            place = f"{pi.city}, {pi.country}" if pi.city else pi.country
            lines.append(f"- {t('summary.location_prefix')} {place}")
        st.markdown(f"**{t('summary.section.about_you')}**")
        st.markdown("\n".join(lines))

    if st.button("Get recommendations", type="primary", use_container_width=True):
        try:
            response = requests.post(
                f"{_API_URL}/recommend",
                params={"top_n": profile.num_recommendations},
                json=profile.to_dict(),
                timeout=120,
            )
            response.raise_for_status()
            recommendations = response.json()
            st.session_state.recommendations = recommendations
        except requests.RequestException as exc:
            st.error(f"Could not reach the recommendation API at {_API_URL}: {exc}")
        except ValueError:
            st.error("The recommendation API returned an invalid response.")

    recommendations = st.session_state.get("recommendations", [])
    if recommendations:
        st.subheader("Recommendations")
        st.dataframe(recommendations, use_container_width=True, hide_index=True)

    with st.expander(t("summary.raw_json")):
        st.json(profile.to_dict())

    st.download_button(
        t("summary.download"),
        data=profile.to_json(),
        file_name="preference_profile.json",
        mime="application/json",
        use_container_width=True,
    )

    if st.button(t("summary.start_over"), type="primary", use_container_width=True):
        reset_wizard()
