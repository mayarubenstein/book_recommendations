from __future__ import annotations

import streamlit as st

from src.data.book_provider import BookProvider
from src.data.mood_catalog import MOOD_CHIPS
from src.i18n.strings import t, translate_genre
from src.profile.schema import Layer1Profile
from src.ui.state import STEP_LAYER2, STEP_SUMMARY, goto


def render(provider: BookProvider) -> None:
    st.write(t("layer1.instruction"))
    draft = st.session_state.draft

    # --- Section 1: The mood I am in for the book (editorial, i18n-keyed) ---
    st.subheader(t("layer1.section.mood"))
    mood_label_by_key = {m: t(f"mood.{m}") for m in MOOD_CHIPS}
    label_to_mood_key = {v: k for k, v in mood_label_by_key.items()}
    prior_mood_labels = [mood_label_by_key[m] for m in draft["l1_moods"] if m in mood_label_by_key]
    selected_mood_labels = (
        st.pills(
            t("layer1.moods.label"),
            options=list(mood_label_by_key.values()),
            selection_mode="multi",
            default=prior_mood_labels,
            label_visibility="collapsed",
        )
        or []
    )
    selected_moods = [label_to_mood_key[label] for label in selected_mood_labels]
    draft["l1_moods"] = selected_moods

    # --- Section 2: The genres I want to enjoy from (raw values from BookProvider) ---
    st.subheader(t("layer1.section.genres"))
    raw_genres = provider.get_available_genres()
    genre_label_by_raw = {g: translate_genre(g) for g in raw_genres}
    label_to_raw_genre = {v: k for k, v in genre_label_by_raw.items()}
    prior_genre_labels = [genre_label_by_raw[g] for g in draft["l1_genres"] if g in genre_label_by_raw]
    selected_genre_labels = (
        st.pills(
            t("layer1.genres.label"),
            options=list(genre_label_by_raw.values()),
            selection_mode="multi",
            default=prior_genre_labels,
            label_visibility="collapsed",
        )
        or []
    )
    selected_genres = [label_to_raw_genre[label] for label in selected_genre_labels]
    draft["l1_genres"] = selected_genres

    # --- Section 3: Books I like ---
    # The real catalog can run to millions of books, so it can never be sent
    # to the browser client-side - search is server-side via the provider and
    # capped to a handful of matches. Results and the running selection are
    # both plain visible chips (like the mood/genre sections above), nothing
    # hidden behind a dropdown the user has to know to open.
    st.subheader(t("layer1.section.books"))
    query = st.text_input(t("layer1.search.input_label"), placeholder=t("layer1.search.placeholder"))
    st.caption(t("layer1.search.limit_note"))
    raw_matches = provider.search_books(query, limit=10) if query.strip() else provider.get_popular_books(limit=10)
    matches = [b for b in raw_matches if b.book_id not in draft["l1_liked_books"]]  # already-added stay in the row below only

    if query.strip() and not raw_matches:
        st.caption(t("layer1.search.no_matches"))
    elif matches:
        book_by_label = {f"{b.title} — {b.authors}": b for b in matches}
        add_labels = (
            st.pills(
                t("layer1.search.add_label"),
                options=list(book_by_label.keys()),
                selection_mode="multi",
                default=[],
            )
            or []
        )
        for label in add_labels:
            book = book_by_label[label]
            draft["l1_liked_books"][book.book_id] = book

    if draft["l1_liked_books"]:
        selected_label_by_id = {book_id: f"{b.title} — {b.authors}" for book_id, b in draft["l1_liked_books"].items()}
        kept_labels = (
            st.pills(
                t("layer1.search.selected_label"),
                options=list(selected_label_by_id.values()),
                selection_mode="multi",
                default=list(selected_label_by_id.values()),
            )
            or []
        )
        draft["l1_liked_books"] = {
            book_id: b for book_id, b in draft["l1_liked_books"].items() if selected_label_by_id[book_id] in kept_labels
        }

    liked_books = list(draft["l1_liked_books"].values())

    has_selection = bool(liked_books or selected_genres or selected_moods)
    if not has_selection:
        st.caption(t("nav.select_at_least_one"))

    def _build_profile() -> Layer1Profile:
        return Layer1Profile(
            liked_books=liked_books,
            selected_genres=selected_genres,
            selected_moods=selected_moods,
        )

    if st.button(t("nav.finish"), type="primary", disabled=not has_selection, use_container_width=True):
        st.session_state.profile["layer1"] = _build_profile()
        goto(STEP_SUMMARY)

    if st.button(t("nav.continue_optional"), type="primary", disabled=not has_selection, use_container_width=True):
        st.session_state.profile["layer1"] = _build_profile()
        goto(STEP_LAYER2)
