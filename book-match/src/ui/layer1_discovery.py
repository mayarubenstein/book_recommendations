from __future__ import annotations

import streamlit as st
from streamlit_searchbox import st_searchbox

from src.data.book_provider import BackendUnavailableError, BookProvider, BookRef, liked_book_key
from src.data.mood_catalog import MOOD_CHIPS
from src.i18n.strings import t, translate_genre
from src.profile.schema import Layer1Profile
from src.ui.state import STEP_LAYER2, STEP_SUMMARY, goto


def _book_label(book: BookRef) -> str:
    return f"{book.title} — {book.authors}" if book.authors else book.title


@st.cache_data(ttl=30, show_spinner=False)
def _cached_search(_provider: BookProvider, query: str) -> list[BookRef]:
    # Leading underscore on _provider excludes it from st.cache_data's hash key
    # (providers aren't hashable/stable-to-hash) -- only `query` keys the cache,
    # so retyping an identical partial string within 30s skips the network call.
    return _provider.search_books(query, limit=10)


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
    # to the browser client-side - search is a live, per-keystroke (debounced)
    # HTTP call to the backend via the provider. Free text is always kept as
    # a fallback ("Add as typed") since a typo-tolerant match still isn't
    # guaranteed to find the right (or any) catalog entry.
    st.subheader(t("layer1.section.books"))
    st.session_state.setdefault("_l1_search_query", "")
    st.session_state.setdefault("_l1_search_backend_down", False)
    st.session_state.setdefault("_l1_search_no_matches", False)

    def _add_book(book: BookRef) -> None:
        draft["l1_liked_books"][liked_book_key(book)] = book

    def _search_options(query: str) -> list[tuple[str, BookRef]]:
        query = query.strip()
        st.session_state["_l1_search_query"] = query
        if not query:
            st.session_state["_l1_search_backend_down"] = False
            st.session_state["_l1_search_no_matches"] = False
            return []
        try:
            results = _cached_search(provider, query)
        except BackendUnavailableError:
            st.session_state["_l1_search_backend_down"] = True
            st.session_state["_l1_search_no_matches"] = False
            return []
        st.session_state["_l1_search_backend_down"] = False
        st.session_state["_l1_search_no_matches"] = not results
        return [(_book_label(book), book) for book in results]

    st_searchbox(
        _search_options,
        key="l1_book_search",
        placeholder=t("layer1.search.placeholder"),
        label=t("layer1.search.input_label"),
        clear_on_submit=True,
        submit_function=_add_book,
    )
    st.caption(t("layer1.search.limit_note"))

    if st.session_state["_l1_search_backend_down"]:
        st.caption(t("layer1.search.backend_unavailable"))
    elif st.session_state["_l1_search_no_matches"]:
        st.caption(t("layer1.search.no_matches"))

    current_query = st.session_state["_l1_search_query"]
    if current_query and st.button(
        t("layer1.search.add_as_typed").format(query=current_query),
        key="l1_add_as_typed",
        use_container_width=True,
    ):
        _add_book(BookRef(book_id=None, title=current_query, authors=""))
        st.session_state["_l1_search_query"] = ""
        st.rerun()

    if draft["l1_liked_books"]:
        label_by_key = {key: _book_label(b) for key, b in draft["l1_liked_books"].items()}
        kept_labels = (
            st.pills(
                t("layer1.search.selected_label"),
                options=list(label_by_key.values()),
                selection_mode="multi",
                default=list(label_by_key.values()),
            )
            or []
        )
        draft["l1_liked_books"] = {
            key: b for key, b in draft["l1_liked_books"].items() if label_by_key[key] in kept_labels
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
