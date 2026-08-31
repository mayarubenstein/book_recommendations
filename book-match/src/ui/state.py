from __future__ import annotations

import uuid

import streamlit as st

STEP_LAYER1 = "layer1"
STEP_LAYER2 = "layer2"
STEP_LAYER3 = "layer3"
STEP_LAYER4 = "layer4"
STEP_SUMMARY = "summary"

# Streamlit drops a widget's own state once that widget isn't instantiated on
# a script run (true for keyed and unkeyed widgets alike) - and since this
# wizard only renders one screen's widgets per run, a screen's selections
# would otherwise vanish the moment the user moves to another screen. `draft`
# is the persisted source of truth instead: each screen seeds its widgets
# with `value=`/`default=` from here and writes the widget's return value
# back immediately, so answers survive navigating away and back.
_DEFAULT_DRAFT = {
    "l1_liked_books": {},  # book_id -> BookRef
    "l1_genres": [],
    "l1_moods": [],
    "l2_plot_character": 50,
    "l2_plot_character_no_pref": True,
    "l2_pace": 50,
    "l2_pace_no_pref": True,
    "l2_complexity": 50,
    "l2_complexity_no_pref": True,
    "l2_length": "medium",
    "l2_length_no_pref": True,
    "l2_tone": 50,
    "l2_tone_no_pref": True,
    "l2_avoid": [],
    "l2_avoid_other": "",
    "l3_free_text": "",
    "l4_age": None,
    "l4_country": None,
    "l4_city": None,
}


def _fresh_draft() -> dict:
    return {k: (v.copy() if isinstance(v, (list, dict)) else v) for k, v in _DEFAULT_DRAFT.items()}


def init_state() -> None:
    if "step" not in st.session_state:
        st.session_state.step = STEP_LAYER1
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    if "locale" not in st.session_state:
        st.session_state.locale = "en"
    if "profile" not in st.session_state:
        st.session_state.profile = {"layer1": None, "layer2": None, "layer3": None, "personal_info": None}
    if "draft" not in st.session_state:
        st.session_state.draft = _fresh_draft()


def goto(step: str) -> None:
    st.session_state.step = step
    st.rerun()


def reset_wizard() -> None:
    """Clear all answers and go back to Layer 1, keeping the same user_id/locale."""
    st.session_state.profile = {"layer1": None, "layer2": None, "layer3": None, "personal_info": None}
    st.session_state.draft = _fresh_draft()
    goto(STEP_LAYER1)
