from __future__ import annotations

import streamlit as st

RTL_LOCALES = {"he", "ar"}


def inject_direction_css() -> None:
    """No-op today (only 'en' is configured), but exercised on every run so
    adding a right-to-left locale (Hebrew, Arabic) later is a content change,
    not a structural one."""
    locale = st.session_state.get("locale", "en")
    if locale not in RTL_LOCALES:
        return
    st.markdown(
        "<style>.stApp { direction: rtl; text-align: right; }</style>",
        unsafe_allow_html=True,
    )
