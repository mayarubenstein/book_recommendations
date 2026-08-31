from __future__ import annotations

import streamlit as st

from src.i18n.strings import t
from src.profile.schema import Layer3Profile
from src.ui.state import STEP_LAYER2, STEP_LAYER4, goto


def render() -> None:
    st.header(t("layer3.title"))
    draft = st.session_state.draft

    free_text = st.text_area(t("layer3.prompt"), value=draft["l3_free_text"], max_chars=1000)
    draft["l3_free_text"] = free_text

    if st.button(t("nav.back"), type="primary", use_container_width=True):
        goto(STEP_LAYER2)

    if st.button(t("nav.skip"), type="primary", use_container_width=True):
        st.session_state.profile["layer3"] = None
        goto(STEP_LAYER4)

    if st.button(t("nav.save_continue"), type="primary", use_container_width=True):
        st.session_state.profile["layer3"] = Layer3Profile(free_text=free_text) if free_text.strip() else None
        goto(STEP_LAYER4)
