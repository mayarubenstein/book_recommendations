from __future__ import annotations

import streamlit as st

from src.data.mock_provider import MockBookProvider
from src.data.profile_store import InMemoryProfileStore
from src.i18n.rtl import inject_direction_css
from src.i18n.strings import t
from src.ui import layer1_discovery, layer2_finetune, layer3_freetext, layer4_about_you, progress_rail, summary
from src.ui.html_utils import render_html
from src.ui.state import STEP_LAYER1, STEP_LAYER2, STEP_LAYER3, STEP_LAYER4, STEP_SUMMARY, goto, init_state

st.set_page_config(page_title="Book Match", page_icon="📚")

init_state()
inject_direction_css()

# Swap these two lines to plug in the real dataset / a persistent store later -
# no other file needs to change.
book_provider = MockBookProvider()
if "profile_store" not in st.session_state:
    st.session_state.profile_store = InMemoryProfileStore()
profile_store = st.session_state.profile_store

render_html(
    f"""
    <div style="margin-bottom: 1rem;">
        <h1 style="font-family: 'Lora', serif; color: #2E6F6E; font-size: 2.5rem;
                   font-weight: 700; line-height: 1.1; margin: 0 0 0.2rem 0;">{t("app.title")}</h1>
        <p style="font-family: 'Outfit', sans-serif; color: #5C8F8E; font-size: 1.35rem;
                  font-weight: 500; line-height: 1.3; margin: 0;">{t("app.subtitle")}</p>
    </div>
    """
)

progress_rail.render()

if st.session_state.step != STEP_LAYER1:
    if st.sidebar.button(t("nav.finish_sidebar"), type="primary", use_container_width=True):
        goto(STEP_SUMMARY)

if st.session_state.step == STEP_LAYER1:
    layer1_discovery.render(book_provider)
elif st.session_state.step == STEP_LAYER2:
    layer2_finetune.render()
elif st.session_state.step == STEP_LAYER3:
    layer3_freetext.render()
elif st.session_state.step == STEP_LAYER4:
    layer4_about_you.render()
elif st.session_state.step == STEP_SUMMARY:
    summary.render(profile_store)
