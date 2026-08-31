from __future__ import annotations

import re

import streamlit as st


def render_html(html: str) -> None:
    """st.markdown(unsafe_allow_html=True) treats any line indented 4+ spaces
    as a Markdown code block, so multi-line HTML written with normal Python
    source indentation silently renders as literal text instead of markup.
    Strip leading whitespace from every line before handing it to Streamlit."""
    flattened = re.sub(r"^[ \t]+", "", html, flags=re.MULTILINE).strip()
    st.markdown(flattened, unsafe_allow_html=True)
