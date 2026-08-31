from __future__ import annotations

import streamlit as st

from src.i18n.strings import t
from src.profile.schema import Layer2Profile
from src.ui.state import STEP_LAYER1, STEP_LAYER3, goto

_LENGTH_OPTIONS = ["short", "medium", "long"]
_AVOID_OPTIONS = [
    "graphic_violence",
    "explicit_content",
    "animal_harm",
    "child_harm",
    "substance_abuse",
    "character_death",
]


def _slider_with_no_preference(draft: dict, label_key: str, draft_key: str, widget_key: str) -> int | None:
    """Title, then a slot reserved for the slider, then the "No preference"
    checkbox - the slider is only drawn into that slot when unchecked, so an
    axis the user hasn't opted into shows no line to set a number on."""
    no_pref_key = f"{draft_key}_no_pref"
    st.markdown(f"**{t(label_key)}**")
    slider_slot = st.empty()
    no_preference = st.checkbox(t("layer2.no_preference"), value=draft[no_pref_key], key=widget_key)
    draft[no_pref_key] = no_preference
    if no_preference:
        return None
    with slider_slot:
        value = st.slider(t(label_key), 0, 100, draft[draft_key], label_visibility="collapsed")
    draft[draft_key] = value
    return value


def _length_with_no_preference(draft: dict, widget_key: str) -> str | None:
    st.markdown(f"**{t('layer2.length')}**")
    slot = st.empty()
    no_preference = st.checkbox(t("layer2.no_preference"), value=draft["l2_length_no_pref"], key=widget_key)
    draft["l2_length_no_pref"] = no_preference
    if no_preference:
        return None
    length_label_by_key = {k: t(f"layer2.length.{k}") for k in _LENGTH_OPTIONS}
    label_to_length_key = {v: k for k, v in length_label_by_key.items()}
    current_key = draft["l2_length"] if draft["l2_length"] in _LENGTH_OPTIONS else "medium"
    with slot:
        length_label = st.select_slider(
            t("layer2.length"),
            options=list(length_label_by_key.values()),
            value=length_label_by_key[current_key],
            label_visibility="collapsed",
        )
    length_preference = label_to_length_key[length_label]
    draft["l2_length"] = length_preference
    return length_preference


def render() -> None:
    st.header(t("layer2.intro"))
    draft = st.session_state.draft

    plot_character = _slider_with_no_preference(draft, "layer2.plot_character", "l2_plot_character", "l2_np_plot_character")
    pace = _slider_with_no_preference(draft, "layer2.pace", "l2_pace", "l2_np_pace")
    complexity = _slider_with_no_preference(draft, "layer2.complexity", "l2_complexity", "l2_np_complexity")
    length_preference = _length_with_no_preference(draft, "l2_np_length")
    tone = _slider_with_no_preference(draft, "layer2.tone", "l2_tone", "l2_np_tone")

    avoid_label_by_key = {k: t(f"layer2.avoid.{k}") for k in _AVOID_OPTIONS}
    label_to_avoid_key = {v: k for k, v in avoid_label_by_key.items()}
    prior_avoid_labels = [avoid_label_by_key[a] for a in draft["l2_avoid"] if a in avoid_label_by_key]
    selected_avoid_labels = (
        st.pills(
            t("layer2.avoid.label"),
            options=list(avoid_label_by_key.values()),
            selection_mode="multi",
            default=prior_avoid_labels,
        )
        or []
    )
    avoid_list = [label_to_avoid_key[label] for label in selected_avoid_labels]
    draft["l2_avoid"] = avoid_list

    avoid_other = st.text_input(
        t("layer2.avoid.other_label"),
        value=draft["l2_avoid_other"],
        placeholder=t("layer2.avoid.other_placeholder"),
    )
    draft["l2_avoid_other"] = avoid_other

    if st.button(t("nav.back"), type="primary", use_container_width=True):
        goto(STEP_LAYER1)

    if st.button(t("nav.skip"), type="primary", use_container_width=True):
        st.session_state.profile["layer2"] = None
        goto(STEP_LAYER3)

    if st.button(t("nav.save_continue"), type="primary", use_container_width=True):
        st.session_state.profile["layer2"] = Layer2Profile(
            plot_vs_character=plot_character,
            pace=pace,
            complexity=complexity,
            length_preference=length_preference,
            tone=tone,
            avoid_list=avoid_list,
            avoid_other=avoid_other.strip() or None,
        )
        goto(STEP_LAYER3)
