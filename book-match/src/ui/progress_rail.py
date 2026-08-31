from __future__ import annotations

import streamlit as st

from src.i18n.strings import t
from src.ui.html_utils import render_html
from src.ui.state import STEP_LAYER1, STEP_LAYER2, STEP_LAYER3, STEP_LAYER4, STEP_SUMMARY

_TEAL = "#2E6F6E"
_TRACK = "#D7E1DC"
_MUTED = "#9AA8A2"
_BG = "#FAFBF9"

# (step id, label key, is this step skippable) - the "optional" badge and the
# fill percentage both come from this being a real, ordered sequence, not
# decoration: it tells the user exactly how many optional steps remain.
_STEPS = [
    (STEP_LAYER1, "progress.step1", False),
    (STEP_LAYER2, "progress.step2", True),
    (STEP_LAYER3, "progress.step3", True),
    (STEP_LAYER4, "progress.step4", True),
    (STEP_SUMMARY, "progress.step5", False),
]


def render() -> None:
    current_index = next(i for i, (step_id, _, _) in enumerate(_STEPS) if step_id == st.session_state.step)
    fill_pct = round(current_index / (len(_STEPS) - 1) * 100)

    items_html = []
    for i, (_step_id, label_key, optional) in enumerate(_STEPS):
        is_current = i == current_index
        is_done = i < current_index
        active = is_current or is_done
        circle_border = _TEAL if active else _TRACK
        circle_bg = _TEAL if is_done else _BG
        circle_color = "#FFFFFF" if is_done else (_TEAL if is_current else _MUTED)
        marker = "&#10003;" if is_done else str(i + 1)
        label_color = _TEAL if active else _MUTED
        label_weight = "600" if is_current else "500"
        optional_html = (
            f'<div style="font-size:0.7rem;color:{_MUTED};">{t("progress.optional")}</div>' if optional else ""
        )
        items_html.append(
            f"""
            <div style="flex:1; text-align:center;">
                <div style="width:1.75rem; height:1.75rem; border-radius:50%;
                            background:{circle_bg}; border:2px solid {circle_border}; color:{circle_color};
                            display:flex; align-items:center; justify-content:center;
                            margin:0 auto 0.35rem auto; font-size:0.8rem; font-weight:600;
                            font-family:'Outfit',sans-serif;">{marker}</div>
                <div style="font-family:'Outfit',sans-serif; font-size:0.82rem;
                            font-weight:{label_weight}; color:{label_color};">{t(label_key)}</div>
                {optional_html}
            </div>
            """
        )

    render_html(
        f"""
        <div style="position:relative; margin: 0 0 1.75rem 0;">
            <div style="position:absolute; top:0.875rem; left:1.75rem; right:1.75rem;
                        height:2px; background:{_TRACK}; z-index:0;"></div>
            <div style="position:absolute; top:0.875rem; left:1.75rem; height:2px; background:{_TEAL}; z-index:0;
                        width:calc((100% - 3.5rem) * {fill_pct} / 100);"></div>
            <div style="position:relative; z-index:1; display:flex; justify-content:space-between;">
                {"".join(items_html)}
            </div>
        </div>
        """
    )
