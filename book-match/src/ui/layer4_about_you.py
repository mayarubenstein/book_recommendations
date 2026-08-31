from __future__ import annotations

import streamlit as st

from src.data.geo import get_cities_for_country, get_country_names
from src.i18n.strings import t
from src.profile.schema import PersonalInfoProfile
from src.ui.state import STEP_LAYER3, STEP_SUMMARY, goto


def render() -> None:
    st.header(t("layer4.title"))
    st.write(t("layer4.intro"))
    draft = st.session_state.draft

    age = st.number_input(
        t("layer4.age.label"),
        min_value=0,
        max_value=120,
        value=draft["l4_age"],
        step=1,
        placeholder=t("layer4.age.placeholder"),
    )
    draft["l4_age"] = age

    country_names = get_country_names()
    country_index = country_names.index(draft["l4_country"]) if draft["l4_country"] in country_names else None
    country = st.selectbox(
        t("layer4.country.label"),
        options=country_names,
        index=country_index,
        placeholder=t("layer4.country.placeholder"),
    )
    draft["l4_country"] = country

    city = None
    if country:
        city_names = get_cities_for_country(country)
        city_index = city_names.index(draft["l4_city"]) if draft["l4_city"] in city_names else None
        city = st.selectbox(
            t("layer4.city.label"),
            options=city_names,
            index=city_index,
            placeholder=t("layer4.city.placeholder"),
        )
        draft["l4_city"] = city
    else:
        draft["l4_city"] = None
        st.caption(t("layer4.city.pick_country_first"))

    if st.button(t("nav.back"), type="primary", use_container_width=True):
        goto(STEP_LAYER3)

    if st.button(t("nav.skip"), type="primary", use_container_width=True):
        st.session_state.profile["personal_info"] = None
        goto(STEP_SUMMARY)

    if st.button(t("nav.save_finish"), type="primary", use_container_width=True):
        has_info = age is not None or country or city
        st.session_state.profile["personal_info"] = (
            PersonalInfoProfile(age=age, country=country, city=city) if has_info else None
        )
        goto(STEP_SUMMARY)
