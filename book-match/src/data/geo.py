from __future__ import annotations

import geonamescache
import streamlit as st


@st.cache_resource
def _cache() -> geonamescache.GeonamesCache:
    return geonamescache.GeonamesCache()


@st.cache_data
def get_country_names() -> list[str]:
    countries = _cache().get_countries()
    return sorted({c["name"] for c in countries.values()})


@st.cache_data
def get_cities_for_country(country_name: str) -> list[str]:
    """Real cities only (geonamescache), scoped to the chosen country - a
    global city list would be too large and too ambiguous (many cities share
    a name across countries) to offer directly."""
    countries = _cache().get_countries()
    iso = next((c["iso"] for c in countries.values() if c["name"] == country_name), None)
    if iso is None:
        return []
    cities = _cache().get_cities()
    return sorted({c["name"] for c in cities.values() if c["countrycode"] == iso})
