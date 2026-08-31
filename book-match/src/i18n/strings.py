from __future__ import annotations

import streamlit as st

# Add a language by adding a new top-level locale entry here - no UI code
# needs to change. Genre labels are handled separately by translate_genre()
# below, since genre values come from the (separately built) BookProvider.
STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "app.title": "Book Match",
        "app.subtitle": "Tell us what you love to read and we'll find your next favorite book.",

        "nav.finish_sidebar": "Finish now — see my profile",
        "nav.back": "← Back",
        "nav.skip": "Skip this step →",
        "nav.finish": "Finish now →",
        "nav.continue_optional": "Continue to fine-tune (optional) →",
        "nav.save_continue": "Save & Continue →",
        "nav.save_finish": "Save & Finish",
        "nav.select_at_least_one": "Pick at least one book, genre, or mood to continue.",

        "progress.step1": "Discover",
        "progress.step2": "Fine-tune",
        "progress.step3": "Anything else",
        "progress.step4": "About you",
        "progress.step5": "Your profile",
        "progress.optional": "optional",

        "layer1.instruction": "Guide me by populating one or more of the following:",
        "layer1.section.books": "Books I like",
        "layer1.section.mood": "The mood I am in for the book",
        "layer1.section.genres": "The genres I want to enjoy from",
        "layer1.search.input_label": "Search for a book or author",
        "layer1.search.placeholder": "Type a title or author, then press Enter…",
        "layer1.search.limit_note": "Showing only the first 10 matching books.",
        "layer1.search.no_matches": "No matches found — try a different title or author.",
        "layer1.search.add_label": "Tap a book to add it to your list",
        "layer1.search.selected_label": "Your selected books (tap to remove)",
        "layer1.genres.label": "Genres you enjoy",
        "layer1.moods.label": "What are you in the mood for?",

        "layer2.intro": "Want better matches? Answer a few quick questions, otherwise skip.",
        "layer2.no_preference": "No preference",
        "layer2.plot_character": "Plot-driven ←→ Character-driven",
        "layer2.plot_character.low": "Plot-driven",
        "layer2.plot_character.high": "Character-driven",
        "layer2.pace": "Fast-paced ←→ Slow-burn",
        "layer2.pace.low": "Fast-paced",
        "layer2.pace.high": "Slow-burn",
        "layer2.complexity": "Light ←→ Dense/complex",
        "layer2.complexity.low": "Light",
        "layer2.complexity.high": "Dense and complex",
        "layer2.length": "Length",
        "layer2.length.short": "Short",
        "layer2.length.medium": "Medium",
        "layer2.length.long": "Long",
        "layer2.tone": "Dark/heavy ←→ Uplifting",
        "layer2.tone.low": "Dark and heavy",
        "layer2.tone.high": "Uplifting",
        "layer2.avoid.label": "Avoid",
        "layer2.avoid.graphic_violence": "Graphic violence",
        "layer2.avoid.explicit_content": "Explicit sexual content",
        "layer2.avoid.animal_harm": "Animal harm",
        "layer2.avoid.child_harm": "Child harm",
        "layer2.avoid.substance_abuse": "Substance abuse",
        "layer2.avoid.character_death": "On-page character death",
        "layer2.avoid.other_label": "Anything else you'd like to avoid?",
        "layer2.avoid.other_placeholder": "e.g. love triangles, unreliable narrators…",

        "layer3.title": "Anything else?",
        "layer3.prompt": "Anything else you want me to consider for the book you want to read?",

        "layer4.title": "A little about you",
        "layer4.intro": "Optional — sharing your age and location can help tailor recommendations (age-appropriate picks, local availability). Skip if you'd rather not say.",
        "layer4.age.label": "Age",
        "layer4.age.placeholder": "e.g. 34",
        "layer4.country.label": "Country",
        "layer4.country.placeholder": "Search for your country…",
        "layer4.city.label": "City",
        "layer4.city.placeholder": "Search for your city…",
        "layer4.city.pick_country_first": "Pick a country above to choose a city (optional).",

        "mood.cozy": "Cozy",
        "mood.dark_gritty": "Dark & gritty",
        "mood.fast_thrilling": "Fast & thrilling",
        "mood.funny_light": "Funny / light",
        "mood.emotional": "Emotional",
        "mood.mind_bending": "Mind-bending",
        "mood.epic_sweeping": "Epic / sweeping",
        "mood.quiet_introspective": "Quiet / introspective",

        "summary.title": "Your preference profile",
        "summary.section.books": "Books you loved",
        "summary.section.books_empty": "You didn't add any specific books — your genre and mood picks still guide the match.",
        "summary.section.vibe": "Genres & mood",
        "summary.section.finetune": "How you like to read",
        "summary.section.finetune_empty": "No fine-tuning preferences set.",
        "summary.length_prefix": "Length:",
        "summary.avoid_prefix": "Prefers to avoid:",
        "summary.section.freetext": "In your words",
        "summary.section.about_you": "About you",
        "summary.age_prefix": "Age:",
        "summary.location_prefix": "Location:",
        "summary.axis.leans": "Leans {direction}",
        "summary.axis.balanced": "Balanced between {low} and {high}",
        "summary.raw_json": "View raw profile data",
        "summary.not_available_yet": "Recommendations aren't available yet — this profile is the input for a future phase.",
        "summary.download": "Download my profile (JSON)",
        "summary.start_over": "← Start over and search again",
    },
    # "he": {...}  # added later, no code changes required
}


def t(key: str) -> str:
    locale = st.session_state.get("locale", "en")
    table = STRINGS.get(locale, STRINGS["en"])
    return table.get(key, STRINGS["en"].get(key, key))


def translate_genre(genre_label: str) -> str:
    """Genre values come from the separately-built BookProvider, so they can't
    be pre-enumerated above. Falls back to the raw provider string when no
    translation exists yet for the active locale."""
    locale = st.session_state.get("locale", "en")
    table = STRINGS.get(locale, {})
    slug = genre_label.strip().lower().replace(" ", "_").replace("-", "_")
    return table.get(f"genre.{slug}", genre_label)
