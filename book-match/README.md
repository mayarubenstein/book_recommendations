# Book Match — Preference-Collection UI

This is the user-interaction layer of a hybrid book recommendation system. It
collects a user's reading preferences through a short, layered wizard and
produces a structured "preference profile" for a future recommendation engine
to consume. No recommender is implemented here yet.

## Design

Four layers, each a self-contained screen the user can stop after, followed
by a summary screen:

- **Layer 1 (required, ~30-60s, no typing needed):** search for books you've
  liked, and/or pick genre and mood chips.
- **Layer 2 (optional):** sliders/toggles for pacing, tone, complexity,
  length, and an avoid-list (fixed set of chips, no free-text option).
- **Layer 3 (optional):** one free-text box for anything else, stored as-is.
- **Layer 4 (optional):** age, country, and city.
- **Summary:** review everything, choose how many book recommendations to
  receive (1-10, defaults to 5), and download the profile as JSON.

A sidebar "Finish now" button is available from Layer 2 onward, so the user
can stop at any point without losing what they've already given.

## Running it

```
pip install -r requirements.txt
streamlit run app.py
```

The app runs out of the box against `MockBookProvider` (`src/data/mock_provider.py`),
a small hardcoded book list — the real book dataset is a separate part of this
project. See `src/data/book_provider.py` for the `BookProvider` interface;
swapping in the real implementation is a one-line change in `app.py`.

## Extension points

- **Book data:** `src/data/book_provider.py` (`BookProvider` protocol) — implement
  it against the real dataset once that part of the project is ready.
- **Persistence / login:** `src/data/profile_store.py` (`ProfileStore` protocol) —
  `InMemoryProfileStore` is session-only; swap in a real store once login exists.
  `UserPreferenceProfile.user_id` already exists (an anonymous session UUID today).
- **Languages / RTL:** `src/i18n/strings.py` — add a new locale dict (e.g. `"he"`)
  to `STRINGS`; no UI code changes needed. `src/i18n/rtl.py` already applies RTL
  layout for right-to-left locales.
- **Recommendations:** `src/recommender/interface.py` — `recommend(profile)` is
  the seam for the future ML phase.
