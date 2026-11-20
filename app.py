"""Streamlit interface for the movie recommendation engine."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from recommender import MovieRecommender


st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1200px; padding-top: 2rem;}
      [data-testid="stMetric"] {background: rgba(120,80,220,.08); padding: 1rem; border-radius: .8rem;}
      .movie-card {border: 1px solid rgba(128,128,128,.25); border-radius: .8rem; padding: 1rem; margin-bottom: .7rem;}
      .muted {opacity: .72; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Building the recommendation index…")
def get_recommender() -> MovieRecommender:
    return MovieRecommender()


engine = get_recommender()
movies = engine.movies
st.title("🎬 CineMatch")
st.caption("Content-based recommendations from a catalog of 12,000 movies")

metrics = st.columns(4)
metrics[0].metric("Movies", f"{len(movies):,}")
metrics[1].metric("Genres", len(engine.genres))
metrics[2].metric("Year range", f"{int(movies.year.min())}–{int(movies.year.max())}")
metrics[3].metric("Feature terms", f"{len(engine.vectorizer.vocabulary_):,}")

with st.sidebar:
    st.header("Recommendation settings")
    result_count = st.slider("Number of results", 5, 25, 10)
    chosen_genres = st.multiselect(
        "Only include these genres", engine.genres,
        help="A result may match any selected genre.",
    )
    min_rating = st.slider("Minimum community rating", 0.0, 10.0, 0.0, 0.5)
    valid_years = movies["year"].dropna().astype(int)
    full_year_range = (int(valid_years.min()), int(valid_years.max()))
    chosen_years = st.slider(
        "Release year", full_year_range[0], full_year_range[1], full_year_range
    )
    st.divider()
    st.caption("Similarity uses genres, plot overviews, and taglines. Ratings are filters, not ranking signals.")

st.subheader("Choose a movie you like")
search = st.text_input(
    "Search the catalog", placeholder="Try The Matrix, Toy Story, Parasite…",
    label_visibility="collapsed",
)
if search.strip():
    choices = engine.find_titles(search, limit=100)
else:
    choices = movies.sort_values(["vote_count", "vote_average"], ascending=False).head(100)[
        ["movie_id", "display_title"]
    ]
if choices.empty:
    st.warning("No matching titles. Try a shorter search phrase.")
    st.stop()

labels = dict(zip(choices["movie_id"].astype(int), choices["display_title"]))
selected_id = st.selectbox(
    "Movie", options=list(labels), format_func=labels.get, label_visibility="collapsed"
)
selected = movies.loc[movies["movie_id"] == selected_id].iloc[0]
st.info(
    f"**{selected['display_title']}**  ·  {selected['genres'].replace('|', ' · ')}  ·  "
    f"⭐ {selected['vote_average']:.1f}/10"
)

recommendations = engine.recommend(
    int(selected_id), n=result_count, genres=chosen_genres,
    min_rating=min_rating, year_range=chosen_years,
)
st.subheader("Your recommendations")
if recommendations.empty:
    st.warning("No movies match every filter. Widen the filters in the sidebar.")
else:
    for rank, movie in recommendations.iterrows():
        runtime = f"{int(movie.runtime)} min" if pd.notna(movie.runtime) else "Runtime unavailable"
        year = str(int(movie.year)) if pd.notna(movie.year) else "Year unavailable"
        overview = html.escape(movie.overview or "No overview is available for this title.")
        title = html.escape(str(movie.title))
        genres = html.escape(movie.genres.replace("|", " · "))
        st.markdown(
            f"""<div class="movie-card"><b>#{rank + 1} &nbsp; {title}</b>
            <div class="muted">{year} · {runtime} · {genres} · ⭐ {movie.vote_average:.1f} · {movie.similarity:.0%} match</div>
            <div style="margin-top:.5rem">{overview}</div></div>""",
            unsafe_allow_html=True,
        )

with st.expander("How recommendations are calculated"):
    st.markdown(
        """The engine turns each movie's genres, tagline, and plot overview into a TF-IDF vector.
        It compares the selected movie with eligible catalog entries using cosine similarity.
        Genre terms receive extra weight, and sidebar filters are applied before results appear.
        No personal data or external API is required."""
    )
