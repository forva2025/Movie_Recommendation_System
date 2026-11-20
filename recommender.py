"""Content-based movie recommendations for the bundled 12,000-film catalog.

Genres, plot overviews, and taglines are represented with TF-IDF features.
Similarity is calculated only for the selected movie, avoiding a memory-heavy
12,000 x 12,000 matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


DATA_PATH = Path(__file__).resolve().parent / "data" / "movies.csv"
REQUIRED_COLUMNS = {
    "movie_id", "title", "genres", "overview", "tagline", "year", "runtime",
    "vote_average", "vote_count", "popularity", "original_language",
}


def load_movie_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and normalize the movie catalog."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Movie catalog not found at {path}. Run scripts/prepare_dataset.py."
        )
    movies = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(movies.columns)
    if missing:
        raise ValueError(f"Movie catalog is missing columns: {sorted(missing)}")

    text_columns = ["title", "genres", "overview", "tagline", "original_language"]
    movies[text_columns] = movies[text_columns].fillna("").astype(str)
    movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="raise").astype(int)
    for column in ["year", "runtime", "vote_average", "vote_count", "popularity"]:
        movies[column] = pd.to_numeric(movies[column], errors="coerce")
    movies = movies.reset_index(drop=True)
    movies["display_title"] = movies.apply(
        lambda row: f"{row['title']} ({int(row['year'])})"
        if pd.notna(row["year"]) else row["title"],
        axis=1,
    )
    return movies


class MovieRecommender:
    """TF-IDF content recommender with optional metadata filters."""

    def __init__(self, movies: pd.DataFrame | None = None) -> None:
        self.movies = load_movie_data() if movies is None else movies.copy().reset_index(drop=True)
        if "display_title" not in self.movies:
            self.movies["display_title"] = self.movies["title"]

        # Repeating genre terms gives them more weight than shared plot words.
        genres = self.movies["genres"].str.replace("|", " ", regex=False)
        features = (
            (genres + " ") * 3
            + self.movies["tagline"].fillna("")
            + " "
            + self.movies["overview"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95,
            max_features=30_000, sublinear_tf=True,
        )
        self.feature_matrix = self.vectorizer.fit_transform(features)
        self._id_to_index = {
            int(movie_id): index for index, movie_id in enumerate(self.movies["movie_id"])
        }

    @property
    def genres(self) -> list[str]:
        """Return all catalog genres in alphabetical order."""
        return sorted({
            genre.strip()
            for genres in self.movies["genres"]
            for genre in genres.split("|")
            if genre.strip()
        })

    def find_titles(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Find titles containing a case-insensitive query."""
        query = query.strip()
        if not query:
            return self.movies.head(0)[["movie_id", "display_title"]]
        matches = self.movies["title"].str.contains(query, case=False, regex=False)
        return self.movies.loc[matches, ["movie_id", "display_title"]].head(limit)

    def recommend(
        self,
        movie_id: int,
        n: int = 10,
        genres: Iterable[str] | None = None,
        min_rating: float = 0.0,
        year_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Return the most similar movies for a catalog movie ID."""
        if movie_id not in self._id_to_index:
            raise KeyError(f"Unknown movie_id: {movie_id}")
        if n < 1:
            raise ValueError("n must be at least 1")

        index = self._id_to_index[movie_id]
        scores = linear_kernel(
            self.feature_matrix[index : index + 1], self.feature_matrix
        ).ravel()
        eligible = np.ones(len(self.movies), dtype=bool)
        eligible[index] = False
        eligible &= self.movies["vote_average"].fillna(0).to_numpy() >= min_rating

        selected_genres = [genre for genre in (genres or []) if genre]
        if selected_genres:
            genre_pattern = "|".join(f"(?:{genre})" for genre in selected_genres)
            eligible &= self.movies["genres"].str.contains(
                genre_pattern, case=False, regex=True
            ).to_numpy()

        if year_range:
            start, end = year_range
            years = self.movies["year"].to_numpy()
            eligible &= np.isfinite(years) & (years >= start) & (years <= end)

        candidates = np.flatnonzero(eligible)
        ranked = candidates[np.argsort(scores[candidates])[::-1]][:n]
        result = self.movies.iloc[ranked].copy()
        result.insert(1, "similarity", scores[ranked])
        return result.reset_index(drop=True)


_default_recommender: MovieRecommender | None = None


def get_recommendations(title: str, n: int = 10) -> pd.Series:
    """Backward-compatible helper returning recommended movie titles."""
    global _default_recommender
    if _default_recommender is None:
        _default_recommender = MovieRecommender()
    exact = _default_recommender.movies[
        _default_recommender.movies["title"].str.casefold() == title.casefold()
    ]
    if exact.empty:
        suggestions = _default_recommender.find_titles(title, limit=5)
        hint = ", ".join(suggestions["display_title"]) or "no close matches"
        raise ValueError(f"Movie not found: {title!r}. Suggestions: {hint}")
    movie_id = int(exact.iloc[0]["movie_id"])
    return _default_recommender.recommend(movie_id, n=n)["title"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", nargs="?", default="The Matrix")
    parser.add_argument("-n", type=int, default=10, help="number of recommendations")
    args = parser.parse_args()
    try:
        recommendations = get_recommendations(args.title, args.n)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Recommendations for {args.title!r}:")
    for rank, title in enumerate(recommendations, start=1):
        print(f"{rank:>2}. {title}")


if __name__ == "__main__":
    main()
