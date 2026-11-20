"""Create the compact movie catalog used by the application.

The source file is the CC0 ``movies_metadata.csv`` from The Movies Dataset:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "movies_metadata_source.csv"
OUTPUT = ROOT / "data" / "movies.csv"
CATALOG_SIZE = 12_000


def parse_names(value: object) -> str:
    """Convert TMDB's list-of-dictionaries fields to pipe-separated names."""
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        items = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return ""
    return "|".join(
        str(item["name"]).strip()
        for item in items
        if isinstance(item, dict) and item.get("name")
    )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE.name}. Download movies_metadata.csv from the URL "
            "in this module's docstring and place it in the project root."
        )

    columns = [
        "id",
        "title",
        "genres",
        "overview",
        "tagline",
        "release_date",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
        "original_language",
    ]
    movies = pd.read_csv(
        SOURCE,
        usecols=columns,
        low_memory=False,
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="skip",
    )

    movies["movie_id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies["vote_count"] = pd.to_numeric(movies["vote_count"], errors="coerce").fillna(0)
    movies["vote_average"] = pd.to_numeric(movies["vote_average"], errors="coerce").fillna(0)
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce").fillna(0)
    movies["runtime"] = pd.to_numeric(movies["runtime"], errors="coerce")
    movies["genres"] = movies["genres"].map(parse_names)
    movies["year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year

    movies = movies.dropna(subset=["movie_id", "title"])
    movies = movies[movies["title"].str.strip().ne("") & movies["genres"].str.strip().ne("")]
    movies = movies.drop_duplicates(subset=["title", "year"], keep="first")
    movies = movies.sort_values(
        ["vote_count", "popularity", "vote_average"], ascending=False
    ).head(CATALOG_SIZE)

    output_columns = [
        "movie_id",
        "title",
        "genres",
        "overview",
        "tagline",
        "year",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
        "original_language",
    ]
    movies = movies[output_columns].sort_values("title", key=lambda series: series.str.lower())
    movies["movie_id"] = movies["movie_id"].astype("int64")
    movies["year"] = movies["year"].astype("Int64")
    movies["runtime"] = movies["runtime"].round().astype("Int64")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    movies.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(movies):,} movies to {OUTPUT}")


if __name__ == "__main__":
    main()
