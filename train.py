"""Reproducibly build the KNN recommender artifacts from the raw CSVs.

Run from the project root:

    python train.py

Inputs (must be present in the working directory):
    Books.csv     ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-{S,M,L}
    Ratings.csv   User-ID, ISBN, Book-Rating
    Users.csv     User-ID, Location, Age          (loaded for parity with the notebook; unused downstream)

Outputs (written next to this script, overwriting any existing copy):
    NNeighbors.pkl   sklearn.neighbors.NearestNeighbors fitted on bookPivot
    bookPivot.pkl    pandas.DataFrame, index=title, columns=user_id, values=rating (NaNs filled with 0.0)
    bookNames.pkl    list[str], == list(bookPivot.index)
    finalRating.pkl  pandas.DataFrame with columns:
                       user_id (int)            — Book-Crossing user id
                       ISBN (str)               — book identifier
                       rating (int)             — explicit/implicit rating, 0–10
                       title (str)              — book title (recommender key)
                       author (str)
                       year (str)
                       publisher (str)
                       img_url (str)            — large Amazon cover URL (may rot; app falls back to Open Library)
                       number_of_ratings (int)  — count of ratings for this title across the active user set

Filtering (mirrors main.ipynb):
    * keep only users with > 200 ratings (the "active" cohort)
    * keep only titles with >= 50 ratings inside that cohort
    * deduplicate (user_id, title) pairs before pivoting
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).parent
ACTIVE_USER_MIN_RATINGS = 200
POPULAR_TITLE_MIN_RATINGS = 50


def build_artifacts() -> None:
    books = pd.read_csv(ROOT / "Books.csv", low_memory=False)
    ratings = pd.read_csv(ROOT / "Ratings.csv")
    _ = pd.read_csv(ROOT / "Users.csv")  # parity with notebook; not used downstream

    books = books.drop(columns=["Image-URL-S", "Image-URL-M"]).rename(
        columns={
            "Book-Title": "title",
            "Book-Author": "author",
            "Year-Of-Publication": "year",
            "Publisher": "publisher",
            "Image-URL-L": "img_url",
        }
    )
    ratings = ratings.rename(columns={"User-ID": "user_id", "Book-Rating": "rating"})

    user_counts = ratings["user_id"].value_counts()
    active_users = user_counts[user_counts > ACTIVE_USER_MIN_RATINGS].index
    ratings = ratings[ratings["user_id"].isin(active_users)]

    ratings_with_books = ratings.merge(books, on="ISBN")
    title_counts = (
        ratings_with_books.groupby("title")["rating"]
        .count()
        .reset_index()
        .rename(columns={"rating": "number_of_ratings"})
    )

    final_rating = ratings_with_books.merge(title_counts, on="title")
    final_rating = final_rating[final_rating["number_of_ratings"] >= POPULAR_TITLE_MIN_RATINGS]
    final_rating = final_rating.drop_duplicates(["user_id", "title"])

    book_pivot = final_rating.pivot_table(
        columns="user_id", index="title", values="rating"
    ).fillna(0)

    model = NearestNeighbors(algorithm="brute")
    model.fit(book_pivot)

    book_names = list(book_pivot.index)

    pickle.dump(model, open(ROOT / "NNeighbors.pkl", "wb"))
    pickle.dump(book_pivot, open(ROOT / "bookPivot.pkl", "wb"))
    pickle.dump(book_names, open(ROOT / "bookNames.pkl", "wb"))
    pickle.dump(final_rating, open(ROOT / "finalRating.pkl", "wb"))

    print(
        f"Wrote artifacts: bookPivot {book_pivot.shape}, "
        f"finalRating {final_rating.shape}, {len(book_names)} titles."
    )


if __name__ == "__main__":
    build_artifacts()
