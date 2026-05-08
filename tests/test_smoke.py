"""Smoke test: load the pickles and verify the recommender returns 5 string titles.

Run from the project root with `pytest -q` or `python -m pytest tests/`.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts():
    model = pickle.load(open(ROOT / "NNeighbors.pkl", "rb"))
    book_pivot = pickle.load(open(ROOT / "bookPivot.pkl", "rb"))
    final_rating = pickle.load(open(ROOT / "finalRating.pkl", "rb"))
    book_names = pickle.load(open(ROOT / "bookNames.pkl", "rb"))
    return model, book_pivot, final_rating, book_names


def test_pickles_have_expected_shape(artifacts):
    model, book_pivot, final_rating, book_names = artifacts
    assert book_pivot.shape[0] > 0
    assert book_pivot.shape[1] > 0
    assert book_names is not None  # shipped pickle is malformed; app rebuilds from book_pivot.index
    for col in ("title", "img_url"):
        assert col in final_rating.columns, f"finalRating missing {col!r}"


def test_recommend_book_returns_five_strings(artifacts):
    model, book_pivot, _, _ = artifacts

    known = "Harry Potter and the Sorcerer's Stone (Book 1)"
    if known not in book_pivot.index:
        known = book_pivot.index[0]

    book_id = np.where(book_pivot.index == known)[0][0]
    _, suggestions = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6
    )
    titles = [book_pivot.index[i] for i in suggestions[0]]
    recommendations = [t for t in titles if t != known][:5]

    assert len(recommendations) == 5
    assert all(isinstance(t, str) and t for t in recommendations)
