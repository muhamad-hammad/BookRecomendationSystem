import pickle
import re

import numpy as np
import requests
import streamlit as st

PLACEHOLDER_IMG = "https://via.placeholder.com/120x180.png?text=No+Cover"


@st.cache_resource
def load_artifacts():
    model = pickle.load(open("NNeighbors.pkl", "rb"))
    final_rating = pickle.load(open("finalRating.pkl", "rb"))
    book_pivot = pickle.load(open("bookPivot.pkl", "rb"))
    return model, final_rating, book_pivot


model, final_rating, book_pivot = load_artifacts()
book_names = list(book_pivot.index)


def _isbn_from_url(url: str) -> str | None:
    if not isinstance(url, str):
        return None
    match = re.search(r"/P/([0-9X]+)\.", url)
    return match.group(1) if match else None


@st.cache_data(show_spinner=False)
def _url_ok(url: str) -> bool:
    try:
        resp = requests.head(url, timeout=2, allow_redirects=True)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def fetch_poster_safe(title: str) -> str:
    """Return a working cover URL for `title`, falling back to Open Library, then a placeholder."""
    matches = np.where(final_rating["title"] == title)[0]
    raw_url = final_rating.iloc[matches[0]]["img_url"] if len(matches) else None

    if isinstance(raw_url, str) and raw_url and _url_ok(raw_url):
        return raw_url

    isbn = _isbn_from_url(raw_url) if raw_url else None
    if isbn:
        ol_url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
        if _url_ok(ol_url):
            return ol_url

    return PLACEHOLDER_IMG


def fetch_posters(suggestions) -> list[str]:
    titles = [book_pivot.index[i] for i in suggestions[0]]
    return [fetch_poster_safe(t) for t in titles]


def recommend_book(book_name: str, n: int = 6):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    _, suggestions = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=n
    )
    titles = [book_pivot.index[i] for i in suggestions[0]]
    posters = fetch_posters(suggestions)
    return titles, posters


def recommend_from_many(selected_titles: list[str], n: int = 6):
    """Average the KNN neighborhoods of multiple liked books and return the top picks."""
    if not selected_titles:
        return [], []

    score_sum = np.zeros(book_pivot.shape[0])
    seed_ids = set()
    for title in selected_titles:
        idx = np.where(book_pivot.index == title)[0]
        if not len(idx):
            continue
        seed_ids.add(int(idx[0]))
        distances, suggestions = model.kneighbors(
            book_pivot.iloc[idx[0], :].values.reshape(1, -1),
            n_neighbors=min(n + len(selected_titles) + 5, book_pivot.shape[0]),
        )
        for d, i in zip(distances[0], suggestions[0]):
            score_sum[i] += 1.0 / (1.0 + d)

    ranked = np.argsort(-score_sum)
    picks = [int(i) for i in ranked if int(i) not in seed_ids][:n]
    titles = [book_pivot.index[i] for i in picks]
    posters = [fetch_poster_safe(t) for t in titles]
    return titles, posters


def render_grid(titles: list[str], posters: list[str]) -> None:
    cols = st.columns(min(5, len(titles)) or 1)
    for i, col in enumerate(cols):
        if i >= len(titles):
            break
        book = titles[i]
        poster = posters[i]
        google_search_url = f"https://www.google.com/search?q={book.replace(' ', '+')}+book"
        with col:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <a href="{google_search_url}" target="_blank">
                        <img src="{poster}" width="120" style="border-radius: 10px;"><br>
                        <p style="font-weight: bold; color: #333;">{book}</p>
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.set_page_config(page_title="Book Recommender", layout="wide")
st.title("📚 Book Recommendation System")
st.markdown("Get book suggestions based on your favorite reads using a machine learning model.")

with st.sidebar:
    st.header("💡 How It Works")
    st.write(
        """
    - Pick a single book or multi-select books you've liked
    - Click 'Show Recommendation'
    - Explore similar books with their covers
    """
    )

st.write(f"📘 **Total Books Available**: `{len(book_names)}`")

mode = st.radio("Recommendation mode", ["Single book", "Books I liked (multi-select)"], horizontal=True)

if mode == "Single book":
    selected_book = st.selectbox("🔍 Type or select a book", book_names)
    if st.button("🎯 Show Recommendation"):
        titles, posters = recommend_book(selected_book)
        st.subheader("📖 Recommended Books")
        render_grid(titles[1:], posters[1:])
else:
    selected_books = st.multiselect("🔍 Books you've liked", book_names)
    if st.button("🎯 Show Recommendation"):
        titles, posters = recommend_from_many(selected_books)
        if not titles:
            st.info("Select at least one book to get recommendations.")
        else:
            st.subheader("📖 Recommended Books")
            render_grid(titles, posters)
