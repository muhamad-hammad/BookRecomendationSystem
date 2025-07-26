import pickle
import streamlit as st
import numpy as np

# Loading models and data
model = pickle.load(open('NNeighbours.pkl', 'rb'))
book_names = pickle.load(open('bookNames.pkl', 'rb'))
final_rating = pickle.load(open('final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('bookPivot.pkl', 'rb'))

# App title and description
st.set_page_config(page_title="Book Recommender", layout="wide")
st.title('📚 Book Recommendation System')
st.markdown("Get book suggestions based on your favorite reads using a machine learning model.")

# Sidebar
with st.sidebar:
    st.header("💡 How It Works")
    st.write("""
    - Select a book from the dropdown
    - Click 'Show Recommendation'
    - Explore similar books with their covers
    """)

st.write(f"📘 **Total Books Available**: `{len(book_names)}`")

# Function to fetch poster URLs
def fetch_posters(suggestions):
    book_titles = [book_pivot.index[book_id] for book_id in suggestions[0]]
    id_indices = [np.where(final_rating['title'] == name)[0][0] for name in book_titles]
    urls = [final_rating.iloc[idx]['img_url'] for idx in id_indices]
    return urls

# Book recommendation logic
def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    _, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    recommended_books = [book_pivot.index[i] for i in suggestions[0]]
    poster_urls = fetch_posters(suggestions)
    return recommended_books, poster_urls

# Book selection
selected_book = st.selectbox("🔍 Type or select a book", book_names)

# Showing recommendations
if st.button("🎯 Show Recommendation"):
    recommended_books, poster_urls = recommend_book(selected_book)
    st.subheader("📖 Recommended Books")

    # Displaying recommendations in columns
    cols = st.columns(5)
    for i, col in enumerate(cols):
        if i + 1 < len(recommended_books):
            book = recommended_books[i + 1]
            poster = poster_urls[i + 1]
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
                    unsafe_allow_html=True
                )
