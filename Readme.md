## 📚 Book Recommendation System

Get personalized book recommendations based on your favorite reads using a machine learning model – all through a sleek and interactive **Streamlit** interface.

---

### 🚀 Features

* 🔍 **Search & Select a Book** from a dropdown list
* ❤️ **Multi-Select "Books I Liked"** mode that averages KNN neighborhoods across several titles
* 🎯 **Get Similar Recommendations** using KNN-based similarity
* 📘 **View Book Covers** with clickable Google search links
* 🛟 **Resilient Cover Fetching** with Open Library fallback and placeholder when URLs rot
* 🧠 **Built with Machine Learning**, Pandas, and Scikit-learn

---

### 🧠 How It Works

1. **User picks a single book** or **multi-selects books they've liked**
2. A **KNN model** finds similar books based on user-item ratings (multi-select mode sums `1 / (1 + distance)` scores across all seeds)
3. Similar books are displayed with their **titles and cover images**
4. Each recommendation links to a **Google search** for more info

---

### 📂 Project Structure

```
├── app.py                 # Streamlit frontend
├── train.py               # Reproducible training script (regenerates the .pkl files)
├── main.ipynb             # Original training notebook
├── NNeighbors.pkl         # Trained scikit-learn NearestNeighbors model
├── bookPivot.pkl          # Title × user_id rating pivot table (used as the KNN feature matrix)
├── bookNames.pkl          # List of book titles (== bookPivot.index)
├── finalRating.pkl        # Filtered rating frame with cover URLs (img_url, title, author, …)
├── Books.csv              # Raw books dataset
├── Ratings.csv            # Raw ratings dataset
├── Users.csv              # Raw users dataset
├── tests/                 # Smoke tests (pickle-load + recommendation regression)
├── Dockerfile             # Container image for deployment
├── .dockerignore          # Excludes .git, notebook, etc. from the image
├── Procfile               # Heroku-style startup command
├── .streamlit/config.toml # Streamlit theme + server config
├── requirements.txt       # Pinned dependencies
├── TODO.md                # Outstanding work backlog
└── Readme.md              # This file
```

---

### 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn (KNN)**
* **Streamlit** – for the interactive web app
* **Pickle** – to save/load ML models

---
📸 Sample Screenshot
![alt text](image.png)
---

### ▶️ Getting Started

1. **Clone this repo**

```bash
git clone https://github.com/yourusername/book-recommendation-system.git
cd book-recommendation-system
```

2. **Install requirements**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

4. **(Optional) Retrain the model** to regenerate the four `.pkl` artifacts from the raw CSVs:

```bash
python train.py
```

5. **(Optional) Run smoke tests**

```bash
pytest
```

---

### 🐳 Docker

```bash
docker build -t book-recommender .
docker run -p 8501:8501 book-recommender
```

Then visit [http://localhost:8501](http://localhost:8501).

---

### 💡 Future Improvements

* 🧾 Add user ratings and review analysis
* 🌐 Add real-time API-based book info
* 🔎 Improve cover quality with official book cover APIs (e.g. Open Library editions endpoint)

---

### 🙌 Acknowledgements

* Dataset inspired by public sources (e.g. Book-Crossing, Goodreads)
* UI built with 💖 using Streamlit

---

### 📬 Contact

Made with ❤️ by **Muhammad Hammad**
📧 Email: [m.hammad.bhatti09@gmail.com](mailto:m.hammad.bhatti09@gmail.com)
