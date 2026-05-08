# TODO — Book Recommendation System

> See [CLAUDE.md](CLAUDE.md) for context. Tasks below are ordered roughly by priority.

## Cleanup
- [ ] Pin versions in `requirements.txt` (`pip freeze` from a working venv → `streamlit`, `pandas`, `scikit-learn`, `numpy`)
- [ ] Drop `pickle-mixin` from `requirements.txt` (stdlib `pickle` is sufficient)
- [ ] Fix `Readme.md` file list — currently lists `model.pkl`, `book_pivot.pkl`, `fetch_posters.py` which don't exist; real files are `NNeighbors.pkl`, `bookPivot.pkl`, `bookNames.pkl`, `finalRating.pkl`

## Reproducibility
- [ ] Extract notebook training logic into `train.py` that reads CSVs, builds the pivot, fits KNN, writes the four `.pkl` files
- [ ] Document pickle schemas (especially columns of `finalRating`) in a `docs/data.md` or in `train.py` docstring

## Robustness
- [ ] Wrap `fetch_posters` in `fetch_poster_safe()` that returns a placeholder image URL when `img_url` is null/unreachable
- [ ] Add `tests/test_smoke.py`: load all pickles, run `recommend_book("<known title>")`, assert 5 string results

## Features (optional)
- [ ] Multi-select "books I liked" → average KNN neighborhoods across the selections
- [ ] Switch poster source to Open Library Covers API for stability

## Deployment
- [ ] Add `Dockerfile` (`python:3.11-slim` → `pip install -r requirements.txt` → `CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0`)
- [ ] Add `.streamlit/config.toml` for theme + server config
- [ ] (Optional) Streamlit Community Cloud config / `Procfile` for Heroku-style hosts
