# Movie Recommender System

This is a content-based movie recommendation web application built using Streamlit. It recommends movies based on multiple metadata features such as overview, cast, crew, genres, and keywords by computing similarity scores using TF-IDF vectorization and cosine similarity.

## Features

- **Content-Based Filtering:** Uses movie metadata to recommend similar movies.
- **Multi-Feature Similarity:** Combines overview, cast, crew, genres, and keywords for richer recommendations.
- **Weighted Feature Vectorization:** Different weights assigned to features to improve recommendation relevance.
- **Interactive Web UI:** Built with Streamlit for easy and fast user interaction.
- **Poster Display:** Fetches movie posters dynamically via TMDB API to enhance user experience.

## Technologies Used

- **Python 3** — Core programming language
- **Streamlit** — For building the interactive web application
- **Pandas & NumPy** — Data manipulation and numerical operations
- **Scikit-learn** — TF-IDF vectorization and cosine similarity calculations
- **TMDB API** — To fetch movie poster images dynamically
- **Git & GitHub** — Version control and project hosting

## Project Structure

- `app.py` — Main Streamlit application script
- `data/` — Movie metadata CSV files (e.g., `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`)
- `utils.py` (optional) — Helper functions for data processing and API calls
- `.gitignore` — To exclude secret files like API keys
- `requirements.txt` — Python dependencies for deployment

## Usage

1. Clone the repository.
2. Install dependencies from `requirements.txt` using `pip install -r requirements.txt`.
3. Set your TMDB API key as a Streamlit secret or environment variable.
4. Run the app locally with `streamlit run app.py`.
5. Enter a movie title to get similar movie recommendations with posters.

## Future Improvements

- Add fuzzy matching for more flexible movie title input.
- Implement year-based filtering to improve recommendation relevance.
- Cache poster images to reduce API calls and speed up loading.
- Extend the recommender to include collaborative filtering or hybrid methods.

---

Feel free to explore, use, and contribute to this project!

---

**Author:** Zaid Hassan
**Date:** May,2025
