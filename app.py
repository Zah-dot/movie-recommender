import streamlit as st
import base64
import requests
import time
from movie_recommender import (
    load_and_process_data,
    vectorize_features,
    build_similarity_matrix,
    get_indices_and_reverse_map,
    get_recommendations,
)


st.set_page_config(layout='wide')
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

def set_background(image_file):
    encoded_img = get_base64_image(image_file)
    css = f"""
<style>
.stApp {{
        background-image: url("data:image/jpg;base64,{encoded_img}");
background-size: 100%;
background-repeat: no-repeat;
background-attachment: scroll;
color: white;
}}
</style>
        """
    st.markdown(css, unsafe_allow_html=True)

set_background("assets/img3.jpeg")

st.title("🎬 Movie Recommender")

@st.cache_resource
def prepare_model():
    df = load_and_process_data()
    vector = vectorize_features(df)
    sim_matrix = build_similarity_matrix(vector)
    index_map, reverse_map = get_indices_and_reverse_map(df)
    return df, sim_matrix, index_map, reverse_map

movie_df, sim, idx_map, rev_map = prepare_model()

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
DEFAULT_POSTER_URL = "https://via.placeholder.com/300x450?text=No+Poster"
#@st.cache_data
def fetch_poster(title, retries=3):
    try:
        for attempt in range(retries):
            try:
                search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
                headers = {
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0"
                }
                response = requests.get(search_url, headers=headers, timeout=5)
                response.raise_for_status()
                data = response.json()

                if data.get("results"):
                    for result in data["results"]:
                        name = result.get("title") or result.get("name")
                        poster_path = result.get("poster_path")
                        if poster_path and name and title.lower() in name.lower():
                            return f"https://image.tmdb.org/t/p/w500{poster_path}"

                return DEFAULT_POSTER_URL

            except requests.exceptions.RequestException as e:
                print(f"[Attempt {attempt+1}] Error fetching poster for '{title}': {e}")
                time.sleep(1 + attempt)

        return DEFAULT_POSTER_URL

    except Exception as e:
        print(f"Final error fetching poster for '{title}': {e}")
        return DEFAULT_POSTER_URL


movie_titles = sorted(movie_df['title'].dropna().unique())
user_input = st.selectbox("Select a movie title:", movie_titles)

if st.button("Recommend"):
    with st.spinner("Fetching recommendations..."):
        results = get_recommendations(user_input, sim, idx_map, rev_map, movie_df)
        if isinstance(results, str):
            st.warning(results)
        else:
            st.subheader("Top Recommendations:")
            cols = st.columns(5)
            for i, (title, rating) in enumerate(results):
                with cols[i % 5]:
                    poster_url = fetch_poster(title)
                    st.image(poster_url, use_container_width=True)
                    st.markdown(f"**{title}**  \n⭐ {rating}")
