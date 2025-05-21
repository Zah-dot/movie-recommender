import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack
import os
import gdown

WEIGHT = {'overview': 1, 'keyword': 1, 'cast': 1, 'dir': 1, 'genre': 1}
stemmer = PorterStemmer()

def stem_tokenizer(text):
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(token) for token in tokens if token.isalpha()]

def extract_names_regex(text):
    if pd.isna(text) or text.strip() == '':
        return []
    return re.findall(r"'name':\s*'([^']*)'", text)

def extract_top5_names(text):
    names = extract_names_regex(text)
    return names[:5]

def extract_director_regex(text):
    if pd.isna(text) or text.strip() == '':
        return ''
    job_pattern = r"'job':\s*'([^']*)'"
    name_pattern = r"'name':\s*'([^']*)'"
    jobs = re.findall(job_pattern, text)
    names = re.findall(name_pattern, text)
    for job, name in zip(jobs, names):
        if job == 'Director':
            return name
    return ''

def clean_id_column(df, id_col='id'):
    df = df.dropna(subset=[id_col])
    df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
    df = df.dropna(subset=[id_col])
    df[id_col] = df[id_col].astype(int)
    return df

def load_and_process_data():
    movie_id = "1jLZxVKH767pZ9vtepykKeoNU1zQB6aGi"
    credits_id = "1l49fqxk7pVtNWnbCffgkwwTVmSPeFXAD"
    keywords_id = "1f_7Syu6n1N_L-3OIRf75tYyttrkxL-hI"
    # Local cache paths
    movie_path = "movies_metadata.csv"
    credits_path = "credits.csv"
    keywords_path = "keywords.csv"
    # Download if not present
    if not os.path.exists(movie_path):
        gdown.download(f"https://drive.google.com/uc?id={movie_id}", movie_path, quiet=False)
    if not os.path.exists(credits_path):
        gdown.download(f"https://drive.google.com/uc?id={credits_id}", credits_path, quiet=False)
    if not os.path.exists(keywords_path):
        gdown.download(f"https://drive.google.com/uc?id={keywords_id}", keywords_path, quiet=False)

    movie = pd.read_csv(movie_path, low_memory=False)
    credit = pd.read_csv(credits_path, low_memory=False)
    keyword = pd.read_csv(keywords_path, low_memory=False)

    movie = clean_id_column(movie)
    credit = clean_id_column(credit)
    keyword = clean_id_column(keyword)

    movie = movie.merge(credit, how='left', on='id')
    movie = movie.merge(keyword, how='left', on='id')

    movie['c'] = movie['cast'].apply(extract_top5_names).apply(lambda x: ' '.join(name.replace(' ', '_') for name in x))
    movie['k'] = movie['keywords'].apply(extract_names_regex).apply(lambda x: ' '.join(stem_tokenizer(' '.join(x))))
    movie['d'] = movie['crew'].apply(extract_director_regex).apply(lambda x: x.replace(' ', '_'))
    movie['g'] = movie['genres'].apply(extract_names_regex).apply(lambda x: ' '.join(x))
    movie['o'] = movie['overview'].fillna('').apply(lambda x: ' '.join(stem_tokenizer(x)))


    movie = movie.drop_duplicates(subset=['title', 'd'], keep='first')
    movie = movie[movie['vote_count'].fillna(0).astype(float) > 100]

    movie = movie.reset_index(drop=True)
    return movie

def vectorize_features(movie_df):
    tfid_overview = TfidfVectorizer(
        max_features=5000, stop_words='english',
        ngram_range=(1, 2), min_df=3, max_df=0.8,
        strip_accents='unicode', lowercase=True
    )
    tfid_cast = TfidfVectorizer(max_features=1000, token_pattern=r'(?u)\b\w+\b', lowercase=False)
    tfid_crew = TfidfVectorizer(max_features=300, lowercase=False)
    tfid_genre = TfidfVectorizer(max_features=20)
    tfid_keyword = TfidfVectorizer(
        max_features=1000, stop_words='english',
        strip_accents='unicode', lowercase=True
    )

    v_overview = tfid_overview.fit_transform(movie_df['o'])
    v_cast = tfid_cast.fit_transform(movie_df['c'])
    v_dir = tfid_crew.fit_transform(movie_df['d'])
    v_genre = tfid_genre.fit_transform(movie_df['g'])
    v_key = tfid_keyword.fit_transform(movie_df['k'])

    v_total = hstack([
        v_overview * WEIGHT['overview'],
        v_cast * WEIGHT['cast'],
        v_dir * WEIGHT['dir'],
        v_genre * WEIGHT['genre'],
        v_key * WEIGHT['keyword'],
    ])

    v_total = normalize(v_total)

    movie_df.drop(columns=['cast', 'crew', 'genres', 'keywords', 'overview'], inplace=True, errors='ignore')
    movie_df.reset_index(drop=True, inplace=True)
    return v_total

def build_similarity_matrix(vectors):
    return linear_kernel(vectors)

def get_indices_and_reverse_map(movie_df):
    movie_df = movie_df.reset_index(drop=True)
    index_map = pd.Series(movie_df.index, index=movie_df['title'].str.lower()).to_dict()
    reverse_map = dict(enumerate(movie_df['title']))
    return index_map, reverse_map

def get_recommendations(title, similarity_matrix, index_map, reverse_map, movie_df):
    idx = index_map.get(title.lower())
    if idx is None:
        return f"'{title}' not found in database."
    if idx >= similarity_matrix.shape[0]:
        return f"Data inconsistency error: index {idx} out of bounds."

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    recommended_indices = [i[0] for i in sim_scores]

    recommendations = []
    for i in recommended_indices:
        if i in reverse_map:
            movie_title = reverse_map[i]
            vote_avg = movie_df.iloc[i]['vote_average']
            recommendations.append((movie_title, vote_avg))

    return recommendations    