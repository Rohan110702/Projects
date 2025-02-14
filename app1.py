import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset
@st.cache_data
def load_data():
    filepath =  "tmdb_5000_movies.csv"
    movies = pd.read_csv(filepath)
    movies = movies[['title', 'genres', 'overview', 'vote_average', 'popularity']]
    movies['genres'] = movies['genres'].apply(lambda x: ", ".join([genre['name'] for genre in ast.literal_eval(x)]))
    movies.dropna(subset=['overview'], inplace=True)
    return movies

movies = load_data()

# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# Streamlit app
st.title("Movie Recommendation System")
st.subheader("Enter a movie title to get recommendations:")

# User input
movie_title = st.text_input("Movie Title")

if movie_title:
    # Convert user input to lowercase for case-insensitive search
    movie_title_lower = movie_title.lower()
    # Check if the movie exists in the dataset (case-insensitive)
    matches = movies[movies['title'].str.lower() == movie_title_lower]
    if not matches.empty:
        st.write(f"Found movie: {matches.iloc[0]['title']}")
        recommendations = get_recommendations(matches.iloc[0]['title'], cosine_sim)
        
        st.write("Here are some movies you might like:")
        for index, row in recommendations.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"Genres: {row['genres']}")
            st.write(f"Rating: {row['vote_average']} | Popularity: {row['popularity']}")
            st.write("---")
    else:
        st.write("Movie not found in the dataset. Please try another title.")
