import streamlit as st
import pandas as pd
import pickle
import os
import time

try:
    import numpy as np
    # Force numpy array import
    np._import_array = lambda: None
    from surprise import Dataset, SVD, Reader
except ImportError as e:
    st.error("Please install compatible versions: pip install numpy==1.21.6 scikit-surprise==1.1.1")
    st.stop()

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e50914;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .rating-badge {
        background-color: #f39c12;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .stSelectbox > div > div > select {
        background-color: #2c3e50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_preprocessed_data():
    try:
        with open('Model/preprocessed_data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        movies_data = data['movies_data']
        movie_com = data['movie_com']
        ratings = data['ratings']
        links = data['links']
        
        st.success("✅ Loaded preprocessed data")
        return movies_data, movie_com, ratings, links
        
    except FileNotFoundError:
        st.warning("Preprocessed data not found, loading from CSV files...")
        try:
            movies_data = pd.read_csv("Data/NewMoviesMetadata.csv")
            movie_com = pd.read_csv("Data/MovieBasedRecommenderData.csv")
            ratings = pd.read_csv("Data/ratings_small.csv")
            links = pd.read_csv("Data/links.csv").dropna()
            links["tmdbId"] = links["tmdbId"].astype("int64")
            
            return movies_data, movie_com, ratings, links
        except FileNotFoundError as e:
            st.error(f"Error loading data files: {e}")
            st.info("Please make sure the model files or CSV files are in the same directory as this app.")
            return None, None, None, None

@st.cache_resource
def load_trained_models():
    try:
        # Load SVD model
        with open('model/svd_model.pkl', 'rb') as f:
            svd = pickle.load(f)
        temp_message(st.success, "✅ Loaded pre-trained SVD model", delay=10)

        # Load cosine similarity matrix
        with open('model/cosine_similarity.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)
        temp_message(st.success, "✅ Loaded cosine similarity matrix", delay=10)

        # Load movie indices
        with open('model/movie_indices.pkl', 'rb') as f:
            indices = pickle.load(f)
        temp_message(st.success, "✅ Loaded movie indices", delay=10)


        return svd, cosine_sim, indices
        
    except FileNotFoundError as e:
        st.error(f"Pre-trained models not found: {e}")
        st.info("Please run the notebook first to generate the model files, or the app will train models from scratch.")
        return None, None, None

@st.cache_resource
def train_svd_model(ratings):
    st.warning("Training SVD model from scratch (this may take a while)...")
    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    svd = SVD()
    svd.fit(trainset)
    st.success("✅ SVD model trained")
    return svd

@st.cache_resource
def create_tfidf_matrix(movie_com):
    st.warning("Creating TF-IDF matrix from scratch (this may take a while)...")
    movie_com["model_feature"] = movie_com["model_feature"].fillna("")
    
    tf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english"
    )
    
    tfidf_matrix = tf.fit_transform(movie_com["model_feature"])
    cosine_sim = linear_kernel(tfidf_matrix[:15000], tfidf_matrix)
    st.success("✅ TF-IDF matrix created")
    return cosine_sim

def hybrid_recommend(userId, title, svd, cosine_sim, movies_data, movie_com, links, indices=None):
    if indices is None:
        indices = pd.Series(movies_data.index, index=movies_data['title'])
    
    try:
        idx = indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        
        sim_scores = list(enumerate(cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]
        movie_indices = [i[0] for i in sim_scores]
        
        movies = movie_com.iloc[movie_indices][["title", "id"]]
        tempLinks = links[links["tmdbId"].isin(movies["id"].tolist())]
        
        preds = []
        for item in tempLinks["movieId"]:
            try:
                pred = svd.predict(userId, item).est
                preds.append(pred)
            except:
                preds.append(2.5) 
        
        if len(preds) > 0:
            tempData = pd.DataFrame({
                "title": movies["title"].iloc[:len(preds)],
                "est": preds,
                "id": movies["id"].iloc[:len(preds)]
            })
            
            tempData = tempData.sort_values("est", ascending=False)
            tempData = tempData[tempData["est"] >= 2.5]
            
            return tempData.head(10)
        else:
            return pd.DataFrame(columns=["title", "est", "id"])
            
    except Exception as e:
        st.error(f"Error in hybrid recommendation: {e}")
        return pd.DataFrame(columns=["title", "est", "id"])

def get_user_recommendations(userID, svd, cosine_sim, movies_data, movie_com, ratings, links, indices=None):
    if indices is None:
        indices = pd.Series(movies_data.index, index=movies_data['title'])
    
    tempRatings = ratings[ratings["userId"] == userID]
    tempRatings = tempRatings[tempRatings["rating"] >= 4]
    
    if tempRatings.empty:
        return pd.DataFrame(columns=["title", "est", "id"])
    
    tempLinks = links[links["movieId"].isin(tempRatings["movieId"].tolist())]
    titlesData = movies_data[movies_data["id"].isin(tempLinks["tmdbId"])]
    
    resultDataFrame = pd.DataFrame()
    
    for title in titlesData["title"].head(5):
        try:
            recommendations = hybrid_recommend(userID, title, svd, cosine_sim, movies_data, movie_com, links, indices)
            if not recommendations.empty:
                resultDataFrame = pd.concat([recommendations, resultDataFrame], ignore_index=True)
        except:
            continue
    
    if not resultDataFrame.empty:
        return resultDataFrame.sort_values("est", ascending=False).drop_duplicates(subset=["title"]).head(10)
    else:
        return pd.DataFrame(columns=["title", "est", "id"])

def temp_message(message_func, text, delay=30):
    placeholder = st.empty()
    getattr(placeholder, message_func.__name__)(text)
    time.sleep(delay)
    placeholder.empty()




def main():
    st.markdown('<h1 class="main-header">🎬 CineMatch</h1>', unsafe_allow_html=True)
    st.markdown("### Discover your next favorite movie with our hybrid recommendation engine!")
    
    # Load data
    with st.spinner("Loading movie database..."):
        movies_data, movie_com, ratings, links = load_preprocessed_data()
    
    if movies_data is None:
        st.stop()
    
    with st.spinner("Loading trained models..."):
        svd, cosine_sim, indices = load_trained_models()
    
    # If models not found, train from scratch
    if svd is None or cosine_sim is None:
        st.warning("🔄 Pre-trained models not found. Training from scratch...")
        with st.spinner("Training recommendation models (this may take a few minutes)..."):
            svd = train_svd_model(ratings)
            cosine_sim = create_tfidf_matrix(movie_com)
            indices = pd.Series(movies_data.index, index=movies_data['title'])
    
    st.success("🚀 Ready to recommend movies!")
    
    st.sidebar.title("🎯 Recommendation Options")
    recommendation_type = st.sidebar.radio(
        "Choose recommendation type:",
        ["Movie-based Recommendations", "User-based Recommendations"]
    )
    
    if recommendation_type == "Movie-based Recommendations":
        st.markdown('<h2 class="sub-header">🎭 Get Recommendations Based on a Movie</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Movie selection
            movie_titles = movies_data['title'].unique()
            selected_movie = st.selectbox(
                "Select a movie you enjoyed:",
                options=[""] + list(movie_titles),
                index=0
            )
            
            user_id = st.number_input(
                "Enter your User ID (1-671):",
                min_value=1,
                max_value=671,
                value=1,
                step=1
            )
        
        with col2:
            st.info("💡 **How it works:**\n\n"
                   "1. Select a movie you liked\n"
                   "2. Enter your user ID\n"
                   "3. Get personalized recommendations!")
        
        if selected_movie and st.button("🔍 Get Recommendations", type="primary"):
            with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
                recommendations = hybrid_recommend(
                    user_id, selected_movie, svd, cosine_sim, 
                    movies_data, movie_com, links, indices
                )
            
            if not recommendations.empty:
                st.success(f"Found {len(recommendations)} recommendations for you!")
                
                # Display recommendations
                for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="movie-card">
                            <h3>#{idx} {row['title']}</h3>
                            <span class="rating-badge">Predicted Rating: {row['est']:.2f}/5</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try selecting a different movie.")
    
    else:  # User-based Recommendations
        st.markdown('<h2 class="sub-header">👤 Get Recommendations Based on Your Profile</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_id = st.number_input(
                "Enter your User ID:",
                min_value=1,
                max_value=671,
                value=1,
                step=1,
                help="This will analyze your rating history to suggest new movies"
            )
        
        with col2:
            st.info("💡 **Profile-based recommendations:**\n\n"
                   "Based on your rating history, we'll find movies similar to ones you've enjoyed!")
        
        if st.button("🎯 Get My Recommendations", type="primary"):
            with st.spinner("Analyzing your movie preferences..."):
                recommendations = get_user_recommendations(
                    user_id, svd, cosine_sim, movies_data, 
                    movie_com, ratings, links, indices
                )
            
            if not recommendations.empty:
                st.success(f"Found {len(recommendations)} personalized recommendations!")
                
                user_ratings = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4)]
                if not user_ratings.empty:
                    user_movies = user_ratings.merge(
                        links[["movieId", "tmdbId"]], on="movieId"
                    ).merge(
                        movies_data[["id", "title"]], left_on="tmdbId", right_on="id"
                    )
                    
                    st.markdown("### 📚 Movies you've enjoyed:")
                    with st.expander("View your highly rated movies"):
                        for _, movie in user_movies.head(5).iterrows():
                            st.write(f"⭐ {movie['title']} - Rating: {movie['rating']}/5")
                
                st.markdown("### 🎬 Recommended for you:")
                
                for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="movie-card">
                            <h3>#{idx} {row['title']}</h3>
                            <span class="rating-badge">Predicted Rating: {row['est']:.2f}/5</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. This user might not have enough rating history.")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🤖 Powered by Hybrid Recommendation System | Collaborative Filtering + Content-Based Filtering</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()