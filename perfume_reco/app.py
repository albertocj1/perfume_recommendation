import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/model_kmeans_k7_metric0.03_date20250219.pkl")
DATA_PATH = os.path.join(BASE_DIR, "../data/raw/final_perfume_data.csv")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load perfume data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# Function to recommend perfumes
def recommend_perfumes(model, data, user_input):
    # Get the cluster of user input
    cluster = model.predict(user_input)[0]
    
    # Filter perfumes in the same cluster
    cluster_perfumes = data[data['Cluster'] == cluster]
    
    # Compute similarity scores
    notes_matrix = cluster_perfumes['notes'].apply(lambda x: x.split())  # Convert to lists
    notes_matrix = notes_matrix.apply(lambda x: ' '.join(x))  # Convert back to strings for vectorization
    vectorized_notes = cluster_perfumes['notes'].str.get_dummies(sep=' ')  # One-hot encode notes
    
    user_vector = user_input.dot(vectorized_notes.T)  # Compute similarity
    cluster_perfumes['Similarity'] = cosine_similarity(user_vector, vectorized_notes)[0]
    
    # Sort by highest similarity
    recommendations = cluster_perfumes.sort_values(by="Similarity", ascending=False).head(5)
    
    return recommendations[['name', 'brand', 'notes', 'Similarity']]

# Streamlit UI
def main():
    st.title("Perfume Recommendation App")
    
    st.write("""
    This app recommends perfumes based on your scent preferences.  
    Adjust the sliders to describe your ideal fragrance, and get recommendations!
    """)

    # User input section for fragrance preferences
    floral = st.slider("Floral (0-10)", 0, 10, 5)
    woody = st.slider("Woody (0-10)", 0, 10, 5)
    citrus = st.slider("Citrus (0-10)", 0, 10, 5)
    spicy = st.slider("Spicy (0-10)", 0, 10, 5)
    musky = st.slider("Musky (0-10)", 0, 10, 5)
    fresh = st.slider("Fresh (0-10)", 0, 10, 5)
    sweet = st.slider("Sweet (0-10)", 0, 10, 5)

    # Convert user input to model input format
    user_input = pd.DataFrame([{
        'floral': floral,
        'woody': woody,
        'citrus': citrus,
        'spicy': spicy,
        'musky': musky,
        'fresh': fresh,
        'sweet': sweet
    }])

    # Load model and data
    model = load_model()
    data = load_data()

    if st.button("Get Recommendation"):
        recommendations = recommend_perfumes(model, data, user_input)
        
        st.subheader("Recommended Perfume(s):")
        st.dataframe(recommendations)

if __name__ == "__main__":
    main()
