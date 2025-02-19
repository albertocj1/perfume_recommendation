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

# Load perfume dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding='ISO-8859-1')

# Find the closest perfume name using fuzzy matching
def get_closest_perfume(user_input, perfume_names):
    match, score = process.extractOne(user_input, perfume_names)
    return match if score > 70 else None  # Accept matches with >70% similarity

# Recommend perfumes based on cosine similarity
def recommend_perfume(perfume_df, user_perfume):
    if user_perfume not in perfume_df["Name"].values:
        return []

    # Select only numeric columns for similarity comparison
    feature_columns = perfume_df.select_dtypes(include=["number"]).columns.tolist()
    
    if not feature_columns:
        raise ValueError("No numerical feature columns found in dataset!")

    # Compute cosine similarity
    feature_matrix = perfume_df[feature_columns].values
    similarity_matrix = cosine_similarity(feature_matrix)

    # Get index of user-selected perfume
    perfume_idx = perfume_df.index[perfume_df["Name"] == user_perfume].tolist()[0]

    # Find top 5 most similar perfumes
    similar_indices = similarity_matrix[perfume_idx].argsort()[::-1][1:6]
    recommendations = perfume_df.iloc[similar_indices]["Name"].tolist()

    return recommendations

# Streamlit UI
def main():
    st.title("Perfume Recommendation System")
    st.write("Find perfumes similar to your favorite scent using AI!")

    # Load model and data
    model = load_model()
    perfume_df = load_data()
    perfume_names = perfume_df["Name"].tolist()

    # User input
    user_input = st.text_input("Enter a perfume you like:")

    if st.button("Find Similar Perfumes"):
        closest_perfume = get_closest_perfume(user_input, perfume_names)

        if closest_perfume:
            st.write(f"Closest match found: **{closest_perfume}**")
            recommendations = recommend_perfume(perfume_df, closest_perfume)
            
            if recommendations:
                st.subheader("Recommended Perfumes:")
                for perfume in recommendations:
                    st.write(f"- {perfume}")
            else:
                st.write("No similar perfumes found in the dataset.")
        else:
            st.write("No similar perfume found. Try another name!")

if __name__ == "__main__":
    main()
