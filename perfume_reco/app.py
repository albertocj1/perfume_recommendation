import os
import streamlit as st
import pandas as pd
import joblib
from fuzzywuzzy import process

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/model_kmeans_k7_metric0.03_date20250219.pkl")
DATA_PATH = os.path.join(BASE_DIR, "../data/raw/final_perfume_data.csv")

# Load the trained KMeans model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    
    # Ensure the dataset has a "Cluster" column
    if "Cluster" not in df.columns:
        raise ValueError("Error: The dataset must contain precomputed cluster labels!")
    
    return df

# Function to find the closest perfume name
def get_closest_perfume(user_input, perfume_names):
    match, score = process.extractOne(user_input, perfume_names)
    return match if score > 70 else None  # Accept matches with >70% similarity

# Function to recommend perfumes based on cluster similarity
def recommend_perfume(user_perfume, perfume_df):
    # Find the cluster of the selected perfume
    perfume_info = perfume_df[perfume_df['Name'] == user_perfume]
    
    if perfume_info.empty:
        return []
    
    user_cluster = perfume_info["Cluster"].values[0]
    
    # Find other perfumes in the same cluster
    similar_perfumes = perfume_df[perfume_df["Cluster"] == user_cluster]["Name"].tolist()
    
    return [p for p in similar_perfumes if p != user_perfume][:5]  # Return top 5

# Streamlit UI
def main():
    st.title("Perfume Recommendation System")
    st.write("Find perfumes similar to your favorite scent based on fragrance clusters!")

    # Load model and data
    model = load_model()
    perfume_df = load_data()
    perfume_names = perfume_df['Name'].tolist()

    user_input = st.text_input("Enter a perfume you like:")

    if st.button("Find Similar Perfumes"):
        closest_perfume = get_closest_perfume(user_input, perfume_names)

        if closest_perfume:
            st.write(f"Closest match found: **{closest_perfume}**")
            recommendations = recommend_perfume(closest_perfume, perfume_df)

            if recommendations:
                st.subheader("Recommended Perfumes:")
                for perfume in recommendations:
                    st.write(f"- {perfume}")
            else:
                st.write("No close matches found in the dataset.")
        else:
            st.write("No similar perfume found. Try another name!")

if __name__ == "__main__":
    main()
