import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/model_kmeans_k7_metric0.03_date20250219.pkl")
DATA_PATH = os.path.join(BASE_DIR, "../data/raw/final_perfume_data.csv")

# Load the trained model and perfume dataset
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding='ISO-8859-1')

# Function to find the closest perfume name
def get_closest_perfume(user_input, perfume_names):
    match, score = process.extractOne(user_input, perfume_names)
    return match if score > 70 else None  # Accept matches with >70% similarity

# Function to recommend perfumes based on cluster similarity
def recommend_perfume(model, user_perfume, perfume_df):
    perfume_idx = perfume_df[perfume_df['Perfume Name'] == user_perfume].index
    if perfume_idx.empty:
        return []
    
    user_cluster = model.predict(perfume_df.iloc[perfume_idx, 1:])  # Exclude name column
    similar_perfumes = perfume_df[model.labels_ == user_cluster[0]]['Perfume Name'].tolist()
    return [p for p in similar_perfumes if p != user_perfume][:5]  # Return top 5 similar

# Streamlit UI
def main():
    st.title("Perfume Recommendation System")
    st.write("Find perfumes similar to your favorite scent based on fragrance clusters!")
    
    model = load_model()
    perfume_df = load_data()
    perfume_names = perfume_df['Name'].tolist()
    
    user_input = st.text_input("Enter a perfume you like:")
    
    if st.button("Find Similar Perfumes"):
        closest_perfume = get_closest_perfume(user_input, perfume_names)
        
        if closest_perfume:
            st.write(f"Closest match found: **{closest_perfume}**")
            recommendations = recommend_perfume(model, closest_perfume, perfume_df)
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
