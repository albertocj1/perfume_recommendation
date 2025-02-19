import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/model_kmeans_k7_metric0.03_date20250219.pkl")
DATA_PATH = os.path.join(BASE_DIR, "../data/raw/perfume_data_encoded.csv")  # Use the encoded dataset

# Load the trained model
@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH)

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# Load model and data
model = load_model()
df = load_data()

# Extract available fragrance notes (assuming the first 5 columns are metadata)
available_notes = list(df.columns[5:])

# Streamlit UI
st.title("Perfume Recommendation App")

st.write("""
This app recommends perfumes based on your fragrance preferences.  
Select the notes you love, and get personalized recommendations!
""")

# User selects fragrance notes
selected_notes = st.multiselect("Choose your favorite fragrance notes:", available_notes)

# Convert user selection into a feature vector
user_vector = pd.DataFrame(0, index=[0], columns=available_notes)
for note in selected_notes:
    user_vector[note] = 1

# Recommendation function
def recommend_perfume(model, user_data):
    # Predict cluster
    cluster = model.predict(user_data)[0]

    # Get perfumes from the same cluster
    cluster_perfumes = df[df["Cluster"] == cluster]  # Assumes a "Cluster" column exists

    # Compute similarity scores
    similarities = cosine_similarity(user_data, cluster_perfumes[available_notes])
    cluster_perfumes["Similarity"] = similarities[0]

    # Sort by similarity and return top 5
    recommendations = cluster_perfumes.sort_values(by="Similarity", ascending=False).head(5)
    return recommendations[["Name", "Brand", "Notes", "Similarity"]]

# Button to get recommendations
if st.button("Get Recommendation"):
    if not selected_notes:
        st.warning("Please select at least one fragrance note!")
    else:
        recommendations = recommend_perfume(model, user_vector)
        st.subheader("Recommended Perfumes:")
        st.write(recommendations)

