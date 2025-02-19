import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/model_kmeans_k7_metric0.03_date20250219.pkl")
DATA_PATH = os.path.join(BASE_DIR, "../data/raw/final_perfume_data.csv")  # Using the original dataset

# Load the trained model
@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Notes"])  # Remove missing values in "Notes"
    df["Notes"] = df["Notes"].str.lower().str.replace(", ", ",")  # Normalize notes format
    return df

# Load model and data
model = load_model()
df = load_data()

# Extract unique fragrance notes from the dataset
all_notes = set()
for notes in df["Notes"]:
    all_notes.update(notes.split(","))  # Split by comma and add unique notes

all_notes = sorted(list(all_notes))  # Sort for better UI display

# Streamlit UI
st.title("Perfume Recommendation App")

st.write("""
This app recommends perfumes based on your fragrance preferences.  
Select the notes you love, and get personalized recommendations!
""")

# User selects fragrance notes
selected_notes = st.multiselect("Choose your favorite fragrance notes:", all_notes)

# Recommendation function
def recommend_perfume(df, selected_notes):
    if not selected_notes:
        return None

    # Convert perfume notes into a bag-of-words matrix
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(","))  # Tokenize by commas
    notes_matrix = vectorizer.fit_transform(df["Notes"])

    # Convert user-selected notes into the same format
    user_notes_str = ",".join(selected_notes)
    user_vector = vectorizer.transform([user_notes_str])

    # Compute cosine similarity
    similarities = cosine_similarity(user_vector, notes_matrix)[0]
    
    # Add similarity scores to dataframe
    df["Similarity"] = similarities

    # Get top 5 recommendations
    recommendations = df.sort_values(by="Similarity", ascending=False).head(5)
    return recommendations[["Name", "Brand", "Notes", "Similarity"]]

# Button to get recommendations
if st.button("Get Recommendation"):
    recommendations = recommend_perfume(df, selected_notes)
    
    if recommendations is None:
        st.warning("Please select at least one fragrance note!")
    else:
        st.subheader("Recommended Perfumes:")
        st.write(recommendations)

