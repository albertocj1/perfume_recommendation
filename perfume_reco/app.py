import streamlit as st
import pandas as pd
import joblib
import os

# Define paths to the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../models/model_kmeans_k7_metric0.03_date20250217.pkl"))

def load_model():
    return joblib.load(MODEL_PATH)

# Load model
kmeans = load_model()

# Load perfume dataset
df = pd.read_csv(os.path.abspath(os.path.join(BASE_DIR, "../data/perfume_data.csv")))
df["Notes"].fillna("", inplace=True)

# Assign clusters (assuming X is precomputed and stored in the dataset)
df["Cluster"] = kmeans.predict(df["Notes"])

def recommend_perfumes(selected_perfume):
    if selected_perfume not in df["Name"].values:
        return []
    
    cluster = df[df["Name"] == selected_perfume]["Cluster"].values[0]
    cluster_perfumes = df[df["Cluster"] == cluster]
    recommendations = cluster_perfumes[cluster_perfumes["Name"] != selected_perfume]
    
    return recommendations.head(5)

def main():
    st.title("Perfume Recommendation System")
    
    selected_perfume = st.selectbox("Select a perfume:", df["Name"].unique())
    
    if st.button("Recommend"):
        recommendations = recommend_perfumes(selected_perfume)
        if not recommendations.empty:
            for _, row in recommendations.iterrows():
                st.image(row["Image URL"], width=100)
                st.write(f"**{row['Name']}** by {row['Brand']}")
                st.write(f"Notes: {row['Notes']}")
                st.write("---")
        else:
            st.write("No recommendations found.")

if __name__ == "__main__":
    main()
