import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# Define paths to the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../models/model_kmeans_k7_metric0.03_date20250219.pkl"))

def load_model():
    model_data = joblib.load(MODEL_PATH)
    return model_data['vectorizer'], model_data['kmeans']

# Load model components
vectorizer, kmeans = load_model()

# Load perfume dataset
df = pd.read_csv(os.path.abspath(os.path.join(BASE_DIR, "../data/raw/final_perfume_data.csv")), encoding='ISO-8859-1')
df["Notes"].fillna("", inplace=True)

# Convert Notes column to numerical representation
X = vectorizer.transform(df["Notes"])

# Assign clusters
df["Cluster"] = kmeans.predict(X)

# Cluster interpretation
def get_cluster_description(cluster_id):
    cluster_descriptions = {
        0: "Fresh & Floral Woody Musk",
        1: "Oriental & Spicy Woody",
        2: "Earthy & Citrus Woody",
        3: "Smoky & Resinous Oriental",
        4: "Fruity Floral Musk",
        5: "Classic Green & Woody Floral",
        6: "Creamy Floral & Sweet Woody"
    }
    return cluster_descriptions.get(cluster_id, "Unknown Cluster")

def recommend_perfumes(selected_perfume):
    if selected_perfume not in df["Name"].values:
        return []
    
    cluster = df[df["Name"] == selected_perfume]["Cluster"].values[0]
    cluster_perfumes = df[df["Cluster"] == cluster]
    
    selected_vector = vectorizer.transform(df[df["Name"] == selected_perfume]["Notes"])
    cluster_vectors = vectorizer.transform(cluster_perfumes["Notes"])
    
    similarities = cosine_similarity(selected_vector, cluster_vectors)[0]
    cluster_perfumes = cluster_perfumes.assign(Similarity=similarities)
    
    recommendations = cluster_perfumes[cluster_perfumes["Name"] != selected_perfume].sort_values(by="Similarity", ascending=False)
    
    return recommendations.head(5)

def main():
    st.title("Perfume Recommendation System")
    
    selected_perfume = st.selectbox("Select a perfume:", df["Name"].unique())
    
    if selected_perfume:
        perfume_details = df[df["Name"] == selected_perfume].iloc[0]
        cluster_desc = get_cluster_description(perfume_details["Cluster"])
        
        st.subheader("Selected Perfume Details")
        st.image(perfume_details["Image URL"], width=150)
        st.write(f"**{perfume_details['Name']}** by {perfume_details['Brand']}")
        st.write(f"Notes: {perfume_details['Notes']}")
        st.write(f"Cluster: {perfume_details['Cluster']} ({cluster_desc})")
        st.write("---")
    
    if st.button("Recommend"):
        recommendations = recommend_perfumes(selected_perfume)
        if not recommendations.empty:
            st.subheader("Recommended Perfumes")
            for _, row in recommendations.iterrows():
                cluster_desc = get_cluster_description(row["Cluster"])
                st.image(row["Image URL"], width=100)
                st.write(f"**{row['Name']}** by {row['Brand']}")
                st.write(f"Notes: {row['Notes']}")
                st.write(f"Cluster: {row['Cluster']} ({cluster_desc})")
                st.write(f"Similarity: {row['Similarity']:.2%}")
                st.write("---")
        else:
            st.write("No recommendations found.")
    
    st.subheader("Cluster Analysis & Interpretation")
    st.write("## **Cluster Interpretations:**")
    st.markdown("""
    - **Cluster 0: Fresh & Floral Woody Musk**  
      - **Dominant Notes:** Musk, Amber, Bergamot, Rose, Jasmine, Sandalwood
      - **Likely Represents:** Soft, warm, and slightly powdery scents with a floral-woody balance.
    
    - **Cluster 1: Oriental & Spicy Woody**  
      - **Dominant Notes:** Patchouli, Saffron, Oud, Amber, Leather, Cardamom  
      - **Likely Represents:** Deep, rich, and exotic fragrances, often found in Middle Eastern perfumery.
    
    - **Cluster 2: Earthy & Citrus Woody**  
      - **Dominant Notes:** Cedarwood, Vetiver, Bergamot, Nutmeg, Grapefruit  
      - **Likely Represents:** Fresh, woody, and slightly spicy colognes, often masculine and grounding.
    
    - **Cluster 3: Smoky & Resinous Oriental**  
      - **Dominant Notes:** Incense, Patchouli, Amber, Leather, Vetiver, Vanilla  
      - **Likely Represents:** Dark, resinous, smoky fragrances, ideal for colder seasons.
    
    - **Cluster 4: Fruity Floral Musk**  
      - **Dominant Notes:** Peony, Lychee, Musk, Vanilla, Rose, Freesia  
      - **Likely Represents:** Playful, sweet, and airy fragrances, often in feminine scents.
    
    - **Cluster 5: Classic Green & Woody Floral**  
      - **Dominant Notes:** Vetiver, Jasmine, Cedar, Oakmoss, Lemon, Iris  
      - **Likely Represents:** Earthy, natural, and vintage-style perfumes with chypre elements.
    
    - **Cluster 6: Creamy Floral & Sweet Woody**  
      - **Dominant Notes:** Vanilla, Sandalwood, Jasmine, Patchouli, Tuberose  
      - **Likely Represents:** Smooth, sweet, and seductive scents with floral-woody warmth.
    """)
    
if __name__ == "__main__":
    main()
