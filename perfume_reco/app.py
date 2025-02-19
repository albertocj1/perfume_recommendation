import streamlit as st
import pandas as pd
import joblib  
import os

# Define path for the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "models\model_kmeans_k7_metric0.03_date20250219.pkl"))

# Load the trained model
def load_model():
    return joblib.load(MODEL_PATH)

# Main function for the Streamlit app
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

    def recommend_perfume(model, user_data):
        # Get predictions from the model
        recommendations = model.predict(user_data)
        return recommendations

    # Load the model
    model = load_model()

    if st.button("Get Recommendation"):
        print(user_input)
        recommendations = recommend_perfume(model, user_input)
        print(recommendations)

        st.subheader("Recommended Perfume(s):")
        st.write(", ".join(recommendations))

if __name__ == "__main__":
    main()
