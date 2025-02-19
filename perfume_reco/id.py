import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("../data/perfume_data.csv")
df["Notes"].fillna("", inplace=True)

# Train TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["Notes"])

# Save the vectorizer
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")

print("TF-IDF vectorizer saved successfully!")
