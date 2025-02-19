# **Fragrance Clusters: Identifying Common Perfume Groups Using Scent Profiles and Recommendation System**

## **Project Overview**  
This project aims to analyze perfumes based on their scent profiles, identify common themes among them, and build a recommendation system. By clustering perfumes based on their notes, we can uncover patterns in fragrance compositions and suggest similar perfumes based on a given sample.

## **Objectives**  

### **Identify Common Themes in Perfumes**  
- Use Natural Language Processing (NLP) techniques to analyze fragrance descriptions.  
- Group perfumes into meaningful clusters based on their dominant notes.  

### **Develop a Perfume Recommendation System**  
- Apply machine learning techniques to find similarities between perfumes.  
- Suggest alternative perfumes based on scent similarity.  

## **Dataset**  
- **Title:** Perfume Recommendation Dataset  
- **Source:** [Kaggle - Perfume Recommendation Dataset](https://www.kaggle.com/datasets/nandini1999/perfume-recommendation-dataset)  
- **Attributes:**  
  - **Name:** Name of the perfume  
  - **Brand:** Brand of the perfume  
  - **Description:** Text description of the perfume  
  - **Notes:** Fragrance notes present in the perfume  
  - **Image URL:** URL of the perfume image  

## **Approach**  

### **Data Preprocessing**  
- Load the dataset and handle missing values.  
- Convert text data to lowercase for uniformity.  
- Remove unnecessary columns (*Description, Image URL*).  
- Tokenize and clean notes by filtering only relevant fragrance terms.  

### **Feature Engineering**  
- Convert fragrance notes into a space-separated string.  
- Use **TF-IDF Vectorization** to transform text into numerical features.  

### **Clustering Analysis**  
- Determine the optimal number of clusters using **Silhouette Score**.  
- Apply **K-Means clustering** to group perfumes with similar scent compositions.  
- Assign cluster labels to each perfume.  

### **Model Export & Storage**  
- Save the trained **TF-IDF vectorizer** and **K-Means model** using **Joblib**.  
- Store model metadata with a standardized naming convention.  

### **Visualization**  
- Reduce dimensionality using **TruncatedSVD** for better visualization.  
- Plot perfume clusters in a 2D space.  

### **Cluster Analysis & Interpretation**  
- Identify top fragrance notes in each cluster.  
- Categorize clusters into broader fragrance families.  

### **Perfume Recommendation System**  
- Implement a **cosine similarity-based** recommendation system.  
- Match user input with the closest perfume names using **fuzzy string matching**.  
- Recommend perfumes based on cluster membership and scent similarity.  

---

## **Tech Stack**  

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit  
- **NLP Tools:** TF-IDF Vectorizer  
- **Clustering Algorithm:** K-Means  
- **Dimensionality Reduction:** TruncatedSVD  
- **Similarity Metric:** Cosine Similarity  
- **Fuzzy Matching:** difflib.get_close_matches  
- **Model Storage:** Joblib  

---

## **Getting Started**  

### **Prerequisites**  
- Python 3.8 or higher  
- Poetry (for dependency management)  

### **Installation**  

#### **Clone the Repository**  
```bash
git clone https://github.com/your-username/perfume-recommendation-system.git
cd perfume-recommendation-system
```

#### **Set Up the Environment**  
Install dependencies using Poetry:
```bash
poetry install
```

#### **Activate the Virtual Environment**  
```bash
poetry shell
```

#### **Run the Streamlit App**  
Start the Streamlit application for interactive exploration:
```bash
streamlit run app.py
```

## **Accessing the Paper**  
For a detailed explanation of the methodology and results, please refer to the project paper:
Fragrance Clusters: Identifying Common Perfume Groups Using Scent Profiles and Recommendation System (Link to be updated soon)

## **Authors**  
- Christian Joshua Alberto
- Mark De Luna
- Rodney Lei Estrada

## **License**  
This project is licensed under the MIT License. See the LICENSE file for details.

