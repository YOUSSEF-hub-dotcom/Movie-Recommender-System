# 🎬 Movie Recommender System

A complete **end-to-end Movie Recommendation System** designed with a **Data Scientist / ML Engineer mindset**, focusing on **reproducibility, scalability, and production readiness**.

This project goes beyond simple similarity search by implementing a structured ML pipeline that includes **EDA, feature engineering, clustering, similarity computation, MLOps practices, API deployment, and an interactive UI**.

---

## 🚀 Project Overview

The goal of this project is to recommend movies that are highly similar in content, style, and metadata using a **hybrid content-based recommendation approach**.

The system is designed to simulate a real-world recommendation engine used by streaming platforms and movie discovery applications.

---

## 🧠 Recommendation Strategy

The recommendation engine combines multiple techniques to improve relevance and robustness:

* **Textual Features**

  * Movie overview
  * Tagline
  * Keywords
  * Cast & crew information

* **Categorical Features**

  * Genres (multi-label encoded)

* **Numerical Features**

  * Budget
  * Popularity
  * Revenue
  * Runtime

### Hybrid Approach:

1. **TF-IDF Vectorization** for textual features
2. **Truncated SVD** for dimensionality reduction
3. **Hierarchical Clustering (Ward linkage)** to group similar movies
4. **Cosine Similarity** within clusters for final recommendations

This approach reduces noise, improves contextual relevance, and avoids global similarity distortions.

---

## 📊 Model Results & Evaluation

| Metric                        | Value       |
| ----------------------------- | ----------- |
| Final Feature Matrix Shape    | (4803, 224) |
| Explained Variance (SVD)      | 25.58%      |
| Optimal Silhouette Score      | 0.0838      |
| Promotion Threshold           | > 0.05      |
| Similarity Score (Test Cases) | > 90%       |

📌 **Evaluation Insight**
Silhouette Score is intentionally moderate due to the **high-dimensional and non-convex nature of content-based recommendation data**.
The true evaluation metric is **recommendation relevance**, which exceeded **90% similarity accuracy** in real test cases.

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was conducted to understand both data quality and industry patterns, including:

* Duplicate movie titles and IDs
* Genre distribution and frequency
* Actor–director collaboration patterns
* Cast size statistics
* Actor genre specialization analysis
* Old vs modern movie casting trends

These insights directly influenced feature weighting and recommendation logic.

---

## 🧪 Text Preprocessing Pipeline

The text preprocessing stage includes:

* Lowercasing
* Tokenization
* Non-alphabetic character removal
* Stopword removal
* Stemming using Porter Stemmer
* Final text aggregation into a unified feature column

---

## 🤖 Model Training & Clustering

* TF-IDF vectorization with configurable `max_features`
* Truncated SVD with configurable `n_components`
* Genre encoding using `MultiLabelBinarizer`
* Feature weighting:

  * Text features × 1.5
  * Genre features × 2.0
* Hierarchical clustering evaluated using Silhouette Score

---

## 🔁 MLflow Lifecycle & Model Registry

The project implements a complete **MLflow lifecycle**:

* Experiment tracking
* Parameter logging
* Metric logging
* Artifact storage (similarity matrices, cluster labels)
* Custom `PyFunc` model for inference
* Model Registry with automatic promotion logic

### 🚦 Model Promotion Logic

* Models are promoted to **Production** only if:

  * Silhouette Score ≥ 0.05
* Otherwise, the model remains in **Staging** for further tuning

---

## 🌐 API Deployment (FastAPI)

The trained model is deployed using **FastAPI**, exposing endpoints for:

* Movie recommendations
* Movie search
* Actor-based exploration
* Director-based exploration

Integration with **TMDB API** enables real-time movie poster retrieval.

---

## 🖥️ Interactive UI (Streamlit)

An interactive Streamlit application allows users to:

* Search for movies
* Get content-based recommendations
* Explore movies by actor or director
* View posters, ratings, and metadata

---

## ⚙️ Project Structure

```
├── data_pipeline.py
├── text_preprocessing.py
├── EDA.py
├── visualization.py
├── model.py
├── mlflow_lifecycle.py
├── api.py
├── app.py
├── main.py
├── MLproject
├── conda.yaml
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* NLTK
* MLflow
* FastAPI
* Streamlit
* Matplotlib

---

## 🎯 Key Takeaways

* Built a **scalable and modular ML pipeline**
* Applied **MLOps best practices** using MLflow
* Designed a **hybrid recommendation system**
* Delivered a **production-ready API and UI**
* Focused on **model stability and real-world relevance**

---

## 🚀 Future Improvements

* User-based personalization
* Online feedback loop
* Approximate Nearest Neighbors (FAISS)
* CI/CD pipeline for automated deployment
----
👨‍💻 Author
Youssef Mahmoud Faculty of Computers & Information Aspiring Data Scientist / ML Engineer

URL Linked in :[https://www.linkedin.com/in/youssef-mahmoud-63b243361?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BDsEvhvY0QxSMfbsidmm2Ww%3D%3D]
