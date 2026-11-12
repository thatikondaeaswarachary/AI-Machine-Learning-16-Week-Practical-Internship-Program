"""
Standalone Movie Recommendation Engine
Next Gen Software Hub Pvt. Ltd. - AI & ML Internship Program
"""

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.df["Director_Clean"] = self.df["Director"].str.replace(" ", "")
        self.df["Content_Soup"] = (
            self.df["Genre"].str.lower() + " " +
            self.df["Director_Clean"].str.lower() + " " +
            self.df["Keywords"].str.lower()
        )
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["Content_Soup"])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.df.index, index=self.df["Title"].str.lower()).drop_duplicates()

    def get_recommendations(self, title: str, top_n: int = 5):
        clean_title = title.strip().lower()
        if clean_title not in self.indices:
            raise KeyError(f"Movie '{title}' was not found in database.")

        idx = self.indices[clean_title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [round(i[1] * 100, 2) for i in sim_scores]

        recs = self.df.iloc[movie_indices][["Title", "Genre", "Director", "Rating"]].copy()
        recs["Match_%"] = scores
        return recs


if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "datasets", "movies_recommendation_data.csv")
    engine = MovieRecommender(csv_file)
    print("Testing Movie Recommender for 'Interstellar':")
    results = engine.get_recommendations("Interstellar", top_n=4)
    print(results.to_string(index=False))
