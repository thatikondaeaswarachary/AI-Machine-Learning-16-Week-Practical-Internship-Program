"""
Week 14 Generator: Recommendation Systems
Next Gen Software Hub Pvt. Ltd. - AI & ML Internship Program
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEEK14_DIR = os.path.join(BASE_DIR, "phase-4-deep-learning/week-14-recommendation-systems")

# 1. Generate Movie Recommendation Dataset
movies = [
    {"Movie_ID": 1, "Title": "Inception", "Genre": "Action Sci-Fi Thriller", "Director": "Christopher Nolan", "Keywords": "dream subconscious heist mind-bending reality architect", "Rating": 8.8},
    {"Movie_ID": 2, "Title": "Interstellar", "Genre": "Adventure Drama Sci-Fi", "Director": "Christopher Nolan", "Keywords": "space black hole time dilation wormhole relativity astronaut father daughter", "Rating": 8.7},
    {"Movie_ID": 3, "Title": "The Dark Knight", "Genre": "Action Crime Drama", "Director": "Christopher Nolan", "Keywords": "batman joker vigilante justice gotham corruption chaos crime fighter", "Rating": 9.0},
    {"Movie_ID": 4, "Title": "The Matrix", "Genre": "Action Sci-Fi", "Director": "Lana Wachowski", "Keywords": "simulation cyber dystopia artificial intelligence hacker martial arts red pill", "Rating": 8.7},
    {"Movie_ID": 5, "Title": "Blade Runner 2049", "Genre": "Action Drama Mystery Sci-Fi", "Director": "Denis Villeneuve", "Keywords": "replicants cyberpunk dystopia neon identity future android detective", "Rating": 8.0},
    {"Movie_ID": 6, "Title": "Arrival", "Genre": "Drama Mystery Sci-Fi", "Director": "Denis Villeneuve", "Keywords": "alien contact linguistics time language extraterrestrial spacecraft future", "Rating": 7.9},
    {"Movie_ID": 7, "Title": "Dune", "Genre": "Action Adventure Drama Sci-Fi", "Director": "Denis Villeneuve", "Keywords": "desert planet spice empire prophecy chosen one sandworm galactic battle", "Rating": 8.0},
    {"Movie_ID": 8, "Title": "The Prestige", "Genre": "Drama Mystery Sci-Fi", "Director": "Christopher Nolan", "Keywords": "magician illusion rivalry obsession cloning secret twist victorian", "Rating": 8.5},
    {"Movie_ID": 9, "Title": "Memento", "Genre": "Mystery Thriller", "Director": "Christopher Nolan", "Keywords": "memory loss revenge non-linear polaroid tattoo murder investigation", "Rating": 8.4},
    {"Movie_ID": 10, "Title": "Shutter Island", "Genre": "Mystery Thriller", "Director": "Martin Scorsese", "Keywords": "mental asylum detective psychological thriller mystery twist island escape", "Rating": 8.2},
    {"Movie_ID": 11, "Title": "Fight Club", "Genre": "Drama", "Director": "David Fincher", "Keywords": "insomnia alter ego underground rebellion anarchy anti-consumerism soap", "Rating": 8.8},
    {"Movie_ID": 12, "Title": "Seven", "Genre": "Crime Drama Mystery", "Director": "David Fincher", "Keywords": "serial killer seven deadly sins detective rainy city detective box dark", "Rating": 8.6},
    {"Movie_ID": 13, "Title": "Zodiac", "Genre": "Crime Drama Mystery", "Director": "David Fincher", "Keywords": "san francisco cipher cryptogram journalist unsolved murder newspaper", "Rating": 7.7},
    {"Movie_ID": 14, "Title": "Pulp Fiction", "Genre": "Crime Drama", "Director": "Quentin Tarantino", "Keywords": "nonlinear hitman mobster briefcase dance dialogue cult classic", "Rating": 8.9},
    {"Movie_ID": 15, "Title": "Django Unchained", "Genre": "Action Drama Western", "Director": "Quentin Tarantino", "Keywords": "bounty hunter vengeance slavery western plantation gunfight justice", "Rating": 8.5},
    {"Movie_ID": 16, "Title": "Inglourious Basterds", "Genre": "Adventure Drama War", "Director": "Quentin Tarantino", "Keywords": "world war occupied france cinema assassination revenge undercover guerilla", "Rating": 8.4},
    {"Movie_ID": 17, "Title": "The Shawshank Redemption", "Genre": "Drama", "Director": "Frank Darabont", "Keywords": "prison hope friendship escape wrongful conviction perseverance banker", "Rating": 9.3},
    {"Movie_ID": 18, "Title": "The Green Mile", "Genre": "Crime Drama Fantasy", "Director": "Frank Darabont", "Keywords": "death row miracle supernatural healing prison guard compassion gentle giant", "Rating": 8.6},
    {"Movie_ID": 19, "Title": "Forrest Gump", "Genre": "Drama Romance", "Director": "Robert Zemeckis", "Keywords": "destiny history running vietnam war ping pong shrimp kindness inspiring", "Rating": 8.8},
    {"Movie_ID": 20, "Title": "Cast Away", "Genre": "Adventure Drama", "Director": "Robert Zemeckis", "Keywords": "desert island plane crash survival isolation fedex volleyball companionship", "Rating": 7.8},
    {"Movie_ID": 21, "Title": "Toy Story", "Genre": "Animation Adventure Comedy", "Director": "John Lasseter", "Keywords": "living toys woody buzz lightyear friendship childhood bedroom jealousy", "Rating": 8.3},
    {"Movie_ID": 22, "Title": "Finding Nemo", "Genre": "Animation Adventure Comedy", "Director": "Andrew Stanton", "Keywords": "ocean clownfish father search barrier reef sydney dentist aquarium", "Rating": 8.2},
    {"Movie_ID": 23, "Title": "WALL-E", "Genre": "Animation Adventure Sci-Fi", "Director": "Andrew Stanton", "Keywords": "robot earth trash cleanup space love plant environmentalism silent", "Rating": 8.4},
    {"Movie_ID": 24, "Title": "Up", "Genre": "Animation Adventure Comedy", "Director": "Pete Docter", "Keywords": "balloons flying house south america wilderness explorer dog friendship grief", "Rating": 8.3},
    {"Movie_ID": 25, "Title": "Spider-Man: Into the Spider-Verse", "Genre": "Animation Action Adventure", "Director": "Peter Ramsey", "Keywords": "multiverse superhero miles morales comic book animation multiverse web", "Rating": 8.4},
]

df_movies = pd.DataFrame(movies)
df_movies["Content_Soup"] = (
    df_movies["Genre"] + " " +
    df_movies["Director"] + " " +
    df_movies["Keywords"]
)

data_dir = os.path.join(WEEK14_DIR, "datasets")
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, "movies_recommendation_data.csv")
df_movies.to_csv(csv_path, index=False)
print(f"[OK] Generated {csv_path} with {len(df_movies)} movies.")

# 2. Week 14 README.md
readme_content = """# Week 14 — Recommendation Systems
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Understand the paradigms of Recommendation Systems:
   - Content-Based Filtering
   - Collaborative Filtering (User-Based & Item-Based)
   - Hybrid Systems
2. Feature Representation: Transforming unstructured metadata (genres, directors, keywords) into high-dimensional vector spaces.
3. Compute vector proximity using **Cosine Similarity**:
   $$\\text{Cosine Similarity}(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|}$$
4. Implement an end-to-end Movie Recommendation Engine capable of returning ranked recommendations for any query title.

---

## 🛠️ Practical Assignment: Content-Based Movie Recommendation Engine
Build a complete content-based movie recommendation system using `datasets/movies_recommendation_data.csv`.

### Requirements:
1. **Metadata Engineering:** Combine `Genre`, `Director`, and `Keywords` into a unified feature representation ("Content Soup").
2. **TF-IDF Vectorization:** Convert the text corpus into a numerical TF-IDF feature matrix.
3. **Similarity Matrix Calculation:** Compute the pairwise Cosine Similarity matrix across all movies.
4. **Recommendation Function:** Write `recommend_movies(title, top_n=5)` that handles case-insensitivity, extracts top-ranked similar movies, and displays match percentages.
5. **Validation & Sample Outputs:** Test recommendations on 3 distinct movie queries (e.g., *Inception*, *The Dark Knight*, *Toy Story*).

---

## 📂 Deliverables
- `datasets/movies_recommendation_data.csv`: Movie metadata database.
- `starter_notebook.ipynb`: Complete step-by-step recommendation pipeline.
- `recommender_engine.py`: Reusable standalone Python module for recommendations.
- Submission report adhering to `templates/weekly_submission_template.md`.
"""

with open(os.path.join(WEEK14_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)
print("[OK] Written Week 14 README.md")

# 3. Week 14 Starter Notebook
nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Week 14 — Recommendation Systems\n",
                "### Building a Content-Based Movie Recommendation Engine\n",
                "**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**\n",
                "\n",
                "---\n",
                "### Objectives:\n",
                "1. Load and inspect the movie dataset.\n",
                "2. Create a unified metadata 'content soup'.\n",
                "3. Compute TF-IDF matrix and Cosine Similarity scores.\n",
                "4. Implement the recommendation ranking engine.\n",
                "5. Evaluate recommendations across multiple genres."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "from sklearn.metrics.pairwise import cosine_similarity\n",
                "\n",
                "# Load dataset\n",
                "df = pd.read_csv('datasets/movies_recommendation_data.csv')\n",
                "print(f'Total Movies Loaded: {len(df)}')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Feature Engineering: Content Soup Construction\n",
                "Combine text attributes (Genre, Director, Keywords) to capture the thematic essence of each title."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clean and combine attributes\n",
                "df['Director_Clean'] = df['Director'].str.replace(' ', '')\n",
                "df['Content_Soup'] = (\n",
                "    df['Genre'].str.lower() + ' ' +\n",
                "    df['Director_Clean'].str.lower() + ' ' +\n",
                "    df['Keywords'].str.lower()\n",
                ")\n",
                "print('Sample Content Soup for movie #1:')\n",
                "print(df['Content_Soup'].iloc[0])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Vectorization & Cosine Similarity Matrix"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tfidf = TfidfVectorizer(stop_words='english')\n",
                "tfidf_matrix = tfidf.fit_transform(df['Content_Soup'])\n",
                "print(f'TF-IDF Matrix Shape: {tfidf_matrix.shape}')\n",
                "\n",
                "# Compute pairwise cosine similarity\n",
                "cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)\n",
                "print(f'Cosine Similarity Matrix Shape: {cosine_sim.shape}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Recommendation Function Implementation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "indices = pd.Series(df.index, index=df['Title'].str.lower()).drop_duplicates()\n",
                "\n",
                "def recommend_movies(title, top_n=5):\n",
                "    title_clean = title.strip().lower()\n",
                "    if title_clean not in indices:\n",
                "        return f\"Movie '{title}' not found in database.\"\n",
                "    \n",
                "    idx = indices[title_clean]\n",
                "    sim_scores = list(enumerate(cosine_sim[idx]))\n",
                "    # Sort by similarity descending, excluding the query movie itself\n",
                "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]\n",
                "    \n",
                "    movie_indices = [i[0] for i in sim_scores]\n",
                "    scores = [round(i[1] * 100, 2) for i in sim_scores]\n",
                "    \n",
                "    results = df.iloc[movie_indices][['Title', 'Genre', 'Director', 'Rating']].copy()\n",
                "    results['Similarity_Score_%'] = scores\n",
                "    return results\n",
                "\n",
                "print('Recommendation engine ready!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Test Sample Recommendations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=== Recommendations for: Inception ===')\n",
                "display(recommend_movies('Inception', top_n=5))\n",
                "\n",
                "print('\\n=== Recommendations for: Toy Story ===')\n",
                "display(recommend_movies('Toy Story', top_n=4))\n",
                "\n",
                "print('\\n=== Recommendations for: Fight Club ===')\n",
                "display(recommend_movies('Fight Club', top_n=4))"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(os.path.join(WEEK14_DIR, "starter_notebook.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("[OK] Generated Week 14 starter_notebook.ipynb")

# 4. Standalone Python Engine
engine_code = '''"""
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
'''

with open(os.path.join(WEEK14_DIR, "recommender_engine.py"), "w", encoding="utf-8") as f:
    f.write(engine_code)
print("[OK] Generated Week 14 recommender_engine.py")
