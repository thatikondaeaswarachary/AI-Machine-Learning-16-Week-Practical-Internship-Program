# Week 14 — Recommendation Systems
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Understand the paradigms of Recommendation Systems:
   - Content-Based Filtering
   - Collaborative Filtering (User-Based & Item-Based)
   - Hybrid Systems
2. Feature Representation: Transforming unstructured metadata (genres, directors, keywords) into high-dimensional vector spaces.
3. Compute vector proximity using **Cosine Similarity**:
   $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$
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
