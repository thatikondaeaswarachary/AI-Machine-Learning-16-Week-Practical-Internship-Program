# Week 13 — Natural Language Processing (NLP)
### Task: Text Classification & Sentiment Analysis
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
By the end of this week, students will master text processing and NLP classification pipelines:
- **Text Preprocessing:** Cleaning raw text, regex substitution, lowercasing, and punctuation stripping.
- **Tokenization:** Segmenting running text into discrete linguistic tokens.
- **Stop-Word Removal:** Eliminating non-discriminative words ("the", "is", "at") to reduce dimensionality.
- **Stemming vs Lemmatization:** Porter Stemmer vs WordNet Lemmatizer (crude heuristic root vs dictionary morphological lemma).
- **Bag of Words (BoW):** Counting token frequency vectors across document corpora (`CountVectorizer`).
- **TF-IDF Vectorization:** Weighting tokens by term frequency and inverse document frequency (`TfidfVectorizer`).
- **Text Classification:** Training supervised models (Naive Bayes, Logistic Regression, or Neural sequence models) on text embeddings.

---

## 🛠️ Practical Assignment (Select One)
1. **Spam Message Classifier:** (Pre-configured in `datasets/sms_spam_text_data.csv` and `starter_notebook.ipynb`)
2. **Sentiment Analysis System:** (Customer reviews / Twitter sentiment dataset)
3. **News Article Topic Classification System:** (AG News / BBC news multi-topic classification)

---

## 📋 Submission Deliverables
- **Dataset:** Raw and processed text data sample.
- **Jupyter Notebook (`starter_notebook.ipynb`):** Complete workflow from text normalization to classification.
- **Text Preprocessing Pipeline:** Reusable Python function or class applying tokenization, stop-word removal, and cleaning.
- **Trained Model:** Serialized vectorizer and classifier (`.joblib` or `.pkl`).
- **Sample Predictions:** At least 5 live predictions with text inputs, predicted category, and confidence probability.
