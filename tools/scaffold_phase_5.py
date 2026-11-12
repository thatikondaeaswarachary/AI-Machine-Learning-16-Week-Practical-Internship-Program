"""
Scaffold Generator for Phase 5: NLP & Generative AI (Weeks 13 & 14)
Next Gen Software Hub Pvt. Ltd. - AI & ML Internship Program
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def make_notebook(cells, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[OK] Generated Notebook: {os.path.relpath(filepath, BASE_DIR)}")

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().split("\n")]}

def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.strip().split("\n")]}

def write_file(rel_path, content):
    full_path = os.path.join(BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Written File: {rel_path}")

# ==============================================================================
# WEEK 13: NLP FOUNDATIONS
# ==============================================================================
def scaffold_week_13():
    # Sample SMS dataset (Spam vs Ham)
    ham_messages = [
        "Hey, are we still meeting for the team project discussion today?",
        "Please review the attached machine learning report before 5 PM.",
        "Can you send me the Python lecture notes from yesterday?",
        "Good morning! Let's schedule the weekly sprint sync at 11 AM.",
        "Your code passed all unit tests. Ready to merge the pull request.",
        "Don't forget to submit the data cleaning assignment on Sunday.",
        "Are you free for lunch near the campus cafeteria?",
        "The model training finished with 92% validation accuracy."
    ] * 15 # 120 ham

    spam_messages = [
        "CONGRATULATIONS! You have won a $1,000 Walmart Gift Card. Click here to claim now!",
        "URGENT: Your account requires immediate verification. Reply with your OTP to avoid suspension.",
        "Exclusive offer! Get 85% discount on luxury watches. Limited stock available call now.",
        "WINNER! As a valued subscriber, you have been selected for a free holiday trip to Paris.",
        "Pre-approved personal loan of $50,000 with 0% interest for 12 months. Apply immediately!"
    ] * 8 # 40 spam

    texts = ham_messages + spam_messages
    labels = ["ham"] * len(ham_messages) + ["spam"] * len(spam_messages)

    df = pd.DataFrame({"Text": texts, "Label": labels}).sample(frac=1.0, random_state=42).reset_index(drop=True)
    csv_path = os.path.join(BASE_DIR, "phase-5-nlp-generative-ai/week-13-nlp-foundations/datasets/sms_spam_text_data.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated Week 13 dataset: {df.shape}")

    write_file("phase-5-nlp-generative-ai/week-13-nlp-foundations/README.md", """# Week 13 — Natural Language Processing (NLP) Foundations
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Implement text normalization: lowercasing, punctuation stripping, regex cleanup.
2. Master tokenization, stop-word removal, and vocabulary mapping.
3. Understand Bag-of-Words and Term Frequency - Inverse Document Frequency (TF-IDF).
4. Build and evaluate an end-to-end Text Classification system for SMS Spam Detection.
""")

    nb_path = os.path.join(BASE_DIR, "phase-5-nlp-generative-ai/week-13-nlp-foundations/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 13 — NLP Foundations
### Text Preprocessing, TF-IDF Vectorization & Spam Classification
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/sms_spam_text_data.csv")
print(f"Total texts: {df.shape[0]} | Spam ratio: {(df['Label'] == 'spam').mean()*100:.1f}%")
df.head()"""),
        md_cell("""## 1. Text Preprocessing Pipeline"""),
        code_cell("""def clean_text(text):
    text = text.lower() # lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # remove URLs
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    text = re.sub(r'\d+', '', text) # remove numbers
    text = text.strip()
    return text

df["Clean_Text"] = df["Text"].apply(clean_text)
df[["Text", "Clean_Text"]].head()"""),
        md_cell("""## 2. TF-IDF Vectorization & Model Training"""),
        code_cell("""X_train, X_test, y_train, y_test = train_test_split(
    df["Clean_Text"], df["Label"], test_size=0.25, random_state=42, stratify=df["Label"]
)

vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF Vocabulary Size: {len(vectorizer.get_feature_names_out())}")

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)
preds = clf.predict(X_test_tfidf)

print(classification_report(y_test, preds))"""),
        md_cell("""## 3. Real-Time Inference Demo"""),
        code_cell("""def predict_sms(message):
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    pred = clf.predict(vec)[0]
    prob = clf.predict_proba(vec).max()
    return pred, prob

sample_msg = "Congratulations! Claim your free $500 gift card today."
result, confidence = predict_sms(sample_msg)
print(f"Message : '{sample_msg}'")
print(f"Result  : {result.upper()} (Confidence: {confidence*100:.1f}%)")""")
    ]
    make_notebook(cells, nb_path)

# ==============================================================================
# WEEK 14: TRANSFORMERS & LLMS
# ==============================================================================
def scaffold_week_14():
    write_file("phase-5-nlp-generative-ai/week-14-transformers-llms/README.md", """# Week 14 — Transformers, Hugging Face & Generative AI
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Understand the Self-Attention mechanism and Transformer Architecture (Encoders vs Decoders).
2. Utilize the Hugging Face `transformers` ecosystem (`pipeline`, tokenizers, models).
3. Apply pre-trained models (e.g. DistilBERT, RoBERTa) for sentiment and text classification.
4. Learn Prompt Engineering methodologies (Zero-shot, Few-shot, Chain-of-Thought).
""")

    nb_path = os.path.join(BASE_DIR, "phase-5-nlp-generative-ai/week-14-transformers-llms/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 14 — Transformers & Generative AI
### Pre-trained Models with Hugging Face & Prompt Engineering
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""from transformers import pipeline
import pandas as pd

print("Hugging Face Transformers imported successfully!")"""),
        md_cell("""## 1. Zero-Shot Sentiment Analysis with DistilBERT"""),
        code_cell("""# Load pre-trained sentiment classifier pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

reviews = [
    "The customer support resolved my issue within minutes. Outstanding service!",
    "The application keeps crashing every time I upload a CSV file. Very disappointed.",
    "Decent product for the price, but shipping was slower than promised."
]

results = sentiment_analyzer(reviews)
for text, res in zip(reviews, results):
    print(f"Review: '{text}'")
    print(f"-> Label: {res['label']}, Confidence: {res['score']*100:.2f}%\n")"""),
        md_cell("""## 2. Zero-Shot Classification Across Custom Categories"""),
        code_cell("""classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text_to_classify = "The latest quarterly revenue exceeded projections due to surging cloud adoption."
candidate_labels = ["Finance", "Healthcare", "Sports", "Technology"]

res = classifier(text_to_classify, candidate_labels)
print(f"Text: '{text_to_classify}'")
for label, score in zip(res["labels"], res["scores"]):
    print(f"  {label:<12}: {score*100:.2f}%")"""),
        md_cell("""## 3. Generative AI Prompt Engineering Exercise
*(Interns: Design few-shot and chain-of-thought prompt templates for summarization and automated data extraction)*""")
    ]
    make_notebook(cells, nb_path)

if __name__ == "__main__":
    print("Scaffolding Phase 5...")
    scaffold_week_13()
    scaffold_week_14()
    print("Phase 5 complete!")
