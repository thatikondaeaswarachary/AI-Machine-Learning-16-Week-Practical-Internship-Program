"""
Scaffold Generator for Phase 3: Advanced Machine Learning (Weeks 9 & 10)
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
# WEEK 9: FEATURE ENGINEERING & MODEL OPTIMIZATION
# ==============================================================================
def scaffold_week_09():
    np.random.seed(42)
    n = 300
    tenure = np.random.randint(1, 60, size=n)
    monthly_bill = np.random.uniform(25, 120, size=n)
    total_bill = tenure * monthly_bill + np.random.normal(0, 50, size=n)
    calls_dropped = np.random.poisson(lam=2.5, size=n)
    support_tickets = np.random.poisson(lam=1.5, size=n)
    
    # Churn probability higher with high dropped calls and tickets
    logits = -3.0 + 0.5 * calls_dropped + 0.7 * support_tickets - 0.05 * tenure + 0.01 * monthly_bill
    prob = 1 / (1 + np.exp(-logits))
    churn = (np.random.rand(n) < prob).astype(int)
    
    df = pd.DataFrame({
        "Tenure_Months": tenure,
        "Monthly_Charges": np.round(monthly_bill, 2),
        "Total_Charges": np.round(total_bill, 2),
        "Calls_Dropped": calls_dropped,
        "Customer_Support_Tickets": support_tickets,
        "Churn": churn
    })
    csv_path = os.path.join(BASE_DIR, "phase-3-advanced-machine-learning/week-09-feature-engineering-optimization/datasets/telecom_churn_features.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated Week 9 dataset: {df.shape}")

    write_file("phase-3-advanced-machine-learning/week-09-feature-engineering-optimization/README.md", """# Week 09 — Feature Engineering & Model Optimization
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Craft impactful engineered features: ratios, interactions, and binning.
2. Select top predictive features using statistical and model-based criteria.
3. Apply Stratified K-Fold Cross-Validation.
4. Execute systematic hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV`.
5. Measure and document quantifiable Before-vs-After optimization improvements.
""")

    nb_path = os.path.join(BASE_DIR, "phase-3-advanced-machine-learning/week-09-feature-engineering-optimization/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 09 — Feature Engineering & Hyperparameter Optimization
### Maximizing Performance: Baseline vs Optimized Model
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv("datasets/telecom_churn_features.csv")
print(f"Loaded {df.shape[0]} records.")
df.head()"""),
        md_cell("""## 1. Feature Engineering
Create domain interaction features:"""),
        code_cell("""# Ratio of dropped calls per month of tenure
df["Dropped_per_Month"] = df["Calls_Dropped"] / (df["Tenure_Months"] + 1)

# Average spend per tenure month
df["Avg_Spend_Ratio"] = df["Total_Charges"] / (df["Tenure_Months"] * df["Monthly_Charges"] + 1e-5)

# High risk interaction flag
df["Frustration_Score"] = df["Calls_Dropped"] * df["Customer_Support_Tickets"]
df.head()"""),
        md_cell("""## 2. Baseline Model Performance"""),
        code_cell("""X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

baseline_rf = RandomForestClassifier(random_state=42)
baseline_rf.fit(X_train, y_train)
baseline_preds = baseline_rf.predict(X_test)
baseline_auc = roc_auc_score(y_test, baseline_rf.predict_proba(X_test)[:, 1])

print(f"Baseline Random Forest ROC-AUC: {baseline_auc:.4f}")
print(classification_report(y_test, baseline_preds))"""),
        md_cell("""## 3. Systematic Hyperparameter Tuning with GridSearchCV"""),
        code_cell("""param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
best_preds = best_rf.predict(X_test)
best_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Tuned Random Forest ROC-AUC: {best_auc:.4f} (Baseline: {baseline_auc:.4f})")
print(f"Net Improvement: +{(best_auc - baseline_auc)*100:.2f}%")""")
    ]
    make_notebook(cells, nb_path)

# ==============================================================================
# WEEK 10: ENSEMBLE LEARNING
# ==============================================================================
def scaffold_week_10():
    np.random.seed(42)
    n = 350
    # Synthetic credit risk
    f1 = np.random.normal(0, 1, size=n)
    f2 = np.random.normal(0, 1, size=n)
    f3 = np.random.uniform(-2, 2, size=n)
    f4 = np.random.exponential(1.5, size=n)
    
    latent = 1.2 * f1 - 1.5 * f2 + 0.8 * (f3**2) + 0.4 * f4
    prob = 1 / (1 + np.exp(-latent))
    target = (np.random.rand(n) < prob).astype(int)
    
    df = pd.DataFrame({
        "Trans_Freq": np.round(f1, 3),
        "Avg_Amount": np.round(f2, 3),
        "Location_Variance": np.round(f3, 3),
        "Velocity_Score": np.round(f4, 3),
        "Fraud_Label": target
    })
    csv_path = os.path.join(BASE_DIR, "phase-3-advanced-machine-learning/week-10-ensemble-learning/datasets/credit_scoring_ensemble.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated Week 10 dataset: {df.shape}")

    write_file("phase-3-advanced-machine-learning/week-10-ensemble-learning/README.md", """# Week 10 — Ensemble Learning Architectures
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Understand the mathematical foundation of Bagging, Boosting, and Stacking.
2. Implement and contrast:
   - Random Forest (Bagging)
   - AdaBoost (Adaptive Boosting)
   - Gradient Boosting & HistGradientBoosting
   - Voting Classifier (Hard vs Soft voting)
3. Evaluate predictive performance vs computational latency trade-offs.
""")

    nb_path = os.path.join(BASE_DIR, "phase-3-advanced-machine-learning/week-10-ensemble-learning/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 10 — Ensemble Learning
### Bagging, Boosting, and Voting Classifier Benchmark
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

df = pd.read_csv("datasets/credit_scoring_ensemble.csv")
X = df.drop(columns=["Fraud_Label"])
y = df["Fraud_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")"""),
        md_cell("""## 1. Benchmarking Individual Learners vs Ensembles"""),
        code_cell("""models = {
    "Base: Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Ensemble: Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Ensemble: AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Ensemble: Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Ensemble: Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42)
}

results = []
for name, clf in models.items():
    start_time = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "F1-Score": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs),
        "Train Time (s)": round(fit_time, 4)
    })

res_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
res_df"""),
        md_cell("""## 2. Building a Soft Voting Classifier"""),
        code_cell("""voting_clf = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("lr", LogisticRegression(random_state=42))
    ],
    voting="soft"
)
voting_clf.fit(X_train, y_train)
v_preds = voting_clf.predict(X_test)
v_probs = voting_clf.predict_proba(X_test)[:, 1]

print("--- Soft Voting Classifier Performance ---")
print(f"Voting ROC-AUC: {roc_auc_score(y_test, v_probs):.4f}")
print(f"Voting F1-Score: {f1_score(y_test, v_preds):.4f}")""")
    ]
    make_notebook(cells, nb_path)

if __name__ == "__main__":
    print("Scaffolding Phase 3...")
    scaffold_week_09()
    scaffold_week_10()
    print("Phase 3 complete!")
