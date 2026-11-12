"""
Scaffold Generator for Phase 2: Machine Learning Core (Weeks 5 to 8)
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
# WEEK 5: ML FUNDAMENTALS
# ==============================================================================
def scaffold_week_05():
    np.random.seed(42)
    n = 150
    exp = np.round(np.random.uniform(0.5, 18.0, size=n), 1)
    base_salary = 35000 + exp * 6200 + np.random.normal(0, 4500, size=n)
    
    df = pd.DataFrame({
        "Years_Experience": exp,
        "Education_Level": np.random.choice(["Bachelor", "Master", "PhD"], size=n, p=[0.6, 0.3, 0.1]),
        "Certifications_Count": np.random.randint(0, 6, size=n),
        "Salary": np.round(base_salary, 2)
    })
    csv_path = os.path.join(BASE_DIR, "phase-2-machine-learning/week-05-ml-fundamentals/datasets/experience_salary_data.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated Week 5 dataset: {df.shape}")

    write_file("phase-2-machine-learning/week-05-ml-fundamentals/README.md", """# Week 05 — Machine Learning Fundamentals
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Understand the core paradigm of Machine Learning (data + answers -> rules).
2. Learn differences between Supervised, Unsupervised, and Reinforcement Learning.
3. Master Train-Test splits, data leakage prevention, and random seeds.
4. Understand Bias-Variance tradeoff, Overfitting vs Underfitting.
5. Train and evaluate your first Scikit-Learn model.

---

## 🛠️ Practical Assignment: First ML Prediction System
Build an end-to-end Salary Prediction Model based on years of experience and education level using `datasets/experience_salary_data.csv`.

### Requirements:
1. Split data into 80% training and 20% test sets.
2. Train a baseline Ordinary Least Squares (OLS) Linear Regression model.
3. Calculate evaluation metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.
4. Visualize regression line vs actual test observations.
5. Save the trained model artifact using `joblib`.
""")

    nb_path = os.path.join(BASE_DIR, "phase-2-machine-learning/week-05-ml-fundamentals/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 05 — Machine Learning Fundamentals
### Building Your First Machine Learning Model
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

df = pd.read_csv("datasets/experience_salary_data.csv")
print(f"Dataset shape: {df.shape}")
df.head()"""),
        md_cell("""## 1. Feature & Target Formulation"""),
        code_cell("""X = df[["Years_Experience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")"""),
        md_cell("""## 2. Model Training & Parameter Inspection"""),
        code_cell("""model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept (beta_0): {model.intercept_:.2f}")
print(f"Slope (beta_1):     {model.coef_[0]:.2f}")
print(f"Model Equation: Salary = {model.intercept_:.2f} + ({model.coef_[0]:.2f} * Experience)")"""),
        md_cell("""## 3. Evaluation Metrics on Test Set"""),
        code_cell("""y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE  : ${mae:.2f}")
print(f"RMSE : ${rmse:.2f}")
print(f"R²   : {r2:.4f}")"""),
        md_cell("""## 4. Visualizing Model Fit"""),
        code_cell("""plt.figure(figsize=(9, 5))
plt.scatter(X_test, y_test, color="#3498db", label="Actual Test Observations", alpha=0.8)
plt.plot(X_test, y_pred, color="#e74c3c", linewidth=2, label="Fitted Regression Line")
plt.title("Experience vs Salary: Actual vs Predicted", fontsize=14, fontweight="bold")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.legend()
plt.show()"""),
        md_cell("""## 5. Model Serialization"""),
        code_cell("""joblib.dump(model, "trained_salary_model.joblib")
print("Model serialized to trained_salary_model.joblib")""")
    ]
    make_notebook(cells, nb_path)

# ==============================================================================
# WEEK 6: CLASSIFICATION ALGORITHMS
# ==============================================================================
def scaffold_week_06():
    np.random.seed(42)
    n = 300
    credit_scores = np.random.randint(550, 850, size=n)
    incomes = np.random.randint(25000, 150000, size=n)
    debt_to_income = np.random.uniform(0.1, 0.65, size=n)
    loan_amounts = np.random.randint(5000, 45000, size=n)
    
    # Probability of default
    prob = 1 / (1 + np.exp(-(-2.5 + (700 - credit_scores) * 0.015 + debt_to_income * 3.5 + (loan_amounts/incomes) * 1.5)))
    defaults = (np.random.rand(n) < prob).astype(int)
    
    df = pd.DataFrame({
        "Credit_Score": credit_scores,
        "Annual_Income": incomes,
        "Debt_to_Income": np.round(debt_to_income, 3),
        "Loan_Amount": loan_amounts,
        "Default": defaults
    })
    csv_path = os.path.join(BASE_DIR, "phase-2-machine-learning/week-06-classification-algorithms/datasets/loan_default_data.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated Week 6 dataset: {df.shape}")

    write_file("phase-2-machine-learning/week-06-classification-algorithms/README.md", """# Week 06 — Classification Algorithms & Multi-Model Benchmark
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Implement and understand key classification algorithms:
   - Logistic Regression
   - Decision Tree Classifier
   - K-Nearest Neighbors (KNN)
   - Random Forest Classifier
   - Support Vector Machine (SVM)
2. Evaluate performance using Confusion Matrix, Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
3. Understand metric trade-offs in high-risk domains (e.g. false negatives in loan default).

---

## 🛠️ Practical Assignment: Multi-Model Loan Default Benchmark
Train and compare 5 classifiers on `datasets/loan_default_data.csv`.

### Requirements:
1. Standardize features for distance-based and margin-based algorithms (KNN, Logistic, SVM).
2. Train all 5 classifiers using identical train-test splits.
3. Compile a comparative metric scorecard (Accuracy, Precision, Recall, F1, ROC-AUC).
4. Plot Confusion Matrices and ROC curves.
5. Select and justify the best model for a risk-averse financial institution.
""")

    nb_path = os.path.join(BASE_DIR, "phase-2-machine-learning/week-06-classification-algorithms/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 06 — Classification Algorithms
### Comparative Multi-Model Benchmark on Loan Default Prediction
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

df = pd.read_csv("datasets/loan_default_data.csv")
print(f"Dataset shape: {df.shape} | Default rate: {df['Default'].mean()*100:.1f}%")
df.head()"""),
        md_cell("""## 1. Feature Preprocessing & Partitioning"""),
        code_cell("""X = df.drop(columns=["Default"])
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)"""),
        md_cell("""## 2. Multi-Algorithm Benchmark Execution"""),
        code_cell("""models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

results = []
for name, clf in models.items():
    # Use scaled data for all for consistency
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1-Score": f1_score(y_test, preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probs)
    })

results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
results_df"""),
        md_cell("""## 3. Confusion Matrix & ROC Curve Comparison"""),
        code_cell("""best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Non-Default", "Default"]).plot(ax=ax, cmap="Blues")
plt.title(f"Confusion Matrix: {best_model_name}", fontsize=13, fontweight="bold")
plt.show()"""),
        md_cell("""## 4. Conclusion & Model Selection
*(Interns: Explain which model should be deployed in production and why)*""")
    ]
    make_notebook(cells, nb_path)

# ==============================================================================
# WEEK 7: REGRESSION ALGORITHMS
# ==============================================================================
def scaffold_week_07():
    np.random.seed(42)
    n = 250
    sqft = np.random.randint(650, 4500, size=n)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.25, 0.4, 0.2, 0.05])
    bathrooms = np.maximum(1, bedrooms - np.random.choice([0, 1], size=n))
    age = np.random.randint(1, 40, size=n)
    location_score = np.random.uniform(1.0, 10.0, size=n)
    
    # House price equation with noise
    prices = 75000 + sqft * 165 + bedrooms * 12000 + bathrooms * 18000 - age * 950 + location_score * 15000 + np.random.normal(0, 22000, size=n)
    
    df = pd.DataFrame({
        "Square_Feet": sqft,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "House_Age": age,
        "Location_Score": np.round(location_score, 1),
        "Price": np.round(prices, 2)
    })
    csv_path = os.path.join(BASE_DIR, "phase-2-machine-learning/week-07-regression-algorithms/datasets/housing_prices_data.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated Week 7 dataset: {df.shape}")

    write_file("phase-2-machine-learning/week-07-regression-algorithms/README.md", """# Week 07 — Regression Algorithms & Prediction Systems
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Implement Linear, Ridge, Lasso, Decision Tree, and Random Forest Regressors.
2. Master evaluation metrics: MAE, MSE, RMSE, and R² Score.
3. Perform residual analysis to check regression assumptions (homoscedasticity, normality).
4. Save and export final prediction pipeline.
""")

    nb_path = os.path.join(BASE_DIR, "phase-2-machine-learning/week-07-regression-algorithms/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 07 — Regression Algorithms
### Building a Real Estate Housing Valuation System
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("datasets/housing_prices_data.csv")
print(f"Loaded {df.shape[0]} housing records.")
df.head()"""),
        md_cell("""## 1. Feature Partitioning & Training Setup"""),
        code_cell("""X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"""),
        md_cell("""## 2. Multi-Model Regressor Benchmark"""),
        code_cell("""regressors = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=100.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []
for name, reg in regressors.items():
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2 Score": r2_score(y_test, preds)
    })

res_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
res_df"""),
        md_cell("""## 3. Residual Diagnostics & Error Distribution"""),
        code_cell("""best_reg = regressors[res_df.iloc[0]["Model"]]
residuals = y_test - best_reg.predict(X_test)

plt.figure(figsize=(9, 5))
sns.scatterplot(x=best_reg.predict(X_test), y=residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.title(f"Residual Plot: {res_df.iloc[0]['Model']}", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.show()""")
    ]
    make_notebook(cells, nb_path)

# ==============================================================================
# WEEK 8: UNSUPERVISED LEARNING & CLUSTERING
# ==============================================================================
def scaffold_week_08():
    np.random.seed(42)
    n = 200
    # Create 4 distinct customer archetypes
    cluster_1 = np.random.normal(loc=[25, 20], scale=[3, 4], size=(50, 2))  # Young, Low Spend
    cluster_2 = np.random.normal(loc=[45, 80], scale=[4, 5], size=(50, 2))  # Middle Age, High Spend (VIP)
    cluster_3 = np.random.normal(loc=[35, 50], scale=[3, 5], size=(50, 2))  # Regular Middle Class
    cluster_4 = np.random.normal(loc=[55, 25], scale=[4, 4], size=(50, 2))  # Senior Budget Conscious
    
    raw = np.vstack([cluster_1, cluster_2, cluster_3, cluster_4])
    annual_incomes = np.clip(raw[:, 0] * 2.2, 15, 140)
    spending_scores = np.clip(raw[:, 1], 1, 100)
    
    df = pd.DataFrame({
        "CustomerID": [f"CUST-{3000 + i}" for i in range(n)],
        "Age": np.random.randint(18, 70, size=n),
        "Annual_Income_k$": np.round(annual_incomes, 1),
        "Spending_Score_1_100": np.round(spending_scores, 1)
    })
    csv_path = os.path.join(BASE_DIR, "phase-2-machine-learning/week-08-unsupervised-clustering/datasets/mall_customers_segmentation.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated Week 8 dataset: {df.shape}")

    write_file("phase-2-machine-learning/week-08-unsupervised-clustering/README.md", """# Week 08 — Unsupervised Learning & Customer Segmentation
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Understand unsupervised clustering paradigms.
2. Implement K-Means Clustering and evaluate cluster stability.
3. Determine optimal clusters using the **Elbow Method (WCSS)** and **Silhouette Score**.
4. Apply Hierarchical Clustering and interpret Dendrograms.
5. Create customer persona profiles and actionable business segments.
""")

    nb_path = os.path.join(BASE_DIR, "phase-2-machine-learning/week-08-unsupervised-clustering/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 08 — Unsupervised Learning
### Customer Segmentation Using K-Means & Hierarchical Clustering
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

df = pd.read_csv("datasets/mall_customers_segmentation.csv")
print(f"Customer records: {df.shape[0]}")
df.head()"""),
        md_cell("""## 1. Feature Selection & Standardization"""),
        code_cell("""features = ["Annual_Income_k$", "Spending_Score_1_100"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)"""),
        md_cell("""## 2. Finding Optimal K: Elbow Method & Silhouette Scores"""),
        code_cell("""wcss = []
silhouette_scores = []
K_range = range(2, 9)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, wcss, "bo-", linewidth=2)
ax1.set_title("Elbow Method (WCSS)", fontweight="bold")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia / WCSS")

ax2.plot(K_range, silhouette_scores, "ro-", linewidth=2)
ax2.set_title("Silhouette Score by K", fontweight="bold")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Silhouette Score")
plt.show()"""),
        md_cell("""## 3. Final K-Means Segmentation & Cluster Profiling"""),
        code_cell("""optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Annual_Income_k$",
    y="Spending_Score_1_100",
    hue="Cluster",
    palette="tab10",
    s=100,
    alpha=0.85
)
plt.title("Identified Customer Segments (K-Means)", fontsize=14, fontweight="bold")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# Segment Profiling
print("--- Cluster Persona Profiling ---")
print(df.groupby("Cluster")[["Annual_Income_k$", "Spending_Score_1_100", "Age"]].mean().round(1))""")
    ]
    make_notebook(cells, nb_path)

if __name__ == "__main__":
    print("Scaffolding Phase 2...")
    scaffold_week_05()
    scaffold_week_06()
    scaffold_week_07()
    scaffold_week_08()
    print("Phase 2 complete!")
