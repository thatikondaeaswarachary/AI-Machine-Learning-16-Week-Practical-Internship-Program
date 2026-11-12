"""
Master Scaffold Generator for 16-Week AI & ML Internship
Next Gen Software Hub Pvt. Ltd.
Builds all weekly guides, synthetic datasets, starter notebooks, and templates.
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
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[OK] Generated Notebook: {os.path.relpath(filepath, BASE_DIR)}")

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

def write_file(rel_path, content):
    full_path = os.path.join(BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Written File: {rel_path}")

# ==============================================================================
# PHASE 1: WEEKS 2 - 4
# ==============================================================================

def scaffold_week_02():
    # Week 2 Notebook
    nb_path = os.path.join(BASE_DIR, "phase-1-python-data-fundamentals/week-02-numpy-pandas-visualization/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 02 — NumPy, Pandas & Data Visualization
### Practical Assignment: E-Commerce Data Analytics
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**

---
### Assignment Objectives:
1. Load and explore `ecommerce_sales_data.csv`.
2. Inspect data types, null values, and summary statistics.
3. Compute groupby aggregations across categories and payment channels.
4. Construct at least 6 distinct, publication-grade visualizations.
5. Formulate key business findings and executive insights."""),
        code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
print("Environment configured successfully!")"""),
        md_cell("""## 1. Dataset Loading & Inspection"""),
        code_cell("""df = pd.read_csv("datasets/ecommerce_sales_data.csv")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()"""),
        md_cell("""## 2. Schema and Missing Value Check"""),
        code_cell("""print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())
print(f"\nDuplicate records: {df.duplicated().sum()}")"""),
        md_cell("""## 3. Summary Statistics & Aggregations"""),
        code_cell("""print(df.describe().T)

category_summary = df.groupby("Product_Category").agg(
    Total_Revenue=("Total_Amount", "sum"),
    Average_Order_Value=("Total_Amount", "mean"),
    Units_Sold=("Quantity", "sum"),
    Avg_Rating=("Customer_Rating", "mean")
).sort_values(by="Total_Revenue", ascending=False)

category_summary"""),
        md_cell("""## 4. Visualizations (6 Required Plots)
### Plot 1: Total Amount Distribution (Histogram + KDE)"""),
        code_cell("""plt.figure(figsize=(9, 5))
sns.histplot(df["Total_Amount"], kde=True, color="#2b5c8f", bins=20)
plt.title("Distribution of Order Total Amount ($)", fontsize=14, fontweight="bold")
plt.xlabel("Total Amount ($)")
plt.ylabel("Frequency")
plt.show()"""),
        md_cell("""### Plot 2: Total Revenue by Category (Bar Chart)"""),
        code_cell("""plt.figure(figsize=(10, 5))
sns.barplot(data=category_summary.reset_index(), x="Product_Category", y="Total_Revenue", palette="Blues_r")
plt.title("Total Revenue by Product Category", fontsize=14, fontweight="bold")
plt.xlabel("Category")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=20)
plt.show()"""),
        md_cell("""### Plot 3: Age vs Spending by Gender (Scatter Plot)"""),
        code_cell("""plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="Age", y="Total_Amount", hue="Gender", s=80, alpha=0.85)
plt.title("Customer Age vs Spending by Gender", fontsize=14, fontweight="bold")
plt.xlabel("Customer Age")
plt.ylabel("Spending ($)")
plt.show()"""),
        md_cell("""### Plot 4: Spending across Payment Methods (Box Plot)"""),
        code_cell("""plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Payment_Method", y="Total_Amount", palette="Set2")
plt.title("Order Amount by Payment Method", fontsize=14, fontweight="bold")
plt.xlabel("Payment Method")
plt.ylabel("Amount ($)")
plt.xticks(rotation=15)
plt.show()"""),
        md_cell("""### Plot 5: Rating Distribution across Product Categories"""),
        code_cell("""plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Product_Category", hue="Customer_Rating", palette="viridis")
plt.title("Customer Ratings by Product Category", fontsize=14, fontweight="bold")
plt.xlabel("Product Category")
plt.ylabel("Number of Ratings")
plt.xticks(rotation=20)
plt.show()"""),
        md_cell("""### Plot 6: Correlation Heatmap of Numeric Variables"""),
        code_cell("""plt.figure(figsize=(8, 6))
num_cols = ["Age", "Quantity", "Unit_Price", "Total_Amount", "Customer_Rating"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f", square=True, linewidths=0.5)
plt.title("Correlation Matrix of Numeric Attributes", fontsize=14, fontweight="bold")
plt.show()"""),
        md_cell("""## 5. Analytical Summary & Business Insights
*(Document your key observations, top-selling categories, and customer spending habits below)*""")
    ]
    make_notebook(cells, nb_path)

def scaffold_week_03():
    # Generate dirty dataset with nulls, outliers, and duplicates
    np.random.seed(42)
    n = 200
    customer_ids = [f"CUST-{1000 + i}" for i in range(n)]
    ages = np.random.randint(18, 70, size=n).astype(float)
    ages[np.random.choice(n, 12, replace=False)] = np.nan # Missing ages
    
    tenures = np.random.randint(1, 72, size=n).astype(float)
    tenures[np.random.choice(n, 8, replace=False)] = np.nan # Missing tenures
    
    monthly_charges = np.random.uniform(20.0, 120.0, size=n)
    # Inject outliers
    monthly_charges[5] = 999.0
    monthly_charges[42] = 1250.0
    
    contract_types = np.random.choice(["Month-to-month", "One year", "Two year"], size=n)
    payment_methods = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n)
    churn = np.random.choice(["No", "Yes"], size=n, p=[0.73, 0.27])
    
    df_raw = pd.DataFrame({
        "CustomerID": customer_ids,
        "Age": ages,
        "Tenure_Months": tenures,
        "Monthly_Charges": np.round(monthly_charges, 2),
        "Contract_Type": contract_types,
        "Payment_Method": payment_methods,
        "Churn": churn
    })
    # Inject 5 duplicate rows
    duplicates = df_raw.iloc[:5].copy()
    df_raw = pd.concat([df_raw, duplicates], ignore_index=True)
    
    csv_path = os.path.join(BASE_DIR, "phase-1-python-data-fundamentals/week-03-data-cleaning-preprocessing/datasets/customer_churn_raw.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_raw.to_csv(csv_path, index=False)
    print(f"[OK] Generated Week 3 raw dataset: {df_raw.shape}")
    
    # Week 3 README
    write_file("phase-1-python-data-fundamentals/week-03-data-cleaning-preprocessing/README.md", """# Week 03 — Data Cleaning & Preprocessing
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Implement missing value imputation (Mean, Median, Mode, and algorithmic imputers).
2. Detect and remove semantic and exact duplicate records.
3. Detect statistical outliers using Z-score and Interquartile Range (IQR) fencing.
4. Apply categorical encoding techniques: Label Encoding vs One-Hot Encoding.
5. Apply feature scaling: Min-Max Normalization and Standardization (`StandardScaler`).
6. Partition datasets into train-test splits without data leakage.

---

## 🛠️ Practical Assignment: Data Preparation Pipeline
Take `datasets/customer_churn_raw.csv` and transform it into a clean, ML-ready dataset.

### Pipeline Requirements:
1. Identify missing value counts and apply justifiable imputation.
2. Remove duplicate rows.
3. Detect outliers in `Monthly_Charges` using IQR fences and handle them (capping/winsorizing).
4. One-Hot Encode categorical features (`Contract_Type`, `Payment_Method`).
5. Label Encode the binary target `Churn` (No -> 0, Yes -> 1).
6. Scale numerical features using `StandardScaler`.
7. Perform an 80/20 stratified train-test split.
8. Export the cleaned dataset to `datasets/customer_churn_cleaned.csv`.
""")

    # Week 3 Notebook
    nb_path = os.path.join(BASE_DIR, "phase-1-python-data-fundamentals/week-03-data-cleaning-preprocessing/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 03 — Data Cleaning & Preprocessing Pipeline
### Turn Raw Noisy Data into an ML-Ready Dataset
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

print("Preprocessing modules loaded!")"""),
        md_cell("""## 1. Load Raw Dataset & Inspect Data Issues"""),
        code_cell("""df = pd.read_csv("datasets/customer_churn_raw.csv")
print(f"Initial Shape: {df.shape}")
df.head(10)"""),
        md_cell("""## 2. Duplicate Detection & Removal"""),
        code_cell("""print(f"Duplicate rows found: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")"""),
        md_cell("""## 3. Missing Value Analysis & Imputation"""),
        code_cell("""print(df.isnull().sum())

# Median imputation for numerical features with skew/outliers
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Tenure_Months"] = df["Tenure_Months"].fillna(df["Tenure_Months"].median())
print("Missing values after imputation:\n", df.isnull().sum())"""),
        md_cell("""## 4. Outlier Detection & Capping (IQR Method)"""),
        code_cell("""Q1 = df["Monthly_Charges"].quantile(0.25)
Q3 = df["Monthly_Charges"].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

print(f"Monthly Charges IQR: {IQR:.2f} | Upper Fence: {upper_fence:.2f}")

# Check outliers
outliers = df[df["Monthly_Charges"] > upper_fence]
print(f"Detected {len(outliers)} outlier(s):")
print(outliers[["CustomerID", "Monthly_Charges"]])

# Cap outliers to the upper fence
df["Monthly_Charges"] = np.where(df["Monthly_Charges"] > upper_fence, upper_fence, df["Monthly_Charges"])
print("Outliers capped successfully!")"""),
        md_cell("""## 5. Categorical Encoding"""),
        code_cell("""# Binary encoding target variable
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])

# One-Hot Encoding nominal features
df_encoded = pd.get_dummies(df, columns=["Contract_Type", "Payment_Method"], drop_first=True, dtype=int)
df_encoded.head()"""),
        md_cell("""## 6. Train-Test Split & Feature Scaling"""),
        code_cell("""X = df_encoded.drop(columns=["CustomerID", "Churn"])
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
num_features = ["Age", "Tenure_Months", "Monthly_Charges"]
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
print(X_train.head())"""),
        md_cell("""## 7. Export Cleaned Dataset & Summary Report"""),
        code_cell("""df_encoded.to_csv("datasets/customer_churn_cleaned.csv", index=False)
print("Saved clean ML-ready dataset to datasets/customer_churn_cleaned.csv")""")
    ]
    make_notebook(cells, nb_path)

def scaffold_week_04():
    # Generate Employee dataset for EDA
    np.random.seed(42)
    n = 250
    departments = ["Engineering", "Sales", "Marketing", "HR", "Product"]
    education = ["Bachelor", "Master", "PhD", "High School"]
    
    ages = np.random.randint(22, 60, size=n)
    salaries = 40000 + (ages - 22) * 2200 + np.random.normal(0, 8000, size=n)
    experience = np.maximum(0, ages - 22 - np.random.randint(0, 4, size=n))
    satisfaction = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.15, 0.3, 0.3, 0.15])
    performance = np.random.choice(["Low", "Medium", "High", "Outstanding"], size=n, p=[0.1, 0.4, 0.35, 0.15])
    attrition = np.where((satisfaction <= 2) & (salaries < 60000), "Yes", 
                         np.random.choice(["No", "Yes"], size=n, p=[0.85, 0.15]))

    df = pd.DataFrame({
        "Emp_ID": [f"EMP-{2000 + i}" for i in range(n)],
        "Age": ages,
        "Department": np.random.choice(departments, size=n),
        "Education": np.random.choice(education, size=n),
        "Years_Experience": experience,
        "Salary": np.round(salaries, 2),
        "Job_Satisfaction": satisfaction,
        "Performance_Rating": performance,
        "Attrition": attrition
    })
    csv_path = os.path.join(BASE_DIR, "phase-1-python-data-fundamentals/week-04-exploratory-data-analysis/datasets/employee_attrition_data.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated Week 4 dataset: {df.shape}")

    write_file("phase-1-python-data-fundamentals/week-04-exploratory-data-analysis/README.md", """# Week 04 — Exploratory Data Analysis (EDA) Project
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Formulate analytical business hypotheses.
2. Conduct univariate, bivariate, and multivariate statistical exploration.
3. Identify correlation patterns, distribution anomalies, and feature importance indicators.
4. Answer 15 rigorous analytical questions using data and visualizations.

---

## 🛠️ Practical Assignment: Employee Attrition & Compensation EDA
Analyze `datasets/employee_attrition_data.csv` to answer 15 critical HR analytics questions:

1. What is the overall employee attrition rate?
2. Which department experiences the highest turnover?
3. What is the median salary across departments?
4. How strongly is years of experience correlated with salary?
5. Does job satisfaction correlate negatively with attrition?
6. Which education level commands the highest compensation?
7. How does performance rating vary across experience levels?
8. Are younger employees (<30 years) more likely to leave the organization?
9. What is the salary distribution shape (skewness & outliers)?
10. Is there a disparity between salary and job satisfaction ratings?
11. Which department boasts the highest percentage of top performers?
12. What is the average age of employees in each department?
13. How does employee retention vary between Master's and Bachelor's degree holders?
14. What are the top 3 distinguishing factors of employees who resigned?
15. What executive compensation or retention strategies should leadership implement?

---

## 📂 Deliverables
- Completed `starter_notebook.ipynb` addressing all 15 questions with 10+ visualizations.
- Comprehensive Executive EDA Report.
""")

    # Week 4 Notebook
    nb_path = os.path.join(BASE_DIR, "phase-1-python-data-fundamentals/week-04-exploratory-data-analysis/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 04 — Exploratory Data Analysis (EDA)
### Employee Retention & Compensation Analytics (15 Analytical Questions)
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams["figure.figsize"] = (10, 6)

df = pd.read_csv("datasets/employee_attrition_data.csv")
print(f"Loaded {df.shape[0]} employee records.")
df.head()"""),
        md_cell("""## Questions 1 to 5: High-Level Workforce & Salary Metrics"""),
        code_cell("""# Q1: Overall Attrition Rate
attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
print(f"Overall Attrition Rate: {attrition_rate:.2f}%")

# Q2: Attrition by Department
dept_attrition = df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
print("\nAttrition by Department (%):")
print(dept_attrition)"""),
        code_cell("""# Visual 1: Attrition Rate by Department
plt.figure(figsize=(9, 5))
dept_attrition.plot(kind='bar', color='#e74c3c')
plt.title("Attrition Rate by Department (%)", fontsize=14, fontweight="bold")
plt.ylabel("Attrition %")
plt.xlabel("Department")
plt.show()"""),
        md_cell("""## Questions 6 to 10: Compensation & Job Satisfaction Analysis"""),
        code_cell("""# Visual 2: Salary Distribution by Department (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Department", y="Salary", palette="Set3")
plt.title("Salary Distribution Across Departments", fontsize=14, fontweight="bold")
plt.show()"""),
        code_cell("""# Visual 3: Experience vs Salary (Scatter Plot with Regression Line)
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x="Years_Experience", y="Salary", scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title("Experience vs Salary Correlation", fontsize=14, fontweight="bold")
plt.show()"""),
        md_cell("""## Questions 11 to 15: Multivariate Relationships & Executive Takeaways
*(Interns: Implement the remaining visual analysis and summarize findings)*""")
    ]
    make_notebook(cells, nb_path)

# ==============================================================================
# RUN SCRIPTS
# ==============================================================================
if __name__ == "__main__":
    print("Scaffolding Phase 1...")
    scaffold_week_02()
    scaffold_week_03()
    scaffold_week_04()
    print("Phase 1 complete!")
