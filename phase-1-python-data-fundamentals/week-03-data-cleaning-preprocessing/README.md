# Week 03 — Data Cleaning & Preprocessing
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
