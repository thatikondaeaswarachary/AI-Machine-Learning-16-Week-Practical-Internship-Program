"""
Scaffold Generator for Phase 6: Model Deployment & Capstone (Weeks 15 & 16)
Next Gen Software Hub Pvt. Ltd. - AI & ML Internship Program
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def write_file(rel_path, content):
    full_path = os.path.join(BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Written File: {rel_path}")

# ==============================================================================
# WEEK 15: MODEL DEPLOYMENT (FASTAPI + STREAMLIT)
# ==============================================================================
def scaffold_week_15():
    # 1. Train a lightweight loan risk model and save it to artifact
    np.random.seed(42)
    n = 200
    credit_score = np.random.randint(550, 850, size=n)
    income = np.random.randint(30000, 150000, size=n)
    loan_amount = np.random.randint(5000, 50000, size=n)
    prob = 1 / (1 + np.exp(-(-2.0 + (720 - credit_score)*0.01 + (loan_amount/income)*1.5)))
    default = (np.random.rand(n) < prob).astype(int)
    
    X = np.column_stack([credit_score, income, loan_amount])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, default)

    model_dir = os.path.join(BASE_DIR, "phase-6-deployment-capstone/week-15-model-deployment-fastapi-streamlit/sample_app/artifacts")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, os.path.join(model_dir, "loan_model.joblib"))
    print(f"[OK] Saved model artifact to {model_dir}/loan_model.joblib")

    # 2. FastAPI Backend
    api_code = '''"""
FastAPI Model Inference Service
Provides RESTful endpoints for loan risk assessment.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="Next Gen AI - Loan Risk Prediction API",
    description="Production-grade model inference service using FastAPI & Pydantic.",
    version="1.0.0"
)

# Load Model Pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "loan_model.joblib")
try:
    bundle = joblib.load(MODEL_PATH)
    scaler = bundle["scaler"]
    model = bundle["model"]
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

class LoanApplication(BaseModel):
    credit_score: int = Field(..., ge=300, le=850, description="Applicant credit score (300-850)")
    annual_income: float = Field(..., gt=1000, description="Annual income in USD")
    loan_amount: float = Field(..., gt=500, description="Requested loan amount in USD")

class PredictionResponse(BaseModel):
    risk_level: str
    default_probability: float
    recommendation: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Loan Risk API", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict_loan_risk(application: LoanApplication):
    try:
        features = np.array([[application.credit_score, application.annual_income, application.loan_amount]])
        features_scaled = scaler.transform(features)
        
        prob = float(model.predict_proba(features_scaled)[0, 1])
        risk_level = "High Risk" if prob >= 0.50 else "Low Risk"
        recommendation = "Reject Loan Application" if prob >= 0.50 else "Approve Loan Application"
        
        return PredictionResponse(
            risk_level=risk_level,
            default_probability=round(prob, 4),
            recommendation=recommendation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
    write_file("phase-6-deployment-capstone/week-15-model-deployment-fastapi-streamlit/sample_app/api.py", api_code)

    # 3. Streamlit Frontend
    streamlit_code = '''"""
Streamlit Interactive Dashboard
Frontend client consuming the FastAPI inference service.
"""

import streamlit as st
import requests

st.set_page_config(
    page_title="Loan Risk Assessment Portal",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Loan Risk Assessment Engine")
st.markdown("**Next Gen Software Hub Pvt. Ltd. — Model Deployment Demo**")
st.write("Enter applicant financial details to generate a real-time risk score.")

API_URL = "http://127.0.0.1:8000/predict"

with st.form("loan_form"):
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=710, step=5)
        annual_income = st.number_input("Annual Income ($)", min_value=5000.0, value=65000.0, step=1000.0)
    with col2:
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000.0, value=15000.0, step=500.0)

    submitted = st.form_submit_button("⚡ Evaluate Risk")

if submitted:
    payload = {
        "credit_score": credit_score,
        "annual_income": annual_income,
        "loan_amount": loan_amount
    }
    
    with st.spinner("Analyzing credit parameters via API..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                prob = result["default_probability"] * 100
                
                st.divider()
                st.subheader("Assessment Result")
                if result["risk_level"] == "Low Risk":
                    st.success(f"✅ **{result['recommendation']}**")
                else:
                    st.error(f"⚠️ **{result['recommendation']}**")
                    
                st.metric("Estimated Default Probability", f"{prob:.1f}%")
                st.progress(min(prob / 100.0, 1.0))
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            st.warning("⚠️ Could not connect to FastAPI backend. Ensure `uvicorn api:app --reload` is running on port 8000.")
'''
    write_file("phase-6-deployment-capstone/week-15-model-deployment-fastapi-streamlit/sample_app/streamlit_app.py", streamlit_code)

    # 4. Week 15 README
    write_file("phase-6-deployment-capstone/week-15-model-deployment-fastapi-streamlit/README.md", """# Week 15 — AI Model Deployment with FastAPI & Streamlit
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Understand model serving architectures and serialization with `joblib`.
2. Build high-performance RESTful API endpoints using **FastAPI** with **Pydantic** input validation.
3. Build an interactive web frontend using **Streamlit** to collect user input and render live predictions.
4. Containerization basics: Understand how Docker encapsulates the frontend and backend.

---

## 🚀 Running the Sample Application

### Step 1: Start the FastAPI Backend
Open a terminal and run:
```bash
cd "phase-6-deployment-capstone/week-15-model-deployment-fastapi-streamlit/sample_app"
uvicorn api:app --reload --port 8000
```
- API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### Step 2: Start the Streamlit Frontend
Open a second terminal and run:
```bash
cd "phase-6-deployment-capstone/week-15-model-deployment-fastapi-streamlit/sample_app"
streamlit run streamlit_app.py
```
- The web app will launch automatically at `http://localhost:8501`.
""")

# ==============================================================================
# WEEK 16: CAPSTONE PROJECT & PORTFOLIO
# ==============================================================================
def scaffold_week_16():
    write_file("phase-6-deployment-capstone/week-16-capstone-project-portfolio/README.md", """# Week 16 — Capstone Project & Portfolio Defense
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Capstone Objective
The Capstone Project is the culmination of your 16-week journey. Interns will architect, build, evaluate, deploy, and defend an end-to-end production AI system that addresses a real-world problem.

---

## 🏆 Approved Capstone Domains
Interns may choose from the following industry domains or propose a custom domain for mentor approval:
1. **Healthcare & Bio-Informatics:** Medical imaging diagnosis (e.g. Chest X-Ray pneumonia classification) or patient readmission prediction.
2. **FinTech & Risk Analytics:** Real-time credit card fraud detection system or automated stock algorithmic signals.
3. **E-Commerce & Retail Intelligence:** Hybrid recommendation engine (collaborative + content-based) with customer churn forecasting.
4. **Autonomous Systems & Vision:** Object detection for safety compliance (PPE kit detection) or automated vehicle damage assessment.
5. **Generative AI & Enterprise Intelligence:** Domain-specific Retrieval-Augmented Generation (RAG) assistant for financial or legal contracts.

---

## 📋 Mandatory Deliverables
1. **GitHub Repository:** Clean modular structure, reproducible environment (`requirements.txt`), comprehensive `README.md` with system architecture diagrams.
2. **Working Web Deployment:** Live interactive application (Streamlit or FastAPI + React).
3. **Technical Capstone Report:** Follow `capstone_template/final_report_template.md`.
4. **Project Defense Presentation:** 10-slide deck followed by live Q&A with the mentorship panel.
""")

    # Capstone Proposal Template
    write_file("phase-6-deployment-capstone/week-16-capstone-project-portfolio/capstone_template/proposal_template.md", """# Capstone Project Proposal
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**

- **Project Title:** [E.g., Automated Chest X-Ray Pathology Detection Engine]
- **Student Name:** [Your Name]
- **Domain:** [Healthcare / FinTech / E-Commerce / Computer Vision / NLP]
- **Date:** [YYYY-MM-DD]

---

### 1. Problem Statement & Motivation
*Describe the real-world problem your system aims to solve and its business or societal impact (150-250 words).*

### 2. Dataset Description
- **Dataset Source:** [Kaggle / Hugging Face / UCI / Custom]
- **Volume & Features:** [Number of rows/images, number of features, target variable]
- **Data Licensing:** [Permitted for academic/internship usage]

### 3. Proposed Methodology & Pipeline
- **Preprocessing:** [Techniques planned: imputation, normalization, tokenization, augmentation]
- **Candidate Models:** [Baseline model vs 2+ advanced/deep models]
- **Primary Metric:** [F1-Score / ROC-AUC / RMSE]
- **Deployment Strategy:** [FastAPI backend + Streamlit frontend]

### 4. 2-Week Sprint Timeline
- **Week 15 Milestone:** Data pipeline complete, baseline model trained and logged.
- **Week 16 Milestone:** Model tuned, API packaged, frontend operational, defense presentation prepared.
""")

    # Capstone Final Report Template
    write_file("phase-6-deployment-capstone/week-16-capstone-project-portfolio/capstone_template/final_report_template.md", """# Capstone Final Technical Report
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

### Project Title: [Your Capstone Project Title]
**Author:** [Your Full Name]  
**Mentorship Track:** AI & Machine Learning  
**Completion Date:** [YYYY-MM-DD]  

---

## Abstract
*A concise 150-word summary of the business problem, dataset, winning architecture, quantitative results, and deployment mechanism.*

---

## 1. Introduction & Problem Scope
*Background context, target user persona, and business value proposition.*

## 2. System Architecture
```
[User / Client] ---> [Streamlit UI] ---> [FastAPI REST Gateway] ---> [ML Inference Pipeline]
                                                                             │
                                                                   [Trained Model Artifact]
```

## 3. Data Pipeline & Exploratory Analysis
- Data ingestion, cleansing, anomaly treatment.
- Key statistical observations and correlation findings.

## 4. Modeling & Experimental Results
| Model Architecture | Accuracy / MAE | Precision / RMSE | Recall | F1 / R² Score | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline Learner | ... | ... | ... | ... | ... |
| Candidate 1 | ... | ... | ... | ... | ... |
| **Winning Architecture** | **...** | **...** | **...** | **...** | **...** |

## 5. Deployment & Production Considerations
- FastAPI endpoint specifications (`/predict`, `/health`).
- Error handling and input validation safeguards.
- Scaling and monitoring recommendations.

## 6. Conclusion & Future Roadmap
*Lessons learned, edge case limitations, and future enhancements.*
""")

if __name__ == "__main__":
    print("Scaffolding Phase 6...")
    scaffold_week_15()
    scaffold_week_16()
    print("Phase 6 complete!")
