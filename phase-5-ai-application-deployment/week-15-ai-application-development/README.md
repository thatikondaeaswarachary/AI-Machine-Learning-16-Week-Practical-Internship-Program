# Week 15 — AI Application Development
### Task: Build an AI-Powered Interactive Application
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
By the end of this week, students will transition from backend modeling to user-facing AI products:
- Creating intuitive web graphical interfaces for AI/ML inference.
- Handling interactive single-instance predictions and batch file uploads.
- Integrating model inference with real-time feedback, loading states, and result visualizations.
- Implementing defensive user input validation and informative error messaging.

---

## 🛠️ Practical Assignment
Take one of your trained models from previous weeks (or the provided pre-trained loan risk / churn model) and build a complete interactive web application.

### Recommended UI Frameworks:
- **Streamlit** *(Recommended for rapid Python-native dashboard and app development)*
- **Flask** *(Lightweight web framework with custom HTML/CSS templates)*
- **Gradio** *(Quick UI prototyping for ML/DL models)*

### Suggested Applications (Select One):
1. **House Price Predictor** (Regression)
2. **Student Performance / Pass-Fail Predictor** (Classification)
3. **Spam Message Detector** (NLP Classification)
4. **Sentiment Analyzer** (NLP Sentiment)
5. **Image Classifier** (Computer Vision CNN)
6. **Customer Churn / Loan Default Risk Predictor** (Pre-built reference in `sample_app/`)

---

## 📋 Mandatory Application Requirements
Your web application must satisfy all six functional criteria:
1. **Enter / Upload Input:** Users must be able to input values manually via sliders/fields OR upload a CSV/image file.
2. **Submit the Input:** A prominent, responsive submission button triggers the evaluation.
3. **Process the Input:** The UI provides visual feedback (e.g. `st.spinner`, progress bar) during inference processing.
4. **Receive a Prediction:** Real-time generation of the predicted label or continuous value.
5. **View Results Clearly:** Results rendered cleanly with confidence percentages, metric indicators, and color-coded status badges (green for pass/low risk, red for high risk/churn).
6. **Descriptive Error Messages:** Friendly, actionable error notifications for out-of-range values, invalid file formats, or disconnected services.

---

## 🚀 Running the Reference Application

The included `sample_app/` provides a production-grade Loan Risk Assessment Engine with a FastAPI REST backend and a Streamlit dashboard:

```bash
# Navigate to the sample app folder
cd "phase-5-ai-application-deployment/week-15-ai-application-development/sample_app"

# Terminal 1: Launch FastAPI Backend Server
uvicorn api:app --reload --port 8000

# Terminal 2: Launch Streamlit Web UI
streamlit run streamlit_app.py
```

---

## 📋 Submission Deliverables
- **Source Code:** Complete application files (`app.py`, `schemas.py`, `streamlit_app.py`, etc.).
- **Trained Model:** Serialized pipeline artifact (`.joblib` or `.pkl`).
- **Screenshots:** High-resolution screenshots of the UI in action (input form, successful prediction, and error handling states).
- **GitHub Repository Link:** Clean repository with setup instructions.
- **Live Demo Link:** Streamlit Community Cloud / Hugging Face Spaces / Render link (if deployed publicly).
