"""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=False)
