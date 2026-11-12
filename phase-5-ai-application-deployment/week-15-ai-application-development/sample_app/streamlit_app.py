"""
Streamlit Interactive AI Application
Next Gen Software Hub Pvt. Ltd. - AI & ML Internship Program
Demonstrates all 6 Application Requirements:
1. Enter/Upload input (Manual sliders + CSV file uploader)
2. Submit the input (Responsive submit triggers)
3. Process the input (Loading spinner & state feedback)
4. Receive prediction (Real-time inference)
5. View results clearly (Confidence gauge, colored metrics, styled badges)
6. Error handling (Defensive checks for missing fields or connection drops)
"""

import streamlit as st
import pandas as pd
import requests
import io

st.set_page_config(
    page_title="AI Risk Predictor | Next Gen Software Hub",
    page_icon="🤖",
    layout="wide"
)

# Header & Branding
st.title("🤖 AI Risk Assessment & Prediction Portal")
st.markdown("**Next Gen Software Hub Pvt. Ltd. — Week 15 AI Application Development**")
st.write("Evaluate loan applicant default risk through live API inference or bulk batch CSV file upload.")

API_URL = "http://127.0.0.1:8001/predict"

tabs = st.tabs(["📝 Single Applicant Prediction", "📁 Batch CSV File Upload", "⚙️ System Health & API Status"])

# TAB 1: Single Applicant Input Form
with tabs[0]:
    st.subheader("Applicant Financial Profile")
    st.info("Requirement 1: Enter applicant parameters using the interactive input controls below.")

    with st.form("single_applicant_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            credit_score = st.slider("Credit Score (300 - 850)", min_value=300, max_value=850, value=710, step=5,
                                     help="Applicant FICO credit score rating")
        with col2:
            annual_income = st.number_input("Annual Income ($ USD)", min_value=1000.0, max_value=1000000.0,
                                            value=65000.0, step=2500.0, help="Verified annual gross salary")
        with col3:
            loan_amount = st.number_input("Requested Loan Amount ($ USD)", min_value=500.0, max_value=500000.0,
                                          value=15000.0, step=1000.0, help="Total requested principal")

        # Requirement 2: Submit the input
        submitted = st.form_submit_button("⚡ Submit for AI Risk Assessment", use_container_width=True)

    if submitted:
        # Requirement 3: Process the input with feedback
        with st.spinner("Processing applicant parameters through AI Risk Inference Engine..."):
            payload = {
                "credit_score": credit_score,
                "annual_income": annual_income,
                "loan_amount": loan_amount
            }
            try:
                response = requests.post(API_URL, json=payload, timeout=6)
                if response.status_code == 200:
                    # Requirement 4 & 5: Receive and view results clearly
                    result = response.json()
                    prob = result["default_probability"] * 100
                    risk_level = result["risk_level"]
                    recommendation = result["recommendation"]

                    st.divider()
                    st.subheader("🎯 Assessment Outcome")
                    res_col1, res_col2, res_col3 = st.columns(3)

                    with res_col1:
                        if risk_level == "Low Risk":
                            st.success(f"### Status: {risk_level}")
                        else:
                            st.error(f"### Status: {risk_level}")
                    with res_col2:
                        st.metric("Estimated Default Probability", f"{prob:.1f}%")
                    with res_col3:
                        st.info(f"**Recommendation:**\n\n{recommendation}")

                    st.write("**Risk Meter:**")
                    st.progress(min(prob / 100.0, 1.0))
                else:
                    # Requirement 6: Receive appropriate error messages
                    st.error(f"⚠️ API Validation Error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                # Requirement 6: Descriptive connection error
                st.error("⚠️ **Connection Error:** Could not reach the FastAPI backend server.")
                st.warning("Please verify that the backend is running at `http://127.0.0.1:8000` via:\n\n"
                           "`uvicorn api:app --reload --port 8000`")

# TAB 2: Batch CSV File Upload
with tabs[1]:
    st.subheader("Batch Evaluation via CSV Upload")
    st.write("Requirement 1: Upload a `.csv` file containing applicant records (`credit_score`, `annual_income`, `loan_amount`).")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head(), use_container_width=True)

            required_cols = {"credit_score", "annual_income", "loan_amount"}
            if not required_cols.issubset(set(batch_df.columns)):
                st.error(f"⚠️ CSV must contain columns: {required_cols}. Found: {list(batch_df.columns)}")
            else:
                if st.button("🚀 Process Batch Predictions", use_container_width=True):
                    with st.spinner("Processing batch records through inference gateway..."):
                        preds = []
                        for _, row in batch_df.iterrows():
                            req_payload = {
                                "credit_score": int(row["credit_score"]),
                                "annual_income": float(row["annual_income"]),
                                "loan_amount": float(row["loan_amount"])
                            }
                            try:
                                r = requests.post(API_URL, json=req_payload, timeout=3)
                                if r.status_code == 200:
                                    res = r.json()
                                    preds.append({
                                        "Risk_Level": res["risk_level"],
                                        "Default_Probability_%": round(res["default_probability"] * 100, 2),
                                        "Decision": res["recommendation"]
                                    })
                                else:
                                    preds.append({"Risk_Level": "Error", "Default_Probability_%": None, "Decision": "API Error"})
                            except requests.exceptions.RequestException:
                                preds.append({"Risk_Level": "Failed", "Default_Probability_%": None, "Decision": "Connection Error"})

                        results_df = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(preds)], axis=1)
                        st.success(f"Batch processing completed for {len(results_df)} applicants!")
                        st.dataframe(results_df, use_container_width=True)

                        # CSV Download button
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Scored Batch Results CSV",
                            data=csv_buffer.getvalue(),
                            file_name="scored_loan_applicants.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"⚠️ Error reading uploaded CSV: {e}")

# TAB 3: System Health Check
with tabs[2]:
    st.subheader("API Gateway Health & Diagnostic Check")
    try:
        health_resp = requests.get("http://127.0.0.1:8001/health", timeout=3)
        if health_resp.status_code == 200:
            st.success("🟢 FastAPI Backend Service: ACTIVE & HEALTHY")
            st.json(health_resp.json())
        else:
            st.warning(f"🟡 Backend returned non-200 status code: {health_resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("🔴 FastAPI Backend Service: OFFLINE / DISCONNECTED")
        st.info("Start the backend server in your terminal: `uvicorn api:app --reload --port 8001`")
