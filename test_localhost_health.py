"""
Diagnostic Script: Verify Localhost 8000 and 8501 Status
"""

import requests
import json
import time

def check_backend():
    print("=" * 65)
    print("1. VERIFYING FASTAPI BACKEND (http://127.0.0.1:8001)")
    print("=" * 65)
    try:
        # Health check
        t0 = time.time()
        resp = requests.get("http://127.0.0.1:8001/health", timeout=3)
        latency = (time.time() - t0) * 1000
        if resp.status_code == 200:
            print(f" [PASS] Health Endpoint : 200 OK ({latency:.1f}ms)")
            print(f"        Payload         : {resp.json()}")
        else:
            print(f" [FAIL] Health Endpoint returned status {resp.status_code}")
            return False

        # Prediction check
        payload = {"credit_score": 750, "annual_income": 80000.0, "loan_amount": 12000.0}
        t0 = time.time()
        resp_pred = requests.post("http://127.0.0.1:8001/predict", json=payload, timeout=3)
        latency_pred = (time.time() - t0) * 1000
        if resp_pred.status_code == 200:
            print(f" [PASS] Predict Endpoint: 200 OK ({latency_pred:.1f}ms)")
            print(f"        Sent Payload    : {payload}")
            print(f"        Model Response  : {resp_pred.json()}")
            return True
        else:
            print(f" [FAIL] Predict Endpoint returned status {resp_pred.status_code}")
            return False
    except Exception as e:
        print(f" [FAIL] Could not connect to FastAPI: {e}")
        return False

def check_frontend():
    print("\n" + "=" * 65)
    print("2. VERIFYING STREAMLIT FRONTEND (http://localhost:8501)")
    print("=" * 65)
    try:
        t0 = time.time()
        resp = requests.get("http://localhost:8501", timeout=5)
        latency = (time.time() - t0) * 1000
        if resp.status_code == 200:
            print(f" [PASS] Streamlit Server: 200 OK ({latency:.1f}ms)")
            print(f"        Page Title Tag  : Found in HTML: {'<title>' in resp.text}")
            print(f"        Content Size    : {len(resp.content):,} bytes")
            return True
        else:
            print(f" [FAIL] Streamlit returned status {resp.status_code}")
            return False
    except Exception as e:
        print(f" [FAIL] Could not connect to Streamlit: {e}")
        return False

def main():
    b_ok = check_backend()
    f_ok = check_frontend()
    print("\n" + "=" * 65)
    print(f" OVERALL STATUS: {'PROPERLY WORKING & OPERATIONAL' if (b_ok and f_ok) else 'ISSUES DETECTED'}")
    print("=" * 65)

if __name__ == "__main__":
    main()
