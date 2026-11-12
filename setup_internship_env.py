"""
Environment Diagnostic Checker
Next Gen Software Hub Pvt. Ltd. - AI & ML Internship Program
Run this script to verify that your Python development environment is configured properly.
"""

import sys
import importlib
import platform

REQUIRED_PACKAGES = {
    # Phase 1: Python & Data Fundamentals
    "numpy": "NumPy (Numerical Computing)",
    "pandas": "Pandas (Data Manipulation)",
    "scipy": "SciPy (Scientific Algorithms)",
    "matplotlib": "Matplotlib (Plotting)",
    "seaborn": "Seaborn (Statistical Data Viz)",
    
    # Phase 2 & 3: Machine Learning & Ensembles
    "sklearn": "Scikit-Learn (Machine Learning)",
    "joblib": "Joblib (Model Serialization)",
    
    # Phase 4: Deep Learning & Vision
    "torch": "PyTorch (Deep Learning Framework)",
    "cv2": "OpenCV (Computer Vision)",
    "PIL": "Pillow (Image Processing)",
    
    # Phase 5: NLP & Transformers
    "transformers": "Hugging Face Transformers (NLP/LLMs)",
    
    # Phase 6: Deployment
    "fastapi": "FastAPI (REST API Development)",
    "uvicorn": "Uvicorn (ASGI Server)",
    "pydantic": "Pydantic (Data Validation)",
    "streamlit": "Streamlit (Interactive UI)",
    
    # Utilities
    "pytest": "Pytest (Unit Testing)"
}

def print_header():
    print("=" * 70)
    print("   NEXT GEN SOFTWARE HUB PVT. LTD. - AI & ML INTERNSHIP")
    print("               ENVIRONMENT READINESS CHECKER           ")
    print("=" * 70)
    print(f" Python Executable : {sys.executable}")
    print(f" Python Version    : {platform.python_version()}")
    print(f" Operating System  : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print("-" * 70)

def check_python_version():
    major, minor = sys.version_info.major, sys.version_info.minor
    if major >= 3 and minor >= 9:
        print(f" [PASS] Python Version: {major}.{minor} is supported (Recommended: 3.10+)")
        return True
    else:
        print(f" [FAIL] Python Version: {major}.{minor} is outdated! Python 3.9 or higher is required.")
        return False

def check_packages():
    print("\nVerifying Core Libraries...")
    print("-" * 70)
    
    passed = []
    missing = []
    
    for module_name, description in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f" [PASS] {description:<42} : v{ver}")
            passed.append(module_name)
        except ImportError:
            print(f" [MISSING] {description:<39} : NOT FOUND")
            missing.append(module_name)
            
    print("-" * 70)
    print(f" Summary: {len(passed)}/{len(REQUIRED_PACKAGES)} packages available.")
    
    if missing:
        print("\n Recommended Action:")
        print(" Some packages are missing. Install them by running:")
        print(f"   pip install -r requirements.txt")
        print(" Or install missing directly:")
        # Map back to pip package name
        pip_map = {"sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow"}
        pip_pkgs = [pip_map.get(m, m) for m in missing]
        print(f"   pip install {' '.join(pip_pkgs)}")
        return False
    else:
        print("\n [SUCCESS] Your environment is fully configured for all 16 weeks!")
        return True

def main():
    print_header()
    py_ok = check_python_version()
    pkg_ok = check_packages()
    print("=" * 70)
    if py_ok and pkg_ok:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
