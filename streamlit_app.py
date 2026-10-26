"""
streamlit_app.py  —  Diabetes Risk Predictor (web UI)
-----------------------------------------------------
A Streamlit front end that calls the FastAPI /predict route.
It never loads model.pkl itself — it talks to the API over HTTP only.

Run (in two terminals):
    uvicorn main:app --reload
    streamlit run streamlit_app.py
"""

import requests
import streamlit as st

# ── 1. Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"
START_HINT = (
    "Could not reach the prediction service. Please start the FastAPI server "
    "first with `uvicorn main:app --reload`, then try again."
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    .hero {
        background: linear-gradient(135deg, #0f4c81 0%, #1b7fbd 100%);
        color: #fff; padding: 1.75rem 2rem; border-radius: 14px; margin-bottom: 1.5rem;
    }
    .hero h1 { color: #fff; margin: 0 0 .35rem 0; font-size: 2.1rem; }
    .hero p  { color: #dceaf7; margin: 0; font-size: 1.02rem; }
    .section-label {
        font-size: .78rem; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; color: #1b7fbd; margin: .25rem 0 .5rem 0;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .9rem 1rem;
    }
    div.stButton > button {
        width: 100%; height: 3rem; font-weight: 600; font-size: 1.05rem; border-radius: 10px;
    }
    .footer { text-align: center; color: #888; font-size: .82rem; margin-top: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_api():
    """Return True if the FastAPI service is up and its model is loaded."""
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except requests.exceptions.RequestException:
        return False


# ── Sidebar: service status + about ──────────────────────────────────────────
with st.sidebar:
    st.header("🩺 Service status")
    if check_api():
        st.success("API online · model loaded")
    else:
        st.error("API offline")
        st.caption("Start it with `uvicorn main:app --reload`")
    st.caption(f"Endpoint: `{API_URL}`")

    st.divider()
    st.subheader("About")
    st.write(
        "This tool estimates diabetes risk using a K-Nearest Neighbours model "
        "trained on the Pima Indians Diabetes dataset (768 patients)."
    )
    st.write(
        "The interface sends your inputs to a separate FastAPI service, "
        "which runs the model and returns the result."
    )
    st.divider()
    st.caption(
        "⚠️ For educational purposes only. Not a medical diagnosis — "
        "consult a healthcare professional."
    )

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Diabetes Risk Predictor")
st.markdown(
    """
    <div class="hero">
        <p>Enter the patient's clinical measurements below and click <b>Predict</b>
        to get an instant risk assessment with a confidence score.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 2. Eight input widgets, grouped into three columns ───────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="section-label">Patient profile</div>', unsafe_allow_html=True)
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=2, step=1,
                                  help="Number of times pregnant")
    age = st.slider("Age (years)", min_value=18, max_value=90, value=30)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5,
                          value=0.5, step=0.01, format="%.3f",
                          help="Family-history diabetes score")

with col2:
    st.markdown('<div class="section-label">Blood measurements</div>', unsafe_allow_html=True)
    glucose = st.slider("Glucose (mg/dL)", min_value=0, max_value=200, value=120,
                        help="Plasma glucose concentration")
    insulin = st.slider("Insulin (mu U/ml)", min_value=0, max_value=850, value=79,
                        help="2-hour serum insulin")
    blood_pressure = st.slider("Blood Pressure (mm Hg)", min_value=0, max_value=130, value=70,
                               help="Diastolic blood pressure")

with col3:
    st.markdown('<div class="section-label">Body measurements</div>', unsafe_allow_html=True)
    bmi = st.slider("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                    help="Body mass index (kg/m²)")
    skin_thickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=20,
                               help="Triceps skin fold thickness")

st.write("")

# ── 3. Predict button ────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    clicked = st.button("🔍 Predict", type="primary")

if clicked:
    # Field names match the PatientFeatures model in main.py exactly.
    payload = {
        "Pregnancies": int(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(dpf),
        "Age": int(age),
    }

    # ── 5. Graceful failure ──────────────────────────────────────────────────
    result = None
    try:
        with st.spinner("Analysing patient data..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
        else:
            st.warning(f"The prediction service responded with status "
                       f"{response.status_code}. {START_HINT}")
    except requests.exceptions.RequestException:
        st.warning(START_HINT)

    # ── 4. Result display ────────────────────────────────────────────────────
    if result:
        st.divider()
        st.subheader("Result")
        res_col, metric_col = st.columns([2, 1], gap="large")

        with res_col:
            if result["result_label"] == "Diabetic":
                st.error("### ⚠️ Diabetic\nThe model indicates a **high risk** of diabetes. "
                         "A follow-up with a healthcare professional is recommended.")
            else:
                st.success("### ✅ Not Diabetic\nThe model indicates a **low risk** of diabetes. "
                           "Keep up healthy habits and regular check-ups.")

        with metric_col:
            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
            st.progress(min(max(result["confidence"], 0.0), 1.0))

        with st.expander("View submitted values"):
            st.table({"Feature": list(payload.keys()),
                      "Value": [str(v) for v in payload.values()]})

st.markdown(
    '<div class="footer">Diabetes Risk Predictor · Streamlit front end + FastAPI model service</div>',
    unsafe_allow_html=True,
)
