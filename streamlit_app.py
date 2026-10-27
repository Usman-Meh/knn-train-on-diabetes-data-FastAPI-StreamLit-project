"""
streamlit_app.py  —  Diabetes Risk Predictor (web UI)
-----------------------------------------------------
A Streamlit front end that calls the FastAPI service.
It never loads model.pkl itself — it talks to the API over HTTP only:
    GET  /         service health
    GET  /stats    dataset + model summary (for the charts)
    POST /predict  the prediction

Run (in two terminals):
    uvicorn main:app --reload
    streamlit run streamlit_app.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── 1. Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "https://lisha-unchanging-dan.ngrok-free.dev"
START_HINT = (
    "Could not reach the prediction service. Please start the FastAPI server "
    "first with `uvicorn main:app --reload`, then try again."
)

# Series colours — each identity keeps its colour across every chart.
C_PATIENT = "#2a78d6"   # this patient
C_DIABETIC = "#eb6834"  # diabetic group
C_HEALTHY = "#1baf7a"   # non-diabetic group
STATUS_COLORS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
                 "critical": "#d03b3b", "na": "#9a9893"}
STATUS_ICONS = {"good": "✓", "warning": "!", "serious": "▲", "critical": "✕", "na": "–"}

# Display name and unit for each API field.
FEATURES = {
    "Pregnancies": ("Pregnancies", ""),
    "Glucose": ("Glucose", "mg/dL"),
    "BloodPressure": ("Blood pressure", "mm Hg"),
    "SkinThickness": ("Skin thickness", "mm"),
    "Insulin": ("Insulin", "µU/ml"),
    "BMI": ("BMI", "kg/m²"),
    "DiabetesPedigreeFunction": ("Pedigree function", ""),
    "Age": ("Age", "years"),
}

DEFAULTS = {"Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
            "Insulin": 79, "BMI": 25.0, "DiabetesPedigreeFunction": 0.5, "Age": 30}
PRESETS = {
    "Low-risk profile": {"Pregnancies": 1, "Glucose": 95, "BloodPressure": 68,
                         "SkinThickness": 22, "Insulin": 70, "BMI": 23.5,
                         "DiabetesPedigreeFunction": 0.25, "Age": 26},
    "High-risk profile": {"Pregnancies": 6, "Glucose": 185, "BloodPressure": 84,
                          "SkinThickness": 35, "Insulin": 200, "BMI": 38.5,
                          "DiabetesPedigreeFunction": 1.1, "Age": 52},
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1250px; }
    .hero {
        background: linear-gradient(135deg, #0f4c81 0%, #1b7fbd 100%);
        color: #fff; padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.25rem;
    }
    .hero p { color: #e3eef9; margin: 0; font-size: 1.02rem; line-height: 1.5; }
    .hero .tag {
        display: inline-block; background: rgba(255,255,255,.16); color: #fff;
        border-radius: 999px; padding: .15rem .65rem; font-size: .75rem;
        font-weight: 600; letter-spacing: .04em; margin: 0 .35rem .6rem 0;
    }
    .section-label {
        font-size: .76rem; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; color: #1b7fbd; margin: .25rem 0 .4rem 0;
    }
    .pill {
        display: inline-flex; align-items: center; gap: .4rem; font-size: .8rem;
        padding: .15rem .6rem; border-radius: 999px; margin: -.4rem 0 .8rem 0;
        border: 1px solid rgba(128,128,128,.28);
    }
    .pill .dot {
        width: 1.05rem; height: 1.05rem; border-radius: 50%; color: #fff;
        font-size: .68rem; font-weight: 700; display: inline-flex;
        align-items: center; justify-content: center;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .8rem 1rem;
    }
    div.stButton > button { font-weight: 600; border-radius: 10px; }
    .footer { text-align: center; color: #888; font-size: .8rem; margin-top: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API helpers ──────────────────────────────────────────────────────────────
def check_api():
    """Return (online, latency_ms) for the FastAPI service."""
    try:
        start = time.perf_counter()
        r = requests.get(f"{API_URL}/", timeout=3)
        latency = (time.perf_counter() - start) * 1000
        return r.status_code == 200 and r.json().get("model_loaded", False), latency
    except requests.exceptions.RequestException:
        return False, None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stats():
    # Raises on failure, so a failed call is never cached.
    r = requests.get(f"{API_URL}/stats", timeout=5)
    r.raise_for_status()
    return r.json()


def load_stats():
    try:
        return fetch_stats()
    except (requests.exceptions.RequestException, ValueError):
        return None


# ── Clinical reference labels ────────────────────────────────────────────────
def classify(feature, value):
    """Return (status, label) from common clinical reference ranges."""
    if value == 0 and feature in ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"):
        return "na", "Not recorded"
    if feature == "Glucose":  # 2-hour oral glucose tolerance test
        if value < 140:
            return "good", "Normal tolerance (<140)"
        if value < 200:
            return "warning", "Impaired tolerance (140–199)"
        return "critical", "Diabetic range (≥200)"
    if feature == "BloodPressure":  # diastolic
        if value < 60:
            return "warning", "Low (<60)"
        if value < 80:
            return "good", "Normal (<80)"
        if value < 90:
            return "serious", "Hypertension stage 1 (80–89)"
        return "critical", "Hypertension stage 2 (≥90)"
    if feature == "BMI":
        if value < 18.5:
            return "warning", "Underweight (<18.5)"
        if value < 25:
            return "good", "Healthy weight (18.5–24.9)"
        if value < 30:
            return "warning", "Overweight (25–29.9)"
        if value < 35:
            return "serious", "Obese class I (30–34.9)"
        return "critical", "Obese class II+ (≥35)"
    if feature == "Age":
        if value < 35:
            return "good", "Under 35"
        if value < 45:
            return "warning", "35–44 · screening advised"
        return "serious", "45+ · higher-risk age group"
    if feature == "Insulin":
        return ("good", "Typical (≤166)") if value <= 166 else ("serious", "Elevated (>166)")
    if feature == "SkinThickness":
        return ("good", "Typical (≤35)") if value <= 35 else ("warning", "Above typical (>35)")
    if feature == "DiabetesPedigreeFunction":
        if value < 0.5:
            return "good", "Low hereditary score"
        if value < 1.0:
            return "warning", "Moderate hereditary score"
        return "serious", "High hereditary score"
    if feature == "Pregnancies":
        return ("good", "Typical") if value < 6 else ("warning", "High parity (≥6)")
    return "na", ""


def pill(status, label):
    color = STATUS_COLORS[status]
    return (f'<div class="pill"><span class="dot" style="background:{color}">'
            f'{STATUS_ICONS[status]}</span>{label}</div>')


def show_pill(feature):
    st.markdown(pill(*classify(feature, st.session_state[feature])), unsafe_allow_html=True)


def risk_tier(p):
    if p < 0.4:
        return "good", "Low risk"
    if p <= 0.6:
        return "warning", "Borderline"
    return "critical", "High risk"


def percentile(stats, feature, value):
    """Percentile rank of value within the training dataset (0–100)."""
    q = np.asarray(stats["features"][feature]["quantiles"])
    rank = (np.searchsorted(q, value, side="left") + np.searchsorted(q, value, side="right")) / 2
    return float(np.clip(rank, 0, 100))


def style_fig(fig, height):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font_size=13),
    )
    return fig


# ── Charts ───────────────────────────────────────────────────────────────────
def gauge_chart(p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(p * 100, 1),
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Probability of diabetes", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickvals": [0, 40, 60, 100]},
            "bar": {"color": C_PATIENT, "thickness": 0.28},
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(12,163,12,.22)"},
                {"range": [40, 60], "color": "rgba(250,178,25,.30)"},
                {"range": [60, 100], "color": "rgba(208,59,59,.25)"},
            ],
            "threshold": {"line": {"color": "#888", "width": 2}, "thickness": 0.8, "value": 50},
        },
    ))
    style_fig(fig, 260)
    fig.update_layout(margin=dict(l=35, r=35, t=50, b=10))
    return fig


def profile_chart(stats, payload):
    """Patient's percentile per feature, against both group averages."""
    names = [FEATURES[f][0] for f in FEATURES][::-1]
    keys = list(FEATURES)[::-1]
    fs = stats["features"]
    patient = [percentile(stats, f, payload[f]) for f in keys]
    healthy = [percentile(stats, f, fs[f]["mean_non_diabetic"]) for f in keys]
    diabetic = [percentile(stats, f, fs[f]["mean_diabetic"]) for f in keys]
    raw = [f"{payload[f]:g} {FEATURES[f][1]}".strip() for f in keys]

    fig = go.Figure()
    fig.add_bar(
        y=names, x=patient, orientation="h", name="This patient", width=0.45,
        marker=dict(color=C_PATIENT, cornerradius=4), customdata=raw,
        hovertemplate="<b>%{y}</b><br>Value: %{customdata}<br>Percentile: %{x:.0f}<extra>This patient</extra>",
    )
    for vals, name, color, key in [(healthy, "Non-diabetic avg", C_HEALTHY, "mean_non_diabetic"),
                                   (diabetic, "Diabetic avg", C_DIABETIC, "mean_diabetic")]:
        fig.add_scatter(
            y=names, x=vals, mode="markers", name=name,
            marker=dict(symbol="diamond", size=12, color=color, line=dict(width=2, color="white")),
            customdata=[f"{fs[f][key]:.1f}" for f in keys],
            hovertemplate="<b>%{y}</b><br>Mean: %{customdata}<br>Percentile: %{x:.0f}<extra>" + name + "</extra>",
        )
    fig.update_xaxes(range=[0, 100], ticksuffix="", title="Percentile within the dataset (0–100)")
    return style_fig(fig, 380)


def distribution_chart(stats, feature, value):
    h = stats["features"][feature]["histogram"]
    edges = np.asarray(h["edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    name, unit = FEATURES[feature]

    fig = go.Figure()
    for counts, label, color in [(h["non_diabetic"], "Non-diabetic", C_HEALTHY),
                                 (h["diabetic"], "Diabetic", C_DIABETIC)]:
        fig.add_bar(
            x=centers, y=counts, name=label, marker=dict(color=color, cornerradius=3),
            customdata=np.stack([edges[:-1], edges[1:]], axis=1),
            hovertemplate="%{customdata[0]:.1f} – %{customdata[1]:.1f}: <b>%{y}</b> patients<extra>" + label + "</extra>",
        )
    fig.add_vline(x=value, line=dict(color=C_PATIENT, width=2, dash="dash"),
                  annotation_text=f"This patient: {value:g}", annotation_position="top")
    fig.update_layout(barmode="group", bargap=0.15, bargroupgap=0.05)
    fig.update_xaxes(title=f"{name} ({unit})" if unit else name)
    fig.update_yaxes(title="Patients")
    return style_fig(fig, 380)


def outcome_donut(stats):
    d = stats["dataset"]
    fig = go.Figure(go.Pie(
        labels=["Non-diabetic", "Diabetic"], values=[d["non_diabetic"], d["diabetic"]],
        hole=0.62, sort=False, marker=dict(colors=[C_HEALTHY, C_DIABETIC], line=dict(color="white", width=2)),
        textinfo="label+percent", hovertemplate="%{label}: <b>%{value}</b> patients<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{d['patients']}</b><br>patients", showarrow=False, font=dict(size=15))
    fig.update_layout(showlegend=False)
    return style_fig(fig, 300)


def history_chart(history):
    df = pd.DataFrame(history)
    x = list(range(1, len(df) + 1))
    fig = go.Figure()
    fig.add_scatter(
        x=x, y=df["P(diabetes) %"], mode="lines+markers", name="P(diabetes)",
        line=dict(color=C_PATIENT, width=2), marker=dict(size=9),
        customdata=df[["Time", "Result"]],
        hovertemplate="Assessment %{x} · %{customdata[0]}<br><b>%{y:.0f}%</b> · %{customdata[1]}<extra></extra>",
    )
    fig.add_hline(y=50, line=dict(color="#888", width=1, dash="dash"),
                  annotation_text="Decision threshold (50%)", annotation_position="top left")
    fig.update_xaxes(title="Assessment", dtick=1)
    fig.update_yaxes(title="Probability of diabetes (%)", range=[-5, 105])
    fig.update_layout(showlegend=False)
    return style_fig(fig, 300)


# ── Sidebar ──────────────────────────────────────────────────────────────────
online, latency = check_api()
stats = load_stats() if online else None

with st.sidebar:
    st.header("🩺 Service status")
    if online:
        st.success(f"API online · model loaded · {latency:.0f} ms")
    else:
        st.error("API offline")
        st.caption("Start it with `uvicorn main:app --reload`")
    st.caption(f"Endpoint: `{API_URL}`")

    st.divider()
    st.subheader("Quick fill")
    st.caption("Load an example patient into the form.")
    for preset_name, values in PRESETS.items():
        st.button(preset_name, key=f"preset_{preset_name}", width="stretch",
                  on_click=lambda v=values: st.session_state.update(v))
    st.button("Reset to defaults", width="stretch",
              on_click=lambda: st.session_state.update(DEFAULTS))

    if stats:
        st.divider()
        st.subheader("Model card")
        m = stats["model"]
        st.markdown(
            f"- **Algorithm:** {m['algorithm']}\n"
            f"- **Held-out accuracy:** {m['test_accuracy']:.1%} ({m['test_size']} patients)\n"
            f"- **Training data:** {stats['dataset']['name']}, {stats['dataset']['patients']} patients"
        )

    st.divider()
    st.caption("⚠️ For educational purposes only. Not a medical diagnosis — "
               "consult a healthcare professional.")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Diabetes Risk Predictor")
st.markdown(
    """
    <div class="hero">
        <span class="tag">MACHINE LEARNING</span><span class="tag">FASTAPI SERVICE</span>
        <span class="tag">CLINICAL DECISION SUPPORT</span>
        <p>Enter the patient's clinical measurements and click <b>Predict</b> for an instant
        risk assessment, a confidence score, and a comparison against 768 reference patients.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
if stats:
    d = stats["dataset"]
    k1.metric("Reference patients", f"{d['patients']}")
    k2.metric("Diabetes prevalence", f"{d['diabetic'] / d['patients']:.1%}")
    k3.metric("Model accuracy", f"{stats['model']['test_accuracy']:.1%}",
              help="Accuracy on the 20% held-out test split")
else:
    k1.metric("Reference patients", "—")
    k2.metric("Diabetes prevalence", "—")
    k3.metric("Model accuracy", "—")
k4.metric("Assessments this session", len(st.session_state.history))

tab_assess, tab_insights, tab_history, tab_about = st.tabs(
    ["🩺 Risk assessment", "📊 Population insights", "🕘 History", "ℹ️ About"]
)

# ── Tab 1: assessment ────────────────────────────────────────────────────────
with tab_assess:
    # 2. Eight input widgets, grouped into three columns, each with a clinical label.
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown('<div class="section-label">Patient profile</div>', unsafe_allow_html=True)
        st.number_input("Pregnancies", min_value=0, max_value=17, step=1, key="Pregnancies",
                        help="Number of times pregnant")
        show_pill("Pregnancies")
        st.slider("Age (years)", min_value=18, max_value=90, key="Age")
        show_pill("Age")
        st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01,
                        format="%.3f", key="DiabetesPedigreeFunction",
                        help="Family-history diabetes score")
        show_pill("DiabetesPedigreeFunction")

    with col2:
        st.markdown('<div class="section-label">Blood measurements</div>', unsafe_allow_html=True)
        st.slider("Glucose (mg/dL)", min_value=0, max_value=200, key="Glucose",
                  help="Plasma glucose 2 hours into an oral glucose tolerance test")
        show_pill("Glucose")
        st.slider("Insulin (µU/ml)", min_value=0, max_value=850, key="Insulin",
                  help="2-hour serum insulin")
        show_pill("Insulin")
        st.slider("Blood Pressure (mm Hg)", min_value=0, max_value=130, key="BloodPressure",
                  help="Diastolic blood pressure")
        show_pill("BloodPressure")

    with col3:
        st.markdown('<div class="section-label">Body measurements</div>', unsafe_allow_html=True)
        st.slider("BMI (kg/m²)", min_value=0.0, max_value=70.0, step=0.1, key="BMI",
                  help="Body mass index")
        show_pill("BMI")
        st.slider("Skin Thickness (mm)", min_value=0, max_value=100, key="SkinThickness",
                  help="Triceps skin fold thickness")
        show_pill("SkinThickness")

    # 3. Predict button
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        clicked = st.button("🔍 Predict", type="primary", width="stretch")

    if clicked:
        # Field names match the PatientFeatures model in main.py exactly.
        payload = {
            "Pregnancies": int(st.session_state.Pregnancies),
            "Glucose": float(st.session_state.Glucose),
            "BloodPressure": float(st.session_state.BloodPressure),
            "SkinThickness": float(st.session_state.SkinThickness),
            "Insulin": float(st.session_state.Insulin),
            "BMI": float(st.session_state.BMI),
            "DiabetesPedigreeFunction": float(st.session_state.DiabetesPedigreeFunction),
            "Age": int(st.session_state.Age),
        }

        # 5. Graceful failure — a friendly message, never a traceback.
        try:
            with st.spinner("Analysing patient data..."):
                start = time.perf_counter()
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                elapsed = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                result = response.json()
                p = result["confidence"] if result["prediction"] == 1 else 1 - result["confidence"]
                st.session_state.last = {"result": result, "payload": payload,
                                         "p": p, "elapsed": elapsed}
                st.session_state.history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    **payload,
                    "Result": result["result_label"],
                    "Confidence %": round(result["confidence"] * 100, 1),
                    "P(diabetes) %": round(p * 100, 1),
                })
            else:
                st.warning(f"The prediction service responded with status "
                           f"{response.status_code}. {START_HINT}")
        except requests.exceptions.RequestException:
            st.warning(START_HINT)

    # 4. Result display
    last = st.session_state.last
    if last:
        result, payload, p = last["result"], last["payload"], last["p"]
        tier_status, tier_label = risk_tier(p)

        st.divider()
        st.subheader("Assessment result")
        res_col, gauge_col = st.columns([3, 2], gap="large")

        with res_col:
            if result["result_label"] == "Diabetic":
                st.error("### ⚠️ Diabetic\nThe model indicates a **high risk** of diabetes. "
                         "A follow-up with a healthcare professional is recommended.")
            else:
                st.success("### ✅ Not Diabetic\nThe model indicates a **low risk** of diabetes. "
                           "Keep up healthy habits and regular check-ups.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{result['confidence'] * 100:.1f}%",
                      help="Probability the model assigns to its predicted class")
            m2.metric("Risk tier", tier_label)
            m3.metric("Response time", f"{last['elapsed']:.0f} ms")

            st.markdown('<div class="section-label">Clinical flags</div>', unsafe_allow_html=True)
            flags = [(f, *classify(f, payload[f])) for f in FEATURES]
            flagged = [x for x in flags if x[1] in ("serious", "critical", "warning")]
            if flagged:
                for f, status, label in sorted(flagged, key=lambda x: ["critical", "serious", "warning"].index(x[1])):
                    st.markdown(pill(status, f"<b>{FEATURES[f][0]}</b> · {payload[f]:g} — {label}"),
                                unsafe_allow_html=True)
            else:
                st.markdown(pill("good", "All measurements within typical reference ranges"),
                            unsafe_allow_html=True)

        with gauge_col:
            st.plotly_chart(gauge_chart(p), width="stretch", config={"displayModeBar": False})
            st.caption(f"Based on the vote of the 5 most similar reference patients · "
                       f"{round(p * 5)} of 5 were diabetic.")

        if stats:
            st.markdown("#### Patient profile vs. reference population")
            st.caption("Where each measurement ranks among the 768 reference patients. Bars that sit "
                       "nearer the orange diamonds look more like the diabetic group.")
            st.plotly_chart(profile_chart(stats, payload), width="stretch",
                            config={"displayModeBar": False})

        with st.expander("Submitted values"):
            table = pd.DataFrame({
                "Feature": [FEATURES[f][0] for f in FEATURES],
                "Value": [f"{payload[f]:g}" for f in FEATURES],
                "Unit": [FEATURES[f][1] for f in FEATURES],
                "Reference status": [classify(f, payload[f])[1] for f in FEATURES],
                "Percentile": [f"{percentile(stats, f, payload[f]):.0f}" if stats else "—"
                               for f in FEATURES],
            })
            st.dataframe(table, hide_index=True, width="stretch")

# ── Tab 2: population insights ───────────────────────────────────────────────
with tab_insights:
    if not stats:
        st.info("Start the FastAPI service to load population insights.")
    else:
        top_left, top_right = st.columns([2, 1], gap="large")
        with top_right:
            st.markdown("#### Outcome split")
            st.plotly_chart(outcome_donut(stats), width="stretch", config={"displayModeBar": False})
        with top_left:
            st.markdown("#### Feature distribution by outcome")
            feature = st.selectbox("Feature", list(FEATURES), index=1,
                                   format_func=lambda f: FEATURES[f][0])
            st.plotly_chart(distribution_chart(stats, feature, st.session_state[feature]),
                            width="stretch", config={"displayModeBar": False})
            st.caption("Dashed line marks the value currently entered in the form.")

        st.markdown("#### Average profile by outcome")
        rows = []
        for f, (name, unit) in FEATURES.items():
            fs = stats["features"][f]
            rows.append({
                "Feature": name, "Unit": unit,
                "Non-diabetic mean": round(fs["mean_non_diabetic"], 2),
                "Diabetic mean": round(fs["mean_diabetic"], 2),
                "Difference": f"{(fs['mean_diabetic'] / fs['mean_non_diabetic'] - 1):+.0%}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ── Tab 3: history ───────────────────────────────────────────────────────────
with tab_history:
    history = st.session_state.history
    if not history:
        st.info("No assessments yet. Run a prediction to start the session log.")
    else:
        st.markdown("#### Probability of diabetes across assessments")
        st.plotly_chart(history_chart(history), width="stretch", config={"displayModeBar": False})
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, hide_index=True, width="stretch")
        c1, c2, _ = st.columns([1, 1, 3])
        c1.download_button("⬇️ Download CSV", hist_df.to_csv(index=False).encode("utf-8"),
                           file_name="diabetes_assessments.csv", mime="text/csv", width="stretch")
        if c2.button("Clear history", width="stretch"):
            st.session_state.history = []
            st.session_state.last = None
            st.rerun()

# ── Tab 4: about ─────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
        #### How it works
        1. **This web app** collects the 8 clinical measurements and sends them as JSON to the API.
        2. **The FastAPI service** (`main.py`) holds the trained model in memory and returns a
           prediction with its confidence.
        3. **The model** is a K-Nearest Neighbours classifier: it finds the 5 most similar patients
           in the Pima Indians Diabetes dataset and takes a majority vote. Inputs are standardised
           first so no single measurement dominates the distance.

        #### Reading the result
        - **Confidence** is the share of the 5 neighbours that agree with the prediction.
        - **Probability of diabetes** is the share of neighbours who were diabetic; 50% is the
          decision threshold.
        - **Clinical flags** use common reference ranges (ADA glucose tolerance, AHA diastolic
          blood pressure, WHO BMI classes). A value of 0 means the measurement was not recorded.

        #### Limitations
        The model was trained on 768 adult women of Pima Indian heritage and reaches about
        70% accuracy on held-out data. It is a teaching tool and **not** a substitute for
        laboratory testing or a clinician's judgement.
        """
    )

st.markdown(
    '<div class="footer">Diabetes Risk Predictor · Streamlit front end + FastAPI model service · '
    'Educational use only</div>',
    unsafe_allow_html=True,
)
