"""
main.py  —  Diabetes Risk API
-----------------------------
A FastAPI app that loads the trained model.pkl and serves predictions
for the diabetes model (8 inputs -> Diabetic / Not Diabetic).

Run with:
    pip install -r requirements.txt
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs to test it.
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "diabetes.csv"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

# The model lives in this dict so your routes can reach it after startup.
ml_model = {}


def build_dataset_stats(model):
    """Summarise the training data so the UI can put a patient in context."""
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES].values, df["Outcome"].values

    # Same split as train_model.py, so this is the honest held-out accuracy.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    accuracy = float((model.predict(X_test) == y_test).mean())

    features = {}
    for col in FEATURES:
        values = df[col]
        edges = np.histogram_bin_edges(values, bins=20)
        features[col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean_non_diabetic": round(float(values[y == 0].mean()), 3),
            "mean_diabetic": round(float(values[y == 1].mean()), 3),
            "quantiles": np.percentile(values, range(101)).round(3).tolist(),
            "histogram": {
                "edges": edges.round(3).tolist(),
                "non_diabetic": np.histogram(values[y == 0], bins=edges)[0].tolist(),
                "diabetic": np.histogram(values[y == 1], bins=edges)[0].tolist(),
            },
        }

    return {
        "model": {
            "algorithm": "K-Nearest Neighbours (k=5) with StandardScaler",
            "test_accuracy": round(accuracy, 4),
            "test_size": int(len(y_test)),
        },
        "dataset": {
            "name": "Pima Indians Diabetes",
            "patients": int(len(df)),
            "diabetic": int((y == 1).sum()),
            "non_diabetic": int((y == 0).sum()),
        },
        "features": features,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup — never inside the route.
    ml_model["predictor"] = joblib.load(MODEL_PATH)
    ml_model["stats"] = build_dataset_stats(ml_model["predictor"])

    yield

    # Release the model on shutdown.
    ml_model.clear()


app = FastAPI(title="Diabetes Risk API", lifespan=lifespan)


# ── Request shape: what the caller SENDS ─────────────────────────────────────
class PatientFeatures(BaseModel):
    # Order matches the column order the model was trained on.
    Pregnancies: int = Field(..., ge=0, le=17)
    Glucose: float = Field(..., ge=0, le=200)
    BloodPressure: float = Field(..., ge=0, le=130)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=850)
    BMI: float = Field(..., ge=0, le=70)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=2.5)
    Age: int = Field(..., ge=18, le=90)


# ── Response shape: what you SEND BACK ───────────────────────────────────────
class PredictionResult(BaseModel):
    prediction: int
    result_label: str
    confidence: float


@app.get("/")
def health():
    # Health check so you can confirm the server is up.
    return {"status": "running", "model_loaded": "predictor" in ml_model}


@app.get("/stats")
def stats():
    # Dataset and model summary used by the UI's charts.
    if "stats" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ml_model["stats"]


@app.post("/predict", response_model=PredictionResult)
def predict(features: PatientFeatures):
    model = ml_model.get("predictor")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build the input in the SAME column order as training.
    X = np.array([[
        features.Pregnancies,
        features.Glucose,
        features.BloodPressure,
        features.SkinThickness,
        features.Insulin,
        features.BMI,
        features.DiabetesPedigreeFunction,
        features.Age,
    ]])

    pred = int(model.predict(X)[0])
    # Confidence = probability of the PREDICTED class.
    conf = round(float(model.predict_proba(X)[0][pred]), 3)

    return PredictionResult(
        prediction=pred,
        result_label="Diabetic" if pred == 1 else "Not Diabetic",
        confidence=conf,
    )
