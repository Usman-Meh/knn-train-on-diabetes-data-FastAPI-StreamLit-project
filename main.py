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

MODEL_PATH = Path(__file__).parent / "model.pkl"

# The model lives in this dict so your routes can reach it after startup.
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup — never inside the route.
    ml_model["predictor"] = joblib.load(MODEL_PATH)

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
