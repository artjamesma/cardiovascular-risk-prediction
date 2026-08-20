from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="CardioRisk AI API",
    description=(
        "FastAPI backend for the cardiovascular risk prediction model. "
        "Educational use only; not a clinical diagnostic system."
    ),
    version="1.0.0",
)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("cardio_xgboost_model.pkl")

FEATURES = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "bmi",
]

# ============================================================
# REQUEST SCHEMA
# ============================================================

class PatientData(BaseModel):
    age: float = Field(..., ge=18, le=100)
    gender: int = Field(..., ge=1, le=2)

    height_cm: float = Field(..., ge=100, le=250)
    weight: float = Field(..., ge=20, le=200)

    ap_hi: int = Field(..., ge=30, le=300)
    ap_lo: int = Field(..., ge=30, le=300)

    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)

    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "message": "CardioRisk AI API is running",
        "status": "ok",
        "model": "XGBoost",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: PatientData):

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    height_m = data.height_cm / 100.0
    bmi = data.weight / (height_m ** 2)

    patient_df = pd.DataFrame(
        [[
            data.age,
            data.gender,
            height_m,
            data.weight,
            data.ap_hi,
            data.ap_lo,
            data.cholesterol,
            data.gluc,
            data.smoke,
            data.alco,
            data.active,
            bmi,
        ]],
        columns=FEATURES,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(patient_df)[0, 1]
    )

    predicted_class = int(probability >= 0.50)

    return {
        "probability": round(probability, 6),
        "probability_percent": round(probability * 100, 2),
        "predicted_class": predicted_class,
        "classification": (
            "cardiovascular_disease"
            if predicted_class == 1
            else "no_cardiovascular_disease"
        ),
        "threshold": 0.50,
        "calculated_bmi": round(bmi, 2),
        "disclaimer": (
            "Educational machine-learning output only. "
            "Not a medical diagnosis or clinical risk score."
        ),
    }