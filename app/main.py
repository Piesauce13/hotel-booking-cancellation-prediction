"""FastAPI service for hotel booking cancellation prediction.

Run from the project root:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs page:  http://<host>:8000/docs
"""
from pathlib import Path

import joblib
from fastapi import FastAPI

from app.schemas import BookingInput, PredictionOutput
from src.preprocessing import predict_from_raw

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "stacking_model.joblib"
artifact = joblib.load(MODEL_PATH)  # loaded once at startup

app = FastAPI(title="Hotel Booking Cancellation API", version="1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "base_models": artifact["base_models"],
        "meta_classifier": artifact["meta_classifier"],
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(booking: BookingInput):
    # use the original column names (aliases) the model was trained with
    row = booking.model_dump(by_alias=True)
    return predict_from_raw(artifact, [row])[0]
