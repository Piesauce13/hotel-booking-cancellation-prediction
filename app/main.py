"""FastAPI service: predict whether a hotel booking will be canceled.

Run from the project root:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
Interactive docs: http://<host>:8000/docs
"""
from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

# Load the saved model once, when the server starts.
saved = joblib.load(Path(__file__).resolve().parent.parent / "models" / "stacking_model.joblib")
pipeline = saved["pipeline"]  # preprocessing + 3 base models + meta-classifier
features = saved["features"]  # the columns (and order) used in training
base_models = saved["base_models"]  # names of the 3 base models

app = FastAPI(title="Hotel Booking Cancellation API")


class Booking(BaseModel):
    """One booking. The JSON keys are the column names of the dataset (they contain spaces, hence the aliases).

    One field per model feature: if you change SELECTED_FEATURES in the notebook, change this class too.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type of meal": "Meal Plan 1",
                "room type": "Room_Type 1",
                "market segment type": "Online",
                "car parking space": 0,
                "number of adults": 2,
                "number of weekend nights": 1,
                "number of week nights": 2,
                "lead time": 120,
                "average price": 95.0,
                "special requests": 0,
                "date of reservation": "2018-06-15",
            }
        }
    )

    type_of_meal: Literal["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"] = Field(alias="type of meal")
    room_type: Literal[
        "Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"
    ] = Field(alias="room type")
    market_segment_type: Literal["Online", "Offline", "Corporate", "Complementary", "Aviation"] = Field(
        alias="market segment type"
    )
    car_parking_space: Literal[0, 1] = Field(alias="car parking space")
    number_of_adults: int = Field(alias="number of adults", ge=0)
    number_of_weekend_nights: int = Field(alias="number of weekend nights", ge=0)
    number_of_week_nights: int = Field(alias="number of week nights", ge=0)
    lead_time: int = Field(alias="lead time", ge=0)
    average_price: float = Field(alias="average price", ge=0)
    special_requests: int = Field(alias="special requests", ge=0)
    date_of_reservation: date = Field(alias="date of reservation")  # yyyy-mm-dd, invalid dates are rejected


@app.get("/health")
def health():
    return {"status": "ok", "base_models": base_models, "meta_classifier": "LogisticRegression"}


@app.post("/predict")
def predict(booking: Booking):
    # 1. request -> one row with the dataset's column names
    row = booking.model_dump(by_alias=True)

    # 2. the ONLY manual preprocessing step: the model was trained on month + weekday, not on the full date.
    #    (scaling and one-hot encoding happen inside the saved pipeline)
    booking_date = row.pop("date of reservation")
    row["reservation month"] = booking_date.month
    row["reservation weekday"] = booking_date.weekday()  # 0 = Monday, same as in the notebook
    X = pd.DataFrame([row])[features]

    # 3. final prediction from the meta-classifier
    probability = float(pipeline.predict_proba(X)[0, 1])

    # 4. what each base model said (these 3 numbers are the input of the meta-classifier)
    base_probabilities = pipeline.named_steps["stack"].transform(pipeline.named_steps["prep"].transform(X))[0]

    return {
        "prediction": "Canceled" if probability >= 0.5 else "Not_Canceled",
        "cancel_probability": round(probability, 4),
        "base_model_probabilities": {name: round(float(p), 4) for name, p in zip(base_models, base_probabilities)},
    }
