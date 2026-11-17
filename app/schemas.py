"""Input / output schemas for the API.

JSON keys are the original dataset column names (with spaces), so a row of
booking.csv can be pasted into Postman almost as-is.
"""
from typing import Dict, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.preprocessing import parse_dates


class BookingInput(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "number of adults": 2,
                "number of children": 0,
                "number of weekend nights": 1,
                "number of week nights": 2,
                "type of meal": "Meal Plan 1",
                "car parking space": 0,
                "room type": "Room_Type 1",
                "lead time": 120,
                "market segment type": "Online",
                "repeated": 0,
                "P-C": 0,
                "P-not-C": 0,
                "average price": 95.0,
                "special requests": 0,
                "date of reservation": "6/15/2018",
            }
        },
    )

    adults: int = Field(alias="number of adults", ge=1, le=10)
    children: int = Field(alias="number of children", ge=0, le=10)
    weekend_nights: int = Field(alias="number of weekend nights", ge=0, le=30)
    week_nights: int = Field(alias="number of week nights", ge=0, le=60)
    meal: Literal["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"] = Field(alias="type of meal")
    parking: Literal[0, 1] = Field(alias="car parking space")
    room: Literal[
        "Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4",
        "Room_Type 5", "Room_Type 6", "Room_Type 7",
    ] = Field(alias="room type")
    lead_time: int = Field(alias="lead time", ge=0, le=1000)
    segment: Literal["Online", "Offline", "Corporate", "Complementary", "Aviation"] = Field(
        alias="market segment type"
    )
    repeated: Literal[0, 1] = Field(alias="repeated")
    p_c: int = Field(alias="P-C", ge=0)
    p_not_c: int = Field(alias="P-not-C", ge=0)
    price: float = Field(alias="average price", ge=0)
    special_requests: int = Field(alias="special requests", ge=0, le=10)
    reservation_date: str = Field(alias="date of reservation")

    @field_validator("reservation_date")
    @classmethod
    def date_must_be_valid(cls, v: str) -> str:
        import pandas as pd

        if pd.isna(parse_dates(pd.Series([v])).iloc[0]):
            raise ValueError("must be a real date like 6/15/2018 or 2018-06-15")
        return v


class PredictionOutput(BaseModel):
    prediction: str
    cancel_probability: float
    base_model_probabilities: Dict[str, float]
