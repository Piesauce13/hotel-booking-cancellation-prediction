"""Shared preprocessing, used by BOTH the notebook (training) and the API (inference).

Keeping this in one place guarantees the API prepares a request exactly the way
the training data was prepared.
"""
import pandas as pd

# ---------------------------------------------------------------- column groups
ID_COL = "Booking_ID"
TARGET = "booking status"
DATE_COL = "date of reservation"

CATEGORICAL_COLS = ["type of meal", "room type", "market segment type"]
BINARY_COLS = ["car parking space", "repeated"]
NUMERIC_COLS = [
    "number of adults",
    "number of children",
    "number of weekend nights",
    "number of week nights",
    "lead time",
    "P-C",
    "P-not-C",
    "average price",
    "special requests",
]
DATE_FEATURES = ["reservation month", "reservation weekday"]

# Columns whose extreme values get capped (cap values are learned in the notebook)
OUTLIER_COLS = [
    "lead time",
    "average price",
    "number of week nights",
    "number of weekend nights",
    "number of children",
]

# What the API receives for one booking (everything except ID and target)
RAW_INPUT_COLS = CATEGORICAL_COLS + BINARY_COLS + NUMERIC_COLS + [DATE_COL]

LABELS = {0: "Not_Canceled", 1: "Canceled"}


# ---------------------------------------------------------------- helpers
def parse_dates(s: pd.Series) -> pd.Series:
    """Parse m/d/yyyy (the dataset format) or yyyy-mm-dd. Invalid dates -> NaT."""
    s = s.astype(str).str.strip()
    d = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    return d.fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the reservation date into month (1-12) and weekday (0=Mon ... 6=Sun)."""
    df = df.copy()
    d = parse_dates(df[DATE_COL])
    df["reservation month"] = d.dt.month
    df["reservation weekday"] = d.dt.dayofweek
    return df


def cap_outliers(df: pd.DataFrame, caps: dict) -> pd.DataFrame:
    """Clip each column in `caps` at its upper cap value."""
    df = df.copy()
    for col, upper in caps.items():
        df[col] = df[col].clip(upper=upper)
    return df


def prepare_input(df: pd.DataFrame, caps: dict, feature_columns: list) -> pd.DataFrame:
    """Raw booking rows -> model-ready feature table (same steps as training)."""
    df = add_date_features(df)
    if df[DATE_FEATURES].isna().any().any():
        raise ValueError("Invalid 'date of reservation' (use m/d/yyyy or yyyy-mm-dd)")
    df = cap_outliers(df, caps)
    return df[feature_columns]


def predict_from_raw(artifact: dict, rows: list) -> list:
    """Predict for a list of raw booking dicts using a loaded model artifact.

    Returns one dict per booking with the final prediction, the cancel probability
    and what each base model said (so the meta-classifier is visible in the output).
    """
    pipe = artifact["pipeline"]
    X = prepare_input(pd.DataFrame(rows), artifact["caps"], artifact["feature_columns"])

    final_proba = pipe.predict_proba(X)[:, 1]

    # What each base model predicted (the inputs of the meta-classifier)
    stack = pipe.named_steps["stack"]
    base_proba = stack.transform(pipe.named_steps["prep"].transform(X))
    base_names = [name for name, _ in stack.estimators]

    results = []
    for i, p in enumerate(final_proba):
        results.append(
            {
                "prediction": LABELS[int(p >= 0.5)],
                "cancel_probability": round(float(p), 4),
                "base_model_probabilities": {
                    n: round(float(base_proba[i, j]), 4) for j, n in enumerate(base_names)
                },
            }
        )
    return results
