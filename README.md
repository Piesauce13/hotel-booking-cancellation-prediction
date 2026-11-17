# Hotel Booking Cancellation Prediction

Cloud ML final project: predict whether a customer will cancel a hotel booking
(target: `booking status`), using a stacking meta-classifier deployed on AWS EC2.

**The documentation is the notebook:** `notebooks/hotel_cancellation.ipynb` holds all
figures, model comparisons and the explanation of every decision. This README only
covers how to run and deploy.

## Project structure

```
hotel-cancellation/
├── data/
│   ├── raw/booking.csv            # original dataset (never edited)
│   └── processed/                 # cleaned data (created by the notebook)
├── notebooks/
│   └── hotel_cancellation.ipynb   # EDA, cleaning, features, models, stacking + explanations
├── src/
│   └── preprocessing.py           # shared cleaning/feature code (notebook + API)
├── models/                        # stacking_model.joblib (created by the notebook)
├── app/
│   ├── main.py                    # FastAPI app: /health, /predict
│   └── schemas.py                 # input validation
├── sample_requests/predict.json   # test payload for Postman
├── requirements.txt               # rewritten with pinned versions by the notebook
└── README.md
```

## 1. Train (on your laptop or Colab)

```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn joblib jupyter
jupyter notebook notebooks/hotel_cancellation.ipynb      # Run All
```

Run all cells (about 2-3 minutes). It creates `models/stacking_model.joblib`,
`data/processed/booking_clean.csv`, `sample_requests/predict.json` and a pinned `requirements.txt`.

## 2. Run the API locally (optional check)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Health check: `GET  http://localhost:8000/health`
- Prediction: `POST http://localhost:8000/predict` (body = `sample_requests/predict.json`)
- Interactive docs: `http://localhost:8000/docs`

Example response:

```json
{
  "prediction": "Not_Canceled",
  "cancel_probability": 0.0631,
  "base_model_probabilities": {"XGBoost": 0.079, "CatBoost": 0.0441, "Random Forest": 0.1107}
}
```

## 3. Deploy on EC2 (fill in as we go)

Copy to the instance: `app/`, `src/`, `models/`, `requirements.txt` (the rest is not needed).

1. Instance type / security group (inbound TCP 8000)
2. Copy project, install requirements
3. Run the API
4. Test with Postman: `POST http://<EC2-public-IP>:8000/predict`
