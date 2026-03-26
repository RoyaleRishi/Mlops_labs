from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import numpy as np
import os

from app.schemas import TransactionInput, PredictionResponse, BatchInput, BatchResponse, ModelInfo

MODEL_PATH = "model/isolation_forest.pkl"
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
    model = joblib.load(MODEL_PATH)
    yield
    model = None


app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="Isolation Forest-based anomaly detection for financial transactions.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfo(
        model_type="IsolationForest",
        n_estimators=model.n_estimators,
        contamination=model.contamination,
        features=TransactionInput.feature_names(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array([[
        transaction.amount,
        transaction.hour_of_day,
        transaction.day_of_week,
        transaction.transactions_last_24h,
        transaction.avg_transaction_amount,
        transaction.distance_from_home_km,
    ]])

    raw = model.predict(features)[0]          # -1 = anomaly, 1 = normal
    score = model.decision_function(features)[0]  # lower = more anomalous

    is_anomaly = bool(raw == -1)
    return PredictionResponse(
        is_anomaly=is_anomaly,
        anomaly_score=round(float(score), 4),
        label="ANOMALY" if is_anomaly else "NORMAL",
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(batch.transactions) > 500:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 500")

    features = np.array([[
        t.amount,
        t.hour_of_day,
        t.day_of_week,
        t.transactions_last_24h,
        t.avg_transaction_amount,
        t.distance_from_home_km,
    ] for t in batch.transactions])

    raw = model.predict(features)
    scores = model.decision_function(features)

    results = [
        PredictionResponse(
            is_anomaly=bool(r == -1),
            anomaly_score=round(float(s), 4),
            label="ANOMALY" if r == -1 else "NORMAL",
        )
        for r, s in zip(raw, scores)
    ]

    return BatchResponse(
        results=results,
        total=len(results),
        anomaly_count=sum(1 for r in results if r.is_anomaly),
    )
