# Transaction Anomaly Detection API

FastAPI service that exposes an Isolation Forest model for real-time financial transaction anomaly detection.  
Based on the [MLOps FastAPI Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs/Lab1) skeleton, extended with a fraud-detection use case.

## Modifications from base lab
| Base Lab 1 | This repo |
|---|---|
| Iris dataset (classification) | Synthetic financial transactions (anomaly detection) |
| Logistic Regression | Isolation Forest |
| Single `/predict` endpoint | `/predict`, `/predict/batch`, `/model-info`, `/health` |
| No input validation beyond types | Field-level constraints (amount > 0, hour 0-23, etc.) |
| No tests | Full pytest suite in `tests/` |

## Features
- **`POST /predict`** — score a single transaction
- **`POST /predict/batch`** — score up to 500 transactions at once
- **`GET /model-info`** — model metadata (n_estimators, contamination, feature list)
- **`GET /health`** — liveness check
- **`GET /docs`** — Swagger UI (auto-generated)

## Setup

```bash
pip install -r requirements.txt

# Train the model (generates model/isolation_forest.pkl)
python train_model.py

# Start the server
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 4500.00,
    "hour_of_day": 2,
    "day_of_week": 6,
    "transactions_last_24h": 80,
    "avg_transaction_amount": 52.00,
    "distance_from_home_km": 1400.0
  }'
```

```json
{
  "is_anomaly": true,
  "anomaly_score": -0.1823,
  "label": "ANOMALY"
}
```

## Running tests

```bash
pytest tests/ -v
```

## Feature schema

| Feature | Description |
|---|---|
| `amount` | Transaction value (USD), must be > 0 |
| `hour_of_day` | Hour of transaction (0–23) |
| `day_of_week` | Day of week (0=Mon, 6=Sun) |
| `transactions_last_24h` | Rolling 24h transaction count |
| `avg_transaction_amount` | User's historical average transaction |
| `distance_from_home_km` | Distance from registered home address |

## Model

Isolation Forest trained on 10,000 synthetic transactions (90% normal, 10% fraud). Anomaly patterns include large late-night transactions far from home and rapid-velocity micro-transactions.
