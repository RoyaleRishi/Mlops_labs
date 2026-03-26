import pytest
import numpy as np
import joblib
import os
from fastapi.testclient import TestClient

# Train a tiny model for testing without needing the real one
@pytest.fixture(scope="session", autouse=True)
def train_test_model():
    from sklearn.ensemble import IsolationForest
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 6))
    clf = IsolationForest(n_estimators=10, contamination=0.1, random_state=0)
    clf.fit(X)
    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/isolation_forest.pkl")
    yield
    # leave the file; real model (if any) would overwrite


@pytest.fixture(scope="session")
def client(train_test_model):
    from app.main import app
    return TestClient(app)


NORMAL_PAYLOAD = {
    "amount": 45.00,
    "hour_of_day": 14,
    "day_of_week": 2,
    "transactions_last_24h": 3,
    "avg_transaction_amount": 50.00,
    "distance_from_home_km": 2.0,
}

ANOMALY_PAYLOAD = {
    "amount": 4500.00,
    "hour_of_day": 2,
    "day_of_week": 6,
    "transactions_last_24h": 80,
    "avg_transaction_amount": 50.00,
    "distance_from_home_km": 1500.0,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert data["model_type"] == "IsolationForest"
    assert len(data["features"]) == 6


def test_predict_returns_valid_schema(client):
    r = client.post("/predict", json=NORMAL_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert data["label"] in ("NORMAL", "ANOMALY")


def test_predict_invalid_amount(client):
    bad = {**NORMAL_PAYLOAD, "amount": -10}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_invalid_hour(client):
    bad = {**NORMAL_PAYLOAD, "hour_of_day": 25}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch_predict(client):
    r = client.post("/predict/batch", json={"transactions": [NORMAL_PAYLOAD, ANOMALY_PAYLOAD]})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert 0 <= data["anomaly_count"] <= 2


def test_batch_size_limit(client):
    big_batch = {"transactions": [NORMAL_PAYLOAD] * 501}
    r = client.post("/predict/batch", json=big_batch)
    assert r.status_code == 400
