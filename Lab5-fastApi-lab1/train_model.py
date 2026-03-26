"""
Train an Isolation Forest on synthetic financial transaction data.
Saves the fitted model to model/isolation_forest.pkl.

Synthetic data schema:
  - amount              : transaction value in USD
  - hour_of_day         : 0-23
  - day_of_week         : 0-6
  - transactions_last_24h: count of prior txns in rolling 24h window
  - avg_transaction_amount: user's historical average
  - distance_from_home_km : haversine distance from registered address
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

SEED = 42
rng = np.random.default_rng(SEED)


def generate_normal_transactions(n: int) -> np.ndarray:
    amount = rng.lognormal(mean=3.8, sigma=0.6, size=n)        # ~$45 median
    hour = rng.integers(7, 22, size=n)                          # daytime-ish
    dow = rng.integers(0, 7, size=n)
    txn_count = rng.integers(1, 10, size=n)
    avg_amount = rng.lognormal(mean=3.8, sigma=0.4, size=n)
    distance = rng.exponential(scale=5.0, size=n)               # usually close to home
    return np.column_stack([amount, hour, dow, txn_count, avg_amount, distance])


def generate_anomalies(n: int) -> np.ndarray:
    # Mix of fraud patterns
    # 1. Large late-night transactions far from home
    n1 = n // 3
    amount1 = rng.uniform(800, 5000, size=n1)
    hour1 = rng.integers(0, 5, size=n1)
    dow1 = rng.integers(0, 7, size=n1)
    txn1 = rng.integers(15, 40, size=n1)
    avg1 = rng.lognormal(mean=3.8, sigma=0.4, size=n1)
    dist1 = rng.uniform(200, 2000, size=n1)

    # 2. Rapid-fire small transactions (velocity fraud)
    n2 = n - n1
    amount2 = rng.uniform(1, 10, size=n2)
    hour2 = rng.integers(0, 23, size=n2)
    dow2 = rng.integers(0, 7, size=n2)
    txn2 = rng.integers(50, 200, size=n2)
    avg2 = rng.lognormal(mean=3.8, sigma=0.4, size=n2)
    dist2 = rng.uniform(0, 50, size=n2)

    block1 = np.column_stack([amount1, hour1, dow1, txn1, avg1, dist1])
    block2 = np.column_stack([amount2, hour2, dow2, txn2, avg2, dist2])
    return np.vstack([block1, block2])


if __name__ == "__main__":
    n_normal = 9000
    n_anomaly = 1000

    X_normal = generate_normal_transactions(n_normal)
    X_fraud = generate_anomalies(n_anomaly)
    X_train = np.vstack([X_normal, X_fraud])

    contamination = n_anomaly / (n_normal + n_anomaly)  # 0.10

    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=SEED,
    )
    clf.fit(X_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/isolation_forest.pkl")

    # Quick sanity check
    test_normal = generate_normal_transactions(200)
    test_fraud = generate_anomalies(200)
    preds_normal = clf.predict(test_normal)
    preds_fraud = clf.predict(test_fraud)

    tp = (preds_fraud == -1).sum()
    tn = (preds_normal == 1).sum()
    print(f"Model saved → model/isolation_forest.pkl")
    print(f"Contamination: {contamination:.2f}")
    print(f"Sanity check — Normal correctly flagged clean : {tn}/200")
    print(f"Sanity check — Fraud correctly flagged anomaly: {tp}/200")
