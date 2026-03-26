from pydantic import BaseModel, Field
from typing import List, ClassVar


class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    transactions_last_24h: int = Field(..., ge=0, description="Number of transactions in the last 24 hours")
    avg_transaction_amount: float = Field(..., gt=0, description="User's historical avg transaction amount")
    distance_from_home_km: float = Field(..., ge=0, description="Distance from user's home location in km")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "amount": 42.50,
                    "hour_of_day": 14,
                    "day_of_week": 2,
                    "transactions_last_24h": 3,
                    "avg_transaction_amount": 55.00,
                    "distance_from_home_km": 1.2,
                }
            ]
        }
    }

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "amount",
            "hour_of_day",
            "day_of_week",
            "transactions_last_24h",
            "avg_transaction_amount",
            "distance_from_home_km",
        ]


class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float = Field(..., description="Decision function score; lower = more anomalous")
    label: str


class BatchInput(BaseModel):
    transactions: List[TransactionInput]


class BatchResponse(BaseModel):
    results: List[PredictionResponse]
    total: int
    anomaly_count: int


class ModelInfo(BaseModel):
    model_type: str
    n_estimators: int
    contamination: float
    features: List[str]
