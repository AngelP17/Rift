from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TransactionRecord(BaseModel):
    tx_id: str
    user_id: str
    merchant_id: str
    device_id: str
    account_id: str
    amount: float = Field(ge=0.0)
    currency: str
    timestamp: datetime
    lat: float
    lon: float
    channel: str
    mcc: str
    is_fraud: int = Field(ge=0, le=1)


class PredictionRequest(BaseModel):
    tx_id: str
    user_id: str
    merchant_id: str
    device_id: str
    account_id: str
    amount: float
    currency: str = "USD"
    timestamp: datetime
    lat: float
    lon: float
    channel: str
    mcc: str


class PredictionResponse(BaseModel):
    decision_id: str
    model_run_id: str
    fraud_probability: float
    calibrated_probability: float
    decision: str
    confidence: float
    explanation: str
