"""FastAPI request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    tx_id: str
    user_id: str
    merchant_id: str
    device_id: str
    account_id: str
    amount: float = Field(ge=0)
    currency: str = "USD"
    timestamp: datetime
    lat: float
    lon: float
    channel: str
    mcc: str


class PredictionResponse(BaseModel):
    decision_id: str
    tx_id: str
    raw_score: float
    calibrated_score: float
    confidence_band: str
    interval_low: float
    interval_high: float
    explanation: Optional[str] = None
    timestamp: str


class ReplayResponse(BaseModel):
    replay_id: str
    decision_id: str
    matched: bool
    stored_prediction: dict
    transaction: Optional[dict] = None
    features: Optional[list] = None
    replayed_at: str
    diff: Optional[dict] = None


class AuditReportResponse(BaseModel):
    decision_id: str
    report_markdown: Optional[str] = None
    report_json: Optional[dict] = None


class MetricsResponse(BaseModel):
    model_type: str
    metrics: dict


class ModelInfoResponse(BaseModel):
    model_id: str
    model_type: str
    version: str
    artifact_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
