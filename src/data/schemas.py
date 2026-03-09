"""Pydantic schemas for Rift data entities."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Channel(str, Enum):
    WEB = "web"
    MOBILE = "mobile"
    POS = "pos"


class Transaction(BaseModel):
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
    channel: Channel
    mcc: str
    is_fraud: int = Field(ge=0, le=1)


class PredictionResult(BaseModel):
    decision_id: str
    tx_id: str
    raw_score: float
    calibrated_score: float
    confidence_band: str
    explanation: str
    timestamp: datetime
    model_id: str
    model_version: str


class AuditReport(BaseModel):
    decision_id: str
    decision_time: datetime
    outcome: str
    confidence: str
    top_drivers: list[str]
    nearest_cases: list[str]
    counterfactual: str
    replay_instructions: str
    recommendation: str


class ConformalOutput(str, Enum):
    HIGH_CONFIDENCE_FRAUD = "high_confidence_fraud"
    REVIEW_NEEDED = "review_needed"
    HIGH_CONFIDENCE_LEGIT = "high_confidence_legit"


class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    version: str
    trained_at: datetime
    metrics: dict
    artifact_path: Optional[str] = None
