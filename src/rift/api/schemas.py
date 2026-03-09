"""API request/response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Single transaction prediction request."""

    transaction: dict = Field(..., description="Transaction payload")


class PredictResponse(BaseModel):
    """Prediction response."""

    raw_score: float
    calibrated_score: float
    decision_band: str
    is_fraud_pred: bool
