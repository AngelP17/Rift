"""FastAPI request and response schemas for the Rift API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Operational response schemas ───────────────────────────────────────


class CurrentModel(BaseModel):
    run_id: Optional[str] = None
    artifact_path: Optional[str] = None


class CurrentMetrics(BaseModel):
    model_type: Optional[str] = None
    sector_profile: Optional[str] = None
    time_split: Optional[bool] = None
    metrics: dict = Field(default_factory=dict)


class RunHistoryPoint(BaseModel):
    run_id: str
    pr_auc: float


class PreparedDatasetSummary(BaseModel):
    summary: Optional[dict] = None


class DashboardSummaryResponse(BaseModel):
    """Shape of the operational snapshot consumed by the Next.js dashboard."""

    version: str
    git_commit: str
    refreshed_at: str
    current_model: Optional[CurrentModel] = None
    current_metrics: Optional[CurrentMetrics] = None
    etl_runs: list[dict] = Field(default_factory=list)
    fairness_audits: list[dict] = Field(default_factory=list)
    drift_reports: list[dict] = Field(default_factory=list)
    federated_runs: list[dict] = Field(default_factory=list)
    prepared_datasets: list[PreparedDatasetSummary] = Field(default_factory=list)
    recent_audits: list[dict] = Field(default_factory=list)
    run_history: list[RunHistoryPoint] = Field(default_factory=list)
    storage_status: Optional[dict] = None
    kpis: dict = Field(default_factory=dict)


class LatestMetricsResponse(BaseModel):
    """Response for `/metrics/latest`.

    When no trained model is registered, the endpoint returns HTTP 200 with
    `status="empty"` and a `message` so the frontend can render a clear
    empty state without interpreting a 404 as a server error.
    """

    status: str = "ok"
    model_type: Optional[str] = None
    metrics: dict = Field(default_factory=dict)
    run_id: Optional[str] = None
    artifact_path: Optional[str] = None
    message: Optional[str] = None


class CurrentModelResponse(BaseModel):
    """Response for `/models/current`.

    Uses the same explicit empty-state pattern as `/metrics/latest` so the
    dashboard can distinguish "no model yet" from "request failed".
    """

    status: str = "ok"
    run_id: Optional[str] = None
    model_type: Optional[str] = None
    artifact_path: Optional[str] = None
    version: Optional[str] = None
    message: Optional[str] = None
