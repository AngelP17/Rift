"""FastAPI server for Rift fraud detection system."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AuditReportResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReplayResponse,
    TransactionRequest,
)
from replay.recorder import DecisionRecorder
from replay.replayer import ReplayEngine
from utils.logging import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

app = FastAPI(
    title="Rift",
    description="Graph ML for Fraud Detection, Replay, and Audit",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_recorder: DecisionRecorder | None = None
_replay_engine: ReplayEngine | None = None


def get_recorder() -> DecisionRecorder:
    global _recorder
    if _recorder is None:
        _recorder = DecisionRecorder()
    return _recorder


def get_replay_engine() -> ReplayEngine:
    global _replay_engine
    if _replay_engine is None:
        _replay_engine = ReplayEngine(get_recorder())
    return _replay_engine


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(tx: TransactionRequest):
    """Run fraud prediction on a transaction."""
    try:
        from models.infer import InferencePipeline

        pipeline = InferencePipeline()
        result = pipeline.predict(tx.model_dump())

        recorder = get_recorder()
        recorder.record_transaction(tx.tx_id, tx.model_dump())
        recorder.record_prediction(result)

        return PredictionResponse(
            decision_id=result["decision_id"],
            tx_id=result["tx_id"],
            raw_score=result["raw_score"],
            calibrated_score=result["calibrated_score"],
            confidence_band=result["confidence_band"],
            interval_low=result["interval_low"],
            interval_high=result["interval_high"],
            timestamp=result["timestamp"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model artifacts not found. Train a model first. {e}")
    except Exception as e:
        log.error("prediction_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/replay/{decision_id}", response_model=ReplayResponse)
async def replay(decision_id: str):
    """Replay a past decision for verification."""
    engine = get_replay_engine()
    result = engine.replay(decision_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return ReplayResponse(**result)


@app.get("/audit/{decision_id}", response_model=AuditReportResponse)
async def audit(decision_id: str):
    """Get the audit report for a decision."""
    recorder = get_recorder()

    report = recorder.conn.execute(
        "SELECT * FROM audit_reports WHERE decision_id = ?", [decision_id]
    ).fetchone()

    if report is None:
        pred = recorder.get_prediction(decision_id)
        if pred is None:
            raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

        from explain.report import generate_report, report_to_markdown
        report_data = generate_report(pred)
        markdown = report_to_markdown(report_data)
        recorder.record_audit_report(decision_id, markdown, report_data)

        return AuditReportResponse(
            decision_id=decision_id,
            report_markdown=markdown,
            report_json=report_data,
        )

    cols = [d[0] for d in recorder.conn.description]
    report_dict = dict(zip(cols, report))

    return AuditReportResponse(
        decision_id=decision_id,
        report_markdown=report_dict.get("report_markdown"),
        report_json=json.loads(report_dict["report_json"]) if report_dict.get("report_json") else None,
    )


@app.get("/metrics/latest", response_model=MetricsResponse)
async def metrics_latest():
    """Get metrics for the latest trained model."""
    recorder = get_recorder()
    result = recorder.conn.execute(
        "SELECT * FROM model_registry ORDER BY registered_at DESC LIMIT 1"
    ).fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail="No models registered")

    cols = [d[0] for d in recorder.conn.description]
    model = dict(zip(cols, result))

    return MetricsResponse(
        model_type=model["model_type"],
        metrics=json.loads(model["metrics"]) if model["metrics"] else {},
    )


@app.get("/models/current", response_model=ModelInfoResponse)
async def models_current():
    """Get info about the currently deployed model."""
    recorder = get_recorder()
    result = recorder.conn.execute(
        "SELECT * FROM model_registry ORDER BY registered_at DESC LIMIT 1"
    ).fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail="No models registered")

    cols = [d[0] for d in recorder.conn.description]
    model = dict(zip(cols, result))

    return ModelInfoResponse(
        model_id=model["model_id"],
        model_type=model["model_type"],
        version=model["version"],
        artifact_path=model.get("artifact_path"),
    )
