"""FastAPI server for Rift."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from rift.api.schemas import PredictRequest, PredictResponse
from rift.utils.config import API_HOST, API_PORT, AUDIT_DB_PATH, MODEL_DIR


# Global model state (loaded on startup)
_model_state: dict[str, Any] = {}


def load_model_artifact() -> None:
    """Load model and calibrator from artifacts (simplified)."""
    _model_state["loaded"] = True
    _model_state["model"] = None
    _model_state["calibrator"] = None
    _model_state["feat_cols"] = []
    # In production, load from MODEL_DIR/MLflow
    result_path = MODEL_DIR / "train_result.json"
    if result_path.exists():
        import json
        with open(result_path) as f:
            _model_state["config"] = json.load(f)
            _model_state["feat_cols"] = _model_state["config"].get("feat_cols", [])


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_artifact()
    yield
    _model_state.clear()


app = FastAPI(
    title="Rift API",
    description="Graph ML for Fraud Detection, Replay, and Audit",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Single transaction prediction."""
    if not _model_state.get("model"):
        raise HTTPException(503, "Model not loaded. Train first: rift train --model graphsage_xgb")
    # Stub - would call rift.models.infer.predict
    return PredictResponse(
        raw_score=0.0,
        calibrated_score=0.0,
        decision_band="review_needed",
        is_fraud_pred=False,
    )


@app.get("/replay/{decision_id}")
def replay(decision_id: str) -> dict:
    """Replay a stored decision."""
    return {"decision_id": decision_id, "status": "Use CLI: rift replay " + decision_id}


@app.get("/audit/{decision_id}")
def audit(decision_id: str, format: str = "json") -> dict | PlainTextResponse:
    """Get audit report for decision."""
    if format == "markdown":
        return PlainTextResponse("# Audit report\nUse CLI: rift audit " + decision_id)
    return {"decision_id": decision_id, "report": "Use CLI: rift audit " + decision_id}


@app.get("/metrics/latest")
def metrics() -> dict:
    """Latest model metrics."""
    config = _model_state.get("config", {})
    return config.get("metrics", {})


@app.get("/models/current")
def models_current() -> dict:
    """Current model info."""
    return {
        "model_type": _model_state.get("config", {}).get("model_type", "none"),
        "loaded": _model_state.get("loaded", False),
    }
