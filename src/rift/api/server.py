from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from rift.lakehouse.sql import build_default_views, query_lakehouse
from rift.data.schemas import PredictionRequest
from rift.dashboard.views import build_dashboard_html, dashboard_snapshot
from rift.datasets.adapters import list_prepared_datasets
from rift.etl.pipeline import list_etl_runs
from rift.federated.simulation import list_federated_runs
from rift.governance.fairness import list_fairness_audits
from rift.explain.report import build_audit_report, build_explanation, report_to_markdown
from rift.models.infer import load_run, payload_to_frame, score_frame
from rift.replay.hashing import decision_hash
from rift.replay.recorder import record_decision
from rift.replay.replayer import replay_decision
from rift.storage.backends import get_storage_backend
from rift.utils.config import get_paths
from rift.utils.io import read_json


app = FastAPI(title="Rift API", version="0.1.0")


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    paths = get_paths()
    artifact = load_run(paths.runs_dir)
    payload = request.model_dump(mode="json")
    frame = payload_to_frame(payload)
    prediction, feature_frame = score_frame(frame, artifact)
    explanation, _ = build_explanation(feature_frame, artifact, prediction)
    prediction["explanation"] = explanation
    decision_id = decision_hash(
        {
            "payload": payload,
            "model_run_id": artifact["_run_id"],
            "prediction": prediction,
        }
    )
    report = build_audit_report(decision_id, feature_frame, artifact, prediction)
    record_decision(
        db_path=paths.audit_db,
        decision_id=decision_id,
        payload=payload,
        feature_frame=feature_frame,
        prediction=prediction,
        report=report,
        markdown=report_to_markdown(report),
        model_run_id=artifact["_run_id"],
    )
    return {
        "decision_id": decision_id,
        "model_run_id": artifact["_run_id"],
        **prediction,
    }


@app.get("/replay/{decision_id}")
def replay(decision_id: str) -> dict:
    try:
        return replay_decision(get_paths().audit_db, decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/audit/{decision_id}")
def audit(decision_id: str) -> dict:
    try:
        result = replay_decision(get_paths().audit_db, decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result["report"]


@app.get("/metrics/latest")
def metrics_latest() -> dict:
    paths = get_paths()
    current = read_json(paths.runs_dir / "current_run.json")
    return read_json(paths.runs_dir / current["run_id"] / "metrics.json")


@app.get("/models/current")
def current_model() -> dict:
    paths = get_paths()
    current = read_json(paths.runs_dir / "current_run.json")
    return {"run_id": current["run_id"], "artifact_path": current["artifact_path"]}


@app.get("/etl/status")
def etl_status(limit: int = 10) -> list[dict]:
    return list_etl_runs(get_paths().warehouse_db, limit=limit)


@app.get("/datasets/status")
def datasets_status(limit: int = 10) -> list[dict]:
    return list_prepared_datasets(get_paths(), limit=limit)


@app.get("/fairness/status")
def fairness_status(limit: int = 10) -> list[dict]:
    return list_fairness_audits(get_paths(), limit=limit)


@app.get("/federated/status")
def federated_status(limit: int = 10) -> list[dict]:
    return list_federated_runs(get_paths(), limit=limit)


@app.get("/dashboard/summary")
def dashboard_summary() -> dict:
    return dashboard_snapshot(get_paths())


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(build_dashboard_html(get_paths()))


@app.get("/storage/status")
def storage_status() -> dict:
    return get_storage_backend(get_paths()).status().to_dict()


@app.get("/lakehouse/status")
def lakehouse_status() -> dict:
    db_path = build_default_views(get_paths())
    return {"lakehouse_db": str(db_path)}


@app.get("/lakehouse/query")
def lakehouse_query(sql: str, limit: int = 1000) -> dict:
    return query_lakehouse(get_paths(), sql=sql, limit=limit).to_dict()
