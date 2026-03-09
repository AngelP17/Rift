from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException

from rift.data.schemas import PredictionRequest
from rift.explain.report import build_audit_report, build_explanation, report_to_markdown
from rift.models.infer import load_run, payload_to_frame, score_frame
from rift.replay.hashing import decision_hash
from rift.replay.recorder import record_decision
from rift.replay.replayer import replay_decision
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
