from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from rift.dashboard.views import (
    build_audits_detail,
    build_dashboard_html,
    build_etl_detail,
    build_governance_detail,
    build_models_detail,
    dashboard_snapshot,
    get_static_dir,
)
from rift.data.schemas import PredictionRequest
from rift.datasets.adapters import list_prepared_datasets
from rift.etl.pipeline import list_etl_runs
from rift.explain.report import build_audit_report, build_explanation, report_to_markdown
from rift.federated.simulation import list_federated_runs
from rift.governance.fairness import list_fairness_audits
from rift.governance.model_cards import generate_model_card
from rift.lakehouse.sql import build_default_views, query_lakehouse
from rift.models.infer import load_run, payload_to_frame, score_frame
from rift.monitoring.drift import list_drift_reports
from rift.monitoring.nl_query import answer_natural_language_query
from rift.replay.hashing import decision_hash
from rift.replay.recorder import record_decision
from rift.replay.replayer import replay_decision
from rift.storage.backends import get_storage_backend
from rift.utils.config import get_paths
from rift.utils.io import read_json


app = FastAPI(title="Rift API", version="1.0.0")

_REPO_ROOT = Path(__file__).resolve().parents[3]

app.mount("/static", StaticFiles(directory=str(get_static_dir())), name="static")


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


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


@app.get("/monitor/drift-status")
def drift_status(limit: int = 10) -> list[dict]:
    return list_drift_reports(get_paths(), limit=limit)


@app.get("/query")
def natural_query(natural: str) -> dict:
    return answer_natural_language_query(get_paths(), natural).to_dict()


# ── Dashboard routes ──────────────────────────────────────────────

@app.get("/dashboard/summary")
def dashboard_summary_json() -> dict:
    return dashboard_snapshot(get_paths())


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_index() -> HTMLResponse:
    return HTMLResponse(build_dashboard_html(get_paths()))


@app.get("/dashboard/etl", response_class=HTMLResponse)
def dashboard_etl() -> HTMLResponse:
    return HTMLResponse(build_etl_detail(get_paths()))


@app.get("/dashboard/governance", response_class=HTMLResponse)
def dashboard_governance() -> HTMLResponse:
    return HTMLResponse(build_governance_detail(get_paths()))


@app.get("/dashboard/audits", response_class=HTMLResponse)
def dashboard_audits() -> HTMLResponse:
    return HTMLResponse(build_audits_detail(get_paths()))


@app.get("/dashboard/models", response_class=HTMLResponse)
def dashboard_models() -> HTMLResponse:
    return HTMLResponse(build_models_detail(get_paths()))


# ── Export / download routes ──────────────────────────────────────

@app.get("/exports/model-card/{run_id}")
def export_model_card(run_id: str, format: str = "md") -> PlainTextResponse:
    paths = get_paths()
    try:
        result = generate_model_card(paths, run_id, repo_root=_REPO_ROOT)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    card_path = Path(result.model_card_path)
    if not card_path.exists():
        raise HTTPException(status_code=404, detail="Model card file not found")

    content = card_path.read_text(encoding="utf-8")
    return PlainTextResponse(
        content,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=model_card_{run_id}.md"},
    )


@app.get("/exports/audit/{decision_id}")
def export_audit(decision_id: str, format: str = "md") -> PlainTextResponse:
    paths = get_paths()
    try:
        result = replay_decision(paths.audit_db, decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if format == "json":
        import json
        content = json.dumps(result["report"], indent=2, default=str)
        return PlainTextResponse(
            content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=audit_{decision_id}.json"},
        )

    markdown = result.get("markdown", "")
    if not markdown:
        markdown = str(result.get("report", {}))
    return PlainTextResponse(
        markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=audit_{decision_id}.md"},
    )


# ── Governance routes ─────────────────────────────────────────────

@app.post("/governance/model-card/{run_id}")
def model_card(run_id: str) -> dict:
    return generate_model_card(get_paths(), run_id, repo_root=_REPO_ROOT).to_dict()


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
