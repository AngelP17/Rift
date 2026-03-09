from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from rift.lakehouse.sql import build_default_views, query_lakehouse
from rift.data.schemas import PredictionRequest
from rift.dashboard.views import build_dashboard_html, build_landing_html, dashboard_snapshot
from rift.datasets.adapters import list_prepared_datasets
from rift.etl.pipeline import list_etl_runs
from rift.federated.simulation import list_federated_runs
from rift.governance.fairness import list_fairness_audits
from rift.governance.model_cards import generate_model_card
from rift.monitoring.drift import list_drift_reports
from rift.monitoring.nl_query import answer_natural_language_query
from rift.explain.report import build_audit_report, build_explanation, report_to_markdown
from rift.models.infer import load_run, payload_to_frame, score_frame
from rift.replay.hashing import decision_hash
from rift.replay.recorder import record_decision
from rift.replay.replayer import replay_decision
from rift.storage.backends import get_storage_backend
from rift.utils.config import get_paths
from rift.utils.io import read_json


app = FastAPI(title="Rift API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(
        content=(
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
            "<rect width='64' height='64' rx='18' fill='#030712'/>"
            "<path d='M18 46V18h15.8c9 0 14.2 4.4 14.2 11.3 0 6.9-5.2 11.5-14.2 11.5H25.4V46H18Zm7.4-11.2h8.1c4.8 0 7.2-1.9 7.2-5.5 0-3.6-2.4-5.3-7.2-5.3h-8.1v10.8Z' fill='#6ea8fe'/>"
            "</svg>"
        ),
        media_type="image/svg+xml",
    )


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_probe() -> dict:
    # Chrome probes this path in some environments; returning an empty payload
    # avoids a noisy but otherwise harmless 404 in the console.
    return {}


@app.get("/", response_class=HTMLResponse)
def landing() -> HTMLResponse:
    return HTMLResponse(build_landing_html(get_paths()))


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


@app.get("/dashboard/summary")
def dashboard_summary() -> dict:
    return dashboard_snapshot(get_paths())


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(build_dashboard_html(get_paths()))


@app.get("/dashboard/export/model-card")
def export_model_card() -> PlainTextResponse:
    """Download the latest model card as a markdown file."""
    paths = get_paths()
    cards_dir = paths.governance_dir / "model_cards"
    if not cards_dir.exists():
        raise HTTPException(status_code=404, detail="No model cards generated yet")
    cards = sorted(cards_dir.glob("*.md"), reverse=True)
    if not cards:
        raise HTTPException(status_code=404, detail="No model cards generated yet")
    content = cards[0].read_text()
    return PlainTextResponse(
        content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{cards[0].name}"'},
    )


@app.get("/dashboard/export/audit")
def export_audit_report() -> PlainTextResponse:
    """Download the latest audit report as a markdown file."""
    paths = get_paths()
    import duckdb
    if not paths.audit_db.exists():
        raise HTTPException(status_code=404, detail="No audit reports recorded yet")
    conn = duckdb.connect(str(paths.audit_db), read_only=True)
    exists = conn.execute(
        "select count(*) from information_schema.tables where table_name = 'audit_reports'"
    ).fetchone()[0]
    if not exists:
        conn.close()
        raise HTTPException(status_code=404, detail="No audit reports recorded yet")
    row = conn.execute("select markdown from audit_reports order by decision_id desc limit 1").fetchone()
    conn.close()
    if row is None or row[0] is None:
        raise HTTPException(status_code=404, detail="No audit reports with markdown found")
    return PlainTextResponse(
        row[0],
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="audit_report.md"'},
    )


@app.post("/governance/model-card/{run_id}")
def model_card(run_id: str) -> dict:
    return generate_model_card(get_paths(), run_id, repo_root=Path("/workspace")).to_dict()


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
