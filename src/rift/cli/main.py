from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl
import typer

from rift.adapters.sectors import DEFAULT_PROFILE, available_sector_profiles, load_sector_profile
from rift.cli.ui import emit_json, emit_markdown, emit_panel, emit_records, emit_summary, emit_syntax, run_staged_progress
from rift.compute.spark_compat import spark_available, summarise_parquet_with_spark
from rift.data.generator import generate_transactions
from rift.datasets.adapters import list_prepared_datasets, prepare_public_dataset
from rift.governance.model_cards import generate_model_card
from rift.lakehouse.sql import build_default_views, query_lakehouse
from rift.etl.pipeline import list_etl_runs, run_etl_pipeline
from rift.federated.simulation import list_federated_runs, train_federated_model
from rift.governance.fairness import list_fairness_audits, run_fairness_audit
from rift.monitoring.drift import list_drift_reports, run_drift_monitor
from rift.monitoring.nl_query import answer_natural_language_query
from rift.orchestration.pipeline import run_end_to_end_pipeline
from rift.explain.report import build_audit_report, build_explanation, report_to_markdown
from rift.models.infer import load_run, payload_to_frame, score_frame
from rift.models.train import train_from_frame
from rift.replay.hashing import decision_hash
from rift.replay.recorder import record_decision
from rift.replay.replayer import replay_decision
from rift.reengineer.simulate import simulate_legacy_migration
from rift.storage.backends import get_storage_backend
from rift.utils.config import get_paths
from rift.utils.io import read_json


app = typer.Typer(help="Rift: graph ML for fraud detection, replay, and audit.")
etl_app = typer.Typer(help="Auditable ETL pipelines for transaction and government-style source data.")
dataset_app = typer.Typer(help="Prepare public datasets into Rift's canonical schema.")
fairness_app = typer.Typer(help="Run governance and fairness audits on scored datasets.")
federated_app = typer.Typer(help="Zero-cost local federated training scaffolding.")
storage_app = typer.Typer(help="Local and S3-compatible storage helpers.")
lakehouse_app = typer.Typer(help="DuckDB lakehouse views and SQL queries over Parquet.")
spark_app = typer.Typer(help="Optional Spark-compatible local compute helpers.")
pipeline_app = typer.Typer(help="End-to-end orchestrated pipeline helpers.")
governance_app = typer.Typer(help="Model cards and governance document generation.")
monitor_app = typer.Typer(help="Monitoring, drift detection, and query workflows.")
sector_app = typer.Typer(help="Sector profiles and adapter metadata.")
reengineer_app = typer.Typer(help="Legacy migration and reengineering helpers.")
app.add_typer(etl_app, name="etl")
app.add_typer(dataset_app, name="dataset")
app.add_typer(fairness_app, name="fairness")
app.add_typer(federated_app, name="federated")
app.add_typer(storage_app, name="storage")
app.add_typer(lakehouse_app, name="lakehouse")
app.add_typer(spark_app, name="spark")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(governance_app, name="governance")
app.add_typer(monitor_app, name="monitor")
app.add_typer(sector_app, name="sector")
app.add_typer(reengineer_app, name="reengineer")


def _read_frame(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    if suffix == ".json":
        return pl.read_json(path)
    raise typer.BadParameter(f"unsupported data format: {path.suffix}")


@app.command()
def generate(
    txns: int = typer.Option(10_000, min=100),
    users: int = typer.Option(1_000, min=10),
    merchants: int = typer.Option(200, min=10),
    fraud_rate: float = typer.Option(0.02, min=0.001, max=0.5),
    seed: int = typer.Option(7),
) -> None:
    paths = get_paths()
    state: dict[str, pl.DataFrame] = {}

    def build_frame() -> pl.DataFrame:
        frame = generate_transactions(txns=txns, users=users, merchants=merchants, fraud_rate=fraud_rate, seed=seed)
        state["frame"] = frame
        return frame

    def persist_frame() -> Path:
        state["frame"].write_parquet(paths.data_path)
        return paths.data_path

    run_staged_progress(
        "Generating synthetic fraud data",
        [
            ("Sampling entities and injected fraud patterns", build_frame),
            ("Writing parquet output", persist_frame),
        ],
    )
    emit_summary(
        "Synthetic Dataset Ready",
        [
            ("Transactions", state["frame"].height),
            ("Users", users),
            ("Merchants", merchants),
            ("Fraud rate", fraud_rate),
            ("Path", paths.data_path),
        ],
    )


@app.command()
def train(
    model: str = typer.Option("graphsage_xgb", help="xgb_tabular, graphsage_only, or graphsage_xgb"),
    time_split: bool = typer.Option(False, help="Use chronological rather than random split."),
    data_path: Path | None = typer.Option(None),
    sector: str = typer.Option(DEFAULT_PROFILE, "--sector"),
    optimize: str = typer.Option("standard", "--optimize", help="Use 'green' for lightweight size optimization metadata."),
) -> None:
    paths = get_paths()
    source = data_path or paths.data_path
    state: dict[str, object] = {}

    def load_frame() -> pl.DataFrame:
        frame = pl.read_parquet(source)
        state["frame"] = frame
        return frame

    def run_training() -> object:
        summary = train_from_frame(
            state["frame"],  # type: ignore[arg-type]
            runs_dir=paths.runs_dir,
            model_type=model,
            time_split=time_split,
            sector_profile=sector,
            optimize_mode=optimize,
        )
        state["summary"] = summary
        return summary

    run_staged_progress(
        "Training fraud model",
        [
            ("Loading training frame", load_frame),
            ("Running hybrid model pipeline", run_training),
        ],
    )
    summary = state["summary"]
    emit_summary(
        "Training Complete",
        [
            ("Run ID", summary.run_id),  # type: ignore[union-attr]
            ("Model", model),
            ("Sector", sector),
            ("Temporal split", time_split),
            ("PR-AUC", f"{summary.metrics.get('pr_auc', 0.0):.4f}"),  # type: ignore[union-attr]
            ("ECE", f"{summary.metrics.get('ece', 0.0):.4f}"),  # type: ignore[union-attr]
        ],
    )
    emit_syntax({"run_id": summary.run_id, "metrics": summary.metrics})  # type: ignore[union-attr]


@app.command()
def predict(tx: Path = typer.Option(..., "--tx", exists=True, readable=True)) -> None:
    paths = get_paths()
    artifact = load_run(paths.runs_dir)
    payload = read_json(tx)
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
    markdown = report_to_markdown(report)
    record_decision(
        db_path=paths.audit_db,
        decision_id=decision_id,
        payload=payload,
        feature_frame=feature_frame,
        prediction=prediction,
        report=report,
        markdown=markdown,
        model_run_id=artifact["_run_id"],
    )
    emit_summary(
        "Prediction Complete",
        [
            ("Decision ID", decision_id),
            ("Run ID", artifact["_run_id"]),
            ("Decision", prediction.get("decision", "unknown")),
            ("Probability", f"{prediction.get('calibrated_probability', 0.0):.4f}"),
            ("Confidence", f"{prediction.get('confidence', 0.0):.4f}"),
        ],
    )
    emit_syntax(
        {
            "decision_id": decision_id,
            "model_run_id": artifact["_run_id"],
            **prediction,
        }
    )


@app.command()
def replay(decision_id: str) -> None:
    paths = get_paths()
    emit_json(replay_decision(paths.audit_db, decision_id))


@app.command()
def audit(decision_id: str, format: str = typer.Option("markdown")) -> None:
    paths = get_paths()
    result = replay_decision(paths.audit_db, decision_id)
    if format == "json":
        emit_json(result["report"])
        return
    emit_markdown(result["markdown"])


@app.command()
def compare() -> None:
    paths = get_paths()
    metrics = []
    for metric_file in sorted(paths.runs_dir.glob("run_*/metrics.json")):
        metrics.append(read_json(metric_file))
    emit_syntax(metrics)


@app.command()
def export(since: int = typer.Option(90), format: str = typer.Option("markdown")) -> None:
    paths = get_paths()
    conn = duckdb.connect(str(paths.audit_db), read_only=True)
    rows = conn.execute(
        """
        select ar.decision_id, ar.markdown, ar.report_json
        from audit_reports ar
        join transactions t on ar.decision_id = t.decision_id
        where t.created_at >= current_timestamp - make_interval(days := ?)
        order by ar.decision_id
        """,
        [since],
    ).fetchall()
    conn.close()
    if format == "json":
        emit_json([json.loads(row[2]) for row in rows])
        return
    emit_markdown("\n\n---\n\n".join(row[1] for row in rows))


@etl_app.command("run")
def etl_run(
    source: Path = typer.Option(..., "--source", exists=True, readable=True),
    source_system: str = typer.Option("government_finance"),
    dataset_name: str = typer.Option("transactions"),
    sector: str = typer.Option(DEFAULT_PROFILE, "--sector"),
) -> None:
    paths = get_paths()
    summary = run_staged_progress(
        "Running ETL pipeline",
        [
            (
                "Validating source and generating warehouse artifacts",
                lambda: run_etl_pipeline(
                    source=source,
                    paths=paths,
                    source_system=source_system,
                    dataset_name=dataset_name,
                    sector=sector,
                    repo_root=Path.cwd(),
                ),
            ),
        ],
    )
    emit_summary(
        "ETL Run Complete",
        [
            ("Run ID", summary.run_id),
            ("Source system", summary.source_system),
            ("Rows valid", summary.rows_valid),
            ("Rows invalid", summary.rows_invalid),
            ("Duplicates removed", summary.duplicates_removed),
        ],
    )
    emit_syntax(summary.to_dict())


@etl_app.command("status")
def etl_status(limit: int = typer.Option(10, min=1, max=100)) -> None:
    paths = get_paths()
    emit_records(
        "ETL Status",
        list_etl_runs(paths.warehouse_db, limit=limit),
        ["run_id", "source_system", "rows_valid", "rows_invalid", "duplicates_removed"],
    )


@dataset_app.command("prepare")
def dataset_prepare(
    adapter: str = typer.Option(..., help="Supported: ieee_cis, credit_card_fraud"),
    source: Path = typer.Option(..., "--source", exists=True, readable=True),
    auto_etl: bool = typer.Option(True, help="Run the ETL pipeline after preparing the canonical dataset."),
) -> None:
    paths = get_paths()
    summary = prepare_public_dataset(source=source, adapter=adapter, paths=paths, auto_etl=auto_etl)
    emit_summary(
        "Dataset Preparation Complete",
        [
            ("Dataset ID", summary.dataset_id),
            ("Adapter", summary.adapter),
            ("Rows prepared", summary.rows_prepared),
            ("Auto ETL", auto_etl),
        ],
    )
    emit_syntax(summary.to_dict())


@dataset_app.command("status")
def dataset_status(limit: int = typer.Option(10, min=1, max=100)) -> None:
    paths = get_paths()
    rows = [item.get("summary", {}) for item in list_prepared_datasets(paths, limit=limit)]
    emit_records("Prepared Datasets", rows, ["dataset_id", "adapter", "rows_prepared", "auto_etl_run_id"])


@fairness_app.command("audit")
def fairness_audit(
    sensitive_column: str = typer.Option(..., "--sensitive-column"),
    data_path: Path | None = typer.Option(None, "--data-path"),
    run_id: str | None = typer.Option(None, "--run-id"),
    threshold: float = typer.Option(0.5, min=0.0, max=1.0),
) -> None:
    paths = get_paths()
    source = data_path or paths.data_path
    frame = _read_frame(source)
    summary = run_fairness_audit(
        frame=frame,
        paths=paths,
        sensitive_column=sensitive_column,
        run_id=run_id,
        threshold=threshold,
        data_path=str(source),
    )
    emit_summary(
        "Fairness Audit Complete",
        [
            ("Audit ID", summary.audit_id),
            ("Sensitive column", summary.sensitive_column),
            ("Demographic parity diff", f"{summary.demographic_parity_difference:.4f}"),
            ("Disparate impact ratio", f"{summary.disparate_impact_ratio:.4f}"),
        ],
    )
    emit_syntax(summary.to_dict())


@fairness_app.command("status")
def fairness_status(limit: int = typer.Option(10, min=1, max=100)) -> None:
    paths = get_paths()
    emit_records(
        "Fairness Audits",
        list_fairness_audits(paths, limit=limit),
        ["audit_id", "sensitive_column", "demographic_parity_difference", "disparate_impact_ratio"],
    )


@federated_app.command("train")
def federated_train(
    data_path: Path | None = typer.Option(None, "--data-path"),
    client_column: str = typer.Option("channel", "--client-column"),
    rounds: int = typer.Option(5, min=1, max=100),
    local_epochs: int = typer.Option(3, min=1, max=100),
    learning_rate: float = typer.Option(0.1, min=0.0001, max=10.0),
    time_split: bool = typer.Option(False),
    optimize: str = typer.Option("standard", "--optimize"),
) -> None:
    paths = get_paths()
    source = data_path or paths.data_path
    frame = _read_frame(source)
    summary = train_federated_model(
        frame=frame,
        paths=paths,
        client_column=client_column,
        rounds=rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        time_split=time_split,
        optimize_mode=optimize,
    )
    emit_summary(
        "Federated Training Complete",
        [
            ("Run ID", summary.run_id),
            ("Client column", summary.client_column),
            ("Client count", summary.client_count),
            ("Rounds", summary.rounds),
        ],
    )
    emit_syntax(summary.to_dict())


@federated_app.command("status")
def federated_status(limit: int = typer.Option(10, min=1, max=100)) -> None:
    paths = get_paths()
    emit_records("Federated Runs", list_federated_runs(paths, limit=limit), ["run_id", "client_column", "client_count", "rounds"])


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000, min=1, max=65535),
) -> None:
    import uvicorn

    uvicorn.run("rift.api.server:app", host=host, port=port, reload=False)


@storage_app.command("status")
def storage_status() -> None:
    paths = get_paths()
    backend = get_storage_backend(paths)
    emit_summary("Storage Status", list(backend.status().to_dict().items()))


@storage_app.command("sync")
def storage_sync(
    object_name: str = typer.Option("data/current_transactions.parquet"),
    source: Path | None = typer.Option(None, "--source"),
) -> None:
    paths = get_paths()
    backend = get_storage_backend(paths)
    frame = _read_frame(source or paths.data_path)
    target = backend.save_parquet(frame, object_name)
    emit_summary(
        "Storage Sync Complete",
        [
            ("Backend", backend.status().backend),
            ("Object", object_name),
            ("Target", target),
        ],
    )


@lakehouse_app.command("build")
def lakehouse_build() -> None:
    paths = get_paths()
    db_path = build_default_views(paths)
    emit_summary("Lakehouse Ready", [("Database", str(db_path))])


@lakehouse_app.command("query")
def lakehouse_query(
    sql: str = typer.Option(..., "--sql"),
    limit: int = typer.Option(1000, min=1, max=10000),
) -> None:
    paths = get_paths()
    result = query_lakehouse(paths, sql=sql, limit=limit)
    emit_json(result.to_dict())


@spark_app.command("summary")
def spark_summary(
    data_path: Path | None = typer.Option(None, "--data-path"),
) -> None:
    source = data_path or get_paths().data_path
    emit_json(
        {
            "spark_available": spark_available(),
            "summary": summarise_parquet_with_spark(source) if spark_available() else None,
        }
    )


@pipeline_app.command("run")
def pipeline_run(
    txns: int = typer.Option(10_000, min=100),
    users: int = typer.Option(1_000, min=10),
    merchants: int = typer.Option(200, min=10),
    fraud_rate: float = typer.Option(0.02, min=0.001, max=0.5),
    model: str = typer.Option("graphsage_xgb"),
    sample_tx: Path = typer.Option(Path("demo/sample_transaction.json"), "--sample-tx", exists=True, readable=True),
    optimize: str = typer.Option("standard", "--optimize"),
) -> None:
    paths = get_paths()
    summary = run_staged_progress(
        "Running end-to-end Rift pipeline",
        [
            (
                "Generating data, training model, and scoring sample transaction",
                lambda: run_end_to_end_pipeline(
                    paths=paths,
                    txns=txns,
                    users=users,
                    merchants=merchants,
                    fraud_rate=fraud_rate,
                    model_type=model,
                    sample_tx_path=sample_tx,
                    optimize_mode=optimize,
                ),
            ),
        ],
    )
    emit_panel(
        "Pipeline Complete",
        "Setup → conflict → resolution: data generated, model trained, sample decision recorded, and artifacts persisted for replay.",
        "green",
    )
    emit_syntax(summary.to_dict())


@governance_app.command("generate-card")
def governance_generate_card(
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    paths = get_paths()
    resolved_run = run_id or read_json(paths.runs_dir / "current_run.json")["run_id"]
    summary = generate_model_card(paths, resolved_run, repo_root=Path.cwd())
    emit_summary("Model Card Generated", [("Run ID", resolved_run), ("Model card", summary.model_card_path)])
    emit_syntax(summary.to_dict())


@monitor_app.command("drift")
def monitor_drift(
    reference_path: Path = typer.Option(..., "--reference-path", exists=True, readable=True),
    current_path: Path = typer.Option(..., "--current-path", exists=True, readable=True),
    threshold: float = typer.Option(0.2, min=0.0, max=1.0),
    trigger_retrain: bool = typer.Option(False, "--trigger-retrain"),
    model: str = typer.Option("graphsage_xgb", "--model"),
) -> None:
    paths = get_paths()
    summary = run_staged_progress(
        "Monitoring drift",
        [
            (
                "Comparing reference and current windows",
                lambda: run_drift_monitor(
                    paths=paths,
                    reference_path=reference_path,
                    current_path=current_path,
                    threshold=threshold,
                    trigger_retrain=trigger_retrain,
                    model_type=model,
                ),
            ),
        ],
    )
    emit_summary(
        "Drift Report Complete",
        [
            ("Report ID", summary.report_id),
            ("Drift score", f"{summary.drift_score:.4f}"),
            ("Is drift", summary.is_drift),
            ("Retrain triggered", summary.retrain_triggered),
        ],
    )
    emit_syntax(summary.to_dict())


@monitor_app.command("drift-status")
def monitor_drift_status(limit: int = typer.Option(10, min=1, max=100)) -> None:
    paths = get_paths()
    emit_records(
        "Drift Reports",
        list_drift_reports(paths, limit=limit),
        ["report_id", "drift_score", "is_drift", "retrain_triggered"],
    )


@app.command()
def query(natural: str = typer.Option(..., "--natural")) -> None:
    paths = get_paths()
    result = answer_natural_language_query(paths, natural)
    emit_json(result.to_dict())


@sector_app.command("list")
def sector_list() -> None:
    emit_json(available_sector_profiles(Path.cwd()))


@sector_app.command("show")
def sector_show(name: str = typer.Option(DEFAULT_PROFILE, "--name")) -> None:
    profile = load_sector_profile(Path.cwd(), name)
    emit_json(
        {
            "name": profile.name,
            "source_system": profile.source_system,
            "field_aliases": profile.field_aliases,
            "privacy_masks": profile.privacy_masks,
            "added_columns": profile.added_columns,
        }
    )


@reengineer_app.command("simulate")
def reengineer_simulate(
    source: Path = typer.Option(..., "--source", exists=True, readable=True),
    output_path: Path = typer.Option(Path(".rift/reengineered/legacy_output.parquet"), "--output-path"),
    source_system: str = typer.Option("legacy_rules_engine", "--source-system"),
    sector: str = typer.Option(DEFAULT_PROFILE, "--sector"),
) -> None:
    paths = get_paths()
    summary = run_staged_progress(
        "Simulating legacy migration",
        [
            (
                "Transforming source export into Rift-compatible output",
                lambda: simulate_legacy_migration(
                    paths=paths,
                    source=source,
                    output_path=output_path,
                    source_system=source_system,
                    sector=sector,
                ),
            ),
        ],
    )
    emit_summary(
        "Legacy Migration Simulation Complete",
        [
            ("Source system", source_system),
            ("Output path", output_path),
            ("Sector", sector),
        ],
    )
    emit_syntax(summary.to_dict())


if __name__ == "__main__":
    app()
