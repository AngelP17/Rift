from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl
import typer

from rift.compute.spark_compat import spark_available, summarise_parquet_with_spark
from rift.data.generator import generate_transactions
from rift.datasets.adapters import list_prepared_datasets, prepare_public_dataset
from rift.lakehouse.sql import build_default_views, query_lakehouse
from rift.etl.pipeline import list_etl_runs, run_etl_pipeline
from rift.federated.simulation import list_federated_runs, train_federated_model
from rift.governance.fairness import list_fairness_audits, run_fairness_audit
from rift.orchestration.pipeline import run_end_to_end_pipeline
from rift.explain.report import build_audit_report, build_explanation, report_to_markdown
from rift.models.infer import load_run, payload_to_frame, score_frame
from rift.models.train import train_from_frame
from rift.replay.hashing import decision_hash
from rift.replay.recorder import record_decision
from rift.replay.replayer import replay_decision
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
app.add_typer(etl_app, name="etl")
app.add_typer(dataset_app, name="dataset")
app.add_typer(fairness_app, name="fairness")
app.add_typer(federated_app, name="federated")
app.add_typer(storage_app, name="storage")
app.add_typer(lakehouse_app, name="lakehouse")
app.add_typer(spark_app, name="spark")
app.add_typer(pipeline_app, name="pipeline")


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
    frame = generate_transactions(txns=txns, users=users, merchants=merchants, fraud_rate=fraud_rate, seed=seed)
    frame.write_parquet(paths.data_path)
    typer.echo(f"wrote {frame.height} transactions to {paths.data_path}")


@app.command()
def train(
    model: str = typer.Option("graphsage_xgb", help="xgb_tabular, graphsage_only, or graphsage_xgb"),
    time_split: bool = typer.Option(False, help="Use chronological rather than random split."),
    data_path: Path | None = typer.Option(None),
) -> None:
    paths = get_paths()
    source = data_path or paths.data_path
    frame = pl.read_parquet(source)
    summary = train_from_frame(frame, runs_dir=paths.runs_dir, model_type=model, time_split=time_split)
    typer.echo(json.dumps({"run_id": summary.run_id, "metrics": summary.metrics}, indent=2))


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
    typer.echo(
        json.dumps(
            {
                "decision_id": decision_id,
                "model_run_id": artifact["_run_id"],
                **prediction,
            },
            indent=2,
        )
    )


@app.command()
def replay(decision_id: str) -> None:
    paths = get_paths()
    typer.echo(json.dumps(replay_decision(paths.audit_db, decision_id), indent=2))


@app.command()
def audit(decision_id: str, format: str = typer.Option("markdown")) -> None:
    paths = get_paths()
    result = replay_decision(paths.audit_db, decision_id)
    if format == "json":
        typer.echo(json.dumps(result["report"], indent=2))
        return
    typer.echo(result["markdown"])


@app.command()
def compare() -> None:
    paths = get_paths()
    metrics = []
    for metric_file in sorted(paths.runs_dir.glob("run_*/metrics.json")):
        metrics.append(read_json(metric_file))
    typer.echo(json.dumps(metrics, indent=2))


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
        typer.echo(json.dumps([json.loads(row[2]) for row in rows], indent=2))
        return
    typer.echo("\n\n---\n\n".join(row[1] for row in rows))


@etl_app.command("run")
def etl_run(
    source: Path = typer.Option(..., "--source", exists=True, readable=True),
    source_system: str = typer.Option("government_finance"),
    dataset_name: str = typer.Option("transactions"),
) -> None:
    paths = get_paths()
    summary = run_etl_pipeline(
        source=source,
        paths=paths,
        source_system=source_system,
        dataset_name=dataset_name,
    )
    typer.echo(json.dumps(summary.to_dict(), indent=2))


@etl_app.command("status")
def etl_status(limit: int = typer.Option(10, min=1, max=100)) -> None:
    paths = get_paths()
    typer.echo(json.dumps(list_etl_runs(paths.warehouse_db, limit=limit), indent=2, default=str))


@dataset_app.command("prepare")
def dataset_prepare(
    adapter: str = typer.Option(..., help="Supported: ieee_cis, credit_card_fraud"),
    source: Path = typer.Option(..., "--source", exists=True, readable=True),
    auto_etl: bool = typer.Option(True, help="Run the ETL pipeline after preparing the canonical dataset."),
) -> None:
    paths = get_paths()
    summary = prepare_public_dataset(source=source, adapter=adapter, paths=paths, auto_etl=auto_etl)
    typer.echo(json.dumps(summary.to_dict(), indent=2))


@dataset_app.command("status")
def dataset_status(limit: int = typer.Option(10, min=1, max=100)) -> None:
    paths = get_paths()
    typer.echo(json.dumps(list_prepared_datasets(paths, limit=limit), indent=2))


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
    typer.echo(json.dumps(summary.to_dict(), indent=2))


@fairness_app.command("status")
def fairness_status(limit: int = typer.Option(10, min=1, max=100)) -> None:
    paths = get_paths()
    typer.echo(json.dumps(list_fairness_audits(paths, limit=limit), indent=2, default=str))


@federated_app.command("train")
def federated_train(
    data_path: Path | None = typer.Option(None, "--data-path"),
    client_column: str = typer.Option("channel", "--client-column"),
    rounds: int = typer.Option(5, min=1, max=100),
    local_epochs: int = typer.Option(3, min=1, max=100),
    learning_rate: float = typer.Option(0.1, min=0.0001, max=10.0),
    time_split: bool = typer.Option(False),
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
    )
    typer.echo(json.dumps(summary.to_dict(), indent=2))


@federated_app.command("status")
def federated_status(limit: int = typer.Option(10, min=1, max=100)) -> None:
    paths = get_paths()
    typer.echo(json.dumps(list_federated_runs(paths, limit=limit), indent=2))


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
    typer.echo(json.dumps(backend.status().to_dict(), indent=2))


@storage_app.command("sync")
def storage_sync(
    object_name: str = typer.Option("data/current_transactions.parquet"),
    source: Path | None = typer.Option(None, "--source"),
) -> None:
    paths = get_paths()
    backend = get_storage_backend(paths)
    frame = _read_frame(source or paths.data_path)
    target = backend.save_parquet(frame, object_name)
    typer.echo(json.dumps({"backend": backend.status().backend, "object_name": object_name, "target": target}, indent=2))


@lakehouse_app.command("build")
def lakehouse_build() -> None:
    paths = get_paths()
    db_path = build_default_views(paths)
    typer.echo(json.dumps({"lakehouse_db": str(db_path)}, indent=2))


@lakehouse_app.command("query")
def lakehouse_query(
    sql: str = typer.Option(..., "--sql"),
    limit: int = typer.Option(1000, min=1, max=10000),
) -> None:
    paths = get_paths()
    result = query_lakehouse(paths, sql=sql, limit=limit)
    typer.echo(json.dumps(result.to_dict(), indent=2, default=str))


@spark_app.command("summary")
def spark_summary(
    data_path: Path | None = typer.Option(None, "--data-path"),
) -> None:
    source = data_path or get_paths().data_path
    typer.echo(
        json.dumps(
            {
                "spark_available": spark_available(),
                "summary": summarise_parquet_with_spark(source) if spark_available() else None,
            },
            indent=2,
        )
    )


@pipeline_app.command("run")
def pipeline_run(
    txns: int = typer.Option(10_000, min=100),
    users: int = typer.Option(1_000, min=10),
    merchants: int = typer.Option(200, min=10),
    fraud_rate: float = typer.Option(0.02, min=0.001, max=0.5),
    model: str = typer.Option("graphsage_xgb"),
    sample_tx: Path = typer.Option(Path("demo/sample_transaction.json"), "--sample-tx", exists=True, readable=True),
) -> None:
    paths = get_paths()
    summary = run_end_to_end_pipeline(
        paths=paths,
        txns=txns,
        users=users,
        merchants=merchants,
        fraud_rate=fraud_rate,
        model_type=model,
        sample_tx_path=sample_tx,
    )
    typer.echo(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    app()
