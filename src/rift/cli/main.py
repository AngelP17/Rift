from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl
import typer

from rift.data.generator import generate_transactions
from rift.explain.report import build_audit_report, build_explanation, report_to_markdown
from rift.models.infer import load_run, payload_to_frame, score_frame
from rift.models.train import train_from_frame
from rift.replay.hashing import decision_hash
from rift.replay.recorder import record_decision
from rift.replay.replayer import replay_decision
from rift.utils.config import get_paths
from rift.utils.io import read_json


app = typer.Typer(help="Rift: graph ML for fraud detection, replay, and audit.")


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
        select decision_id, markdown, report_json
        from audit_reports
        order by decision_id
        """
    ).fetchall()
    conn.close()
    if format == "json":
        typer.echo(json.dumps([json.loads(row[2]) for row in rows], indent=2))
        return
    typer.echo("\n\n---\n\n".join(row[1] for row in rows))


if __name__ == "__main__":
    app()
