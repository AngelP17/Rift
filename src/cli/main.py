"""Rift CLI: command-line interface for the fraud detection system."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(
    name="rift",
    help="Rift: Graph ML for Fraud Detection, Replay, and Audit",
    add_completion=False,
)


@app.command()
def generate(
    txns: int = typer.Option(100_000, help="Number of transactions to generate"),
    users: int = typer.Option(5_000, help="Number of users"),
    merchants: int = typer.Option(1_200, help="Number of merchants"),
    fraud_rate: float = typer.Option(0.02, help="Fraud rate (0 to 1)"),
    seed: int = typer.Option(42, help="Random seed"),
    output: str = typer.Option("transactions", help="Output file name (without extension)"),
):
    """Generate synthetic transaction data."""
    from data.generator import generate_transactions
    from utils.io import save_parquet

    console.print(f"[bold blue]Generating {txns:,} transactions...[/]")
    df = generate_transactions(
        n_txns=txns, n_users=users, n_merchants=merchants,
        fraud_rate=fraud_rate, seed=seed,
    )
    path = save_parquet(df, output)
    console.print(f"[green]Saved {len(df):,} transactions to {path}[/]")
    console.print(f"  Fraud rate: {df['is_fraud'].mean():.4f}")


@app.command()
def train(
    model: str = typer.Option("graphsage_xgb", help="Model type: xgb_tabular, graphsage_only, graphsage_xgb, gat_xgb"),
    time_split: bool = typer.Option(False, "--time-split", help="Use temporal split"),
    window: str = typer.Option("7d", help="Window for rolling evaluation"),
    calibration: str = typer.Option("isotonic", help="Calibration method: isotonic, platt"),
    seed: int = typer.Option(42, help="Random seed"),
    epochs: int = typer.Option(50, help="Training epochs for GNN"),
    booster: str = typer.Option("xgboost", help="Booster type: xgboost, lightgbm"),
    data_file: str = typer.Option("transactions", help="Input data file name"),
    mlflow_backend: str = typer.Option("sqlite", help="MLflow backend: sqlite, file, none"),
    tracker: str = typer.Option("mlflow", help="Experiment tracker: mlflow, clearml, none"),
):
    """Train a fraud detection model."""
    from models.train import train_pipeline
    from utils.io import load_parquet

    if mlflow_backend != "none" and tracker == "mlflow":
        from monitoring.mlflow_setup import init_mlflow
        init_mlflow(backend=mlflow_backend)

    strategy = "temporal" if time_split else "random"
    window_days = int(window.replace("d", ""))

    console.print(f"[bold blue]Training {model} model...[/]")
    df = load_parquet(data_file)

    result = train_pipeline(
        df, model_type=model, split_strategy=strategy,
        window_days=window_days, calibration_method=calibration,
        seed=seed, epochs=epochs, booster=booster,
    )

    all_metrics = {**result.get("raw_metrics", {}), **result.get("calibrated_metrics", {})}

    if tracker == "mlflow":
        from monitoring.mlflow_setup import log_training_run
        run_id = log_training_run(
            model_type=model,
            params={"split": strategy, "window": window, "calibration": calibration,
                     "seed": seed, "epochs": epochs, "booster": booster},
            metrics=all_metrics,
            artifacts={"model": result.get("model_path", "")},
        )
        if run_id:
            console.print(f"[dim]MLflow run: {run_id}[/]")
    elif tracker == "clearml":
        from monitoring.clearml_tracker import log_training_to_clearml
        log_training_to_clearml(
            model_type=model,
            params={"split": strategy, "window": window, "calibration": calibration,
                     "seed": seed, "epochs": epochs, "booster": booster},
            metrics=all_metrics,
        )

    console.print("[green]Training complete![/]")
    _print_metrics(result.get("raw_metrics", {}), "Raw Metrics")
    _print_metrics(result.get("calibrated_metrics", {}), "Calibrated Metrics")


@app.command()
def predict(
    tx: str = typer.Option(..., help="Path to transaction JSON file"),
    model_type: str = typer.Option("graphsage_xgb", help="Model type to use"),
):
    """Run prediction on a transaction."""
    from models.infer import predict_single

    tx_data = json.loads(Path(tx).read_text())
    result = predict_single(tx_data, model_type=model_type)

    console.print(f"\n[bold]Decision ID:[/] {result['decision_id']}")
    console.print(f"[bold]Raw Score:[/] {result['raw_score']:.4f}")
    console.print(f"[bold]Calibrated Score:[/] {result['calibrated_score']:.4f}")
    console.print(f"[bold]Confidence Band:[/] {result['confidence_band']}")


@app.command()
def replay(
    decision_id: str = typer.Argument(help="Decision ID to replay"),
):
    """Replay a past decision for verification."""
    from replay.recorder import DecisionRecorder
    from replay.replayer import ReplayEngine

    engine = ReplayEngine(DecisionRecorder())
    result = engine.replay(decision_id)

    if "error" in result:
        console.print(f"[red]{result['error']}[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Replay ID:[/] {result['replay_id']}")
    console.print(f"[bold]Decision ID:[/] {result['decision_id']}")
    match_color = "green" if result["matched"] else "red"
    console.print(f"[bold]Match:[/] [{match_color}]{result['matched']}[/]")

    pred = result["stored_prediction"]
    console.print(f"[bold]Raw Score:[/] {pred['raw_score']:.4f}")
    console.print(f"[bold]Calibrated Score:[/] {pred['calibrated_score']:.4f}")
    console.print(f"[bold]Confidence Band:[/] {pred['confidence_band']}")


@app.command()
def audit(
    decision_id: str = typer.Argument(help="Decision ID to audit"),
    format: str = typer.Option("markdown", help="Output format: markdown, json"),
):
    """Generate an audit report for a decision."""
    from explain.report import generate_report, report_to_json, report_to_markdown
    from replay.recorder import DecisionRecorder

    recorder = DecisionRecorder()
    pred = recorder.get_prediction(decision_id)

    if pred is None:
        console.print(f"[red]Decision {decision_id} not found[/]")
        raise typer.Exit(1)

    report = generate_report(pred)
    if format == "markdown":
        md = report_to_markdown(report)
        console.print(md)
        recorder.record_audit_report(decision_id, md, report)
    else:
        j = report_to_json(report)
        console.print(j)
        recorder.record_audit_report(decision_id, "", report)


@app.command()
def compare(
    metrics: str = typer.Option("pr_auc,recall_at_1pct_fpr,ece", help="Comma-separated metrics to compare"),
    data_file: str = typer.Option("transactions", help="Input data file name"),
):
    """Compare metrics across model types."""
    console.print("[bold blue]Model comparison[/]")
    console.print(f"Metrics: {metrics}")
    console.print("[yellow]Train models first, then compare their stored metrics.[/]")


@app.command()
def export(
    since: str = typer.Option("90d", help="Export decisions since (e.g., 90d)"),
    format: str = typer.Option("markdown", help="Output format: markdown, json"),
):
    """Export decisions for audit review."""
    from audit.export import export_decisions

    days = int(since.replace("d", ""))
    path = export_decisions(since_days=days, format=format)
    console.print(f"[green]Exported to {path}[/]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
):
    """Start the FastAPI server."""
    import uvicorn

    console.print(f"[bold blue]Starting Rift API on {host}:{port}[/]")
    uvicorn.run("api.server:app", host=host, port=port, reload=False)


@app.command()
def validate(
    suite: str = typer.Option("deepchecks", help="Validation suite: deepchecks"),
    ref: str = typer.Option("data/transactions.parquet", help="Reference dataset path"),
    cur: str = typer.Option("data/transactions.parquet", help="Current dataset path"),
    output: str = typer.Option("", help="Output report path (auto-generated if empty)"),
):
    """Run data and model validation checks."""
    from validate.deepchecks_suite import run_data_validation

    console.print("[bold blue]Running validation suite...[/]")
    out = output if output else None
    result = run_data_validation(ref, cur, out)

    if "error" in result:
        console.print(f"[yellow]{result['error']}[/]")
        if "install" in result:
            console.print(f"[dim]Install: {result['install']}[/]")
    else:
        console.print(f"[green]Integrity: {'PASS' if result['integrity_passed'] else 'FAIL'}[/]")
        console.print(f"[green]Validation: {'PASS' if result['validation_passed'] else 'FAIL'}[/]")
        console.print(f"[dim]Report: {result['report_path']}[/]")


@app.command()
def monitor(
    ui: str = typer.Option("evidently", help="Dashboard: evidently, streamlit"),
    ref: str = typer.Option("data/transactions.parquet", help="Reference dataset"),
    cur: str = typer.Option("data/transactions.parquet", help="Current dataset"),
):
    """Generate drift monitoring reports or launch dashboard."""
    if ui == "streamlit":
        from monitoring.evidently_dashboard import launch_streamlit_dashboard
        cmd = launch_streamlit_dashboard()
        console.print(f"[bold blue]Launch dashboard with:[/]\n  {cmd}")
    else:
        from monitoring.evidently_dashboard import generate_drift_report
        console.print("[bold blue]Generating drift report...[/]")
        result = generate_drift_report(ref, cur)
        if "error" in result:
            console.print(f"[yellow]{result['error']}[/]")
        else:
            drift = result.get("dataset_drift_detected", False)
            color = "red" if drift else "green"
            console.print(f"[{color}]Drift detected: {drift}[/]")
            console.print(f"[dim]Report: {result['report_path']}[/]")


@app.command()
def query(
    natural: str = typer.Option(..., help="Natural language query about audit data"),
    chat: bool = typer.Option(False, help="Enable multi-turn chat mode"),
):
    """Ask natural language questions about audit decisions (powered by Ollama)."""
    from explain.ollama_chat import AuditChatAssistant

    assistant = AuditChatAssistant()

    if chat:
        console.print("[bold blue]Rift Audit Chat (type 'exit' to quit)[/]\n")
        while True:
            user_input = console.input("[bold cyan]You:[/] ")
            if user_input.lower() in ("exit", "quit", "q"):
                break
            response = assistant.ask(user_input)
            console.print(f"\n[bold green]Rift:[/] {response}\n")
    else:
        response = assistant.ask(natural)
        console.print(response)


@app.command(name="search-audits")
def search_audits(
    query_text: str = typer.Option(..., "--query", help="Semantic search query"),
    k: int = typer.Option(5, help="Number of results"),
):
    """Search audit records by semantic similarity."""
    from search.vector_search import AuditVectorSearch

    console.print(f"[bold blue]Searching audits: '{query_text}'...[/]")
    searcher = AuditVectorSearch()
    results = searcher.search(query_text, k=k)

    if not results:
        console.print("[yellow]No results found.[/]")
        return

    if "error" in results[0]:
        console.print(f"[yellow]{results[0]['error']}[/]")
        return

    table = Table(title=f"Top {k} Similar Audits")
    table.add_column("Rank", style="dim")
    table.add_column("Decision ID", style="cyan")
    table.add_column("Band", style="yellow")
    table.add_column("Score", style="green")
    table.add_column("Distance", style="dim")

    for r in results:
        table.add_row(
            str(r.get("rank", "")),
            r.get("decision_id", ""),
            r.get("confidence_band", ""),
            f"{r.get('calibrated_score', 0):.4f}" if r.get("calibrated_score") else "N/A",
            f"{r.get('distance', 0):.4f}",
        )
    console.print(table)


def _print_metrics(metrics: dict, title: str) -> None:
    if not metrics:
        return
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in sorted(metrics.items()):
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


if __name__ == "__main__":
    app()
