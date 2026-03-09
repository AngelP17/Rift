"""Rift CLI entrypoint."""

from pathlib import Path
from typing import Optional

import typer

from rift.data.generator import generate_transactions
from rift.data.splits import SplitStrategy
from rift.models.train import train as train_model
from rift.utils.config import ARTIFACTS_DIR, MODEL_DIR
from rift.utils.io import save_parquet
from rift.utils.logging import setup_logging
from rift.utils.seeds import set_seed

app = typer.Typer(name="rift", help="Rift: Graph ML for Fraud Detection, Replay, and Audit")
logger = setup_logging()


@app.command()
def generate(
    txns: int = typer.Option(100_000, "--txns", "-n", help="Number of transactions"),
    users: int = typer.Option(5_000, "--users", help="Number of users"),
    merchants: int = typer.Option(1_200, "--merchants", help="Number of merchants"),
    fraud_rate: float = typer.Option(0.02, "--fraud-rate", help="Fraud rate"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
):
    """Generate synthetic transaction data."""
    set_seed(seed)
    df = generate_transactions(
        n_txns=txns,
        n_users=users,
        n_merchants=merchants,
        fraud_rate=fraud_rate,
        seed=seed,
    )
    out_path = output or ARTIFACTS_DIR / "transactions.parquet"
    save_parquet(df, out_path)
    logger.info(f"Generated {len(df)} transactions to {out_path}")
    typer.echo(f"Generated {len(df)} transactions → {out_path}")


@app.command()
def train(
    model: str = typer.Option("graphsage_xgb", "--model", "-m", help="Model type"),
    time_split: bool = typer.Option(False, "--time-split", help="Use chronological split"),
    window: Optional[str] = typer.Option(None, "--window", help="Rolling window (e.g. 7d)"),
    data: Optional[Path] = typer.Option(None, "--data", "-d", help="Input data path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output dir"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
):
    """Train fraud detection model."""
    import polars as pl
    from rift.utils.io import load_parquet

    data_path = data or ARTIFACTS_DIR / "transactions.parquet"
    if not data_path.exists():
        typer.echo(f"Data not found at {data_path}. Run: rift generate --txns 10000")
        raise typer.Exit(1)
    df = load_parquet(data_path)
    split = SplitStrategy.CHRONOLOGICAL if time_split else SplitStrategy.RANDOM
    window_days = int(window.replace("d", "")) if window else None
    out_dir = output or MODEL_DIR
    result = train_model(
        df,
        model_type=model,
        split=split,
        time_window_days=window_days,
        output_dir=out_dir,
        seed=seed,
    )
    typer.echo(f"Training complete. Metrics: {result['metrics']}")


@app.command()
def predict_cmd(
    input_file: Path = typer.Argument(..., help="Transaction JSON file"),
):
    """Predict fraud for a single transaction."""
    from rift.utils.io import load_json

    tx = load_json(input_file)
    typer.echo("Prediction requires loaded model. Use API or train first.")


@app.command()
def replay(
    decision_id: str = typer.Argument(..., help="Decision ID to replay"),
):
    """Replay a stored decision."""
    typer.echo(f"Replay {decision_id}: requires model + DB. See AUDIT_GUIDE.md")


@app.command()
def audit(
    decision_id: str = typer.Argument(..., help="Decision ID"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format"),
):
    """Generate audit report for a decision."""
    typer.echo(f"Audit report for {decision_id} (format={format})")


@app.command()
def compare(
    metrics: Optional[str] = typer.Option(None, "--metrics", help="Metrics to compare"),
):
    """Compare models (pr_auc, recall@0.01fpr, ece)."""
    typer.echo("Compare: run experiments and aggregate metrics from train_result.json")


@app.command()
def export(
    since: str = typer.Option("90d", "--since", help="Time range (e.g. 90d)"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format"),
):
    """Export audit reports."""
    from rift.audit.export import export_audit_reports
    from rift.utils.config import AUDIT_DB_PATH

    days = int(since.replace("d", ""))
    out = export_audit_reports(str(AUDIT_DB_PATH), since_days=days, format=format)
    typer.echo(out[:500] + "..." if len(out) > 500 else out)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
