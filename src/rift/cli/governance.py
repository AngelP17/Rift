"""Governance CLI – model card generation (stub for Big Four workflow)."""

import json
from pathlib import Path
from typing import Optional

import typer

from rift.utils.config import MODEL_DIR

governance_app = typer.Typer(help="Governance and model card utilities")


@governance_app.command("generate-card")
def generate_card(
    run_id: str = typer.Argument("latest", help="MLflow run ID or 'latest'"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
) -> None:
    """Generate model card from train_result.json (Big Four style)."""
    result_path = MODEL_DIR / "train_result.json"
    if not result_path.exists():
        typer.echo("No train_result.json found. Run: rift train --model graphsage_xgb")
        raise typer.Exit(1)

    with open(result_path) as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    model_type = data.get("model_type", "unknown")
    feat_cols = data.get("feat_cols", [])

    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()[:12]
    except Exception:
        git_commit = "unknown"

    card = f"""# Model Card: {model_type} (Run {run_id})

**Generated:** Auto-generated from train_result.json

## Intended Use
Fraud detection in financial transactions using graph ML.

## Performance Metrics
| Metric | Value |
|--------|-------|
| PR-AUC | {metrics.get('pr_auc', 'N/A')} |
| Recall @ 1% FPR | {metrics.get('recall_at_1pct_fpr', 'N/A')} |
| ECE (calibrated) | {metrics.get('ece_calibrated', 'N/A')} |
| Brier (calibrated) | {metrics.get('brier_calibrated', 'N/A')} |

## Reproducibility
- Git commit: {git_commit}
- Model type: {model_type}
- Feature columns: {len(feat_cols)} features

## Limitations
- Synthetic data only
- See docs/GOVERNANCE.md for full limitations
"""

    out_path = output or Path(f"model_card_{run_id}.md")
    out_path.write_text(card)
    typer.echo(f"Model card written to {out_path}")
