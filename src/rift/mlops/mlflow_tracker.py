from __future__ import annotations

from pathlib import Path
from typing import Any


def log_run_metrics(
    tracking_dir: Path,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, Any] | None = None,
) -> str | None:
    try:
        import mlflow
    except Exception:
        return None

    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags({key: str(value) for key, value in tags.items()})
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(metrics)
        return mlflow.active_run().info.run_id
