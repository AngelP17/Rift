"""MLflow integration with SQLite backend for persistent experiment tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)

_TRACKING_URI: str | None = None


def init_mlflow(backend: str = "sqlite") -> None:
    """Initialize MLflow with the specified backend.

    Args:
        backend: 'sqlite' for persistent SQLite DB, 'file' for file-based store.
    """
    global _TRACKING_URI
    try:
        import mlflow

        if backend == "sqlite":
            db_path = cfg.data_dir / "mlflow.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _TRACKING_URI = f"sqlite:///{db_path}"
        else:
            _TRACKING_URI = str(cfg.data_dir / "mlruns")

        mlflow.set_tracking_uri(_TRACKING_URI)
        mlflow.set_experiment("rift-fraud-experiments")
        log.info("mlflow_initialized", backend=backend, uri=_TRACKING_URI)
    except ImportError:
        log.warning("mlflow_not_installed", msg="pip install mlflow to enable tracking")


def log_training_run(
    model_type: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Log a complete training run to MLflow.

    Returns the MLflow run_id or None if MLflow is unavailable.
    """
    try:
        import mlflow

        if _TRACKING_URI:
            mlflow.set_tracking_uri(_TRACKING_URI)

        with mlflow.start_run(run_name=f"{model_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}") as run:
            mlflow.set_tag("model_type", model_type)
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, v)

            mlflow.log_params(params)

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

            if artifacts:
                for name, path in artifacts.items():
                    if Path(path).exists():
                        mlflow.log_artifact(path, artifact_path=name)

            log.info("mlflow_run_logged", run_id=run.info.run_id, model=model_type)
            return run.info.run_id

    except ImportError:
        log.warning("mlflow_not_available")
        return None
    except Exception as e:
        log.warning("mlflow_log_failed", error=str(e))
        return None


def register_model(run_id: str, model_name: str, stage: str = "Staging") -> None:
    """Register a model version in the MLflow model registry."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        if _TRACKING_URI:
            mlflow.set_tracking_uri(_TRACKING_URI)

        client = MlflowClient()

        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        result = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )
        log.info("model_registered", name=model_name, version=result.version, stage=stage)

    except ImportError:
        log.warning("mlflow_not_available")
    except Exception as e:
        log.warning("mlflow_register_failed", error=str(e))


def get_experiment_summary() -> list[dict]:
    """Retrieve a summary of all experiment runs."""
    try:
        import mlflow

        if _TRACKING_URI:
            mlflow.set_tracking_uri(_TRACKING_URI)

        runs = mlflow.search_runs(experiment_names=["rift-fraud-experiments"], output_format="list")
        summary = []
        for run in runs:
            summary.append({
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            })
        return summary
    except ImportError:
        return []
    except Exception:
        return []
