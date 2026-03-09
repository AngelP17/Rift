"""ClearML experiment tracking integration as MLflow alternative/complement."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logging import get_logger

log = get_logger(__name__)


class ClearMLTracker:
    """Experiment tracker using ClearML for rich UI and sweep support.

    ClearML provides a self-hosted web UI at localhost:8080 with:
    - Experiment comparison and visualization
    - Hyperparameter sweep orchestration
    - Model versioning and serving
    - Dataset versioning
    """

    def __init__(self, project_name: str = "Rift-Fraud", auto_connect: bool = True):
        self.project_name = project_name
        self.auto_connect = auto_connect
        self.task = None
        self._available: bool | None = None

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import clearml  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False
            log.warning("clearml_not_installed", msg="pip install clearml")
        return self._available

    def start_run(self, task_name: str, task_type: str = "training") -> bool:
        """Start a new ClearML task/run."""
        if not self._check_available():
            return False

        try:
            from clearml import Task

            type_map = {
                "training": Task.TaskTypes.training,
                "testing": Task.TaskTypes.testing,
                "inference": Task.TaskTypes.inference,
                "data_processing": Task.TaskTypes.data_processing,
            }

            self.task = Task.init(
                project_name=self.project_name,
                task_name=task_name,
                task_type=type_map.get(task_type, Task.TaskTypes.training),
                auto_connect_frameworks=self.auto_connect,
            )
            log.info("clearml_task_started", task_name=task_name)
            return True

        except Exception as e:
            log.warning("clearml_init_failed", error=str(e))
            return False

    def log_params(self, params: dict[str, Any]) -> None:
        if self.task:
            self.task.connect_configuration(params, name="hyperparameters")

    def log_metrics(self, metrics: dict[str, float], series: str = "evaluation") -> None:
        if not self.task:
            return
        logger = self.task.get_logger()
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.report_scalar(title=series, series=name, value=value, iteration=0)

    def log_artifact(self, name: str, path: str | Path) -> None:
        if self.task and Path(path).exists():
            self.task.upload_artifact(name, artifact_object=str(path))

    def finish(self) -> None:
        if self.task:
            self.task.close()
            self.task = None


def log_training_to_clearml(
    model_type: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, str] | None = None,
) -> bool:
    """Convenience function to log a full training run to ClearML."""
    tracker = ClearMLTracker()
    task_name = f"Train-{model_type}-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"

    if not tracker.start_run(task_name):
        return False

    tracker.log_params({"model_type": model_type, **params})
    tracker.log_metrics(metrics)

    if artifacts:
        for name, path in artifacts.items():
            tracker.log_artifact(name, path)

    tracker.finish()
    return True
