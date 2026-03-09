"""Deepchecks continuous validation suite for data integrity, model quality, and bias."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)


def run_data_validation(
    reference_path: str | Path,
    current_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run Deepchecks data validation suite comparing reference and current datasets.

    Returns a summary dict and saves an HTML report.
    """
    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import data_integrity, train_test_validation

        ref_df = pd.read_parquet(reference_path)
        cur_df = pd.read_parquet(current_path)

        label_col = "is_fraud" if "is_fraud" in ref_df.columns else None

        cat_features = [c for c in ["channel", "mcc", "currency"] if c in ref_df.columns]
        num_features = [
            c for c in ref_df.columns
            if c not in cat_features
            and c not in ["tx_id", "user_id", "merchant_id", "device_id", "account_id", "timestamp"]
            and c != label_col
            and ref_df[c].dtype in ["float64", "float32", "int64", "int32"]
        ]

        ref_ds = Dataset(ref_df, label=label_col, cat_features=cat_features, features=num_features + cat_features)
        cur_ds = Dataset(cur_df, label=label_col, cat_features=cat_features, features=num_features + cat_features)

        integrity_result = data_integrity().run(cur_ds)
        validation_result = train_test_validation().run(ref_ds, cur_ds)

        output_path = output_path or (cfg.data_dir / "reports" / "deepchecks_validation.html")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        validation_result.save_as_html(str(output_path))

        summary = {
            "integrity_passed": integrity_result.passed(),
            "validation_passed": validation_result.passed(),
            "integrity_results": len(integrity_result.get_not_passed_checks()),
            "validation_results": len(validation_result.get_not_passed_checks()),
            "report_path": str(output_path),
        }

        log.info("deepchecks_validation_complete", **summary)
        return summary

    except ImportError:
        log.warning("deepchecks_not_installed", msg="pip install deepchecks")
        return {"error": "deepchecks not installed", "install": "pip install deepchecks"}


def run_model_validation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str] | None = None,
    X_test: np.ndarray | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run model performance validation checks."""
    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import model_evaluation

        if X_test is not None:
            df = pd.DataFrame(X_test, columns=feature_names or [f"f{i}" for i in range(X_test.shape[1])])
            df["label"] = y_true
            ds = Dataset(df, label="label")

            class _MockModel:
                def predict(self, X):
                    return (y_pred > 0.5).astype(int)[:len(X)]

                def predict_proba(self, X):
                    n = len(X)
                    return np.column_stack([1 - y_pred[:n], y_pred[:n]])

            result = model_evaluation().run(ds, _MockModel())

            output_path = output_path or (cfg.data_dir / "reports" / "deepchecks_model.html")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result.save_as_html(str(output_path))

            return {
                "passed": result.passed(),
                "failed_checks": len(result.get_not_passed_checks()),
                "report_path": str(output_path),
            }

        return _basic_model_checks(y_true, y_pred)

    except ImportError:
        log.warning("deepchecks_not_installed")
        return _basic_model_checks(y_true, y_pred)


def _basic_model_checks(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Fallback model validation without Deepchecks."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    checks = {}
    pr_auc = average_precision_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)

    checks["pr_auc"] = pr_auc
    checks["roc_auc"] = roc
    checks["brier_score"] = brier
    checks["pr_auc_pass"] = pr_auc > 0.5
    checks["roc_auc_pass"] = roc > 0.6
    checks["brier_pass"] = brier < 0.25
    checks["all_passed"] = all(v for k, v in checks.items() if k.endswith("_pass"))

    return checks
