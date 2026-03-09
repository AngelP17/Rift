"""Evaluation metrics for fraud detection models."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from utils.logging import get_logger

log = get_logger(__name__)


def compute_all_metrics(
    y_true: np.ndarray, y_score: np.ndarray, prefix: str = "",
) -> dict:
    """Compute comprehensive fraud detection metrics."""
    metrics = {}
    pf = f"{prefix}_" if prefix else ""

    metrics[f"{pf}pr_auc"] = float(average_precision_score(y_true, y_score))
    metrics[f"{pf}roc_auc"] = float(roc_auc_score(y_true, y_score))
    metrics[f"{pf}brier_score"] = float(brier_score_loss(y_true, y_score))
    metrics[f"{pf}ece"] = float(expected_calibration_error(y_true, y_score))
    metrics[f"{pf}recall_at_1pct_fpr"] = float(recall_at_fpr(y_true, y_score, target_fpr=0.01))

    log.info("metrics_computed", prefix=prefix, **metrics)
    return metrics


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = y_score[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (n_bin / total) * abs(avg_conf - avg_acc)

    return ece


def recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01,
) -> float:
    """Recall at a given false positive rate threshold."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid][-1])


def reliability_curve(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reliability (calibration) curve data."""
    fraction_pos, mean_predicted = calibration_curve(y_true, y_score, n_bins=n_bins)
    return fraction_pos, mean_predicted


def compare_models(results: dict[str, dict]) -> dict:
    """Compare metrics across multiple models. Returns a summary dict."""
    summary = {}
    key_metrics = ["pr_auc", "roc_auc", "brier_score", "ece", "recall_at_1pct_fpr"]

    for metric in key_metrics:
        vals = {}
        for model_name, model_metrics in results.items():
            for k, v in model_metrics.items():
                if k.endswith(metric):
                    vals[model_name] = v
        if vals:
            best = max(vals, key=vals.get) if metric != "brier_score" and metric != "ece" else min(vals, key=vals.get)
            summary[metric] = {"values": vals, "best": best}

    return summary
