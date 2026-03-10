"""Centralized KPI threshold logic for the Rift operations dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KpiCard:
    label: str
    value: str
    raw_value: float | int
    status: str
    color_var: str
    help_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value,
            "raw_value": self.raw_value,
            "status": self.status,
            "color_var": self.color_var,
            "help_text": self.help_text,
        }


@dataclass(frozen=True)
class ActionLink:
    label: str
    href: str
    icon: str = ""


@dataclass(frozen=True)
class EmptyState:
    message: str
    action_label: str = ""
    action_href: str = ""
    cli_command: str = ""


THRESHOLDS = {
    "pr_auc": {"green": 0.85, "yellow": 0.70, "lower_is_better": False},
    "ece": {"green": 0.05, "yellow": 0.10, "lower_is_better": True},
    "brier_score": {"green": 0.12, "yellow": 0.20, "lower_is_better": True},
    "recall_at_1pct_fpr": {"green": 0.60, "yellow": 0.40, "lower_is_better": False},
    "coverage": {"green": 0.93, "yellow": 0.88, "lower_is_better": False},
    "disparity_ratio": {"green": 0.80, "yellow": 0.65, "lower_is_better": False},
}


def _evaluate_threshold(value: float, metric_name: str) -> tuple[str, str]:
    """Return (status, css_color_variable) for a metric value."""
    cfg = THRESHOLDS.get(metric_name)
    if cfg is None:
        return "neutral", "var(--accent)"

    green = cfg["green"]
    yellow = cfg["yellow"]
    lower = cfg["lower_is_better"]

    if lower:
        if value <= green:
            return "good", "var(--good)"
        if value <= yellow:
            return "warning", "var(--warn)"
        return "critical", "var(--danger)"
    else:
        if value >= green:
            return "good", "var(--good)"
        if value >= yellow:
            return "warning", "var(--warn)"
        return "critical", "var(--danger)"


def build_kpi_cards(
    kpis: dict[str, int],
    current_metrics: dict[str, Any] | None,
) -> list[KpiCard]:
    cards = [
        KpiCard("ETL Runs", str(kpis.get("etl_runs", 0)), kpis.get("etl_runs", 0),
                "neutral", "var(--accent)", "Total auditable ETL pipeline executions"),
        KpiCard("Fairness Audits", str(kpis.get("fairness_audits", 0)), kpis.get("fairness_audits", 0),
                "neutral", "var(--accent)", "Demographic parity and disparity ratio checks"),
        KpiCard("Drift Reports", str(kpis.get("drift_reports", 0)), kpis.get("drift_reports", 0),
                "neutral", "var(--accent)", "Data drift monitoring reports"),
        KpiCard("Federated Runs", str(kpis.get("federated_runs", 0)), kpis.get("federated_runs", 0),
                "neutral", "var(--accent)", "Zero-cost federated training scaffolds"),
        KpiCard("Recorded Audits", str(kpis.get("recent_audits", 0)), kpis.get("recent_audits", 0),
                "neutral", "var(--accent)", "SHA-256 hashed decision records in DuckDB"),
    ]

    if current_metrics and "metrics" in current_metrics:
        m = current_metrics["metrics"]
        for metric_key, display_name, help_text in [
            ("pr_auc", "PR-AUC", "Precision-Recall AUC on time-split test set"),
            ("ece", "ECE", "Expected Calibration Error (lower is better)"),
            ("brier_score", "Brier Score", "Mean squared calibration error (lower is better)"),
            ("recall_at_1pct_fpr", "Recall@1%FPR", "Fraud recall at 1% false positive rate"),
        ]:
            val = m.get(metric_key)
            if val is not None:
                status, color = _evaluate_threshold(val, metric_key)
                cards.append(KpiCard(
                    display_name, f"{val:.3f}", val, status, color, help_text,
                ))

    return cards


QUICK_ACTIONS: list[ActionLink] = [
    ActionLink("Run Prediction", "/predict"),
    ActionLink("Latest Model Card", "/governance/model-card/latest"),
    ActionLink("Check Drift", "/monitor/drift-status"),
    ActionLink("Fairness Status", "/fairness/status"),
    ActionLink("Ask Question", "/query?natural=show+recent+flagged+transactions"),
    ActionLink("Lakehouse", "/lakehouse/status"),
    ActionLink("ETL Runs", "/etl/status"),
    ActionLink("Storage", "/storage/status"),
]

EMPTY_STATES: dict[str, EmptyState] = {
    "etl": EmptyState(
        "No ETL runs recorded yet.",
        "Run ETL Pipeline", "/etl/status",
        "rift etl run --source data/raw.csv --source-system treasury",
    ),
    "fairness": EmptyState(
        "No fairness audits recorded yet.",
        "Run Fairness Audit", "/fairness/status",
        "rift fairness audit --sensitive-column channel",
    ),
    "drift": EmptyState(
        "No drift reports recorded yet.",
        "Run Drift Check", "/monitor/drift-status",
        "rift monitor drift --reference-path ref.parquet --current-path cur.parquet",
    ),
    "federated": EmptyState(
        "No federated training runs recorded yet.",
        "Run Federated Training", "/federated/status",
        "rift federated train --client-column channel --rounds 3",
    ),
    "audits": EmptyState(
        "No audit decisions recorded yet.",
        "Run a Prediction", "/predict",
        "rift predict --tx demo/sample_transaction.json",
    ),
    "datasets": EmptyState(
        "No public datasets prepared yet.",
        "Prepare Dataset", "/datasets/status",
        "rift dataset prepare --adapter ieee_cis --source demo/ieee_cis_sample.csv",
    ),
    "models": EmptyState(
        "No model runs recorded yet.",
        "Train a Model", "",
        "rift train --model graphsage_xgb --time-split",
    ),
}
