"""Prometheus metrics instrumentation for Rift platform observability.

Exposes counters, histograms, and gauges for API, ETL, training,
fairness, drift, audit, and NL query subsystems.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

RIFT_INFO = Info("rift", "Rift platform metadata")
RIFT_INFO.info({"version": "1.0.0", "component": "fraud-detection-platform"})

# ── API metrics ───────────────────────────────────────────────────
API_REQUESTS = Counter(
    "rift_api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
API_LATENCY = Histogram(
    "rift_api_request_duration_seconds", "API request latency", ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ── ETL metrics ───────────────────────────────────────────────────
ETL_RUNS = Counter("rift_etl_runs_total", "Total ETL pipeline executions", ["source_system"])
ETL_FAILURES = Counter("rift_etl_failures_total", "Total ETL pipeline failures", ["source_system"])
ETL_ROWS_EXTRACTED = Counter("rift_etl_rows_extracted_total", "Rows extracted by ETL")
ETL_ROWS_LOADED = Counter("rift_etl_rows_loaded_total", "Rows loaded into warehouse")
ETL_DURATION = Histogram(
    "rift_etl_duration_seconds", "ETL pipeline duration",
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
)

# ── Training metrics ──────────────────────────────────────────────
TRAINING_RUNS = Counter("rift_training_runs_total", "Total model training runs", ["model_type"])
TRAINING_DURATION = Histogram(
    "rift_training_duration_seconds", "Training duration", ["model_type"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)
LATEST_PR_AUC = Gauge("rift_latest_pr_auc", "Latest PR-AUC from most recent training run")
LATEST_ECE = Gauge("rift_latest_ece", "Latest ECE from most recent training run")
LATEST_BRIER = Gauge("rift_latest_brier_score", "Latest Brier score from most recent training run")

# ── Fairness metrics ──────────────────────────────────────────────
FAIRNESS_AUDITS = Counter("rift_fairness_audits_total", "Total fairness audits executed")
FAIRNESS_DPD = Gauge(
    "rift_fairness_demographic_parity_difference",
    "Latest demographic parity difference",
)
FAIRNESS_DIR = Gauge(
    "rift_fairness_disparate_impact_ratio",
    "Latest disparate impact ratio",
)

# ── Drift metrics ─────────────────────────────────────────────────
DRIFT_CHECKS = Counter("rift_drift_checks_total", "Total drift monitoring checks")
DRIFT_DETECTED = Counter("rift_drift_detected_total", "Drift detections triggered")
LATEST_DRIFT_SCORE = Gauge("rift_latest_drift_score", "Latest drift score")

# ── Audit / replay metrics ────────────────────────────────────────
PREDICTIONS_RECORDED = Counter("rift_predictions_recorded_total", "Total predictions recorded to DuckDB")
REPLAY_REQUESTS = Counter("rift_replay_requests_total", "Total replay requests", ["matched"])

# ── NL query metrics ──────────────────────────────────────────────
NL_QUERIES = Counter("rift_nl_queries_total", "Total natural-language queries")
NL_QUERY_LATENCY = Histogram(
    "rift_nl_query_duration_seconds", "NL query latency",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30),
)
NL_QUERY_LLM_USED = Counter("rift_nl_query_llm_used_total", "NL queries served by LLM")
NL_QUERY_FALLBACK_USED = Counter("rift_nl_query_fallback_used_total", "NL queries using SQL fallback")
