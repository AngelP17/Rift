"""Tests for the Rift operations dashboard: KPI logic, views, and route rendering."""

from __future__ import annotations

import os
from pathlib import Path


from rift.dashboard.kpis import (
    EMPTY_STATES,
    QUICK_ACTIONS,
    KpiCard,
    build_kpi_cards,
)
from rift.utils.config import get_paths


def _paths(tmp_path: Path):
    os.environ["RIFT_HOME"] = str(tmp_path / ".rift")
    os.environ["RIFT_STORAGE_BACKEND"] = "local"
    return get_paths()


class TestKpiThresholds:
    def test_pr_auc_green(self):
        cards = build_kpi_cards(
            {"etl_runs": 0, "fairness_audits": 0, "drift_reports": 0, "federated_runs": 0, "recent_audits": 0},
            {"metrics": {"pr_auc": 0.92}},
        )
        pr_card = [c for c in cards if c.label == "PR-AUC"][0]
        assert pr_card.status == "good"
        assert "good" in pr_card.color_var

    def test_pr_auc_yellow(self):
        cards = build_kpi_cards(
            {"etl_runs": 0, "fairness_audits": 0, "drift_reports": 0, "federated_runs": 0, "recent_audits": 0},
            {"metrics": {"pr_auc": 0.75}},
        )
        pr_card = [c for c in cards if c.label == "PR-AUC"][0]
        assert pr_card.status == "warning"

    def test_pr_auc_red(self):
        cards = build_kpi_cards(
            {"etl_runs": 0, "fairness_audits": 0, "drift_reports": 0, "federated_runs": 0, "recent_audits": 0},
            {"metrics": {"pr_auc": 0.50}},
        )
        pr_card = [c for c in cards if c.label == "PR-AUC"][0]
        assert pr_card.status == "critical"

    def test_ece_lower_is_better(self):
        cards = build_kpi_cards(
            {"etl_runs": 0, "fairness_audits": 0, "drift_reports": 0, "federated_runs": 0, "recent_audits": 0},
            {"metrics": {"ece": 0.03}},
        )
        ece_card = [c for c in cards if c.label == "ECE"][0]
        assert ece_card.status == "good"

    def test_count_cards_always_present(self):
        cards = build_kpi_cards(
            {"etl_runs": 5, "fairness_audits": 2, "drift_reports": 1, "federated_runs": 0, "recent_audits": 10},
            None,
        )
        labels = {c.label for c in cards}
        assert "ETL Runs" in labels
        assert "Recorded Audits" in labels
        assert len(cards) == 5

    def test_no_metrics_no_metric_cards(self):
        cards = build_kpi_cards(
            {"etl_runs": 0, "fairness_audits": 0, "drift_reports": 0, "federated_runs": 0, "recent_audits": 0},
            None,
        )
        assert all(c.label in ("ETL Runs", "Fairness Audits", "Drift Reports", "Federated Runs", "Recorded Audits") for c in cards)


class TestEmptyStates:
    def test_all_empty_states_have_message(self):
        for key, state in EMPTY_STATES.items():
            assert state.message, f"Empty state '{key}' has no message"
            assert state.cli_command, f"Empty state '{key}' has no CLI command"

    def test_quick_actions_are_complete(self):
        assert len(QUICK_ACTIONS) >= 5
        labels = {a.label for a in QUICK_ACTIONS}
        assert "Run Prediction" in labels
        assert "Check Drift" in labels


class TestDashboardRendering:
    def test_build_dashboard_html(self, tmp_path: Path):
        paths = _paths(tmp_path)
        from rift.dashboard.views import build_dashboard_html
        html_out = build_dashboard_html(paths)
        assert "Rift Operations Dashboard" in html_out
        assert "Platform Health Overview" in html_out
        assert "quick-link" in html_out
        assert "No records" in html_out or "empty-state" in html_out

    def test_build_etl_detail(self, tmp_path: Path):
        paths = _paths(tmp_path)
        from rift.dashboard.views import build_etl_detail
        html_out = build_etl_detail(paths)
        assert "ETL Pipeline Runs" in html_out
        assert "Dashboard" in html_out

    def test_build_governance_detail(self, tmp_path: Path):
        paths = _paths(tmp_path)
        from rift.dashboard.views import build_governance_detail
        html_out = build_governance_detail(paths)
        assert "Governance" in html_out

    def test_build_audits_detail(self, tmp_path: Path):
        paths = _paths(tmp_path)
        from rift.dashboard.views import build_audits_detail
        html_out = build_audits_detail(paths)
        assert "Audit Decision Records" in html_out

    def test_build_models_detail(self, tmp_path: Path):
        paths = _paths(tmp_path)
        from rift.dashboard.views import build_models_detail
        html_out = build_models_detail(paths)
        assert "Model Runs" in html_out

    def test_kpi_card_serialization(self):
        card = KpiCard("Test", "0.95", 0.95, "good", "var(--good)", "Test help")
        d = card.to_dict()
        assert d["label"] == "Test"
        assert d["status"] == "good"


class TestApiRoutes:
    def test_dashboard_route(self, tmp_path: Path):
        _paths(tmp_path)
        from fastapi.testclient import TestClient
        from rift.api.server import app
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_dashboard_html_route(self, tmp_path: Path):
        _paths(tmp_path)
        from fastapi.testclient import TestClient
        from rift.api.server import app
        client = TestClient(app)

        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "Rift Operations Dashboard" in response.text

    def test_dashboard_summary_route(self, tmp_path: Path):
        _paths(tmp_path)
        from fastapi.testclient import TestClient
        from rift.api.server import app
        client = TestClient(app)

        response = client.get("/dashboard/summary")
        assert response.status_code == 200
        data = response.json()
        assert "kpis" in data
