"""Dashboard helpers for Rift operational visibility."""

from rift.dashboard.views import (
    build_audits_detail,
    build_dashboard_html,
    build_detail_html,
    build_etl_detail,
    build_governance_detail,
    build_models_detail,
    dashboard_snapshot,
    get_static_dir,
)

__all__ = [
    "build_audits_detail",
    "build_dashboard_html",
    "build_detail_html",
    "build_etl_detail",
    "build_governance_detail",
    "build_models_detail",
    "dashboard_snapshot",
    "get_static_dir",
]
