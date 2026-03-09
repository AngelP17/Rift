from __future__ import annotations

import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from rift.datasets.adapters import list_prepared_datasets
from rift.etl.pipeline import list_etl_runs
from rift.federated.simulation import list_federated_runs
from rift.governance.fairness import list_fairness_audits
from rift.monitoring.drift import list_drift_reports
from rift.storage.backends import get_storage_backend
from rift.utils.config import RiftPaths
from rift.utils.io import read_json

_VERSION = "1.0.0"


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
    except Exception:
        return "unknown"


def _collect_run_history(runs_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Collect PR-AUC history from past training runs for the sparkline chart."""
    history: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return history
    for metrics_file in sorted(runs_dir.glob("run_*/metrics.json"))[-limit:]:
        try:
            data = read_json(metrics_file)
            run_id = metrics_file.parent.name
            pr_auc = data.get("metrics", {}).get("pr_auc")
            if pr_auc is not None:
                history.append({"run_id": run_id, "pr_auc": round(float(pr_auc), 4)})
        except Exception:
            continue
    return history


def _recent_audits(paths: RiftPaths, limit: int = 10) -> list[dict[str, Any]]:
    if not paths.audit_db.exists():
        return []
    conn = duckdb.connect(str(paths.audit_db), read_only=True)
    exists = conn.execute(
        "select count(*) from information_schema.tables where table_name = 'predictions'"
    ).fetchone()[0]
    if not exists:
        conn.close()
        return []
    rows = conn.execute(
        """
        select p.decision_id, p.model_run_id, p.prediction_json, ar.markdown
        from predictions p
        left join audit_reports ar on ar.decision_id = p.decision_id
        order by p.decision_id desc
        limit ?
        """,
        [limit],
    ).fetchall()
    conn.close()
    output: list[dict[str, Any]] = []
    for decision_id, model_run_id, prediction_json, markdown in rows:
        prediction = json.loads(prediction_json)
        output.append(
            {
                "decision_id": decision_id,
                "model_run_id": model_run_id,
                "decision": prediction.get("decision"),
                "calibrated_probability": prediction.get("calibrated_probability"),
                "confidence": prediction.get("confidence"),
                "markdown": markdown,
            }
        )
    return output


def dashboard_snapshot(paths: RiftPaths) -> dict[str, Any]:
    current_run = _safe_read_json(paths.runs_dir / "current_run.json")
    current_metrics = None
    if current_run:
        current_metrics = _safe_read_json(paths.runs_dir / current_run["run_id"] / "metrics.json")
    etl_runs = list_etl_runs(paths.warehouse_db, limit=10)
    fairness_runs = list_fairness_audits(paths, limit=10)
    drift_reports = list_drift_reports(paths, limit=10)
    federated_runs = list_federated_runs(paths, limit=10)
    prepared_datasets = list_prepared_datasets(paths, limit=10)
    recent_audits = _recent_audits(paths, limit=10)
    storage_status = get_storage_backend(paths).status().to_dict()
    run_history = _collect_run_history(paths.runs_dir)
    return {
        "version": _VERSION,
        "git_commit": _git_commit(),
        "refreshed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current_model": current_run,
        "current_metrics": current_metrics,
        "etl_runs": etl_runs,
        "fairness_audits": fairness_runs,
        "drift_reports": drift_reports,
        "federated_runs": federated_runs,
        "prepared_datasets": prepared_datasets,
        "recent_audits": recent_audits,
        "storage_status": storage_status,
        "run_history": run_history,
        "kpis": {
            "etl_runs": len(etl_runs),
            "fairness_audits": len(fairness_runs),
            "drift_reports": len(drift_reports),
            "federated_runs": len(federated_runs),
            "recent_audits": len(recent_audits),
        },
    }


# ── KPI card rendering ──────────────────────────────────────────────


def _kpi_status(value: float, thresholds: tuple[float, float], invert: bool = False) -> str:
    """Return a CSS class for green/amber/red based on value and thresholds."""
    good, warn = thresholds
    if invert:
        if value <= good:
            return "status-good"
        if value <= warn:
            return "status-warn"
        return "status-bad"
    if value >= good:
        return "status-good"
    if value >= warn:
        return "status-warn"
    return "status-bad"


def _render_card(label: str, value: str, extra_class: str = "") -> str:
    cls = f"card {extra_class}".strip()
    return (
        f"<div class='{cls}'>"
        f"<div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div>"
        f"</div>"
    )


def _render_cards(kpis: dict[str, Any], current_metrics: dict[str, Any] | None) -> str:
    cards = [
        _render_card("ETL Runs", str(kpis["etl_runs"])),
        _render_card("Fairness Audits", str(kpis["fairness_audits"])),
        _render_card("Drift Reports", str(kpis["drift_reports"])),
        _render_card("Federated Runs", str(kpis["federated_runs"])),
        _render_card("Recorded Audits", str(kpis["recent_audits"])),
    ]
    if current_metrics:
        pr_auc = current_metrics["metrics"].get("pr_auc", 0.0)
        ece = current_metrics["metrics"].get("ece", 0.0)
        cards.append(_render_card(
            "Current PR-AUC",
            f"{pr_auc:.3f}",
            _kpi_status(pr_auc, (0.85, 0.7)),
        ))
        cards.append(_render_card(
            "Current ECE",
            f"{ece:.3f}",
            _kpi_status(ece, (0.02, 0.05), invert=True),
        ))
    return "".join(cards)


# ── Table rendering ──────────────────────────────────────────────────


def _render_table(title: str, columns: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"<section class='panel'><h2>{html.escape(title)}</h2><p class='empty'>No records yet.</p></section>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
            + "</tr>"
        )
    return (
        f"<section class='panel'><h2>{html.escape(title)}</h2>"
        f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
        "</section>"
    )


# ── Quick Actions ────────────────────────────────────────────────────


def _render_quick_actions(current_run: dict[str, Any] | None) -> str:
    run_id = current_run["run_id"] if current_run else ""
    return f"""
    <section class="actions" id="quick-actions">
      <h2>Quick Actions</h2>
      <div class="actions-row">
        <a class="btn btn-green" href="/docs#/default/predict_predict_post" target="_blank">
          <span class="btn-icon">▶</span> Run Prediction
        </a>
        <a class="btn btn-purple" href="/governance/model-card/{html.escape(run_id)}" target="_blank"
           {'style="pointer-events:none;opacity:0.4"' if not run_id else ''}>
          <span class="btn-icon">📋</span> Generate Model Card
        </a>
        <a class="btn btn-amber" href="/monitor/drift-status" target="_blank">
          <span class="btn-icon">📊</span> Check Drift
        </a>
        <a class="btn btn-blue" href="/query?natural=show+recent+flagged+transactions" target="_blank">
          <span class="btn-icon">💬</span> Ask NL Question
        </a>
      </div>
    </section>
    """


# ── Sparkline chart ──────────────────────────────────────────────────


def _render_sparkline(run_history: list[dict[str, Any]]) -> str:
    if len(run_history) < 1:
        return ""
    labels = json.dumps([h["run_id"].replace("run_", "") for h in run_history])
    values = json.dumps([h["pr_auc"] for h in run_history])
    return f"""
    <section class="panel sparkline-panel" id="pr-auc-trend">
      <h2>PR-AUC Trend</h2>
      <canvas id="prAucChart" height="80"></canvas>
      <script>
        document.addEventListener('DOMContentLoaded', function() {{
          const ctx = document.getElementById('prAucChart').getContext('2d');
          new Chart(ctx, {{
            type: 'line',
            data: {{
              labels: {labels},
              datasets: [{{
                label: 'PR-AUC',
                data: {values},
                borderColor: '#6ea8fe',
                backgroundColor: 'rgba(110,168,254,0.10)',
                borderWidth: 2,
                pointBackgroundColor: '#6ea8fe',
                pointBorderColor: '#1a2340',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.3
              }}]
            }},
            options: {{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {{
                legend: {{ display: false }},
                tooltip: {{
                  backgroundColor: 'rgba(18,26,49,0.95)',
                  borderColor: '#6ea8fe',
                  borderWidth: 1,
                  titleColor: '#f3f6ff',
                  bodyColor: '#aab5d1',
                  padding: 10,
                  cornerRadius: 8,
                }}
              }},
              scales: {{
                x: {{
                  ticks: {{ color: '#aab5d1', font: {{ size: 11 }} }},
                  grid: {{ color: 'rgba(255,255,255,0.04)' }}
                }},
                y: {{
                  min: 0, max: 1,
                  ticks: {{ color: '#aab5d1', font: {{ size: 11 }} }},
                  grid: {{ color: 'rgba(255,255,255,0.04)' }}
                }}
              }}
            }}
          }});
        }});
      </script>
    </section>
    """


# ── Full page build ──────────────────────────────────────────────────


def build_dashboard_html(paths: RiftPaths) -> str:
    snapshot = dashboard_snapshot(paths)
    current_run = snapshot["current_model"]
    current_metrics = snapshot["current_metrics"]
    prepared_dataset_rows = [item.get("summary", {}) for item in snapshot["prepared_datasets"]]
    storage_status = snapshot["storage_status"]
    run_history = snapshot["run_history"]
    version = snapshot["version"]
    git_commit = snapshot["git_commit"]
    refreshed_at = snapshot["refreshed_at"]

    hero_run = "No active model run"
    run_id_display = "—"
    if current_run:
        hero_run = current_run["run_id"]
        run_id_display = current_run["run_id"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rift Operations Dashboard</title>
  <meta name="description" content="Rift — open-source, zero-cost, local-first auditable fraud ML platform operations dashboard." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121a31;
      --panel-2: #17223f;
      --text: #f3f6ff;
      --muted: #aab5d1;
      --accent: #6ea8fe;
      --border: #2a365e;
      --good: #2fbf71;
      --warn: #f0ad4e;
      --bad: #e5534b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #08101d, #10192e 35%, #0a1221);
      color: var(--text);
      min-height: 100vh;
    }}

    /* ── Fade-in animation ── */
    @keyframes fadeInUp {{
      from {{ opacity: 0; transform: translateY(12px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 0 24px 48px;
      animation: fadeInUp 0.5s ease-out;
    }}

    /* ── Top branding bar ── */
    .topbar {{
      background: rgba(13,17,23,0.95);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 100;
    }}
    .topbar-left {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .topbar-logo {{
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.01em;
    }}
    .topbar-logo span {{
      background: linear-gradient(135deg, var(--accent), var(--good));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 20px;
      padding: 3px 10px;
      white-space: nowrap;
    }}
    .pill code {{
      color: var(--accent);
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 11px;
    }}
    .topbar-right {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .export-btn {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 8px;
      padding: 6px 12px;
      text-decoration: none;
      transition: all 0.2s;
      cursor: pointer;
    }}
    .export-btn:hover {{
      background: rgba(255,255,255,0.12);
      color: var(--text);
      border-color: rgba(255,255,255,0.18);
    }}

    /* ── Hero ── */
    .hero {{
      background: linear-gradient(135deg, rgba(110,168,254,0.15), rgba(47,191,113,0.10));
      border: 1px solid rgba(110,168,254,0.20);
      border-radius: 20px;
      padding: 28px 32px;
      margin: 24px 0;
      box-shadow: 0 20px 50px rgba(0,0,0,0.25);
      position: relative;
      overflow: hidden;
    }}
    .hero::before {{
      content: '';
      position: absolute;
      top: -50%;
      right: -20%;
      width: 400px;
      height: 400px;
      background: radial-gradient(circle, rgba(110,168,254,0.08) 0%, transparent 70%);
      pointer-events: none;
    }}
    .hero h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; position: relative; }}
    .hero-sub {{
      color: var(--muted);
      font-size: 14px;
      position: relative;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .hero-sub .dot {{
      width: 4px;
      height: 4px;
      border-radius: 50%;
      background: var(--muted);
      display: inline-block;
    }}

    /* ── KPI cards ── */
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .card {{
      background: rgba(18,26,49,0.92);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.15);
      padding: 18px;
      min-height: 110px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .card:hover {{
      transform: translateY(-3px);
      box-shadow: 0 14px 36px rgba(0,0,0,0.25);
    }}
    .label {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .value {{ font-size: 28px; font-weight: 700; }}

    /* ── Status-coded cards ── */
    .status-good {{
      border-color: rgba(47,191,113,0.35);
      background: linear-gradient(135deg, rgba(47,191,113,0.12), rgba(18,26,49,0.92));
    }}
    .status-good .value {{ color: var(--good); }}
    .status-warn {{
      border-color: rgba(240,173,78,0.35);
      background: linear-gradient(135deg, rgba(240,173,78,0.12), rgba(18,26,49,0.92));
    }}
    .status-warn .value {{ color: var(--warn); }}
    .status-bad {{
      border-color: rgba(229,83,75,0.35);
      background: linear-gradient(135deg, rgba(229,83,75,0.12), rgba(18,26,49,0.92));
    }}
    .status-bad .value {{ color: var(--bad); }}

    /* ── Quick Actions ── */
    .actions {{
      margin-bottom: 20px;
    }}
    .actions h2 {{
      font-size: 15px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 12px;
    }}
    .actions-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 10px 18px;
      border: none;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 600;
      color: white;
      text-decoration: none;
      cursor: pointer;
      transition: all 0.2s;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}
    .btn:hover {{
      transform: translateY(-1px);
      filter: brightness(1.15);
      box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }}
    .btn:active {{
      transform: translateY(0);
    }}
    .btn-icon {{ font-size: 14px; }}
    .btn-green  {{ background: linear-gradient(135deg, #238636, #2ea043); }}
    .btn-purple {{ background: linear-gradient(135deg, #8957e5, #a371f7); }}
    .btn-amber  {{ background: linear-gradient(135deg, #d29922, #e3b341); color: #1a1a2e; }}
    .btn-blue   {{ background: linear-gradient(135deg, #0366d6, #388bfd); }}

    /* ── Sparkline ── */
    .sparkline-panel {{
      margin-bottom: 20px;
    }}
    .sparkline-panel canvas {{
      max-height: 120px;
      margin-top: 8px;
    }}

    /* ── Data grid ── */
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
    }}
    .two-up {{ grid-column: span 6; }}
    .full   {{ grid-column: 1 / -1; }}

    /* ── Panels & tables ── */
    .panel {{
      background: rgba(18,26,49,0.92);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.15);
      padding: 18px;
    }}
    .panel h2 {{ margin-bottom: 14px; font-size: 16px; font-weight: 600; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 9px 12px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
    th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      position: sticky;
      top: 0;
      background: var(--panel);
    }}
    tr {{ transition: background 0.15s; }}
    tr:hover {{ background: rgba(110,168,254,0.04); }}
    .empty {{ color: var(--muted); font-style: italic; }}

    /* ── Footer / notes ── */
    .note {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }}
    .footer {{
      margin-top: 24px;
      padding: 16px 0;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .footer-text {{
      font-size: 12px;
      color: var(--muted);
    }}
    .footer-text code {{
      color: var(--accent);
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 11px;
    }}

    @media (max-width: 1024px) {{
      .two-up {{ grid-column: 1 / -1; }}
      .topbar {{ flex-direction: column; align-items: flex-start; }}
      .hero h1 {{ font-size: 22px; }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-left">
      <div class="topbar-logo"><span>Rift</span></div>
      <span class="pill">v{html.escape(version)}</span>
      <span class="pill">Commit <code>{html.escape(git_commit)}</code></span>
      <span class="pill">Run <code>{html.escape(run_id_display)}</code></span>
    </div>
    <div class="topbar-right">
      <a class="export-btn" href="/dashboard/export/model-card" title="Download latest model card">
        📄 Model Card
      </a>
      <a class="export-btn" href="/dashboard/export/audit" title="Download latest audit report">
        📋 Audit Report
      </a>
    </div>
  </header>

  <div class="shell">
    <section class="hero">
      <h1>Rift Operations Dashboard</h1>
      <div class="hero-sub">
        <span>{html.escape(hero_run)}</span>
        <span class="dot"></span>
        <span>Open-source, zero-cost, local-first auditable ML platform</span>
      </div>
    </section>

    <section class="cards">
      {_render_cards(snapshot["kpis"], current_metrics)}
    </section>

    {_render_quick_actions(current_run)}

    {_render_sparkline(run_history)}

    <section class="grid">
      <div class="two-up">
        {_render_table("Latest ETL Runs", ["run_id", "source_system", "rows_valid", "rows_invalid", "duplicates_removed"], snapshot["etl_runs"])}
      </div>
      <div class="two-up">
        {_render_table("Recent Fairness Audits", ["audit_id", "sensitive_column", "demographic_parity_difference", "disparate_impact_ratio"], snapshot["fairness_audits"])}
      </div>
      <div class="two-up">
        {_render_table("Recent Drift Reports", ["report_id", "drift_score", "is_drift", "retrain_triggered"], snapshot["drift_reports"])}
      </div>
      <div class="two-up">
        {_render_table("Federated Training Runs", ["run_id", "client_column", "client_count", "rounds"], snapshot["federated_runs"])}
      </div>
      <div class="two-up">
        {_render_table("Prepared Public Datasets", ["dataset_id", "adapter", "rows_prepared", "auto_etl_run_id"], prepared_dataset_rows)}
      </div>
      <div class="full">
        {_render_table("Recent Audit Decisions", ["decision_id", "model_run_id", "decision", "calibrated_probability", "confidence"], snapshot["recent_audits"])}
      </div>
      <div class="full panel">
        <h2>Operating Notes</h2>
        <p class="note">Storage backend: <strong>{html.escape(str(storage_status.get("backend")))}</strong>. {html.escape(str(storage_status.get("details")))}</p>
        <p class="note">This dashboard is served locally from Rift's FastAPI application and reads only from local Parquet, DuckDB, JSON artifacts, and optional MinIO-compatible object storage under <code>.rift/</code>. No paid SaaS, proprietary cloud service, or closed-source UI dependency is required.</p>
      </div>
    </section>

    <footer class="footer">
      <span class="footer-text">Last refreshed: <code>{html.escape(refreshed_at)}</code></span>
      <span class="footer-text">Rift v{html.escape(version)} · {html.escape(git_commit)} · Zero-cost governance</span>
    </footer>
  </div>
</body>
</html>
"""


def _hero_graph_spec() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "user-1", "label": "User", "type": "user", "x": -0.88, "y": -0.2, "z": 0.22},
            {"id": "user-2", "label": "User", "type": "user", "x": -0.62, "y": 0.46, "z": 0.08},
            {"id": "txn-1", "label": "Transaction", "type": "transaction", "x": -0.26, "y": 0.1, "z": 0.42},
            {"id": "txn-2", "label": "Transaction", "type": "transaction", "x": 0.12, "y": -0.34, "z": 0.35},
            {"id": "txn-3", "label": "Transaction", "type": "transaction", "x": 0.42, "y": 0.28, "z": 0.5},
            {"id": "merchant-1", "label": "Merchant", "type": "merchant", "x": 0.7, "y": 0.48, "z": 0.04},
            {"id": "merchant-2", "label": "Merchant", "type": "merchant", "x": 0.84, "y": -0.1, "z": 0.16},
            {"id": "device-1", "label": "Device", "type": "device", "x": 0.18, "y": 0.76, "z": 0.26},
            {"id": "device-2", "label": "Device", "type": "device", "x": -0.08, "y": -0.78, "z": 0.28},
            {"id": "account-1", "label": "Account", "type": "account", "x": -0.46, "y": -0.62, "z": 0.1},
            {"id": "account-2", "label": "Account", "type": "account", "x": 0.56, "y": -0.66, "z": 0.18},
        ],
        "edges": [
            ["user-1", "txn-1"],
            ["user-2", "txn-1"],
            ["user-1", "txn-2"],
            ["txn-1", "merchant-1"],
            ["txn-1", "device-1"],
            ["txn-1", "account-1"],
            ["txn-2", "merchant-2"],
            ["txn-2", "device-2"],
            ["txn-2", "account-1"],
            ["txn-3", "merchant-1"],
            ["txn-3", "device-1"],
            ["txn-3", "account-2"],
            ["user-2", "device-1"],
            ["user-1", "device-2"],
            ["user-2", "merchant-1"],
            ["account-1", "device-2"],
            ["account-2", "device-1"],
        ],
    }


def _build_landing_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    current_model = snapshot.get("current_model") or {}
    current_metrics = snapshot.get("current_metrics") or {}
    metrics = current_metrics.get("metrics", {})
    review_rate = float(metrics.get("review_rate") or 0.05)
    coverage = max(0.0, min(1.0, 1.0 - review_rate))
    latest_audit = snapshot["recent_audits"][0] if snapshot.get("recent_audits") else {}
    latest_markdown = latest_audit.get("markdown", "")
    excerpt_lines = [
        line.strip("- ").strip()
        for line in latest_markdown.splitlines()
        if line and not line.startswith("#") and not line.startswith("##") and not line.startswith("`")
    ]
    excerpt = " ".join(excerpt_lines[:2])[:220] or "Every decision is hashed, recorded, and replayable for audit review."
    history = snapshot.get("run_history") or []
    history_values = [float(point.get("pr_auc", 0.0)) for point in history]
    history_min = min(history_values, default=0.0)
    history_max = max(history_values, default=1.0)
    return {
        "version": snapshot.get("version", _VERSION),
        "gitCommit": snapshot.get("git_commit", "unknown"),
        "refreshedAt": snapshot.get("refreshed_at", ""),
        "runId": current_model.get("run_id", "offline"),
        "modelType": current_metrics.get("model_type", "graphsage_xgb"),
        "sectorProfile": current_metrics.get("sector_profile", "fintech"),
        "timeSplit": bool(current_metrics.get("time_split", False)),
        "metrics": {
            "pr_auc": float(metrics.get("pr_auc") or 0.0),
            "ece": float(metrics.get("ece") or 0.0),
            "brier": float(metrics.get("brier") or 0.0),
            "recall_at_1pct_fpr": float(metrics.get("recall_at_1pct_fpr") or 0.0),
            "review_rate": review_rate,
            "coverage": coverage,
        },
        "kpis": snapshot.get("kpis", {}),
        "history": history,
        "historyRange": {"min": history_min, "max": history_max},
        "latestAudit": {
            "decisionId": latest_audit.get("decision_id", "No recorded audits yet"),
            "decision": latest_audit.get("decision", "review_needed"),
            "confidence": float(latest_audit.get("confidence") or 0.0),
            "calibratedProbability": float(latest_audit.get("calibrated_probability") or 0.0),
            "excerpt": excerpt,
        },
        "graph": _hero_graph_spec(),
    }


def build_landing_html(paths: RiftPaths) -> str:
    snapshot = dashboard_snapshot(paths)
    payload = _build_landing_payload(snapshot)
    metrics = payload["metrics"]
    latest_audit = payload["latestAudit"]
    quick_start = (
        "curl -X POST http://localhost:8000/predict \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d @demo/sample_transaction.json"
    )
    page = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rift | Graph-Aware Fraud Detection</title>
  <meta name="description" content="Rift is a graph-aware fraud detection platform for calibrated scoring, conformal triage, replay, and audit-ready governance." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@500;600;700;800&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet" />
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='18' fill='%23030712'/%3E%3Cpath d='M18 46V18h15.8c9 0 14.2 4.4 14.2 11.3 0 6.9-5.2 11.5-14.2 11.5H25.4V46H18Zm7.4-11.2h8.1c4.8 0 7.2-1.9 7.2-5.5 0-3.6-2.4-5.3-7.2-5.3h-8.1v10.8Z' fill='%236ea8fe'/%3E%3C/svg%3E" />
  <style>
    :root {
      --bg: #030712;
      --bg-soft: #08101c;
      --surface: rgba(9, 16, 34, 0.76);
      --surface-strong: rgba(15, 24, 47, 0.92);
      --surface-elevated: rgba(17, 29, 57, 0.94);
      --border: rgba(110, 168, 254, 0.16);
      --border-strong: rgba(110, 168, 254, 0.3);
      --text: #f3f6ff;
      --muted: #9ba8c9;
      --accent: #6ea8fe;
      --accent-2: #2fbf71;
      --danger: #e74c3c;
      --amber: #f39c12;
      --violet: #8e44ad;
      --user: #4A90D9;
      --transaction: #E74C3C;
      --merchant: #27AE60;
      --device: #F39C12;
      --account: #8E44AD;
      --card-radius: 28px;
      --section-radius: 34px;
      --shadow: 0 30px 80px rgba(0, 0, 0, 0.34);
      --grid: linear-gradient(rgba(110, 168, 254, 0.075) 1px, transparent 1px), linear-gradient(90deg, rgba(110, 168, 254, 0.075) 1px, transparent 1px);
      --hero-gradient: radial-gradient(circle at 10% 10%, rgba(110, 168, 254, 0.18), transparent 34%), radial-gradient(circle at 88% 20%, rgba(47, 191, 113, 0.14), transparent 30%), linear-gradient(180deg, rgba(6, 12, 26, 0.94), rgba(5, 10, 22, 0.98));
      --ease-out: cubic-bezier(0.25, 0.46, 0.45, 0.94);
      --ease-in: cubic-bezier(0.55, 0.06, 0.68, 0.19);
      --content-max: 1240px;
    }

    * {
      box-sizing: border-box;
    }

    html {
      scroll-behavior: smooth;
    }

    body {
      margin: 0;
      min-height: 100vh;
      overflow-x: hidden;
      font-family: 'Manrope', sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top, rgba(110, 168, 254, 0.08), transparent 35%),
        radial-gradient(circle at 80% 15%, rgba(47, 191, 113, 0.08), transparent 24%),
        linear-gradient(180deg, #02050c 0%, #040914 40%, #02050b 100%);
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.22;
      background-image: var(--grid);
      background-size: 78px 78px;
      mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.9), transparent 88%);
    }

    body::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.18;
      background-image:
        radial-gradient(rgba(255,255,255,0.7) 0.6px, transparent 0.6px),
        radial-gradient(rgba(255,255,255,0.45) 0.5px, transparent 0.5px);
      background-size: 34px 34px, 58px 58px;
      background-position: 0 0, 14px 20px;
      mix-blend-mode: soft-light;
    }

    a {
      color: inherit;
      text-decoration: none;
    }

    .page-shell {
      position: relative;
      z-index: 1;
    }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      backdrop-filter: blur(18px);
      background: rgba(2, 7, 18, 0.62);
      border-bottom: 1px solid rgba(110, 168, 254, 0.12);
    }

    .topbar-inner {
      width: min(calc(100% - 32px), var(--content-max));
      margin: 0 auto;
      padding: 18px 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 24px;
    }

    .brand {
      display: inline-flex;
      align-items: center;
      gap: 16px;
    }

    .brand-mark {
      width: 42px;
      height: 42px;
      display: grid;
      place-items: center;
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(110, 168, 254, 0.18), rgba(47, 191, 113, 0.16));
      border: 1px solid rgba(110, 168, 254, 0.24);
      box-shadow: 0 16px 34px rgba(3, 7, 18, 0.5);
      font-family: 'Syne', sans-serif;
      font-size: 1.2rem;
      font-weight: 800;
      letter-spacing: -0.05em;
    }

    .brand-copy {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .brand-copy strong {
      font-family: 'Syne', sans-serif;
      font-size: 1.15rem;
      letter-spacing: -0.04em;
    }

    .brand-copy span {
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .nav-links {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: flex-end;
    }

    .nav-links a,
    .nav-links button {
      border: 0;
      color: var(--muted);
      background: transparent;
      font: inherit;
      padding: 10px 14px;
      border-radius: 999px;
      cursor: pointer;
      transition: color 220ms var(--ease-out), background 220ms var(--ease-out), transform 220ms var(--ease-out);
    }

    .nav-links a:hover,
    .nav-links a:focus-visible,
    .nav-links button:hover,
    .nav-links button:focus-visible {
      color: var(--text);
      background: rgba(110, 168, 254, 0.1);
      outline: none;
      transform: translateY(-1px);
    }

    .nav-links .nav-cta {
      color: var(--text);
      background: linear-gradient(135deg, rgba(110, 168, 254, 0.2), rgba(47, 191, 113, 0.16));
      border: 1px solid rgba(110, 168, 254, 0.18);
    }

    .content {
      width: min(calc(100% - 32px), var(--content-max));
      margin: 0 auto;
      padding: 28px 0 120px;
    }

    .hero {
      position: relative;
      padding: 42px;
      border-radius: 42px;
      background: var(--hero-gradient);
      border: 1px solid rgba(110, 168, 254, 0.16);
      overflow: hidden;
      box-shadow: var(--shadow);
      isolation: isolate;
    }

    .hero::before,
    .hero::after {
      content: "";
      position: absolute;
      inset: auto;
      border-radius: 999px;
      pointer-events: none;
      filter: blur(14px);
      opacity: 0.85;
    }

    .hero::before {
      top: -120px;
      right: -80px;
      width: 320px;
      height: 320px;
      background: radial-gradient(circle, rgba(110, 168, 254, 0.16), transparent 66%);
    }

    .hero::after {
      left: -100px;
      bottom: -120px;
      width: 280px;
      height: 280px;
      background: radial-gradient(circle, rgba(47, 191, 113, 0.14), transparent 64%);
    }

    .hero-grid {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(420px, 0.95fr);
      gap: 28px;
      align-items: stretch;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 9px 14px;
      border-radius: 999px;
      color: #d2defa;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-size: 0.72rem;
      background: rgba(110, 168, 254, 0.08);
      border: 1px solid rgba(110, 168, 254, 0.16);
      margin-bottom: 20px;
    }

    .eyebrow::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: linear-gradient(135deg, #7fd0ff, #2fbf71);
      box-shadow: 0 0 0 6px rgba(110, 168, 254, 0.08);
    }

    .hero-copy h1,
    h2,
    h3 {
      font-family: 'Syne', sans-serif;
    }

    .hero-copy h1 {
      margin: 0;
      font-size: clamp(3rem, 6vw, 5.2rem);
      line-height: 0.92;
      letter-spacing: -0.06em;
      max-width: 9ch;
    }

    .hero-copy .lede {
      max-width: 64ch;
      margin: 18px 0 0;
      font-size: 1.06rem;
      line-height: 1.82;
      color: var(--muted);
    }

    .hero-runbar {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 24px;
    }

    .hero-pill {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 11px 14px;
      border-radius: 18px;
      background: rgba(8, 14, 29, 0.55);
      border: 1px solid rgba(110, 168, 254, 0.14);
      color: var(--muted);
      font-size: 0.84rem;
    }

    .hero-pill code {
      font-family: 'JetBrains Mono', monospace;
      color: var(--text);
      font-size: 0.76rem;
    }

    .hero-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 28px;
    }

    .button {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      padding: 14px 18px;
      border-radius: 18px;
      border: 1px solid transparent;
      font-weight: 700;
      transition: transform 220ms var(--ease-out), border-color 220ms var(--ease-out), background 220ms var(--ease-out), box-shadow 220ms var(--ease-out);
      will-change: transform, opacity;
      transform: translateZ(0);
      backface-visibility: hidden;
    }

    .button:hover,
    .button:focus-visible {
      outline: none;
      transform: translateY(-2px) scale(1.01);
      box-shadow: 0 18px 36px rgba(4, 8, 18, 0.4);
    }

    .button.primary {
      background: linear-gradient(135deg, rgba(110, 168, 254, 0.26), rgba(47, 191, 113, 0.22));
      border-color: rgba(110, 168, 254, 0.24);
    }

    .button.secondary {
      background: rgba(9, 16, 34, 0.44);
      border-color: rgba(110, 168, 254, 0.16);
    }

    .button .arrow {
      transition: transform 150ms ease-in-out;
    }

    .button:hover .arrow,
    .button:focus-visible .arrow {
      transform: translateX(4px);
    }

    .metrics-grid {
      margin-top: 32px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .metric-card {
      position: relative;
      min-height: 170px;
      padding: 18px 18px 20px;
      border-radius: 26px;
      overflow: hidden;
      background: linear-gradient(160deg, rgba(14, 23, 45, 0.96), rgba(8, 13, 27, 0.88));
      border: 1px solid rgba(110, 168, 254, 0.14);
      box-shadow: 0 20px 42px rgba(2, 6, 16, 0.34);
      transition: transform 240ms var(--ease-out), box-shadow 240ms var(--ease-out), border-color 240ms var(--ease-out);
      transform: translateZ(0);
      will-change: transform, opacity;
    }

    .metric-card:hover {
      transform: translateY(-4px) scale(1.015);
      box-shadow: 0 24px 54px rgba(2, 6, 16, 0.4);
      border-color: rgba(110, 168, 254, 0.28);
    }

    .metric-card::before {
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at top right, rgba(255,255,255,0.12), transparent 45%);
      pointer-events: none;
    }

    .metric-card[data-status="good"] {
      border-color: rgba(47, 191, 113, 0.22);
    }

    .metric-card[data-status="warn"] {
      border-color: rgba(243, 156, 18, 0.22);
    }

    .metric-card[data-status="bad"] {
      border-color: rgba(231, 76, 60, 0.24);
    }

    .metric-eyebrow {
      position: relative;
      z-index: 1;
      font-size: 0.76rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .metric-value {
      position: relative;
      z-index: 1;
      margin-top: 12px;
      font-family: 'JetBrains Mono', monospace;
      font-size: clamp(2rem, 3vw, 2.6rem);
      line-height: 1;
      font-variant-numeric: tabular-nums;
      will-change: contents;
    }

    .metric-foot {
      position: relative;
      z-index: 1;
      margin-top: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 0.88rem;
    }

    .metric-track {
      width: 100%;
      height: 8px;
      margin-top: 18px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      overflow: hidden;
    }

    .metric-fill {
      width: 0;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, rgba(110, 168, 254, 0.9), rgba(47, 191, 113, 0.92));
      transition: width 800ms var(--ease-out);
    }

    .hero-visual {
      position: relative;
      min-height: 720px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .hero-canvas-shell,
    .mini-panel,
    .section-shell,
    .feature-card,
    .sequence-card,
    .quickstart-shell,
    .audit-card,
    .comparison-shell,
    .heatmap-shell {
      background: linear-gradient(160deg, rgba(11, 18, 38, 0.88), rgba(7, 12, 24, 0.78));
      border: 1px solid rgba(110, 168, 254, 0.14);
      box-shadow: 0 24px 60px rgba(2, 6, 16, 0.34);
      backdrop-filter: blur(14px);
    }

    .hero-canvas-shell {
      position: relative;
      flex: 1;
      min-height: 480px;
      border-radius: 30px;
      overflow: hidden;
    }

    .hero-canvas {
      width: 100%;
      height: 100%;
      display: block;
    }

    .hero-legend {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    .mini-panel {
      padding: 14px 16px;
      border-radius: 20px;
    }

    .mini-label {
      color: var(--muted);
      font-size: 0.74rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .mini-value {
      margin-top: 10px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.35rem;
      color: var(--text);
    }

    .mini-note {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.6;
    }

    .legend-dot {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 0.8rem;
    }

    .legend-dot::before {
      content: "";
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--legend-color, var(--accent));
      box-shadow: 0 0 20px color-mix(in srgb, var(--legend-color, var(--accent)) 50%, transparent);
    }

    .section-shell {
      position: relative;
      margin-top: 26px;
      padding: 34px;
      border-radius: var(--section-radius);
      overflow: hidden;
    }

    .section-shell::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(180deg, rgba(255,255,255,0.04), transparent 30%);
    }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 24px;
      margin-bottom: 24px;
    }

    .section-header h2 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3rem);
      letter-spacing: -0.05em;
      line-height: 0.98;
      max-width: 10ch;
    }

    .section-header p {
      margin: 0;
      max-width: 58ch;
      color: var(--muted);
      line-height: 1.82;
    }

    .section-kicker {
      color: #c0cff4;
      font-size: 0.74rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }

    .viz-grid,
    .feature-grid,
    .architecture-grid,
    .audit-grid,
    .footer-grid {
      display: grid;
      gap: 18px;
    }

    .viz-grid {
      grid-template-columns: minmax(0, 1.08fr) minmax(320px, 0.92fr);
      align-items: stretch;
    }

    .calibration-card,
    .comparison-shell,
    .heatmap-shell,
    .confidence-shell,
    .audit-card,
    .quickstart-shell {
      border-radius: 28px;
      padding: 24px;
    }

    .panel-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 18px;
      margin-bottom: 18px;
    }

    .panel-title h3 {
      margin: 0;
      font-size: 1.4rem;
      letter-spacing: -0.04em;
    }

    .panel-title span {
      color: var(--muted);
      font-size: 0.84rem;
    }

    .chart-frame {
      position: relative;
      min-height: 320px;
      border-radius: 24px;
      border: 1px solid rgba(110, 168, 254, 0.1);
      background: linear-gradient(180deg, rgba(4, 8, 18, 0.72), rgba(7, 12, 26, 0.3));
      overflow: hidden;
    }

    #calibrationChart,
    #trendChart {
      width: 100%;
      height: 100%;
      display: block;
    }

    .comparison-shell {
      position: relative;
    }

    .comparison-stage {
      position: relative;
      min-height: 320px;
      border-radius: 24px;
      overflow: hidden;
      border: 1px solid rgba(110, 168, 254, 0.1);
      background: linear-gradient(180deg, rgba(4, 8, 18, 0.72), rgba(7, 12, 26, 0.36));
    }

    .comparison-half {
      position: absolute;
      inset: 0;
      padding: 20px;
    }

    .comparison-flat {
      background:
        repeating-linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.045) 12px, transparent 12px, transparent 34px),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.045), rgba(255,255,255,0.045) 1px, transparent 1px, transparent 22%);
      filter: saturate(0.4);
    }

    .comparison-network {
      width: calc(var(--split, 64) * 1%);
      overflow: hidden;
      border-right: 1px solid rgba(255,255,255,0.14);
      background:
        radial-gradient(circle at 35% 30%, rgba(110,168,254,0.2), transparent 30%),
        radial-gradient(circle at 70% 75%, rgba(47,191,113,0.18), transparent 26%),
        linear-gradient(180deg, rgba(6, 11, 25, 0.8), rgba(8, 13, 28, 0.42));
    }

    .comparison-network::before,
    .comparison-network::after {
      content: "";
      position: absolute;
      inset: auto;
      background: currentColor;
      border-radius: 999px;
      opacity: 0.86;
    }

    .comparison-network::before {
      top: 86px;
      left: 72px;
      width: 14px;
      height: 14px;
      color: var(--user);
      box-shadow:
        116px 32px 0 var(--transaction),
        244px -12px 0 var(--merchant),
        168px 110px 0 var(--device),
        310px 128px 0 var(--account),
        80px 180px 0 var(--transaction);
    }

    .comparison-network::after {
      left: 90px;
      top: 104px;
      width: 290px;
      height: 2px;
      color: rgba(110, 168, 254, 0.28);
      box-shadow:
        0 0 0 rgba(0,0,0,0),
        28px 52px 0 rgba(110, 168, 254, 0.28),
        124px 18px 0 rgba(110, 168, 254, 0.28),
        198px 100px 0 rgba(110, 168, 254, 0.28);
      transform: rotate(8deg);
      transform-origin: left center;
    }

    .comparison-handle {
      position: absolute;
      top: 16px;
      bottom: 16px;
      left: calc(var(--split, 64) * 1%);
      width: 42px;
      transform: translateX(-50%);
      display: grid;
      place-items: center;
      pointer-events: none;
    }

    .comparison-handle::before {
      content: "";
      position: absolute;
      inset: 0;
      width: 2px;
      margin: 0 auto;
      background: linear-gradient(180deg, transparent, rgba(255,255,255,0.42), transparent);
    }

    .comparison-knob {
      position: relative;
      z-index: 1;
      width: 42px;
      height: 42px;
      border-radius: 999px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(110, 168, 254, 0.2);
      background: rgba(12, 20, 40, 0.92);
      box-shadow: 0 12px 24px rgba(0,0,0,0.24);
      color: var(--text);
    }

    .comparison-slider {
      width: 100%;
      margin-top: 18px;
      accent-color: #6ea8fe;
      background: transparent;
    }

    .comparison-meta {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.84rem;
    }

    .feature-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-top: 22px;
    }

    .feature-card {
      position: relative;
      padding: 22px;
      border-radius: 24px;
      transition: transform 240ms var(--ease-out), border-color 240ms var(--ease-out), box-shadow 240ms var(--ease-out);
      will-change: transform, opacity;
      transform: translateZ(0);
    }

    .feature-card:hover {
      transform: translateY(-4px) scale(1.018);
      border-color: rgba(110, 168, 254, 0.28);
      box-shadow: 0 26px 52px rgba(2, 6, 16, 0.38);
    }

    .feature-icon {
      width: 52px;
      height: 52px;
      border-radius: 18px;
      display: grid;
      place-items: center;
      font-size: 1.2rem;
      margin-bottom: 18px;
      background: rgba(110, 168, 254, 0.1);
      border: 1px solid rgba(110, 168, 254, 0.16);
    }

    .feature-card h3 {
      margin: 0 0 12px;
      font-size: 1.2rem;
      letter-spacing: -0.04em;
    }

    .feature-card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.75;
    }

    .heatmap-shell {
      margin-top: 26px;
    }

    .heatmap-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 18px;
      align-items: stretch;
    }

    .heatmap-canvas-wrap {
      min-height: 360px;
      border-radius: 24px;
      overflow: hidden;
      border: 1px solid rgba(110, 168, 254, 0.1);
      background:
        linear-gradient(180deg, rgba(4, 8, 18, 0.76), rgba(7, 12, 26, 0.38)),
        repeating-linear-gradient(0deg, rgba(255,255,255,0.02), rgba(255,255,255,0.02) 1px, transparent 1px, transparent 40px),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.02) 1px, transparent 1px, transparent 40px);
    }

    #heatmapCanvas {
      width: 100%;
      height: 100%;
      display: block;
    }

    .heatmap-list {
      display: grid;
      gap: 12px;
    }

    .heatmap-signal {
      padding: 16px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(110,168,254,0.08), rgba(255,255,255,0.035));
      border: 1px solid rgba(110,168,254,0.12);
    }

    .heatmap-signal strong {
      display: block;
      font-size: 0.94rem;
      margin-bottom: 6px;
      color: #eaf1ff;
    }

    .heatmap-signal span {
      color: #c2d0ef;
      font-size: 0.84rem;
      line-height: 1.7;
    }

    .sequence-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }

    .sequence-card {
      position: relative;
      padding: 18px;
      border-radius: 22px;
      min-height: 180px;
      transition: transform 220ms var(--ease-out), border-color 220ms var(--ease-out), background 220ms var(--ease-out);
    }

    .sequence-card::after {
      content: "";
      position: absolute;
      inset: auto;
      left: calc(100% - 10px);
      top: 50%;
      width: 20px;
      height: 2px;
      background: linear-gradient(90deg, rgba(110,168,254,0.36), transparent);
    }

    .sequence-card:last-child::after {
      display: none;
    }

    .sequence-card.is-active {
      transform: translateY(-4px);
      border-color: rgba(110, 168, 254, 0.24);
      background: linear-gradient(160deg, rgba(18, 30, 58, 0.94), rgba(8, 13, 27, 0.84));
    }

    .sequence-step {
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent);
      font-size: 0.78rem;
    }

    .sequence-card h3 {
      margin: 18px 0 10px;
      font-size: 1.05rem;
    }

    .sequence-card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.65;
      font-size: 0.92rem;
    }

    .architecture-grid {
      grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
      align-items: stretch;
      margin-top: 18px;
    }

    .confidence-shell {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .confidence-band {
      position: relative;
      overflow: hidden;
      padding: 18px 18px 22px;
      border-radius: 22px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.03);
    }

    .confidence-band strong {
      display: block;
      font-size: 0.98rem;
      margin-bottom: 8px;
    }

    .confidence-band span {
      display: block;
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.7;
      margin-bottom: 14px;
    }

    .confidence-bar {
      height: 12px;
      width: 100%;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255,255,255,0.06);
    }

    .confidence-bar > i {
      display: block;
      width: 0;
      height: 100%;
      border-radius: inherit;
      transition: width 900ms var(--ease-out);
    }

    .confidence-band[data-tone="fraud"] i {
      background: linear-gradient(90deg, rgba(231,76,60,0.65), rgba(231,76,60,0.96));
    }

    .confidence-band[data-tone="review"] i {
      background: linear-gradient(90deg, rgba(243,156,18,0.58), rgba(243,156,18,0.92));
    }

    .confidence-band[data-tone="legit"] i {
      background: linear-gradient(90deg, rgba(47,191,113,0.6), rgba(47,191,113,0.92));
    }

    .architecture-map {
      padding: 24px;
      border-radius: 28px;
      background: linear-gradient(180deg, rgba(6, 11, 25, 0.82), rgba(8, 13, 27, 0.54));
      border: 1px solid rgba(110, 168, 254, 0.12);
    }

    .layer {
      padding: 16px 18px;
      border-radius: 20px;
      margin-top: 14px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.026);
    }

    .layer:first-child {
      margin-top: 0;
    }

    .layer strong {
      display: block;
      margin-bottom: 8px;
      font-size: 1rem;
    }

    .layer span {
      color: var(--muted);
      line-height: 1.7;
      font-size: 0.9rem;
    }

    .audit-grid {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      align-items: stretch;
      margin-top: 18px;
    }

    .audit-card h3,
    .quickstart-shell h3 {
      margin: 0 0 16px;
      font-size: 1.35rem;
      letter-spacing: -0.04em;
    }

    .audit-card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.8;
    }

    .audit-meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0 0;
    }

    .audit-meta .mini-panel {
      min-height: 104px;
    }

    .audit-id {
      margin-top: 18px;
      padding: 16px;
      border-radius: 20px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.76rem;
      line-height: 1.7;
      color: #d6e0fb;
      word-break: break-all;
    }

    .quickstart-shell pre {
      margin: 0;
      padding: 20px;
      border-radius: 22px;
      background: #040915;
      border: 1px solid rgba(110, 168, 254, 0.12);
      color: #d5def8;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      line-height: 1.78;
      overflow-x: auto;
    }

    .quickstart-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }

    .footer-grid {
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      margin-top: 30px;
      padding: 0 4px;
    }

    .footer-copy {
      color: var(--muted);
      line-height: 1.8;
    }

    .footer-copy code {
      font-family: 'JetBrains Mono', monospace;
      color: #d7e4ff;
    }

    .reveal {
      opacity: 0;
      transform: translate3d(0, 40px, 0);
      transition: opacity 720ms var(--ease-out), transform 720ms var(--ease-out);
      will-change: transform, opacity;
    }

    .reveal.is-visible {
      opacity: 1;
      transform: translate3d(0, 0, 0);
    }

    [data-parallax] {
      will-change: transform;
      transform: translateZ(0);
    }

    .loading-note {
      color: var(--muted);
      font-size: 0.78rem;
      margin-top: 12px;
      min-height: 1.3em;
    }

    @media (max-width: 1180px) {
      .hero-grid,
      .viz-grid,
      .architecture-grid,
      .heatmap-grid,
      .audit-grid,
      .footer-grid {
        grid-template-columns: 1fr;
      }

      .sequence-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }

      .hero-visual {
        min-height: auto;
      }
    }

    @media (max-width: 860px) {
      .topbar-inner,
      .content {
        width: min(calc(100% - 22px), var(--content-max));
      }

      .hero {
        padding: 26px;
        border-radius: 30px;
      }

      .metrics-grid,
      .feature-grid,
      .audit-meta {
        grid-template-columns: 1fr;
      }

      .hero-legend {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .sequence-grid {
        grid-template-columns: 1fr;
      }

      .section-shell,
      .comparison-shell,
      .heatmap-shell,
      .calibration-card,
      .quickstart-shell,
      .audit-card {
        padding: 20px;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      html {
        scroll-behavior: auto;
      }

      *,
      *::before,
      *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
      }
    }
  </style>
</head>
<body>
  <div class="page-shell">
    <header class="topbar">
      <div class="topbar-inner">
        <a class="brand" href="#hero" aria-label="Rift home">
          <div class="brand-mark">R</div>
          <div class="brand-copy">
            <strong>Rift</strong>
            <span>Graph-Aware Fraud Detection</span>
          </div>
        </a>
        <nav class="nav-links" aria-label="Primary">
          <a href="#why-rift">Why Rift</a>
          <a href="#features">Features</a>
          <a href="#how-it-works">How It Works</a>
          <a href="#architecture">Architecture</a>
          <a href="#audit">Audit</a>
          <a href="/dashboard">Operations</a>
          <a class="nav-cta" href="#quick-start">Quick Start</a>
        </nav>
      </div>
    </header>

    <main class="content">
      <section class="hero reveal is-visible" id="hero">
        <div class="hero-grid">
          <div class="hero-copy" data-parallax="0.12">
            <div class="eyebrow">Calibrated graph ML for high-stakes payments</div>
            <h1>Trace fraud the way it actually moves.</h1>
            <p class="lede">
              Rift fuses behavioral features, heterogeneous graph structure, calibration, and conformal prediction into one auditable operating surface. It is built for teams that want fewer false alarms, clearer uncertainty, and replayable evidence when a decision matters.
            </p>

            <div class="hero-runbar">
              <div class="hero-pill">Current run <code id="runBadge">__RUN_ID__</code></div>
              <div class="hero-pill">Model <code id="modelBadge">__MODEL_TYPE__</code></div>
              <div class="hero-pill">Sector <code>__SECTOR_PROFILE__</code></div>
              <div class="hero-pill">Temporal split <code id="timeSplitBadge">__TIME_SPLIT__</code></div>
            </div>

            <div class="hero-actions">
              <a class="button primary" href="#quick-start">Run a prediction <span class="arrow">→</span></a>
              <a class="button secondary" href="/docs">Explore API docs <span class="arrow">→</span></a>
            </div>

            <div class="metrics-grid">
              <article class="metric-card" data-metric-card data-key="pr_auc" data-target="0.85" data-direction="up">
                <div class="metric-eyebrow">PR-AUC</div>
                <div class="metric-value" data-value>__PRAUC__</div>
                <div class="metric-foot">
                  <span>Target above 0.85</span>
                  <span data-delta></span>
                </div>
                <div class="metric-track"><div class="metric-fill"></div></div>
              </article>

              <article class="metric-card" data-metric-card data-key="ece" data-target="0.05" data-direction="down">
                <div class="metric-eyebrow">Expected Calibration Error</div>
                <div class="metric-value" data-value>__ECE__</div>
                <div class="metric-foot">
                  <span>Target below 0.05</span>
                  <span data-delta></span>
                </div>
                <div class="metric-track"><div class="metric-fill"></div></div>
              </article>

              <article class="metric-card" data-metric-card data-key="brier" data-target="0.12" data-direction="down">
                <div class="metric-eyebrow">Brier Score</div>
                <div class="metric-value" data-value>__BRIER__</div>
                <div class="metric-foot">
                  <span>Target below 0.12</span>
                  <span data-delta></span>
                </div>
                <div class="metric-track"><div class="metric-fill"></div></div>
              </article>

              <article class="metric-card" data-metric-card data-key="coverage" data-target="0.95" data-direction="up">
                <div class="metric-eyebrow">Coverage</div>
                <div class="metric-value" data-value>__COVERAGE__</div>
                <div class="metric-foot">
                  <span>Target near 95%</span>
                  <span data-delta></span>
                </div>
                <div class="metric-track"><div class="metric-fill"></div></div>
              </article>
            </div>
          </div>

          <div class="hero-visual" data-parallax="0.05">
            <div class="hero-canvas-shell">
              <canvas class="hero-canvas" id="heroNetwork" aria-label="Animated graph network visualization"></canvas>
            </div>

            <div class="hero-legend">
              <div class="mini-panel">
                <div class="mini-label">Node Types</div>
                <div class="mini-note">
                  <span class="legend-dot" style="--legend-color: var(--user)">User</span><br />
                  <span class="legend-dot" style="--legend-color: var(--transaction)">Transaction</span><br />
                  <span class="legend-dot" style="--legend-color: var(--merchant)">Merchant</span>
                </div>
              </div>
              <div class="mini-panel">
                <div class="mini-label">Flow</div>
                <div class="mini-value" id="flowPulse">Realtime</div>
                <div class="mini-note">Particles move through shared devices, merchants, and accounts to surface coordinated fraud signals.</div>
              </div>
              <div class="mini-panel">
                <div class="mini-label">Decision Band</div>
                <div class="mini-value" id="decisionBandHero">Review needed</div>
                <div class="mini-note">Conformal uncertainty keeps ambiguous cases out of auto-action paths.</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="section-shell reveal" id="why-rift">
        <div class="section-header">
          <div>
            <div class="section-kicker">Why Rift</div>
            <h2>From raw scores to defensible decisions.</h2>
          </div>
          <p>
            Fraud teams do not need another opaque score. They need calibrated probabilities, confidence-aware triage, and a model that can see the connective tissue between accounts, devices, merchants, and transactions.
          </p>
        </div>

        <div class="viz-grid">
          <div class="calibration-card">
            <div class="panel-title">
              <h3>Calibration Curve</h3>
              <span>Raw scores vs calibrated probabilities</span>
            </div>
            <div class="chart-frame">
              <svg id="calibrationChart" viewBox="0 0 700 360" role="img" aria-label="Calibration curve visualization"></svg>
            </div>
          </div>

          <div class="comparison-shell">
            <div class="panel-title">
              <h3>Model Comparison</h3>
              <span>Tabular-only vs graph-aware</span>
            </div>
            <div class="comparison-stage" id="comparisonStage" style="--split: 64;">
              <div class="comparison-half comparison-flat"></div>
              <div class="comparison-half comparison-network"></div>
              <div class="comparison-handle"><div class="comparison-knob">↔</div></div>
            </div>
            <input class="comparison-slider" id="comparisonSlider" type="range" min="25" max="85" value="64" aria-label="Graph-aware comparison slider" />
            <div class="comparison-meta">
              <span>Tabular baseline underweights relationships between shared devices and clustered merchants.</span>
              <span id="comparisonLift">Graph-aware view improves context, triage quality, and recall at low FPR.</span>
            </div>
          </div>
        </div>
      </section>

      <section class="section-shell reveal" id="features">
        <div class="section-header">
          <div>
            <div class="section-kicker">Features</div>
            <h2>A product surface built for operators, not demos.</h2>
          </div>
          <p>
            Every interaction is designed around clarity: what changed, how confident the system is, and what evidence exists if a case needs to be replayed or defended later.
          </p>
        </div>

        <div class="feature-grid">
          <article class="feature-card">
            <div class="feature-icon">◎</div>
            <h3>Graph-aware embeddings</h3>
            <p>Transaction records inherit context from the accounts, devices, merchants, and people around them. That is where coordinated fraud leaves its fingerprints.</p>
          </article>

          <article class="feature-card">
            <div class="feature-icon">◔</div>
            <h3>Calibration-first outputs</h3>
            <p>Rift turns raw model scores into probabilities the operations team can reason about, compare, and threshold without guessing what a “0.83” actually means.</p>
          </article>

          <article class="feature-card">
            <div class="feature-icon">⌘</div>
            <h3>Replay and audit trails</h3>
            <p>Each decision is hashed, stored, and replayable. The latest audit payload is available in the landing page, the operations console, and the API.</p>
          </article>
        </div>
      </section>

      <section class="heatmap-shell reveal">
        <div class="panel-title">
          <h3>Fraud Pattern Heatmap</h3>
          <span>Shared-device clusters and merchant bursts</span>
        </div>
        <div class="heatmap-grid">
          <div class="heatmap-canvas-wrap">
            <canvas id="heatmapCanvas" aria-label="Fraud clustering heatmap"></canvas>
          </div>
          <div class="heatmap-list">
            <div class="heatmap-signal">
              <strong>Coordinated device reuse</strong>
              <span>High-risk points link back to the same devices and cluster in short time windows.</span>
            </div>
            <div class="heatmap-signal">
              <strong>Merchant concentration</strong>
              <span>Suspicious subgraphs often collapse onto a few merchants before spreading to adjacent accounts.</span>
            </div>
            <div class="heatmap-signal">
              <strong>Review queue protection</strong>
              <span>Uncertain transactions remain visible without being forced into false-positive auto-blocks.</span>
            </div>
          </div>
        </div>
      </section>

      <section class="section-shell reveal" id="how-it-works">
        <div class="section-header">
          <div>
            <div class="section-kicker">How It Works</div>
            <h2>An orchestrated detection sequence, not a single score.</h2>
          </div>
          <p>
            The pipeline explains itself step by step: relational evidence enters, signals propagate, a calibrated probability is produced, uncertainty wraps the score, and a decision band is emitted.
          </p>
        </div>

        <div class="sequence-grid" id="sequenceGrid">
          <article class="sequence-card">
            <div class="sequence-step">01</div>
            <h3>Transaction enters</h3>
            <p>Raw payload arrives with amount, location, channel, merchant, device, and account context.</p>
          </article>
          <article class="sequence-card">
            <div class="sequence-step">02</div>
            <h3>Graph propagation</h3>
            <p>Nodes connected by shared entities amplify suspicious structure and reduce blind spots from flat tables.</p>
          </article>
          <article class="sequence-card">
            <div class="sequence-step">03</div>
            <h3>Feature extraction</h3>
            <p>Behavioral windows, geo jumps, device sharing, and merchant prevalence are refreshed in parallel.</p>
          </article>
          <article class="sequence-card">
            <div class="sequence-step">04</div>
            <h3>Model scoring</h3>
            <p>Hybrid inference combines graph embeddings with tabular signals to produce a raw fraud score.</p>
          </article>
          <article class="sequence-card">
            <div class="sequence-step">05</div>
            <h3>Calibration</h3>
            <p>Isotonic or Platt calibration maps the raw score into a probability the operator can trust.</p>
          </article>
          <article class="sequence-card">
            <div class="sequence-step">06</div>
            <h3>Decision band</h3>
            <p>Conformal intervals decide whether the case is fraud, legit, or review-needed before action is taken.</p>
          </article>
        </div>
      </section>

      <section class="section-shell reveal" id="architecture">
        <div class="section-header">
          <div>
            <div class="section-kicker">Architecture</div>
            <h2>Layered for trust, not just throughput.</h2>
          </div>
          <p>
            Rift treats graph reasoning, calibration, conformal uncertainty, explainability, and audit persistence as one pipeline. The model is only one layer inside the system.
          </p>
        </div>

        <div class="architecture-grid">
          <div class="architecture-map">
            <div class="layer">
              <strong>Feature Layer</strong>
              <span>Rolling windows, geo distance, device reuse, merchant prevalence, and behavioral z-scores generate high-signal context.</span>
            </div>
            <div class="layer">
              <strong>Graph Layer</strong>
              <span>Five node types and seven edge types convert fraud from a row problem into a relational one.</span>
            </div>
            <div class="layer">
              <strong>Model Layer</strong>
              <span>GraphSAGE or GAT embeddings merge with gradient boosting for hybrid scoring and low-FPR recall.</span>
            </div>
            <div class="layer">
              <strong>Trust Layer</strong>
              <span>Calibrated probabilities, conformal bands, plain-English explanations, and deterministic replay close the governance loop.</span>
            </div>
          </div>

          <div class="confidence-shell">
            <div class="panel-title">
              <h3>Confidence Bands</h3>
              <span>Coverage and triage visualization</span>
            </div>
            <div class="confidence-band" data-tone="fraud" data-fill="18">
              <strong>High-confidence fraud</strong>
              <span>Red band is reserved for transactions whose interval sits entirely above the decision threshold.</span>
              <div class="confidence-bar"><i></i></div>
            </div>
            <div class="confidence-band" data-tone="review" data-fill="28">
              <strong>Review needed</strong>
              <span>Amber band catches ambiguous cases and protects the system from brittle yes-or-no automation.</span>
              <div class="confidence-bar"><i></i></div>
            </div>
            <div class="confidence-band" data-tone="legit" data-fill="54">
              <strong>High-confidence legit</strong>
              <span>Green coverage reflects the share of cases cleared without violating the conformal uncertainty boundary.</span>
              <div class="confidence-bar"><i></i></div>
            </div>
          </div>
        </div>
      </section>

      <section class="section-shell reveal" id="audit">
        <div class="section-header">
          <div>
            <div class="section-kicker">Audit</div>
            <h2>Evidence stays attached to the decision.</h2>
          </div>
          <p>
            Every score can be reconstructed. The latest recorded decision below comes from the live audit store and links directly back to replay and governance endpoints.
          </p>
        </div>

        <div class="audit-grid">
          <article class="audit-card">
            <h3>Latest audit decision</h3>
            <p id="auditExcerpt">__AUDIT_EXCERPT__</p>
            <div class="audit-meta">
              <div class="mini-panel">
                <div class="mini-label">Decision</div>
                <div class="mini-value" id="auditDecision">__AUDIT_DECISION__</div>
                <div class="mini-note">Live replayable outcome from the DuckDB audit store.</div>
              </div>
              <div class="mini-panel">
                <div class="mini-label">Confidence</div>
                <div class="mini-value" id="auditConfidence">__AUDIT_CONFIDENCE__</div>
                <div class="mini-note">Confidence from the latest recorded model decision.</div>
              </div>
            </div>
            <div class="audit-id" id="auditDecisionId">__AUDIT_ID__</div>
          </article>

          <article class="quickstart-shell" id="quick-start">
            <h3>Quick Start</h3>
            <pre><code>__QUICKSTART__</code></pre>
            <div class="quickstart-actions">
              <a class="button primary" href="/docs#/default/predict_predict_post">Open predict endpoint <span class="arrow">→</span></a>
              <a class="button secondary" href="/dashboard">Open operations dashboard <span class="arrow">→</span></a>
            </div>
            <div class="loading-note" id="liveStatus">Pulling live metrics from <code>/metrics/latest</code> and <code>/dashboard/summary</code>.</div>
          </article>
        </div>
      </section>

      <footer class="footer-grid reveal">
        <div class="footer-copy">
          Rift combines graph ML, calibration, conformal prediction, replay, and audit tooling into one local-first system. Latest refresh: <code id="refreshStamp">__REFRESHED_AT__</code>.
        </div>
        <div class="hero-runbar">
          <div class="hero-pill">Version <code>__VERSION__</code></div>
          <div class="hero-pill">Commit <code>__GIT_COMMIT__</code></div>
        </div>
      </footer>
    </main>
  </div>

  <script>
    const INITIAL_DATA = __INITIAL_DATA__;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const nodeColors = {
      user: '#4A90D9',
      transaction: '#E74C3C',
      merchant: '#27AE60',
      device: '#F39C12',
      account: '#8E44AD',
    };

    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    function formatPercent(value, digits = 1) {
      return `${(value * 100).toFixed(digits)}%`;
    }

    function formatMetric(key, value) {
      if (key === 'pr_auc' || key === 'coverage' || key === 'recall_at_1pct_fpr') {
        return formatPercent(value, 1);
      }
      return value.toFixed(3);
    }

    function metricStatus(direction, value, target) {
      if (direction === 'up') {
        if (value >= target) return 'good';
        if (value >= target * 0.82) return 'warn';
        return 'bad';
      }
      if (value <= target) return 'good';
      if (value <= target * 1.5) return 'warn';
      return 'bad';
    }

    function animateCount(element, nextValue, formatter) {
      const startValue = Number(element.dataset.current || 0);
      const start = performance.now();
      const duration = prefersReducedMotion ? 0 : 1300;

      function frame(now) {
        const progress = duration === 0 ? 1 : clamp((now - start) / duration, 0, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = startValue + (nextValue - startValue) * eased;
        element.textContent = formatter(current);
        if (progress < 1) {
          requestAnimationFrame(frame);
        } else {
          element.dataset.current = String(nextValue);
          element.textContent = formatter(nextValue);
        }
      }

      requestAnimationFrame(frame);
    }

    function updateMetricCards(metrics) {
      document.querySelectorAll('[data-metric-card]').forEach((card) => {
        const key = card.dataset.key;
        const target = Number(card.dataset.target);
        const direction = card.dataset.direction;
        const value = Number(metrics[key] ?? 0);
        const ratio = direction === 'up'
          ? clamp(value / Math.max(target, 0.0001), 0.08, 1)
          : clamp(1 - value / Math.max(target * 1.65, 0.0001), 0.08, 1);
        card.dataset.status = metricStatus(direction, value, target);
        card.querySelector('.metric-fill').style.width = `${ratio * 100}%`;
        const delta = direction === 'up'
          ? `${value >= target ? 'Above' : 'Below'} target by ${formatPercent(Math.abs(value - target), 1)}`
          : `${value <= target ? 'Under' : 'Over'} target by ${(Math.abs(value - target)).toFixed(3)}`;
        card.querySelector('[data-delta]').textContent = delta;
        const valueNode = card.querySelector('[data-value]');
        if (!valueNode.dataset.current) {
          valueNode.dataset.current = String(value);
          valueNode.textContent = formatMetric(key, value);
        } else {
          animateCount(valueNode, value, (number) => formatMetric(key, number));
        }
      });

      document.getElementById('decisionBandHero').textContent = metrics.coverage >= 0.95 ? 'Legit / fraud split stable' : 'Review load elevated';
    }

    function renderCalibration(svg, metrics) {
      const width = 700;
      const height = 360;
      const padding = 52;
      const innerWidth = width - padding * 2;
      const innerHeight = height - padding * 2;
      const ece = clamp(metrics.ece || 0.02, 0.01, 0.18);
      const calibratedLift = clamp((metrics.pr_auc || 0.4) * 0.14, 0.04, 0.12);
      const rawPoints = [];
      const calibratedPoints = [];

      for (let step = 0; step <= 10; step += 1) {
        const x = step / 10;
        const raw = clamp(x + Math.sin(x * Math.PI * 1.15) * (ece * 1.7) + (0.08 - x * 0.04), 0, 1);
        const calibrated = clamp(x + (raw - x) * 0.34 - calibratedLift * 0.06, 0, 1);
        rawPoints.push({ x, y: raw });
        calibratedPoints.push({ x, y: calibrated });
      }

      function toPath(points) {
        return points.map((point, index) => {
          const px = padding + point.x * innerWidth;
          const py = height - padding - point.y * innerHeight;
          return `${index === 0 ? 'M' : 'L'}${px.toFixed(1)} ${py.toFixed(1)}`;
        }).join(' ');
      }

      const diagonal = `M${padding} ${height - padding} L${width - padding} ${padding}`;
      const rawPath = toPath(rawPoints);
      const calibratedPath = toPath(calibratedPoints);
      const axisTicks = Array.from({ length: 6 }, (_, index) => index / 5);

      svg.innerHTML = `
        <defs>
          <linearGradient id="calibratedGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#6ea8fe" />
            <stop offset="100%" stop-color="#2fbf71" />
          </linearGradient>
        </defs>
        <rect width="${width}" height="${height}" fill="transparent"></rect>
        ${axisTicks.map((tick) => {
          const y = height - padding - tick * innerHeight;
          const x = padding + tick * innerWidth;
          return `
            <line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="rgba(255,255,255,0.06)" />
            <line x1="${x}" y1="${padding}" x2="${x}" y2="${height - padding}" stroke="rgba(255,255,255,0.04)" />
            <text x="${x}" y="${height - 18}" fill="#92a3c7" font-size="11" text-anchor="middle">${tick.toFixed(1)}</text>
            <text x="${padding - 16}" y="${y + 4}" fill="#92a3c7" font-size="11" text-anchor="end">${tick.toFixed(1)}</text>
          `;
        }).join('')}
        <path d="${diagonal}" stroke="rgba(255,255,255,0.24)" stroke-dasharray="7 9" fill="none"></path>
        <path d="${rawPath}" stroke="rgba(231,76,60,0.82)" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round"></path>
        <path d="${calibratedPath}" stroke="url(#calibratedGradient)" stroke-width="5" fill="none" stroke-linecap="round" stroke-linejoin="round"></path>
        <text x="${padding}" y="24" fill="#f3f6ff" font-size="13">Observed fraud rate</text>
        <text x="${width - padding}" y="${height - 18}" fill="#f3f6ff" font-size="13" text-anchor="end">Predicted probability</text>
        <text x="${width - padding}" y="${padding + 12}" fill="#d4def8" font-size="12" text-anchor="end">Calibrated</text>
        <text x="${width - padding}" y="${padding + 32}" fill="rgba(231,76,60,0.82)" font-size="12" text-anchor="end">Raw score</text>
      `;
    }

    function initHeroGraph(canvas, graph) {
      if (!canvas || !graph) return;
      const ctx = canvas.getContext('2d');
      const mouse = { x: 0, y: 0, active: false };
      const baseNodes = graph.nodes.map((node, index) => ({
        ...node,
        drift: index * 0.9 + 1.4,
      }));
      const particles = graph.edges.map((edge, index) => ({
        edge,
        t: (index * 0.17) % 1,
        speed: 0.0028 + (index % 4) * 0.00055,
      }));

      function sizeCanvas() {
        const ratio = window.devicePixelRatio || 1;
        const bounds = canvas.getBoundingClientRect();
        canvas.width = Math.floor(bounds.width * ratio);
        canvas.height = Math.floor(bounds.height * ratio);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      }

      function project(node, time) {
        const bounds = canvas.getBoundingClientRect();
        const driftX = Math.sin(time * 0.00034 * node.drift + node.z * 6) * 22;
        const driftY = Math.cos(time * 0.00028 * node.drift + node.x * 4) * 18;
        const px = (node.x * 0.5 + 0.5) * bounds.width + driftX + (mouse.active ? (mouse.x - bounds.width / 2) * 0.03 * (node.z + 0.2) : 0);
        const py = (node.y * 0.44 + 0.5) * bounds.height + driftY + (mouse.active ? (mouse.y - bounds.height / 2) * 0.02 * (node.z + 0.2) : 0);
        return {
          x: px,
          y: py,
          r: 7 + node.z * 12,
        };
      }

      function draw(time) {
        const bounds = canvas.getBoundingClientRect();
        ctx.clearRect(0, 0, bounds.width, bounds.height);
        ctx.fillStyle = 'rgba(2, 7, 18, 0.18)';
        ctx.fillRect(0, 0, bounds.width, bounds.height);

        const projected = new Map();
        baseNodes.forEach((node) => projected.set(node.id, project(node, time)));

        graph.edges.forEach((edge, edgeIndex) => {
          const from = projected.get(edge[0]);
          const to = projected.get(edge[1]);
          const gradient = ctx.createLinearGradient(from.x, from.y, to.x, to.y);
          gradient.addColorStop(0, 'rgba(110,168,254,0.08)');
          gradient.addColorStop(0.5, 'rgba(110,168,254,0.35)');
          gradient.addColorStop(1, 'rgba(47,191,113,0.08)');
          ctx.strokeStyle = gradient;
          ctx.lineWidth = 1.1 + (edgeIndex % 3) * 0.2;
          ctx.setLineDash(prefersReducedMotion ? [] : [8, 8]);
          ctx.lineDashOffset = prefersReducedMotion ? 0 : -(time * 0.018 + edgeIndex * 6);
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.stroke();
        });
        ctx.setLineDash([]);

        particles.forEach((particle) => {
          particle.t = (particle.t + (prefersReducedMotion ? 0.001 : particle.speed)) % 1;
          const from = projected.get(particle.edge[0]);
          const to = projected.get(particle.edge[1]);
          const x = from.x + (to.x - from.x) * particle.t;
          const y = from.y + (to.y - from.y) * particle.t;
          ctx.beginPath();
          ctx.fillStyle = 'rgba(110,168,254,0.96)';
          ctx.shadowColor = 'rgba(110,168,254,0.7)';
          ctx.shadowBlur = 18;
          ctx.arc(x, y, 2.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        });

        let nearestDistance = Infinity;
        let nearestId = null;
        if (mouse.active) {
          baseNodes.forEach((node) => {
            const point = projected.get(node.id);
            const distance = Math.hypot(point.x - mouse.x, point.y - mouse.y);
            if (distance < nearestDistance) {
              nearestDistance = distance;
              nearestId = node.id;
            }
          });
        }

        baseNodes.forEach((node) => {
          const point = projected.get(node.id);
          const emphasis = nearestId === node.id && nearestDistance < 64;
          ctx.beginPath();
          ctx.fillStyle = nodeColors[node.type];
          ctx.shadowColor = emphasis ? nodeColors[node.type] : 'rgba(0,0,0,0)';
          ctx.shadowBlur = emphasis ? 28 : 18;
          ctx.arc(point.x, point.y, emphasis ? point.r * 0.86 : point.r * 0.72, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
          ctx.beginPath();
          ctx.strokeStyle = emphasis ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.18)';
          ctx.lineWidth = emphasis ? 2.4 : 1;
          ctx.arc(point.x, point.y, point.r + 4, 0, Math.PI * 2);
          ctx.stroke();
        });

        requestAnimationFrame(draw);
      }

      canvas.addEventListener('mousemove', (event) => {
        const bounds = canvas.getBoundingClientRect();
        mouse.x = event.clientX - bounds.left;
        mouse.y = event.clientY - bounds.top;
        mouse.active = true;
      });

      canvas.addEventListener('mouseleave', () => {
        mouse.active = false;
      });

      sizeCanvas();
      requestAnimationFrame(draw);
      window.addEventListener('resize', sizeCanvas);
    }

    function initHeatmap(canvas, metrics) {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      let points = [];

      function mulberry32(seed) {
        let value = seed >>> 0;
        return function next() {
          value += 0x6D2B79F5;
          let t = value;
          t = Math.imul(t ^ (t >>> 15), t | 1);
          t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }

      function resize() {
        const ratio = window.devicePixelRatio || 1;
        const bounds = canvas.getBoundingClientRect();
        canvas.width = Math.floor(bounds.width * ratio);
        canvas.height = Math.floor(bounds.height * ratio);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      }

      function buildPoints(bounds) {
        const random = mulberry32(20260309);
        const clusterStrength = clamp((metrics.pr_auc || 0.5) + (metrics.coverage || 0.9) * 0.2, 0.45, 1);
        const clusters = [
          { x: bounds.width * 0.18, y: bounds.height * 0.34, risk: 0.22, radius: 78 },
          { x: bounds.width * 0.46, y: bounds.height * 0.2, risk: 0.58, radius: 92 },
          { x: bounds.width * 0.72, y: bounds.height * 0.62, risk: 0.94, radius: 72 },
        ];

        points = [];
        clusters.forEach((cluster, clusterIndex) => {
          const total = 30 + clusterIndex * 14;
          for (let index = 0; index < total; index += 1) {
            const angle = (index / total) * Math.PI * 2;
            const radius = cluster.radius * (0.22 + random() * 0.72);
            const x = cluster.x + Math.cos(angle) * radius + (random() - 0.5) * 18;
            const y = cluster.y + Math.sin(angle) * radius * 0.78 + (random() - 0.5) * 18;
            const risk = clamp(cluster.risk + (random() - 0.5) * 0.18 * clusterStrength, 0, 1);
            points.push({ x, y, risk, clusterIndex });
          }
        });
      }

      function draw() {
        const bounds = canvas.getBoundingClientRect();
        ctx.clearRect(0, 0, bounds.width, bounds.height);
        ctx.fillStyle = 'rgba(4, 11, 24, 0.92)';
        ctx.fillRect(0, 0, bounds.width, bounds.height);

        const haze = ctx.createRadialGradient(bounds.width * 0.22, bounds.height * 0.3, 0, bounds.width * 0.22, bounds.height * 0.3, bounds.width * 0.3);
        haze.addColorStop(0, 'rgba(110,168,254,0.14)');
        haze.addColorStop(1, 'rgba(110,168,254,0)');
        ctx.fillStyle = haze;
        ctx.fillRect(0, 0, bounds.width, bounds.height);

        const hotZone = ctx.createRadialGradient(bounds.width * 0.72, bounds.height * 0.62, 0, bounds.width * 0.72, bounds.height * 0.62, bounds.width * 0.22);
        hotZone.addColorStop(0, 'rgba(231,76,60,0.24)');
        hotZone.addColorStop(1, 'rgba(231,76,60,0)');
        ctx.fillStyle = hotZone;
        ctx.fillRect(0, 0, bounds.width, bounds.height);

        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        for (let index = 0; index < points.length; index += 1) {
          const point = points[index];
          if (point.risk < 0.7 || index % 9 !== 0) continue;
          const target = points[(index + 7) % points.length];
          ctx.beginPath();
          ctx.moveTo(point.x, point.y);
          ctx.lineTo(target.x, target.y);
          ctx.stroke();
        }

        points.forEach((point) => {
          const color = point.risk > 0.74 ? '#ff6b57' : point.risk > 0.45 ? '#ffbf47' : '#2fbf71';
          ctx.beginPath();
          ctx.fillStyle = color;
          ctx.shadowColor = color;
          ctx.shadowBlur = 22;
          ctx.arc(point.x, point.y, 4 + point.risk * 5.2, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        });

        ctx.fillStyle = 'rgba(243,246,255,0.72)';
        ctx.font = '600 12px JetBrains Mono';
        ctx.fillText('Low risk', 18, bounds.height - 18);
        ctx.fillStyle = '#ffbf47';
        ctx.fillText('Review cluster', bounds.width * 0.42, 28);
        ctx.fillStyle = '#ff6b57';
        ctx.fillText('Coordinated fraud', bounds.width * 0.72, bounds.height - 22);
      }

      resize();
      buildPoints(canvas.getBoundingClientRect());
      draw();
      window.addEventListener('resize', () => {
        resize();
        buildPoints(canvas.getBoundingClientRect());
        draw();
      });
    }

    function initSequence() {
      const cards = [...document.querySelectorAll('#sequenceGrid .sequence-card')];
      if (!cards.length) return;
      let activeIndex = 0;
      function tick() {
        cards.forEach((card, index) => card.classList.toggle('is-active', index === activeIndex));
        activeIndex = (activeIndex + 1) % cards.length;
      }
      tick();
      if (!prefersReducedMotion) {
        setInterval(tick, 950);
      }
    }

    function initConfidenceBands() {
      document.querySelectorAll('.confidence-band').forEach((band) => {
        const fill = Number(band.dataset.fill || 0);
        const bar = band.querySelector('.confidence-bar > i');
        if (bar) {
          requestAnimationFrame(() => {
            bar.style.width = `${fill}%`;
          });
        }
      });
    }

    function initReveal() {
      const elements = document.querySelectorAll('.reveal');
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
          }
        });
      }, { threshold: 0.15 });

      elements.forEach((element) => observer.observe(element));
    }

    function initParallax() {
      if (prefersReducedMotion) return;
      const elements = [...document.querySelectorAll('[data-parallax]')];
      if (!elements.length) return;
      let ticking = false;
      function update() {
        const scrollTop = window.scrollY;
        elements.forEach((element) => {
          const speed = Number(element.dataset.parallax || 0.08);
          element.style.transform = `translate3d(0, ${scrollTop * speed * -0.3}px, 0)`;
        });
        ticking = false;
      }
      window.addEventListener('scroll', () => {
        if (!ticking) {
          ticking = true;
          requestAnimationFrame(update);
        }
      }, { passive: true });
    }

    function initComparison() {
      const slider = document.getElementById('comparisonSlider');
      const stage = document.getElementById('comparisonStage');
      if (!slider || !stage) return;
      slider.addEventListener('input', () => {
        stage.style.setProperty('--split', slider.value);
      });
    }

    async function fetchJson(url) {
      const response = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if (!response.ok) throw new Error(`Failed: ${url}`);
      return response.json();
    }

    function applyLiveData(metricsResponse, summaryResponse) {
      if (metricsResponse) {
        const nextMetrics = {
          ...INITIAL_DATA.metrics,
          ...Object.fromEntries(Object.entries(metricsResponse.metrics || {}).map(([key, value]) => [key, Number(value)])),
        };
        nextMetrics.coverage = clamp(1 - Number(nextMetrics.review_rate || 0), 0, 1);
        INITIAL_DATA.metrics = nextMetrics;
        updateMetricCards(nextMetrics);
        renderCalibration(document.getElementById('calibrationChart'), nextMetrics);
      }

      if (summaryResponse) {
        if (summaryResponse.current_model?.run_id) {
          document.getElementById('runBadge').textContent = summaryResponse.current_model.run_id;
        }
        if (summaryResponse.current_metrics?.model_type) {
          document.getElementById('modelBadge').textContent = summaryResponse.current_metrics.model_type;
        }
        if (summaryResponse.current_metrics?.time_split !== undefined) {
          document.getElementById('timeSplitBadge').textContent = summaryResponse.current_metrics.time_split ? 'enabled' : 'disabled';
        }
        if (summaryResponse.recent_audits?.length) {
          const latestAudit = summaryResponse.recent_audits[0];
          document.getElementById('auditDecision').textContent = String(latestAudit.decision || 'review_needed').replaceAll('_', ' ');
          document.getElementById('auditConfidence').textContent = formatPercent(Number(latestAudit.confidence || 0), 1);
          document.getElementById('auditDecisionId').textContent = latestAudit.decision_id || '';
        }
        document.getElementById('refreshStamp').textContent = summaryResponse.refreshed_at || INITIAL_DATA.refreshedAt;
        if (summaryResponse.run_history?.length) {
          document.getElementById('flowPulse').textContent = `${summaryResponse.run_history.length} runs tracked`;
        }
      }
    }

    async function refreshLiveData() {
      const liveStatus = document.getElementById('liveStatus');
      try {
        const [metricsResponse, summaryResponse] = await Promise.all([
          fetchJson('/metrics/latest'),
          fetchJson('/dashboard/summary'),
        ]);
        applyLiveData(metricsResponse, summaryResponse);
        liveStatus.innerHTML = 'Live metrics refreshed from <code>/metrics/latest</code> and <code>/dashboard/summary</code>.';
      } catch (error) {
        liveStatus.textContent = 'Live API refresh unavailable. Showing the latest server-rendered snapshot.';
      }
    }

    document.addEventListener('DOMContentLoaded', () => {
      initReveal();
      initParallax();
      initComparison();
      initSequence();
      initConfidenceBands();
      initHeroGraph(document.getElementById('heroNetwork'), INITIAL_DATA.graph);
      initHeatmap(document.getElementById('heatmapCanvas'), INITIAL_DATA.metrics);
      updateMetricCards(INITIAL_DATA.metrics);
      renderCalibration(document.getElementById('calibrationChart'), INITIAL_DATA.metrics);
      refreshLiveData();
      setInterval(refreshLiveData, 30000);
    });
  </script>
</body>
</html>
"""
    return (
        page
        .replace("__INITIAL_DATA__", json.dumps(payload))
        .replace("__RUN_ID__", html.escape(str(payload["runId"])))
        .replace("__MODEL_TYPE__", html.escape(str(payload["modelType"])))
        .replace("__SECTOR_PROFILE__", html.escape(str(payload["sectorProfile"])))
        .replace("__TIME_SPLIT__", "enabled" if payload["timeSplit"] else "disabled")
        .replace("__PRAUC__", f"{metrics['pr_auc'] * 100:.1f}%")
        .replace("__ECE__", f"{metrics['ece']:.3f}")
        .replace("__BRIER__", f"{metrics['brier']:.3f}")
        .replace("__COVERAGE__", f"{metrics['coverage'] * 100:.1f}%")
        .replace("__AUDIT_EXCERPT__", html.escape(str(latest_audit["excerpt"])))
        .replace("__AUDIT_DECISION__", html.escape(str(latest_audit["decision"]).replace("_", " ")))
        .replace("__AUDIT_CONFIDENCE__", f"{latest_audit['confidence'] * 100:.1f}%")
        .replace("__AUDIT_ID__", html.escape(str(latest_audit["decisionId"])))
        .replace("__QUICKSTART__", html.escape(quick_start))
        .replace("__REFRESHED_AT__", html.escape(str(payload["refreshedAt"])))
        .replace("__VERSION__", html.escape(str(payload["version"])))
        .replace("__GIT_COMMIT__", html.escape(str(payload["gitCommit"])))
    )
