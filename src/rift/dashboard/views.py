from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import duckdb

from rift.datasets.adapters import list_prepared_datasets
from rift.etl.pipeline import list_etl_runs
from rift.federated.simulation import list_federated_runs
from rift.governance.fairness import list_fairness_audits
from rift.utils.config import RiftPaths
from rift.utils.io import read_json


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


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
    federated_runs = list_federated_runs(paths, limit=10)
    prepared_datasets = list_prepared_datasets(paths, limit=10)
    recent_audits = _recent_audits(paths, limit=10)
    return {
        "current_model": current_run,
        "current_metrics": current_metrics,
        "etl_runs": etl_runs,
        "fairness_audits": fairness_runs,
        "federated_runs": federated_runs,
        "prepared_datasets": prepared_datasets,
        "recent_audits": recent_audits,
        "kpis": {
            "etl_runs": len(etl_runs),
            "fairness_audits": len(fairness_runs),
            "federated_runs": len(federated_runs),
            "recent_audits": len(recent_audits),
        },
    }


def _render_cards(kpis: dict[str, Any], current_metrics: dict[str, Any] | None) -> str:
    cards = [
        ("ETL Runs", str(kpis["etl_runs"])),
        ("Fairness Audits", str(kpis["fairness_audits"])),
        ("Federated Runs", str(kpis["federated_runs"])),
        ("Recorded Audits", str(kpis["recent_audits"])),
    ]
    if current_metrics:
        cards.append(("Current PR-AUC", f"{current_metrics['metrics'].get('pr_auc', 0.0):.3f}"))
        cards.append(("Current ECE", f"{current_metrics['metrics'].get('ece', 0.0):.3f}"))
    return "".join(
        f"<div class='card'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(value)}</div></div>"
        for label, value in cards
    )


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


def build_dashboard_html(paths: RiftPaths) -> str:
    snapshot = dashboard_snapshot(paths)
    current_run = snapshot["current_model"]
    current_metrics = snapshot["current_metrics"]
    prepared_dataset_rows = [item.get("summary", {}) for item in snapshot["prepared_datasets"]]
    hero = "No active model run"
    if current_run:
        hero = f"Current model run: {current_run['run_id']}"

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rift Operations Dashboard</title>
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
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #08101d, #10192e 35%, #0a1221);
      color: var(--text);
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(110,168,254,0.18), rgba(47,191,113,0.12));
      border: 1px solid rgba(110,168,254,0.25);
      border-radius: 20px;
      padding: 28px;
      margin-bottom: 24px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.25);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 18px;
    }}
    .cards {{
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
    }}
    .card, .panel {{
      background: rgba(18,26,49,0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 16px 32px rgba(0,0,0,0.18);
    }}
    .card {{
      padding: 18px;
      min-height: 118px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .value {{ font-size: 30px; font-weight: 700; }}
    .panel {{ padding: 18px; }}
    .panel h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
    th {{ color: var(--muted); font-weight: 600; position: sticky; top: 0; background: var(--panel); }}
    .two-up {{
      grid-column: span 6;
    }}
    .full {{
      grid-column: 1 / -1;
    }}
    .empty {{ color: var(--muted); }}
    .note {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 1024px) {{
      .two-up {{ grid-column: 1 / -1; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Rift Operations Dashboard</h1>
      <p>{html.escape(hero)}. Open-source, zero-cost, local-first oversight for ETL lineage, fairness governance, federated runs, and audit review.</p>
    </section>
    <section class="cards">
      {_render_cards(snapshot["kpis"], current_metrics)}
    </section>
    <section class="grid">
      <div class="two-up">
        {_render_table("Latest ETL Runs", ["run_id", "source_system", "rows_valid", "rows_invalid", "duplicates_removed"], snapshot["etl_runs"])}
      </div>
      <div class="two-up">
        {_render_table("Recent Fairness Audits", ["audit_id", "sensitive_column", "demographic_parity_difference", "disparate_impact_ratio"], snapshot["fairness_audits"])}
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
        <h2>Operating notes</h2>
        <p class="note">This dashboard is served locally from Rift's FastAPI application and reads only from local Parquet, DuckDB, and JSON artifacts under <code>.rift/</code>. No paid SaaS, proprietary cloud service, or closed-source UI dependency is required.</p>
      </div>
    </section>
  </div>
</body>
</html>
"""
