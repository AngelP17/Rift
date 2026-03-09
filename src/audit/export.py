"""Audit export: bulk export decisions and reports."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from replay.recorder import DecisionRecorder
from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)


def export_decisions(
    since_days: int = 90,
    format: str = "markdown",
    output_dir: Path | None = None,
    recorder: DecisionRecorder | None = None,
) -> Path:
    """Export all decisions since a given number of days ago."""
    recorder = recorder or DecisionRecorder()
    output_dir = output_dir or (cfg.data_dir / "exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)

    results = recorder.conn.execute(
        "SELECT * FROM predictions WHERE recorded_at >= ? ORDER BY recorded_at DESC",
        [cutoff],
    ).fetchall()
    cols = [d[0] for d in recorder.conn.description]
    decisions = [dict(zip(cols, row)) for row in results]

    if format == "markdown":
        return _export_markdown(decisions, output_dir)
    elif format == "json":
        return _export_json(decisions, output_dir)
    else:
        raise ValueError(f"Unknown format: {format}")


def _export_markdown(decisions: list[dict], output_dir: Path) -> Path:
    lines = [
        "# Rift Decision Export",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Total Decisions:** {len(decisions)}",
        "",
        "| Decision ID | TX ID | Band | Score | Recorded |",
        "|---|---|---|---|---|",
    ]

    for d in decisions:
        lines.append(
            f"| {d['decision_id']} | {d['tx_id']} | {d['confidence_band']} | "
            f"{d['calibrated_score']:.4f} | {d['recorded_at']} |"
        )

    lines.extend(["", "---", f"*Exported at {datetime.now(timezone.utc).isoformat()}*"])

    out = output_dir / "decisions_export.md"
    out.write_text("\n".join(lines))
    log.info("exported_markdown", path=str(out), n=len(decisions))
    return out


def _export_json(decisions: list[dict], output_dir: Path) -> Path:
    out = output_dir / "decisions_export.json"
    out.write_text(json.dumps(decisions, indent=2, default=str))
    log.info("exported_json", path=str(out), n=len(decisions))
    return out
