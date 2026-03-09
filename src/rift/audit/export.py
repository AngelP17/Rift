"""Audit export utilities."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb


def export_audit_reports(
    db_path: str,
    since_days: int = 90,
    format: str = "markdown",
    output_path: Optional[Path] = None,
) -> str:
    """Export audit reports since given days."""
    con = duckdb.connect(db_path)
    cutoff = (datetime.utcnow() - timedelta(days=since_days)).isoformat()
    rows = con.execute(
        """
        SELECT ar.decision_id, ar.report_md, ar.report_json, p.created_at
        FROM audit_reports ar
        JOIN predictions p ON ar.decision_id = p.decision_id
        WHERE p.created_at >= ?
        ORDER BY p.created_at DESC
        """,
        [cutoff],
    ).fetchall()
    con.close()

    if format == "markdown":
        parts = [f"# Rift Audit Export\nGenerated: {datetime.utcnow().isoformat()}Z\n"]
        for decision_id, md, _, created in rows:
            parts.append(f"\n---\n## {decision_id}\n{created}\n\n{md or 'No report'}\n")
        out = "\n".join(parts)
    else:
        import json
        out = json.dumps(
            [{"decision_id": r[0], "report": r[2], "created_at": r[3]} for r in rows],
            indent=2,
            default=str,
        )

    if output_path:
        Path(output_path).write_text(out)
    return out
