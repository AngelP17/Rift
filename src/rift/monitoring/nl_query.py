from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib import error, request

import duckdb

from rift.utils.config import RiftPaths
from rift.utils.io import write_json


@dataclass(frozen=True)
class NaturalLanguageQueryResult:
    query: str
    sql: str
    rows: int
    preview: list[dict[str, Any]]
    llm_used: bool
    answer: str
    result_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _heuristic_sql(query: str) -> tuple[str, str, str]:
    lower = query.lower()
    if "flagged" in lower or "fraud" in lower:
        sql = """
        select p.decision_id, p.model_run_id, p.prediction_json
        from predictions p
        where lower(p.prediction_json) like '%high_confidence_fraud%'
        order by p.decision_id desc
        limit 10
        """
        return sql, "Recent flagged transactions", "audit"
    if "fairness" in lower or "bias" in lower:
        sql = """
        select audit_id, sensitive_column, demographic_parity_difference, disparate_impact_ratio
        from fairness_audits
        order by created_at desc
        limit 10
        """
        return sql, "Recent fairness audit results", "governance"
    if "drift" in lower:
        sql = """
        select report_id, drift_score, is_drift, retrain_triggered
        from drift_reports
        order by created_at desc
        limit 10
        """
        return sql, "Recent drift monitoring results", "governance"
    sql = """
    select decision_id, model_run_id, prediction_json
    from predictions
    order by decision_id desc
    limit 10
    """
    return sql, "Recent audit decisions", "audit"


def _run_sql(paths: RiftPaths, sql: str, database: str) -> list[dict[str, Any]]:
    target_db = paths.governance_db if database == "governance" else paths.audit_db
    conn = duckdb.connect(str(target_db), read_only=False)
    try:
        rows = conn.execute(sql).fetchdf().to_dict(orient="records")
    except Exception:
        rows = []
    finally:
        conn.close()
    return rows


def _ollama_summarize(prompt: str) -> str | None:
    payload = json.dumps({"model": "llama3", "prompt": prompt, "stream": False}).encode("utf-8")
    req = request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
            return str(body.get("response", "")).strip() or None
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def answer_natural_language_query(paths: RiftPaths, query: str) -> NaturalLanguageQueryResult:
    sql, heuristic, database = _heuristic_sql(query)
    rows = _run_sql(paths, sql, database)
    preview = rows[:10]
    summary_prompt = (
        "Summarize these Rift audit query results for a non-technical reviewer.\n"
        f"Question: {query}\nResults: {json.dumps(preview, default=str)}"
    )
    llm_answer = _ollama_summarize(summary_prompt)
    answer = llm_answer or f"{heuristic}: returned {len(preview)} row(s)."
    result_path = paths.query_dir / f"query_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    result = NaturalLanguageQueryResult(
        query=query,
        sql=sql.strip(),
        rows=len(rows),
        preview=preview,
        llm_used=llm_answer is not None,
        answer=answer,
        result_path=str(result_path),
    )
    write_json(result_path, result.to_dict())
    return result
