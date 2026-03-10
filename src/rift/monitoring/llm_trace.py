"""First-party local LLM query trace store.

Records every natural-language query with prompt, generated SQL, latency,
path taken (LLM vs fallback), and result summary. Stored in DuckDB for
zero-cost local tracing. Optional TruLens adapter for advanced evaluation.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

TRACE_SCHEMA = """
CREATE TABLE IF NOT EXISTS nl_query_traces (
    trace_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP,
    prompt TEXT,
    generated_sql TEXT,
    path VARCHAR,
    latency_seconds DOUBLE,
    result_row_count INTEGER,
    answer_summary_length INTEGER,
    model_name VARCHAR,
    error TEXT,
    metadata JSON
);
"""


@dataclass
class QueryTrace:
    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    prompt: str = ""
    generated_sql: str = ""
    path: str = "unknown"
    latency_seconds: float = 0.0
    result_row_count: int = 0
    answer_summary_length: int = 0
    model_name: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryTraceStore:
    """DuckDB-backed trace store for NL query observability."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(db_path))
        conn.execute(TRACE_SCHEMA)
        conn.close()

    def record(self, trace: QueryTrace) -> None:
        conn = duckdb.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO nl_query_traces
            (trace_id, timestamp, prompt, generated_sql, path,
             latency_seconds, result_row_count, answer_summary_length,
             model_name, error, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                trace.trace_id, trace.timestamp, trace.prompt,
                trace.generated_sql, trace.path, trace.latency_seconds,
                trace.result_row_count, trace.answer_summary_length,
                trace.model_name, trace.error, json.dumps(trace.metadata),
            ],
        )
        conn.close()

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        conn = duckdb.connect(str(self.db_path), read_only=True)
        rows = conn.execute(
            "SELECT * FROM nl_query_traces ORDER BY timestamp DESC LIMIT ?", [limit]
        ).fetchall()
        cols = [d[0] for d in conn.description]
        conn.close()
        return [dict(zip(cols, r)) for r in rows]

    def summary(self) -> dict[str, Any]:
        conn = duckdb.connect(str(self.db_path), read_only=True)
        row = conn.execute("""
            SELECT
                count(*) as total_queries,
                count(CASE WHEN path = 'llm' THEN 1 END) as llm_queries,
                count(CASE WHEN path = 'fallback' THEN 1 END) as fallback_queries,
                avg(latency_seconds) as avg_latency,
                max(latency_seconds) as max_latency,
                count(CASE WHEN error != '' THEN 1 END) as error_count
            FROM nl_query_traces
        """).fetchone()
        conn.close()
        return {
            "total_queries": row[0],
            "llm_queries": row[1],
            "fallback_queries": row[2],
            "avg_latency_seconds": round(row[3] or 0, 3),
            "max_latency_seconds": round(row[4] or 0, 3),
            "error_count": row[5],
        }


class TracedQueryExecutor:
    """Wraps the NL query path with automatic trace recording."""

    def __init__(self, trace_store: QueryTraceStore, model_name: str = ""):
        self.store = trace_store
        self.model_name = model_name

    def execute(self, prompt: str, query_fn: Any) -> tuple[Any, QueryTrace]:
        trace = QueryTrace(prompt=prompt, model_name=self.model_name)
        start = time.monotonic()
        try:
            result = query_fn(prompt)
            trace.latency_seconds = round(time.monotonic() - start, 4)
            trace.path = getattr(result, "path", "unknown")
            trace.generated_sql = getattr(result, "sql", "") or ""
            trace.result_row_count = getattr(result, "rows", 0) or 0
            answer = getattr(result, "answer", "") or ""
            trace.answer_summary_length = len(answer)
        except Exception as e:
            trace.latency_seconds = round(time.monotonic() - start, 4)
            trace.error = str(e)
            trace.path = "error"
            result = None

        self.store.record(trace)
        return result, trace
