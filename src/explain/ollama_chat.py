"""Ollama-powered audit chat assistant with structured SQL planning and chat history."""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb

from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)


class AuditChatAssistant:
    """Conversational audit assistant using a local Ollama LLM.

    Supports structured SQL generation, multi-turn chat history,
    and automatic query execution against the DuckDB audit store.
    Fully offline, zero-cost.
    """

    SYSTEM_PROMPT_TEMPLATE = """You are an audit assistant for the Rift fraud detection system.
You help auditors, compliance reviewers, and risk teams understand fraud decisions.

Available DuckDB tables:
- predictions (decision_id, tx_id, raw_score, calibrated_score, confidence_band, model_id, decision_hash, recorded_at)
- transactions (tx_id, payload JSON, recorded_at)
- features (tx_id, feature_vector JSON, recorded_at)
- audit_reports (decision_id, report_markdown, report_json, generated_at)
- replay_events (replay_id, decision_id, matched, diff, replayed_at)
- model_registry (model_id, model_type, version, artifact_path, metrics JSON, registered_at)

Recent decisions context (JSON):
{context}

Rules:
1. When the user asks for data, generate a valid DuckDB SQL query inside ```sql ... ``` blocks.
2. For explanations, be concise, cite decision_id or tx_id, and avoid ML jargon.
3. Never hallucinate data -- use only what is in the provided context or retrievable via SQL.
4. If you cannot answer, say so honestly.
5. For "why was X flagged?" questions, reference the confidence_band, calibrated_score, and suggest running `rift audit <decision_id>` for the full report."""

    def __init__(self, model: str = "llama3.1:8b", db_path: Path | None = None):
        self.model = model
        self.db_path = db_path or cfg.audit_db
        self.history: list[dict[str, str]] = []
        self._ollama_available: bool | None = None

    def _check_ollama(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            import ollama

            ollama.list()
            self._ollama_available = True
        except Exception:
            self._ollama_available = False
        return self._ollama_available

    def _get_context(self, limit: int = 10) -> str:
        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)
            rows = conn.execute(
                "SELECT decision_id, tx_id, calibrated_score, confidence_band, recorded_at "
                "FROM predictions ORDER BY recorded_at DESC LIMIT ?",
                [limit],
            ).fetchall()
            conn.close()
            cols = ["decision_id", "tx_id", "calibrated_score", "confidence_band", "recorded_at"]
            return json.dumps([dict(zip(cols, r)) for r in rows], default=str, indent=2)
        except Exception:
            return "[]"

    def ask(self, query: str) -> str:
        """Ask a natural language question about audit data.

        If Ollama is available, uses the LLM. Otherwise falls back to
        direct SQL pattern matching.
        """
        if not self._check_ollama():
            return self._fallback_query(query)

        import ollama

        context = self._get_context()
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(context=context)

        messages = [
            {"role": "system", "content": system_prompt},
            *self.history[-6:],
            {"role": "user", "content": query},
        ]

        try:
            response = ollama.chat(model=self.model, messages=messages)
            content = response["message"]["content"]

            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": content})

            sql_result = self._try_execute_sql(content)
            if sql_result:
                return f"{content}\n\n**Query Results:**\n{sql_result}"

            return content

        except Exception as e:
            log.warning("ollama_query_failed", error=str(e))
            return self._fallback_query(query)

    def _try_execute_sql(self, response: str) -> str | None:
        sql_match = re.search(r"```sql\s*(.*?)```", response, re.DOTALL)
        if not sql_match:
            return None

        sql = sql_match.group(1).strip()
        if not sql.upper().startswith("SELECT"):
            return None

        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)
            result = conn.execute(sql).fetchall()
            cols = [d[0] for d in conn.description]
            conn.close()
            rows = [dict(zip(cols, r)) for r in result[:20]]
            return json.dumps(rows, indent=2, default=str)
        except Exception as e:
            return f"SQL execution error: {e}"

    def _fallback_query(self, query: str) -> str:
        """Direct query handling without Ollama."""
        query_lower = query.lower()

        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)

            if "how many" in query_lower and "fraud" in query_lower:
                result = conn.execute(
                    "SELECT confidence_band, COUNT(*) as cnt FROM predictions GROUP BY confidence_band"
                ).fetchall()
                conn.close()
                lines = [f"  {r[0]}: {r[1]}" for r in result]
                return "Decision breakdown:\n" + "\n".join(lines)

            if "latest" in query_lower or "recent" in query_lower:
                result = conn.execute(
                    "SELECT decision_id, tx_id, calibrated_score, confidence_band "
                    "FROM predictions ORDER BY recorded_at DESC LIMIT 5"
                ).fetchall()
                conn.close()
                lines = []
                for r in result:
                    lines.append(f"  {r[0]} | {r[1]} | score={r[2]:.4f} | {r[3]}")
                return "Latest decisions:\n" + "\n".join(lines)

            if "explain" in query_lower or "why" in query_lower:
                conn.close()
                return (
                    "To get a detailed explanation, run:\n"
                    "  rift audit <decision_id> --format markdown\n\n"
                    "This generates a plain-English report with risk drivers, "
                    "similar cases, and counterfactual analysis."
                )

            conn.close()
            return (
                "I can help with audit queries. Try:\n"
                '  - "How many fraud decisions?"\n'
                '  - "Show latest decisions"\n'
                '  - "Explain decision DEC_XXX"\n\n'
                "For full LLM-powered queries, ensure Ollama is running: ollama serve"
            )

        except Exception as e:
            return f"Database query failed: {e}. Ensure you have run `rift train` first."

    def clear_history(self) -> None:
        self.history.clear()
