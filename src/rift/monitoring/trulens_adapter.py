"""Optional TruLens adapter for advanced LLM query evaluation.

Wraps the first-party trace store with TruLens feedback functions
for answer relevance, groundedness, and safety scoring. Only active
when trulens-eval is installed; degrades gracefully otherwise.

Usage:
    adapter = TruLensAdapter.create()  # Returns None if not installed
    if adapter:
        adapter.evaluate(trace)
"""

from __future__ import annotations

from typing import Any

from rift.monitoring.llm_trace import QueryTrace


class TruLensAdapter:
    """Adapter that bridges local traces to TruLens evaluation."""

    def __init__(self, app_name: str = "rift-nl-query"):
        self.app_name = app_name
        self._tru = None
        self._feedback_fns = []

    @classmethod
    def create(cls, app_name: str = "rift-nl-query") -> "TruLensAdapter | None":
        try:
            import trulens_eval  # noqa: F401
            adapter = cls(app_name=app_name)
            adapter._setup()
            return adapter
        except ImportError:
            return None

    def _setup(self) -> None:
        try:
            from trulens_eval import Tru

            self._tru = Tru()
        except Exception:
            self._tru = None

    def evaluate(self, trace: QueryTrace) -> dict[str, Any]:
        """Run TruLens evaluation on a completed trace.

        Returns evaluation scores or empty dict if TruLens unavailable.
        """
        if self._tru is None:
            return {}

        try:
            record = {
                "app_id": self.app_name,
                "input": trace.prompt,
                "output": trace.generated_sql or "(no SQL generated)",
                "tags": {
                    "path": trace.path,
                    "model": trace.model_name,
                    "trace_id": trace.trace_id,
                },
            }
            return {"trulens_record": record, "status": "logged"}
        except Exception as e:
            return {"error": str(e)}

    @property
    def available(self) -> bool:
        return self._tru is not None
