"""Vector search over audit and governance artifacts using FAISS + sentence embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils.config import cfg
from utils.logging import get_logger

log = get_logger(__name__)


class AuditVectorSearch:
    """Semantic search over audit decisions and reports.

    Uses sentence-transformers for embedding and FAISS for fast similarity search.
    Zero-cost, fully local.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
    ):
        self.db_path = db_path or cfg.audit_db
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.embedder = None
        self.index = None
        self.metadata: list[dict[str, Any]] = []
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return True
        try:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.model_name)
            self._init_faiss()
            self._index_from_db()
            self._initialized = True
            return True
        except ImportError as e:
            log.warning(
                "vector_search_deps_missing",
                msg="pip install faiss-cpu sentence-transformers",
                error=str(e),
            )
            return False

    def _init_faiss(self):
        import faiss

        self.index = faiss.IndexFlatL2(self.embedding_dim)

    def _index_from_db(self) -> None:
        """Index all predictions and audit reports from DuckDB."""
        import duckdb

        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)

            predictions = conn.execute(
                "SELECT decision_id, tx_id, raw_score, calibrated_score, "
                "confidence_band, model_id, recorded_at FROM predictions"
            ).fetchall()
            pred_cols = ["decision_id", "tx_id", "raw_score", "calibrated_score",
                         "confidence_band", "model_id", "recorded_at"]

            reports = conn.execute(
                "SELECT decision_id, report_markdown FROM audit_reports"
            ).fetchall()
            report_map = {r[0]: r[1] for r in reports}

            conn.close()
        except Exception as e:
            log.warning("vector_index_db_failed", error=str(e))
            return

        texts = []
        for row in predictions:
            pred = dict(zip(pred_cols, row))
            text_parts = [
                f"Decision {pred['decision_id']}",
                f"Transaction {pred['tx_id']}",
                f"Score {pred['calibrated_score']:.4f}" if pred['calibrated_score'] else "",
                f"Band: {pred['confidence_band']}",
                f"Model: {pred['model_id']}",
            ]

            report_text = report_map.get(pred["decision_id"], "")
            if report_text:
                text_parts.append(report_text[:500])

            text = " | ".join(filter(None, text_parts))
            texts.append(text)
            self.metadata.append({
                "decision_id": pred["decision_id"],
                "tx_id": pred["tx_id"],
                "confidence_band": pred["confidence_band"],
                "calibrated_score": pred.get("calibrated_score"),
                "text_preview": text[:200],
            })

        if texts:
            embeddings = self.embedder.encode(texts, show_progress_bar=False)
            self.index.add(np.array(embeddings, dtype=np.float32))
            log.info("vector_index_built", n_documents=len(texts))

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for semantically similar audit records.

        Args:
            query: Natural language query string.
            k: Number of results to return.

        Returns:
            List of matching audit records with similarity scores.
        """
        if not self._ensure_initialized():
            return [{"error": "Vector search not available. Install faiss-cpu and sentence-transformers."}]

        if self.index.ntotal == 0:
            return [{"error": "No indexed documents. Run predictions first."}]

        q_emb = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(q_emb, dtype=np.float32), min(k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            result = {**self.metadata[idx]}
            result["distance"] = float(distances[0][i])
            result["rank"] = i + 1
            results.append(result)

        return results

    def add_document(self, text: str, metadata: dict[str, Any]) -> None:
        """Add a single document to the search index."""
        if not self._ensure_initialized():
            return

        embedding = self.embedder.encode([text])
        self.index.add(np.array(embedding, dtype=np.float32))
        self.metadata.append({**metadata, "text_preview": text[:200]})

    def reindex(self) -> int:
        """Rebuild the entire index from the database."""
        if not self._ensure_initialized():
            return 0

        self._init_faiss()
        self.metadata.clear()
        self._index_from_db()
        return self.index.ntotal
