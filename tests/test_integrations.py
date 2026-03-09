"""Tests for new integration modules: MLflow, Ollama chat, vector search, validation."""


import numpy as np


class TestMLflowSetup:
    def test_init_mlflow_sqlite(self):
        from monitoring.mlflow_setup import init_mlflow

        init_mlflow(backend="sqlite")

    def test_log_training_run_without_mlflow(self):
        from monitoring.mlflow_setup import log_training_run

        log_training_run(
            model_type="xgb_tabular",
            params={"seed": 42, "epochs": 10},
            metrics={"pr_auc": 0.85, "ece": 0.04},
        )
        # Returns None or a run_id depending on mlflow availability

    def test_experiment_summary(self):
        from monitoring.mlflow_setup import get_experiment_summary

        runs = get_experiment_summary()
        assert isinstance(runs, list)


class TestOllamaChat:
    def test_fallback_query_help(self, tmp_db):
        from explain.ollama_chat import AuditChatAssistant

        assistant = AuditChatAssistant(db_path=tmp_db)
        response = assistant._fallback_query("help me")
        assert "audit" in response.lower() or "rift" in response.lower()

    def test_fallback_query_recent(self, tmp_db):
        from replay.recorder import DecisionRecorder

        recorder = DecisionRecorder(tmp_db)
        recorder.record_prediction({
            "decision_id": "DEC_CHAT01",
            "tx_id": "TX_CHAT01",
            "raw_score": 0.8,
            "calibrated_score": 0.78,
            "confidence_band": "high_confidence_fraud",
            "model_type": "test",
        })
        recorder.close()

        from explain.ollama_chat import AuditChatAssistant

        assistant = AuditChatAssistant(db_path=tmp_db)
        response = assistant._fallback_query("show latest decisions")
        assert "DEC_CHAT01" in response

    def test_clear_history(self, tmp_db):
        from explain.ollama_chat import AuditChatAssistant

        assistant = AuditChatAssistant(db_path=tmp_db)
        assistant.history.append({"role": "user", "content": "test"})
        assistant.clear_history()
        assert len(assistant.history) == 0


class TestVectorSearch:
    def test_init_without_deps(self, tmp_db):
        from search.vector_search import AuditVectorSearch

        searcher = AuditVectorSearch(db_path=tmp_db)
        assert searcher.metadata == []

    def test_search_empty(self, tmp_db):
        from search.vector_search import AuditVectorSearch

        searcher = AuditVectorSearch(db_path=tmp_db)
        results = searcher.search("test query", k=3)
        assert isinstance(results, list)


class TestDeepchecksSuite:
    def test_basic_model_checks(self):
        from validate.deepchecks_suite import _basic_model_checks

        np.random.default_rng(42)
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        result = _basic_model_checks(y_true, y_pred)
        assert "pr_auc" in result
        assert "roc_auc" in result
        assert "brier_score" in result
        assert result["pr_auc"] > 0.5

    def test_run_data_validation_without_deepchecks(self, tmp_path):
        from validate.deepchecks_suite import run_data_validation

        result = run_data_validation(
            tmp_path / "nonexistent_ref.parquet",
            tmp_path / "nonexistent_cur.parquet",
        )
        assert isinstance(result, dict)


class TestDecisionHashing:
    def test_hash_consistency(self):
        from replay.hashing import decision_hash

        payload = {
            "tx_id": "TX_001",
            "raw_score": 0.85,
            "calibrated_score": 0.82,
            "confidence_band": "high_confidence_fraud",
            "model_type": "graphsage_xgb",
        }
        h1 = decision_hash(payload)
        h2 = decision_hash(payload)
        assert h1 == h2
        assert len(h1) == 64


class TestClearMLTracker:
    def test_check_available(self):
        from monitoring.clearml_tracker import ClearMLTracker

        tracker = ClearMLTracker()
        result = tracker._check_available()
        assert isinstance(result, bool)
