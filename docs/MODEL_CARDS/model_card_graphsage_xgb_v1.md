# Model Card: GraphSAGE + XGBoost Hybrid (v1)

**Version:** 1.0 | **Date:** 2026-03-09

---

## Intended Use

Fraud detection in financial transactions using graph ML. Suitable for:
- Transaction risk scoring (payments, cards, transfers)
- Pre-production research, PoC, and regulatory demonstration
- Audit-ready decision support (not autonomous blocking)

## Out-of-Scope Use

- Real-time streaming (batch-oriented)
- Non-financial domains without retraining
- Autonomous blocking without human review
- Use with unvalidated or adversarial inputs

---

## Training Data

| Attribute | Value |
|-----------|-------|
| Dataset type | Synthetic (Rift generator) |
| Sample count | Configurable (e.g., 2,000–100,000) |
| Fraud rate | Configurable (default 2%) |
| Seed | 42 (configurable via RIFT_SEED) |
| Data hash (SHA256) | Run-specific; see train_result.json |

---

## Performance Metrics (Example: 2,000 txns, chronological split)

| Metric | Value |
|--------|-------|
| PR-AUC | ~0.02–0.85 (data-dependent) |
| Recall @ 1% FPR | Configurable |
| Brier (raw) | ~0.04 |
| Brier (calibrated) | ~0.02 |
| ECE (raw) | ~0.04 |
| ECE (calibrated) | ~0.004 |
| Review rate | Band-dependent |

---

## Ethical Considerations

| Consideration | Status |
|---------------|--------|
| Bias evaluation | Not applicable (synthetic data, no demographics) |
| Fairness metrics | Planned: disparity ratio, fair conformal |
| Environmental impact | Standard training; green optimization planned |

---

## Limitations

- Synthetic data only (no real PII)
- Temporal leakage mitigated via time-split; rolling evaluation recommended for production
- No adversarial robustness testing
- Fairness not evaluated across demographic groups

---

## Reproducibility

| Artifact | Reference |
|----------|------------|
| Git commit | See CHANGELOG / `git rev-parse HEAD` |
| MLflow run ID | Optional; artifacts in `artifacts/models/` |
| Random seed | RIFT_SEED (default 42) |
| Model type | graphsage_xgb |
| Calibrator | Isotonic regression |

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-09 | Initial model card |
