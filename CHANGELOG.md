# Changelog

All notable changes to Rift will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-09

### Added

- Big Four–aligned documentation structure (GOVERNANCE.md, MODEL_CARDS, COMPLIANCE_MAPPINGS)
- EU AI Act, NIST AI RMF, and ISO 42001 compliance mapping tables
- Model card Jinja template and `rift governance generate-card` CLI
- Full demo script (`scripts/full_demo.sh`)
- Executive summaries in README and AUDIT_GUIDE
- Reproducibility section with git hash, seed, data version references

### Changed

- README: executive summary, CLI reference table, documentation quality badge
- AUDIT_GUIDE: expanded plain-language sections, compliance references
- docs/ folder structure reorganized for audit workflow

### Documentation

- Added CHANGELOG.md
- Added docs/GOVERNANCE.md with ethics, risk, limitations
- Added docs/COMPLIANCE_MAPPINGS/eu_ai_act_high_risk_mapping.md
- Added docs/COMPLIANCE_MAPPINGS/nist_ai_rmf_mapping.md
- Added docs/MODEL_CARDS/ and model card template

---

## [0.1.0] - 2026-03-09

### Added

- Synthetic transaction generator with fraud patterns (velocity bursts, geo jumps, new device, etc.)
- Polars-based feature engine (rolling counts, distance from centroid, merchant fraud rate, etc.)
- Heterogeneous and homogeneous graph builder for transactions
- GraphSAGE and GAT encoders
- XGBoost baseline and GraphSAGE+XGBoost / GAT+XGBoost hybrid ensemble
- Calibration (isotonic regression, Platt scaling)
- Conformal prediction for decision bands (high_confidence_fraud, review_needed, high_confidence_legit)
- DuckDB audit store with deterministic decision recording
- Deterministic replay engine for audit verification
- SHAP and counterfactual explainability
- Plain-English audit report generator
- FastAPI endpoints: /predict, /replay, /audit, /metrics/latest
- CLI: rift generate, train, predict, replay, audit, export, compare
- Time-aware splits (random, chronological, rolling)
- Evaluation metrics: PR-AUC, Recall@1% FPR, ECE, Brier

### Fixed

- Polars sqrt deprecation (use .sqrt() on expression)
- Deterministic transaction IDs for reproducible data generation
- DuckDB recorder INSERT OR REPLACE compatibility

### Security

- SHA-256 canonical hashing for decision IDs and audit trail
- Redaction of sensitive fields in audit exports
