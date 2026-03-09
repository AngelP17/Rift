# Rift Governance, Ethics & Compliance

**Version:** 0.2.0 | **Last Updated:** 2026-03-09

---

## Executive Summary

- **Purpose:** Rift is an auditable fraud detection system combining graph neural networks, calibrated risk scoring, and deterministic replay for high-stakes financial decisions.
- **Target users:** ML engineers, fintech risk teams, auditors, and compliance reviewers.
- **Key differentiators:** Every decision is recorded like a receipt; replay verifies same outcome; plain-English reports for non-technical stakeholders.
- **Regulatory alignment:** Designed to support EU AI Act (high-risk), NIST AI RMF, and SOC 2–style audit requirements.
- **Scope:** Synthetic transaction data; production deployment requires real data integration and sector-specific validation.

---

## 1. Intended Use

| Aspect | Description |
|--------|-------------|
| **Primary use** | Fraud detection and risk scoring on financial transactions (payments, cards, transfers) |
| **Decision types** | High-confidence fraud, review needed, high-confidence legitimate |
| **Outputs** | Calibrated probability, confidence band, SHAP drivers, counterfactuals, audit report |
| **Deployment context** | Pre-production research, PoC, regulatory demonstration; production use requires integration hardening |

---

## 2. Assumptions

- Transaction data includes user, merchant, device, account, amount, timestamp, and geo fields.
- Fraud is a minority class (typically 1–5%).
- Historical labels (is_fraud) are available for training and calibration.
- Graph structure (user–transaction–merchant–device edges) captures relational fraud signals.
- Model artifacts (weights, calibrator, conformal thresholds) are versioned and retrievable for replay.

---

## 3. Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Synthetic data only** | Real-world performance may differ; no PII in training | Use real anonymized data for production validation |
| **Static graph** | Does not model real-time graph updates | Rolling-window graphs planned for v2 |
| **No adversarial robustness** | Evasion attacks not explicitly tested | Input validation, monitoring recommended |
| **Single decision type** | Binary fraud/legit; no multi-class or severity | Extend for AML/severity tiers if needed |
| **Calibration on validation set** | May drift over time | Periodic recalibration, drift monitoring |

---

## 4. Known Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-------------|
| False positives block legitimate users | Medium | High (customer friction) | Confidence-based triage; human-in-the-loop for review_needed |
| False negatives allow fraud | Medium | High (financial loss) | Recall@low-FPR tuning; ensemble diversity |
| Model drift | High (over time) | Medium | Time-split evaluation; planned drift detection |
| Bias across groups | Unknown | High (fairness, regulatory) | Synthetic data limits bias eval; Fair conformal planned |
| Misuse (e.g., discriminatory profiling) | Low | High | Out-of-scope use documented; governance review |

---

## 5. Fairness & Bias

- **Current status:** Rift uses synthetic data; demographic attributes are not modeled. Fairness metrics (e.g., disparity ratio across groups) are not yet computed.
- **Planned:** Fair conformal prediction (group-level coverage) and fairness audits in roadmap.
- **Recommendation:** For production, evaluate fairness on real data across relevant subgroups (geography, channel, merchant category).

---

## 6. Privacy & Data Protection

| Practice | Implementation |
|----------|-----------------|
| **Sensitive field redaction** | user_id, device_id, account_id, lat, lon redacted in audit exports |
| **No PII in training (default)** | Synthetic generator produces no real PII |
| **Audit trail** | Decision ID = SHA-256 of canonical payload; no plaintext PII in hash input if redacted before storage |
| **Data retention** | Audit store (DuckDB) retention policy configurable; recommend alignment with regulatory requirements |

---

## 7. Transparency & Explainability

| Requirement | How Satisfied |
|-------------|---------------|
| **Feature importance** | SHAP values for top drivers |
| **Counterfactuals** | “What would need to change to flip the decision” |
| **Plain-language summary** | Audit report includes reviewer recommendation |
| **Replay** | Deterministic replay of any decision for verification |

---

## 8. Human Oversight

- **Confidence bands:** `review_needed` forces human review before action.
- **Replay:** Auditors can verify any decision independently.
- **No autonomous blocking:** System recommends; human or policy decides final action.

---

## 9. Regulatory & Standards Mapping

See:

- [EU AI Act High-Risk Mapping](COMPLIANCE_MAPPINGS/eu_ai_act_high_risk_mapping.md)
- [NIST AI RMF Mapping](COMPLIANCE_MAPPINGS/nist_ai_rmf_mapping.md)
- [ISO 42001 Mapping](COMPLIANCE_MAPPINGS/iso_42001_mapping.md)

---

## 10. Version Control & Traceability

| Artifact | Traceability |
|----------|--------------|
| Code | Git commit hash in model cards and reproducibility section |
| Data | Synthetic generator seed; data hash (SHA256) for file-based datasets |
| Model | Model type, calibrator version, conformal parameters in audit store |
| Decisions | Decision ID (SHA-256) links to transaction, features, prediction |

---

## 11. Approval & Change Log

| Version | Date | Author | Summary |
|---------|------|--------|---------|
| 0.1.0 | 2026-03-09 | Rift Team | Initial governance document |
| 0.2.0 | 2026-03-09 | Rift Team | Big Four alignment; compliance mappings; expanded limitations |

---

*For detailed change history, see [CHANGELOG.md](../CHANGELOG.md).*
