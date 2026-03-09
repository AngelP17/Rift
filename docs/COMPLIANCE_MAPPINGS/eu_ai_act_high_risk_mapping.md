# EU AI Act High-Risk Mapping (Rift v0.2.0)

**Document version:** 1.0 | **Date:** 2026-03-09

Fraud detection in credit/financial services may qualify as a **high-risk AI system** under the EU AI Act (Annex III). This table maps Rift capabilities to relevant provisions.

---

## Mapping Table

| Article / Requirement | How Satisfied in Rift | Evidence / Location |
|-----------------------|------------------------|----------------------|
| **Art. 9 – Risk management system** | Conformal uncertainty bands reduce overconfidence; time-split evaluation surfaces temporal risk; planned drift detection | `src/rift/models/conformal.py`, `src/rift/data/splits.py`, `docs/GOVERNANCE.md` |
| **Art. 10 – Data and data governance** | Synthetic data generator with configurable fraud patterns; redaction for PII in audit exports; data versioning via seed/hash | `src/rift/data/generator.py`, `src/rift/audit/redact.py`, `src/rift/replay/hashing.py` |
| **Art. 13 – Transparency and explainability** | SHAP feature importance; counterfactual explanations; plain-English audit reports; NL-accessible via AUDIT_GUIDE | `src/rift/explain/shap_explainer.py`, `src/rift/explain/counterfactuals.py`, `src/rift/explain/report.py`, `AUDIT_GUIDE.md` |
| **Art. 14 – Human oversight** | Confidence-based triage (`review_needed`); deterministic replay for audit; no autonomous blocking | `src/rift/models/conformal.py`, `src/rift/replay/replayer.py`, `docs/GOVERNANCE.md` |
| **Art. 15 – Accuracy, robustness, cybersecurity** | Calibration (ECE, Brier); temporal evaluation; input validation in feature engine; adversarial robustness planned | `src/rift/models/calibrate.py`, `src/rift/models/metrics.py`, `src/rift/features/engine.py` |
| **Art. 17 – Quality management system** | Model cards; governance document; changelog; reproducibility (seed, commit hash) | `docs/MODEL_CARDS/`, `docs/GOVERNANCE.md`, `CHANGELOG.md` |
| **Art. 29 – Logging** | DuckDB audit store; decision ID; transaction, features, prediction, model ref stored | `src/rift/replay/recorder.py`, `src/rift/replay/replayer.py` |

---

## Gaps and Roadmap

| Gap | Planned Mitigation |
|-----|---------------------|
| No formal risk management process documentation | Expand GOVERNANCE.md with risk register |
| Adversarial robustness not tested | Add adversarial evaluation in v2 |
| No group-level fairness metrics | Fair conformal prediction in roadmap |
| Drift detection not automated | Alibi-Detect or custom drift module planned |

---

## References

- [EU AI Act – Consolidated text](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689)
- [Rift GOVERNANCE.md](GOVERNANCE.md)
