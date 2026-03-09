# ISO/IEC 42001 (AI Management System) Mapping (Rift v0.2.0)

**Document version:** 1.0 | **Date:** 2026-03-09

ISO/IEC 42001 specifies requirements for an AI management system (AIMS). This table maps Rift to representative controls.

---

## Mapping Table

| ISO 42001 Clause | Requirement | How Satisfied in Rift | Evidence |
|------------------|-------------|------------------------|----------|
| **4.1** | Context of organization | Intended use, scope, limitations in GOVERNANCE.md | `docs/GOVERNANCE.md` |
| **4.2** | Interested parties | Auditors, risk teams, ML engineers addressed in docs | README, AUDIT_GUIDE |
| **5.1** | Leadership & commitment | Governance document; model cards; accountability via replay | `docs/GOVERNANCE.md` |
| **6.1** | Risk assessment | Known risks, limitations, mitigation in GOVERNANCE | `docs/GOVERNANCE.md` §4–5 |
| **7.2** | Competence | Model cards document training data, metrics, reproducibility | `docs/MODEL_CARDS/` |
| **8.1** | Operational planning | Train/predict/replay pipeline; calibration; conformal bands | `src/rift/models/` |
| **8.2** | Transparency | SHAP, counterfactuals, plain-English reports | `src/rift/explain/` |
| **9.1** | Monitoring | Metrics API; audit store; replay for verification | `rift api`, `src/rift/replay/` |
| **10.2** | Nonconformity | Limitations, out-of-scope use documented | Model cards, GOVERNANCE |

---

## References

- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html)
- [Rift GOVERNANCE.md](GOVERNANCE.md)
