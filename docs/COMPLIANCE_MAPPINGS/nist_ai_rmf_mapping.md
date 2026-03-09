# NIST AI Risk Management Framework Mapping (Rift v0.2.0)

**Document version:** 1.0 | **Date:** 2026-03-09

The [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) organizes AI risk management into four functions: **Govern**, **Map**, **Manage**, **Measure**. This table maps Rift to relevant sub-categories.

---

## Mapping Table

| NIST Function | Sub-category | How Satisfied in Rift | Evidence |
|---------------|-------------|------------------------|----------|
| **Govern – Governance** | Policies for AI risk | GOVERNANCE.md; assumptions, limitations, known risks documented | `docs/GOVERNANCE.md` |
| **Govern – Accountability** | Roles and responsibilities | Decision ID, replay, audit trail enable accountability | `src/rift/replay/`, `AUDIT_GUIDE.md` |
| **Map – Context** | Intended use, scope | Model cards; GOVERNANCE §1 | `docs/MODEL_CARDS/`, `docs/GOVERNANCE.md` |
| **Map – Categorization** | Risk characterization | High-risk (fraud); confidence bands; human oversight | `src/rift/models/conformal.py` |
| **Manage – Allocation** | Risk allocation | Confidence-based triage; review_needed band | `src/rift/models/conformal.py` |
| **Manage – Response** | Mitigation strategies | Calibration; time-split; limitations documented | `src/rift/models/calibrate.py`, `docs/GOVERNANCE.md` |
| **Measure – Performance** | Metrics, monitoring | PR-AUC, Recall@FPR, ECE, Brier; metrics API | `src/rift/models/metrics.py`, `rift api` |
| **Measure – Trustworthiness** | Explainability, transparency | SHAP, counterfactuals, plain-English reports, replay | `src/rift/explain/`, `AUDIT_GUIDE.md` |

---

## Crosswalk to AI RMF Profile

| Profile Element | Rift Implementation |
|-----------------|----------------------|
| **Transparency** | Model cards; audit reports; explainability |
| **Explainability** | SHAP drivers, counterfactuals, similar cases |
| **Safety** | Calibration; conformal uncertainty; human-in-loop for review |
| **Accountability** | Decision ID; replay; audit trail |
| **Fairness** | Planned: group-level metrics, fair conformal |

---

## References

- [NIST AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework)
- [Rift GOVERNANCE.md](GOVERNANCE.md)
