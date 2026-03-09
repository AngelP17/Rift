# Theoretical Foundations

## Why Graph Models for Fraud

Fraud is fundamentally relational. A transaction is not an isolated event -- it connects a user, a merchant, a device, and an account. Graph neural networks can propagate information across these connections to detect patterns invisible to tabular models.

Key supporting work:

- **NVIDIA 2025 Financial Fraud GNN Blueprint** demonstrates production-grade GNN workflows for transaction networks in financial services.
- **GE-GNN (IEEE 2025)** and **HMOA-GNN (PMC 2025)** show that attention-based graph architectures achieve 5-15% lift in fraud detection on heterogeneous graphs.
- **FinGuard-GNN (2025)** demonstrates hybrid GNN + gradient boosting pipelines for scalable fraud detection.

## Temporal Evaluation

Random train/test splits in fraud detection leak future behavioral patterns into training, overstating performance. Rift implements three splitting strategies:

1. **Random split** -- standard but flawed baseline
2. **Chronological split** -- realistic deployment scenario
3. **Rolling-window evaluation** -- tests model robustness over time

Supporting work:
- Non-exchangeable conformal prediction for temporal GNNs (ACM SIGKDD 2025) addresses temporal distribution shift in graph predictions.
- 2026 transaction risk research combining GNNs with efficient probabilistic prediction validates temporal evaluation necessity.

## Calibration

A risk score of 0.90 should mean that approximately 90% of transactions at that score level are truly fraudulent. Without calibration, model outputs are often poorly aligned with actual probabilities.

Rift supports:
- **Platt scaling**: logistic regression on raw scores
- **Isotonic regression**: monotone nonparametric calibration

Measured by:
- Expected Calibration Error (ECE)
- Brier score
- Reliability curves

## Conformal Prediction

Conformal prediction provides a distribution-free framework for constructing prediction sets with guaranteed coverage. Instead of a single label, Rift produces a confidence band:

- **High confidence fraud**: prediction set contains only "fraud"
- **Review needed**: prediction set contains both "fraud" and "legit"
- **High confidence legit**: prediction set contains only "legit"

At significance level alpha = 0.05, the true label is contained in the prediction set at least 95% of the time, regardless of the underlying distribution.

Supporting work:
- 2026 trustworthy fraud framework combining explainability and conformal prediction for production fraud systems.
- WWW 2025 work on fair conformal prediction for group coverage guarantees.

## Explainability

Rift layers multiple explanation methods:

1. **SHAP (SHapley Additive exPlanations)**: game-theoretic feature attribution
2. **Nearest historical analogs**: cosine-similarity retrieval of similar past cases
3. **Counterfactual analysis**: minimal feature changes that would flip the decision
4. **Plain-English narrative**: combines all signals into human-readable text

Supporting work:
- 2026 human-centered trust and explainability evaluation shows that technical SHAP plots alone are insufficient for non-technical stakeholders.
- 2025 systematic review of deep learning in financial fraud detection emphasizes the need for interpretable fraud systems.

## References

1. NVIDIA (2025). Financial Fraud Detection with Graph Neural Networks. NVIDIA AI Enterprise Blueprint.
2. Transaction risk assessment combining GNNs with probabilistic prediction (2026).
3. Trustworthy fraud detection: explainability meets conformal prediction (2026).
4. Systematic review of deep learning approaches for financial fraud detection (2025).
5. Human-centered evaluation of trust and explainability in graph-based decision systems (2026).
6. Non-exchangeable conformal prediction for temporal graph neural networks. ACM SIGKDD (2025).
7. GE-GNN: Graph-Enhanced Fraud Detection. IEEE (2025).
8. HMOA-GNN: Hierarchical Multi-Order Attention GNN for fraud detection. PMC (2025).
9. FinGuard-GNN: Financial Fraud Detection with GNN. Atlantis Press (2025).
