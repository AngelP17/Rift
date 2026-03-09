# Glossary

| Term | Definition |
|---|---|
| **Calibration** | Adjusting model output probabilities so that a score of 0.90 truly means ~90% probability of fraud |
| **Conformal Prediction** | A statistical framework for creating prediction sets with guaranteed coverage |
| **Counterfactual** | An explanation showing what minimal changes to the input would flip the decision |
| **Decision ID** | A unique identifier assigned to every prediction, used for replay and audit |
| **DuckDB** | An embedded analytical database used to store the audit trail |
| **ECE** | Expected Calibration Error -- measures how well predicted probabilities match actual outcomes |
| **GAT** | Graph Attention Network -- a GNN that uses attention to weight neighbor contributions |
| **GNN** | Graph Neural Network -- a model that operates on graph-structured data |
| **GraphSAGE** | A GNN architecture that learns by sampling and aggregating neighbor features |
| **Heterogeneous Graph** | A graph with multiple node types and edge types |
| **Isotonic Regression** | A non-parametric calibration method that fits a monotone function |
| **MCC** | Merchant Category Code -- classifies the type of business |
| **Platt Scaling** | A calibration method using logistic regression on raw model outputs |
| **PR-AUC** | Precision-Recall Area Under Curve -- key metric for imbalanced fraud detection |
| **Recall@1%FPR** | Recall (fraud caught) when the false positive rate is limited to 1% |
| **Replay** | Deterministically re-running a past decision to verify reproducibility |
| **SHAP** | SHapley Additive exPlanations -- game-theoretic feature importance method |
| **Temporal Split** | Splitting data by time to prevent future information from leaking into training |
