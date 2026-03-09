# Rift Architecture

## System Overview

```mermaid
flowchart LR
    A[Transactions] --> B[Feature Engine]
    A --> C[Graph Builder]
    C --> D[GraphSAGE]
    D --> E[Embeddings]
    B --> F[XGBoost Hybrid]
    E --> F
    F --> G[Calibration]
    G --> H[Conformal]
    H --> I[Decision Recorder]
    H --> J[Explainer]
    I --> K[DuckDB]
    J --> L[Audit Report]
```

## Components

- **Feature Engine**: Polars-based rolling aggregates, geo features, behavioral signals
- **Graph Builder**: Heterogeneous transaction graph (user, merchant, device, account, transaction nodes)
- **GraphSAGE/GAT**: Node embeddings for relational fraud signals
- **Hybrid Classifier**: GNN embeddings + tabular features → XGBoost
- **Calibration**: Isotonic/Platt for trustworthy probabilities
- **Conformal**: Uncertainty bands (high_conf_fraud, review_needed, high_conf_legit)
- **Recorder**: DuckDB audit store with deterministic hashing
- **Explainer**: SHAP + counterfactuals → plain-English reports
