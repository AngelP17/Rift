# Rift

**Graph ML for Fraud Detection, Replay, and Audit**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> An auditable fraud detection system that combines graph neural networks, calibrated risk scoring, conformal uncertainty, deterministic replay, and plain-English audit reports.

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

## 30-Second Quickstart

```bash
pip install -e ".[dev]"
export PYTHONPATH=src

# Generate data, train, predict
python -m cli.main generate --txns 10000 --fraud-rate 0.03
python -m cli.main train --model xgb_tabular --time-split
python -m cli.main predict --tx demo/sample_transaction.json
```

---

## 1. Why Rift Exists

Fraud detection is a high-stakes domain where model outputs directly affect people and businesses. Most fraud ML projects stop at a model notebook. Rift goes further by treating every prediction as an auditable, replayable decision with calibrated uncertainty and plain-English explanations.

## 2. What It Proves

Rift demonstrates five core principles:

1. **Fraud is relational, not just tabular.** Graph structure captures relationships between users, devices, merchants, and accounts that flat feature tables miss.
2. **Time-aware evaluation matters.** Random splits leak future patterns into training. Temporal splits give realistic performance estimates.
3. **Probabilities must be calibrated.** A score of 0.90 should behave like 90% risk. Isotonic and Platt calibration make this concrete.
4. **High-stakes decisions need uncertainty.** Conformal prediction adds confidence bands (fraud / review / legit) instead of raw labels.
5. **Explanations must be usable by non-technical people.** SHAP values, counterfactuals, and nearest-neighbor analogs are translated into plain English.

## 3. Architecture

```mermaid
flowchart TD
    A[Raw Transactions] --> B[Polars Feature Pipeline]
    A --> C[Graph Builder]
    C --> D[Heterogeneous Transaction Graph]
    D --> E[GraphSAGE Encoder]
    E --> F[Node / Transaction Embeddings]
    B --> G[Tabular Features]
    F --> H[XGBoost Hybrid Classifier]
    G --> H
    H --> I[Raw Fraud Score]
    I --> J[Calibration Layer]
    J --> K[Calibrated Probability]
    K --> L[Conformal Prediction Layer]
    L --> M[Decision + Confidence Band]
    M --> N[SHAP + Counterfactual Explainer]
    N --> O[Plain-English Audit Report]
    M --> P[Deterministic Recorder]
    P --> Q[DuckDB Audit Store]
    Q --> R[Replay Engine]
    R --> S[CLI / FastAPI]
    O --> S
```

### Components

| Module | Purpose |
|---|---|
| `data/` | Synthetic transaction generator with 7 fraud patterns |
| `features/` | Polars-based feature engineering (rolling windows, geo, z-scores) |
| `graph/` | Heterogeneous graph builder (5 node types, 7 edge types) |
| `models/` | XGBoost, GraphSAGE, GAT, hybrid ensemble, calibration, conformal |
| `replay/` | DuckDB-backed decision recording, deterministic replay, lineage |
| `explain/` | SHAP, counterfactuals, nearest neighbors, plain-English reports |
| `audit/` | Bulk export, PII redaction, Jinja2 report templates |
| `api/` | FastAPI server with predict, replay, audit endpoints |
| `cli/` | Typer CLI with generate, train, predict, replay, audit commands |

## 4. Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/AngelP17/Rift.git
cd Rift
pip install -e ".[dev]"
export PYTHONPATH=src
```

### Generate Synthetic Data

```bash
python -m cli.main generate --txns 100000 --users 5000 --merchants 1200 --fraud-rate 0.02
```

### Train Models

```bash
# Tabular baseline
python -m cli.main train --model xgb_tabular --time-split

# GraphSAGE only
python -m cli.main train --model graphsage_only --time-split

# Hybrid (flagship)
python -m cli.main train --model graphsage_xgb --time-split --window 7d

# GAT variant
python -m cli.main train --model gat_xgb --time-split
```

### Predict

```bash
python -m cli.main predict --tx demo/sample_transaction.json
```

### Replay & Audit

```bash
python -m cli.main replay <decision_id>
python -m cli.main audit <decision_id> --format markdown
python -m cli.main export --since 90d --format markdown
```

### Start API Server

```bash
python -m cli.main serve --port 8000
```

## 5. Demo Flow

```bash
# Full end-to-end demo
python -m cli.main generate --txns 10000 --fraud-rate 0.03
python -m cli.main train --model graphsage_xgb --time-split
python -m cli.main predict --tx demo/sample_transaction.json
# Copy the decision_id from output, then:
python -m cli.main replay <decision_id>
python -m cli.main audit <decision_id>
```

## 6. Experiments

```mermaid
flowchart TD
    A[Synthetic Data] --> B[Time-based Split]
    B --> C1[Tabular XGBoost]
    B --> C2[GraphSAGE]
    B --> C3[GraphSAGE + XGBoost]
    C1 --> D[Metrics]
    C2 --> D
    C3 --> D
    D --> E[Calibration]
    E --> F[Conformal]
    F --> G[Experiment Report]
```

| Experiment | Claim |
|---|---|
| **Relational vs Tabular** | Graph structure improves fraud detection over tabular-only |
| **Temporal Leakage** | Random splits inflate performance vs chronological splits |
| **Calibration** | Isotonic/Platt calibration improves operational trustworthiness |
| **Conformal Uncertainty** | Confidence bands reduce unnecessary manual review |
| **Explainability** | Plain-English reports are more usable than raw SHAP plots |

### Target Metrics

- PR-AUC > 0.85 on time split
- Recall@1% FPR > 0.60
- Brier score < 0.12
- ECE < 0.05 after isotonic calibration
- Conformal coverage near 95%, set size < 1.4

## 7. Audit Mode

Rift records every model decision like a receipt. See [AUDIT_GUIDE.md](AUDIT_GUIDE.md) for details.

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as Rift CLI
    participant DB as DuckDB
    participant M as Model Artifacts
    participant R as Report Engine
    U->>CLI: rift replay decision_id
    CLI->>DB: fetch stored transaction, features, model refs
    CLI->>M: load exact model/calibrator/conformal artifacts
    CLI->>CLI: rerun prediction deterministically
    CLI->>R: regenerate explanation and summary
    CLI-->>U: same score, confidence, explanation, report
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Run fraud prediction |
| GET | `/replay/{decision_id}` | Replay a past decision |
| GET | `/audit/{decision_id}` | Get audit report |
| GET | `/metrics/latest` | Latest model metrics |
| GET | `/models/current` | Current model info |
| GET | `/health` | Health check |

## 8. Related Work

- NVIDIA 2025 Financial Fraud GNN Blueprint
- 2026 transaction risk work combining GNNs with probabilistic prediction
- 2026 trustworthy fraud framework (explainability + conformal prediction)
- 2025 systematic review of deep learning in financial fraud detection
- 2026 human-centered trust and explainability for graph-based systems

See [docs/theory.md](docs/theory.md) for full citations.

## 9. Roadmap

- [x] Synthetic data generator with 7 fraud patterns
- [x] Polars feature engineering pipeline
- [x] Heterogeneous graph builder
- [x] XGBoost baseline, GraphSAGE, GAT, hybrid ensemble
- [x] Isotonic and Platt calibration
- [x] Conformal prediction (3-class triage)
- [x] SHAP + counterfactual explainability
- [x] Plain-English audit reports
- [x] Deterministic replay engine
- [x] DuckDB audit store
- [x] FastAPI + CLI
- [x] Docker support
- [ ] Temporal GNN extension (TGAT)
- [ ] LightGBM booster option
- [ ] PDF report export
- [ ] Fairness audit module

## License

[MIT](LICENSE)
