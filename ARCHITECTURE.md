# Rift System Architecture

> Comprehensive technical architecture of the Rift fraud detection platform.
> All diagrams use [Mermaid.js](https://mermaid.js.org/) for inline rendering on GitHub.

---

## High-Level System Overview

```mermaid
flowchart TB
    subgraph ingestion ["Data Ingestion"]
        A[Raw Transactions] --> B[Polars Feature Pipeline]
        A --> C[Graph Builder]
    end

    subgraph features ["Feature Layer"]
        B --> D[Rolling Windows<br/>1h / 24h / 7d]
        B --> E[Geo Distance<br/>from Centroid]
        B --> F[Behavioral<br/>Z-scores]
        D & E & F --> G[Tabular Feature Matrix]
    end

    subgraph graph ["Graph Layer"]
        C --> H[Heterogeneous Graph<br/>5 Node Types / 7 Edge Types]
        H --> I[GraphSAGE Encoder]
        H --> J[GAT Encoder]
        I & J --> K[Transaction Embeddings<br/>16-dim]
    end

    subgraph model ["Model Layer"]
        G & K --> L[XGBoost / LightGBM<br/>Hybrid Classifier]
        L --> M[Raw Fraud Score]
        M --> N[Calibration<br/>Isotonic / Platt]
        N --> O[Calibrated Probability]
        O --> P[Conformal Prediction<br/>α = 0.05]
        P --> Q{Decision Band}
    end

    subgraph trust ["Trust & Audit Layer"]
        Q -->|high_confidence_fraud| R[Block & Investigate]
        Q -->|review_needed| S[Manual Review Queue]
        Q -->|high_confidence_legit| T[Auto-Approve]
        Q --> U[SHAP Explainer]
        Q --> V[Counterfactual Engine]
        Q --> W[Nearest Neighbor Finder]
        U & V & W --> X[Plain-English Report]
    end

    subgraph audit ["Persistence & Replay"]
        Q --> Y[Decision Recorder]
        Y --> Z[(DuckDB Audit Store)]
        Z --> AA[Replay Engine]
        Z --> AB[Lineage Tracker]
        X --> Z
    end

    subgraph mlops ["MLOps Layer"]
        L --> AC[(MLflow<br/>SQLite Backend)]
        AC --> AD[Experiment Tracking]
        AC --> AE[Model Registry]
        AG[Deepchecks] --> AH[Validation Reports]
        AI[Evidently] --> AJ[Drift Monitoring]
    end

    subgraph surface ["Product Surface"]
        AA --> AK[FastAPI Server]
        AB --> AK
        X --> AK
        AK --> AL[REST API<br/>predict / replay / audit]
        AK --> AM[Typer CLI<br/>generate / train / predict]
        AL --> AN[Docker Container]
    end
```

---

## Data Generation & Feature Engineering Pipeline

```mermaid
flowchart LR
    subgraph gen ["Synthetic Generator"]
        A[Config<br/>n_txns, fraud_rate, seed] --> B[Entity Creation<br/>Users, Merchants,<br/>Devices, Accounts]
        B --> C[Transaction Sampling<br/>Amounts, Timestamps,<br/>Geo, Channels]
        C --> D[Fraud Pattern Injection]
    end

    subgraph patterns ["7 Fraud Patterns"]
        D --> P1[Velocity Bursts]
        D --> P2[Merchant Shifts]
        D --> P3[Geo Jumps]
        D --> P4[New Device]
        D --> P5[Coordinated Reuse]
        D --> P6[Account Takeover]
        D --> P7[Testing Transactions]
    end

    subgraph feat ["Polars Feature Engine"]
        D --> F1[Rolling Counts<br/>1h / 24h / 7d]
        D --> F2[Rolling Spend<br/>1h / 24h / 7d]
        D --> F3[Haversine Geo Distance<br/>from User Centroid]
        D --> F4[New Merchants<br/>per Window]
        D --> F5[Device Sharing<br/>Degree]
        D --> F6[Merchant Fraud<br/>Prevalence]
        D --> F7[Time Since<br/>Last Transaction]
        D --> F8[Amount Z-Score<br/>within User History]
        F1 & F2 & F3 & F4 & F5 & F6 & F7 & F8 --> FM[Feature Matrix<br/>20 columns]
    end
```

---

## Heterogeneous Graph Schema

```mermaid
graph LR
    U((User)) -->|initiates| TX((Transaction))
    TX -->|at| M((Merchant))
    TX -->|via| D((Device))
    TX -->|from| A((Account))
    U -->|uses| D
    U -->|shops_at| M
    A -->|linked| D

    style U fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style TX fill:#E74C3C,stroke:#A93226,color:#fff
    style M fill:#27AE60,stroke:#1E8449,color:#fff
    style D fill:#F39C12,stroke:#D68910,color:#fff
    style A fill:#8E44AD,stroke:#6C3483,color:#fff
```

| Node Type | Count (100K txns) | Features |
|---|---|---|
| `user` | ~5,000 | Identity (degree computed) |
| `merchant` | ~1,200 | Identity (fraud rate computed) |
| `device` | ~8,000 | Identity (sharing degree) |
| `account` | ~6,000 | Identity (device count) |
| `transaction` | 100,000 | 20-dim engineered features |

---

## Model Architecture Comparison

```mermaid
flowchart TD
    subgraph baseA ["Baseline A: Tabular XGBoost"]
        A1[Tabular Features<br/>20-dim] --> A2[XGBoost<br/>depth=5, η=0.1]
        A2 --> A3[Raw Score]
    end

    subgraph baseB ["Baseline B: GraphSAGE Only"]
        B1[Node Features] --> B2[GraphSAGE<br/>2 layers, 32→16]
        B2 --> B3[MLP Head<br/>16→8→1]
        B3 --> B4[Raw Score]
    end

    subgraph hybrid ["Flagship: GraphSAGE + XGBoost"]
        C1[Node Features] --> C2[GraphSAGE<br/>2 layers, 32→16]
        C2 --> C3[Embeddings<br/>16-dim]
        C4[Tabular Features<br/>20-dim] --> C5[Concatenate]
        C3 --> C5
        C5 --> C6[XGBoost<br/>36-dim input]
        C6 --> C7[Raw Score]
    end

    subgraph alt ["Alternative: GAT + XGBoost"]
        D1[Node Features] --> D2[GAT<br/>4 heads, 2 layers]
        D2 --> D3[Attention-Weighted<br/>Embeddings 16-dim]
        D4[Tabular Features] --> D5[Concatenate]
        D3 --> D5
        D5 --> D6[XGBoost / LightGBM]
        D6 --> D7[Raw Score]
    end
```

### GraphSAGE Layer Detail

```mermaid
flowchart LR
    X[Node Feature x_v] --> LS[Linear_self]
    N[Neighbor Mean<br/>MEAN{x_u : u ∈ N_v}] --> LN[Linear_neigh]
    LS --> ADD[+]
    LN --> ADD
    ADD --> RELU[ReLU]
    RELU --> DROP[Dropout 0.1]
    DROP --> OUT[h_v]
```

### GAT Attention Mechanism

```mermaid
flowchart LR
    XV[Wh_v] --> CAT["[Wh_v || Wh_u]"]
    XU[Wh_u] --> CAT
    CAT --> ATT[a^T · LeakyReLU]
    ATT --> SOFT[Softmax over N_v]
    SOFT --> AGG["Σ α_vu · Wh_u"]
    AGG --> OUT[h_v]
```

---

## Calibration & Conformal Prediction Pipeline

```mermaid
flowchart LR
    subgraph calibration ["Calibration"]
        RS[Raw Score<br/>0.0 – 1.0] --> ISO[Isotonic<br/>Regression]
        RS --> PL[Platt<br/>Scaling]
        ISO --> CS[Calibrated<br/>Score]
        PL --> CS
    end

    subgraph conformal ["Conformal Prediction"]
        CS --> QH["Quantile q̂<br/>from cal set"]
        QH --> INT["Interval<br/>[p - q̂, p + q̂]"]
        INT --> DEC{Decision}
        DEC -->|"interval > 0.5<br/>entirely"| HCF[high_confidence_fraud]
        DEC -->|"interval < 0.5<br/>entirely"| HCL[high_confidence_legit]
        DEC -->|"interval spans 0.5"| RN[review_needed]
    end

    subgraph metrics ["Quality Metrics"]
        CS --> ECE[ECE < 0.05]
        CS --> BRIER[Brier < 0.12]
        DEC --> COV[Coverage ≈ 95%]
        DEC --> SS[Set Size < 1.4]
    end
```

---

## Audit & Replay Architecture

```mermaid
sequenceDiagram
    participant TX as Transaction
    participant FE as Feature Engine
    participant Model as Model Stack
    participant CAL as Calibrator
    participant CP as Conformal
    participant EXP as Explainer
    participant REC as Recorder
    participant DB as DuckDB

    TX->>FE: raw transaction
    FE->>Model: tabular features + graph
    Model->>CAL: raw score
    CAL->>CP: calibrated score
    CP->>REC: decision + band
    EXP->>REC: SHAP + counterfactual
    REC->>DB: store (tx, features, prediction, explanation, hash)

    Note over DB: SHA-256 decision hash ensures integrity
```

```mermaid
sequenceDiagram
    participant U as User / Auditor
    participant CLI as Rift CLI / API
    participant DB as DuckDB
    participant ART as Model Artifacts
    participant RPT as Report Engine

    U->>CLI: rift replay <decision_id>
    CLI->>DB: fetch stored transaction, features, model refs
    CLI->>ART: load exact model + calibrator + conformal artifacts
    CLI->>CLI: rerun prediction deterministically
    CLI->>RPT: regenerate explanation
    CLI-->>U: verified: same score ✓, same band ✓, same explanation ✓
```

### DuckDB Schema

```mermaid
erDiagram
    TRANSACTIONS {
        varchar tx_id PK
        json payload
        timestamp recorded_at
    }
    FEATURES {
        varchar tx_id PK
        json feature_vector
        timestamp recorded_at
    }
    PREDICTIONS {
        varchar decision_id PK
        varchar tx_id FK
        double raw_score
        double calibrated_score
        varchar confidence_band
        double interval_low
        double interval_high
        varchar model_id
        varchar decision_hash
        timestamp recorded_at
    }
    MODEL_REGISTRY {
        varchar model_id PK
        varchar model_type
        varchar version
        varchar artifact_path
        json metrics
        timestamp registered_at
    }
    AUDIT_REPORTS {
        varchar decision_id PK
        text report_markdown
        json report_json
        timestamp generated_at
    }
    REPLAY_EVENTS {
        varchar replay_id PK
        varchar decision_id FK
        boolean matched
        json diff
        timestamp replayed_at
    }

    TRANSACTIONS ||--o{ PREDICTIONS : "tx_id"
    FEATURES ||--o{ PREDICTIONS : "tx_id"
    MODEL_REGISTRY ||--o{ PREDICTIONS : "model_id"
    PREDICTIONS ||--o{ AUDIT_REPORTS : "decision_id"
    PREDICTIONS ||--o{ REPLAY_EVENTS : "decision_id"
```

---

## MLOps & Monitoring Architecture

```mermaid
flowchart TB
    subgraph tracking ["Experiment Tracking"]
        TR[Training Run] --> MLF[(MLflow<br/>SQLite Backend)]
        MLF --> PARAMS[Parameters<br/>model_type, lr, epochs]
        MLF --> METRICS[Metrics<br/>PR-AUC, ECE, Brier]
        MLF --> ARTS[Artifacts<br/>model.pkl, calibrator.pkl]
        MLF --> REG[Model Registry<br/>staging → production]
    end

    subgraph validation ["Continuous Validation"]
        REF[Reference Data] --> DC[Deepchecks Suite]
        CUR[Current Data] --> DC
        DC --> INT[Data Integrity]
        DC --> PERF[Model Performance]
        DC --> BIAS[Bias Detection]
        DC --> HTML[HTML Report]
    end

    subgraph monitoring ["Drift Monitoring"]
        PROD[Production Data] --> EV[Evidently AI]
        BASELINE[Training Data] --> EV
        EV --> DD[Data Drift]
        EV --> TD[Target Drift]
        EV --> QM[Quality Metrics]
        EV --> DASH[Streamlit Dashboard]
    end

    subgraph search ["Semantic Audit Search"]
        AUDITS[Audit Reports] --> EMB[Sentence-BERT<br/>all-MiniLM-L6-v2]
        EMB --> VEC[(FAISS Index<br/>384-dim)]
        QUERY[Natural Language Query] --> VEC
        VEC --> TOP[Top-K Similar Audits]
    end

    subgraph cicd ["CI/CD Gates"]
        PR[Pull Request] --> GHA[GitHub Actions]
        GHA --> LINT[Ruff Lint]
        GHA --> TEST[Pytest Suite]
        GHA --> VALID[Deepchecks Gate]
        VALID --> MERGE{Merge?}
    end
```

---

## Explainability Stack

```mermaid
flowchart TD
    PRED[Prediction Result] --> SHAP[SHAP TreeExplainer]
    PRED --> CF[Counterfactual Engine]
    PRED --> NN[Nearest Neighbor Finder]

    SHAP --> TOP5[Top 5 Feature<br/>Attributions]
    CF --> CHANGES[Minimal Changes<br/>to Flip Decision]
    NN --> SIM[Top 3 Similar<br/>Historical Cases]

    TOP5 & CHANGES & SIM --> NL[Plain-English<br/>Narrative Builder]
    NL --> MD[Markdown Report]
    NL --> JSON[JSON Report]

    subgraph redact ["PII Redaction"]
        MD --> REDACT[Regex Redaction<br/>user_id, device_id,<br/>account_id, lat/lon]
        REDACT --> SAFE[Auditor-Safe Report]
    end
```

---

## Training Experiment Flow

```mermaid
flowchart TD
    DATA[100K Synthetic Transactions<br/>2% Fraud Rate] --> SPLIT{Split Strategy}

    SPLIT -->|Random| RS[Random 70/15/15]
    SPLIT -->|Temporal| TS[Chronological 70/15/15]
    SPLIT -->|Rolling| RW[7-day Windows]

    RS & TS & RW --> MODELS

    subgraph MODELS ["Model Training"]
        M1[XGBoost Tabular]
        M2[GraphSAGE Only]
        M3[GraphSAGE + XGBoost]
        M4[GAT + XGBoost]
    end

    MODELS --> EVAL

    subgraph EVAL ["Evaluation"]
        E1[PR-AUC]
        E2["Recall @ 1% FPR"]
        E3[ECE]
        E4[Brier Score]
    end

    EVAL --> CAL[Calibration<br/>Isotonic / Platt]
    CAL --> CONF[Conformal Prediction<br/>Coverage / Set Size]
    CONF --> REPORT[Experiment Report<br/>+ MLflow Logging]
```

---

## Ollama Audit Chat Flow

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as Rift CLI
    participant OL as Ollama (Local LLM)
    participant DB as DuckDB
    participant VS as FAISS Vector Search

    U->>CLI: rift query "Why was TX_001 flagged?"
    CLI->>DB: fetch recent decisions (context)
    CLI->>VS: semantic search for similar audits
    CLI->>OL: prompt with context + history
    OL-->>CLI: structured response
    alt Contains SQL
        CLI->>DB: execute generated SQL
        DB-->>CLI: query results
        CLI-->>U: formatted answer + data
    else Plain text
        CLI-->>U: explanation
    end
```

---

## Deployment Architecture

```mermaid
flowchart LR
    subgraph container ["Docker Container"]
        API[FastAPI Server<br/>port 8000]
        CLI[Typer CLI]
        MODELS[Model Artifacts]
        DB[(DuckDB)]
    end

    subgraph monitoring ["Monitoring Stack"]
        EVIDENTLY[Evidently Dashboard<br/>Streamlit port 8501]
        MLFLOW[MLflow UI<br/>port 5000]
    end

    CLIENT[Client / Auditor] -->|POST /predict| API
    CLIENT -->|GET /replay/id| API
    CLIENT -->|GET /audit/id| API
    CLIENT -->|Browser| EVIDENTLY
    CLIENT -->|Browser| MLFLOW
    API --> MODELS
    API --> DB
```

---

## Repository Structure

```mermaid
flowchart TD
    ROOT[rift/] --> SRC[src/]
    ROOT --> TESTS[tests/]
    ROOT --> DOCS[docs/]
    ROOT --> DOCKER[docker/]
    ROOT --> DEMO[demo/]

    SRC --> DATA[data/<br/>generator, schemas, splits]
    SRC --> FEAT[features/<br/>engine, aggregates, temporal]
    SRC --> GRAPH[graph/<br/>builder, hetero_graph,<br/>windows, motifs]
    SRC --> MOD[models/<br/>xgb, graphsage, gat,<br/>ensemble, calibrate,<br/>conformal, metrics,<br/>train, infer]
    SRC --> REPLAY[replay/<br/>recorder, replayer,<br/>lineage, hashing]
    SRC --> EXP[explain/<br/>shap, counterfactuals,<br/>report, nearest_neighbors,<br/>ollama_chat]
    SRC --> AUD[audit/<br/>export, redact, templates]
    SRC --> APID[api/<br/>server, schemas]
    SRC --> CLID[cli/<br/>main]
    SRC --> UTIL[utils/<br/>config, logging,<br/>seeds, io]
    SRC --> VAL[validate/<br/>deepchecks_suite]
    SRC --> MON[monitoring/<br/>evidently_dashboard,<br/>mlflow_setup]
    SRC --> SEARCH[search/<br/>vector_search]
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Feature Engineering | Polars | Fast columnar feature computation |
| Graph Neural Networks | PyTorch + custom GNN layers | GraphSAGE, GAT encoders |
| Gradient Boosting | XGBoost / LightGBM | Tabular + embedding classification |
| Calibration | scikit-learn | Isotonic regression, Platt scaling |
| Conformal Prediction | Custom (distribution-free) | Uncertainty-aware triage |
| Explainability | SHAP | Feature importance attribution |
| Audit Store | DuckDB | Embedded analytical database |
| Experiment Tracking | MLflow (SQLite backend) | Params, metrics, artifacts, registry |
| Validation | Deepchecks | Data integrity, bias, performance |
| Monitoring | Evidently AI + Streamlit | Drift detection dashboards |
| Vector Search | FAISS + sentence-transformers | Semantic audit search |
| LLM Chat | Ollama (local) | Natural language audit queries |
| API | FastAPI | REST endpoints |
| CLI | Typer + Rich | Command-line interface |
| Containerization | Docker + Compose | Reproducible deployment |
| CI/CD | GitHub Actions | Lint, test, validation gates |
