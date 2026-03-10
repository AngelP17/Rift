# Roadmap

## v1.0 (Current)

### Core ML Pipeline
- [x] Synthetic transaction generator (7 fraud patterns)
- [x] Polars feature engineering pipeline (20 features)
- [x] Heterogeneous graph builder (5 node types, 7 edge types)
- [x] XGBoost tabular baseline
- [x] GraphSAGE encoder and classifier
- [x] GAT encoder and classifier
- [x] GraphSAGE/GAT + XGBoost hybrid ensemble
- [x] Platt and isotonic calibration
- [x] Conformal prediction (3-class triage)
- [x] Time-aware data splitting (random, temporal, rolling-window)

### Audit & Explainability
- [x] SHAP feature importance
- [x] Counterfactual analysis
- [x] Nearest neighbor analogs
- [x] Plain-English audit reports
- [x] DuckDB decision recorder (SHA-256 hashed)
- [x] Deterministic replay engine
- [x] Decision lineage tracking
- [x] PII redaction

### Governance
- [x] Fairness audits (demographic parity, disparity ratio)
- [x] Drift monitoring (Alibi-Detect + fallback)
- [x] Model card generation (Jinja2 templates)
- [x] Sector profiles (YAML-driven field aliasing + privacy masking)
- [x] Legacy reengineering simulator
- [x] Green optimization (downcasting + artifact size tracking)
- [x] Federated training scaffolding

### Operations
- [x] ETL pipeline (bronze/silver/gold)
- [x] DuckDB lakehouse with SQL queries
- [x] Local + S3-compatible storage backends
- [x] MLflow SQLite experiment tracking
- [x] Deepchecks continuous validation
- [x] Evidently drift monitoring + Streamlit dashboard
- [x] FAISS semantic audit search
- [x] Ollama audit chat assistant
- [x] Natural language query interface

### Product Surface
- [x] FastAPI server (20+ endpoints)
- [x] Operations dashboard (server-rendered HTML with KPI cards, tables, drill-downs)
- [x] Next.js React frontend (optional)
- [x] Typer CLI (12 commands for src/cli + 15+ for src/rift/cli)
- [x] Docker and docker-compose support
- [x] Airflow DAG scaffolding
- [x] JupyterHub scaffolding
- [x] GitHub Actions CI/CD (lint, test, validate)

### Documentation & Demos
- [x] 6 Colab-compatible demo notebooks
- [x] Colab setup helper with GPU detection
- [x] ARCHITECTURE.md with Mermaid diagrams
- [x] AUDIT_GUIDE.md for non-technical reviewers
- [x] docs/ (architecture, theory, experiments, dashboard, notebooks, glossary, roadmap)
- [x] CONTRIBUTING.md
- [x] Demo files (sample_transaction.json, full_audit.sh)
- [x] 97 passing tests

## v1.1 (Near-term)

- [ ] PDF export for audit reports (WeasyPrint)
- [ ] Notebook-based experiment dashboards with visualization
- [ ] Performance benchmarks on public datasets (IEEE-CIS, Yelp)
- [ ] API authentication and rate limiting
- [ ] Webhook notifications for drift alerts
- [ ] Dashboard dark/light theme toggle

## v2.0 (Medium-term)

- [ ] Temporal Graph Attention Network (TGAT) encoder
- [ ] Fair conformal prediction (group coverage guarantees)
- [ ] Streaming prediction mode
- [ ] Kubernetes deployment manifests
- [ ] Multi-tenant support

## Not Planned

The following are explicitly out of scope to maintain credibility:
- Blockchain or ZK proofs
- Autonomous agents
- Federated learning (beyond simulation scaffolding)
- Deepfake detection
- Kafka (until streaming is needed)
- RL as the main system
- Quantum computing
