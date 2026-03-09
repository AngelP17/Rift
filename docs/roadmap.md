# Roadmap

## v1.0 (Current)

- [x] Synthetic transaction generator (7 fraud patterns)
- [x] Polars feature engineering pipeline
- [x] Heterogeneous graph builder (5 node types, 7 edge types)
- [x] XGBoost tabular baseline
- [x] GraphSAGE encoder and classifier
- [x] GAT encoder and classifier
- [x] GraphSAGE/GAT + XGBoost hybrid ensemble
- [x] Platt and isotonic calibration
- [x] Conformal prediction (3-class triage)
- [x] SHAP feature importance
- [x] Counterfactual analysis
- [x] Nearest neighbor analogs
- [x] Plain-English audit reports
- [x] DuckDB decision recorder
- [x] Deterministic replay engine
- [x] Decision lineage tracking
- [x] PII redaction
- [x] Bulk export (markdown/JSON)
- [x] FastAPI server
- [x] Typer CLI
- [x] Docker support
- [x] Comprehensive test suite

## v1.1 (Near-term)

- [ ] LightGBM as alternative booster in hybrid
- [ ] Graph motif features (triangles, centrality)
- [ ] PDF export for audit reports
- [ ] MLflow model registry integration
- [ ] Windowed graph evaluation in experiments
- [ ] Notebook-based experiment dashboards

## v2.0 (Medium-term)

- [ ] Temporal Graph Attention Network (TGAT) encoder
- [ ] Time-aware edge embeddings
- [ ] Fair conformal prediction (group coverage)
- [ ] Fairness audit module (`rift fairness-audit`)
- [ ] Streaming prediction mode
- [ ] Kubernetes deployment manifests

## Not Planned

The following are explicitly out of scope to maintain credibility:
- Blockchain or ZK proofs
- Autonomous agents
- Federated learning
- Deepfake detection
- Kafka (until streaming is needed)
- RL as the main system
- Quantum computing
