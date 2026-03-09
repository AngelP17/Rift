# Contributing to Rift

## Development setup

```bash
python3 -m pip install -e ".[dev]"
pytest
```

## Project principles

- keep the pipeline deterministic where possible;
- prefer explicit data contracts over implicit assumptions;
- preserve replayability for decision records;
- keep audit-facing language understandable to non-technical readers.

## Open-source and zero-cost policy

Rift's core path must remain open-source and zero-cost to run.

That means:

- do not introduce required paid cloud services for core workflows;
- do not make proprietary SDKs or closed-source platforms mandatory;
- prefer local-first OSS components such as Parquet, DuckDB, FastAPI, Polars, and scikit-learn;
- keep optional integrations optional and document the local fallback path.

When adding cloud-like architecture patterns:

- provide a zero-cost local equivalent first;
- prefer checked-in infrastructure as code and scripts over manual setup steps;
- make sure local validation remains possible without vendor credentials.

When adding governance features:

- prefer generated artifacts over handwritten status notes where possible;
- keep model cards, drift reports, and fairness outputs reproducible from local artifacts;
- ensure governance templates remain usable without hosted model registries.

## Documentation policy

Documentation changes are required whenever shipped behavior changes.

Please update the relevant Markdown files when you change:

- CLI commands or flags;
- API endpoints or payloads;
- audit or replay behavior;
- dashboard behavior;
- dataset adapters or governance workflows;
- storage backends, orchestration, or lakehouse behavior;
- model cards, drift monitoring, sector profiles, or query features;
- setup instructions;
- model or artifact expectations that users rely on.

For diagrams:

- use Mermaid fenced blocks only;
- do not add ASCII art diagrams.

## Pull requests

Please include:

- a summary of the behavior change;
- tests for new functionality;
- notes about any artifact or schema changes;
- screenshots or example output for audit/reporting changes when relevant;
- corresponding doc updates when the change affects users or contributors.
