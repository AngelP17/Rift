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

## Documentation policy

Documentation changes are required whenever shipped behavior changes.

Please update the relevant Markdown files when you change:

- CLI commands or flags;
- API endpoints or payloads;
- audit or replay behavior;
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
