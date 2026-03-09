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

## Pull requests

Please include:

- a summary of the behavior change;
- tests for new functionality;
- notes about any artifact or schema changes;
- screenshots or example output for audit/reporting changes when relevant.
