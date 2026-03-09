# Rift Governance Features

Rift includes local, open-source governance tooling intended for compliance-oriented reviews without requiring paid managed platforms.

## Included artifacts

- fairness audit reports
- drift monitoring reports
- generated model cards
- governance summary documents
- replayable audit reports

## Template sources

- `docs/templates/model_card.md.j2`
- `docs/templates/governance_summary.md.j2`

## Generated outputs

- `.rift/governance/fairness/`
- `.rift/governance/drift/`
- `.rift/governance/model_cards/`

## Example commands

```bash
rift fairness audit --sensitive-column channel
rift monitor drift --reference-path .rift/data/transactions.parquet --current-path .rift/data/transactions.parquet
rift governance generate-card
```

## Notes

- The default path remains local-first and zero-cost.
- Optional integrations such as local Ollama summarization or Alibi-Detect drift scoring should remain optional enhancements rather than hard requirements.
