# Audit Guide

Rift records every model decision like a receipt. That receipt can be replayed later to verify the same outcome and explanation without needing to inspect model code.

## What the system does

Rift scores a transaction for fraud risk, calibrates that score so it behaves more like a usable probability, and then assigns one of three operational outcomes:

- `high_confidence_fraud`
- `review_needed`
- `high_confidence_legit`

## What a decision ID means

Each prediction is stored with a deterministic SHA-256 decision ID. The ID is computed from a canonical JSON record of the transaction payload, model run, prediction, and explanation.

## How to replay a decision

Use the CLI:

```bash
rift replay <decision_id>
```

Replay fetches the stored transaction, features, model references, and prediction record from DuckDB and verifies the stored output.

## What "confidence" means

Confidence in Rift is not just the raw model score. It combines:

1. a calibrated probability, and
2. a conformal prediction band that indicates whether the system is confident enough to make a direct decision.

## What "review needed" means

`review_needed` means the model did not have enough evidence to place the transaction confidently into a single class. The transaction should be routed to manual review rather than silently approved or blocked.

## How personal data is redacted

The MVP stores synthetic data by default. When real data is used, personal information should be redacted before export. The audit export functions are structured so sensitive fields can be filtered before they leave the audit store.

## Example report

An audit report includes:

- decision ID
- decision time
- outcome
- calibrated fraud probability
- top drivers
- a plain-English explanation
- replay instructions

Example wording:

> This transaction was flagged because it came from a new device, was far from the user's recent activity centroid, and followed an elevated transaction velocity window. The system is confident enough to recommend manual review or intervention depending on policy.
