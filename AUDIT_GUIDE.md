# Rift Audit Guide

*For non-technical auditors, compliance reviewers, and risk teams*

## What the system does

Rift is a fraud detection system that scores transactions and produces **auditable decisions**. Every decision is recorded like a receipt: you can later re-run the exact same inputs through the same model and get the same outcome. This supports compliance, governance, and dispute resolution.

## What a Decision ID means

Each prediction gets a unique **Decision ID** — a 32-character hash derived from the transaction, features, and model outputs. Think of it as a receipt number. You can use this ID to:

- Retrieve the full audit report
- Replay the decision to verify it reproduces
- Export for compliance archives

## How to replay a decision

From the command line:

```bash
rift replay <decision_id>
```

This loads the stored transaction and features, runs the same model, and confirms you get the same score, confidence band, and explanation. If anything differs, the replay reports a mismatch.

## What "high confidence fraud" means

The model assigns each transaction to one of three bands:

- **High confidence fraud**: The system is highly confident this is fraudulent. Recommend blocking and investigation.
- **Review needed**: Uncertainty is too high to auto-decide. Recommend manual review.
- **High confidence legit**: The system is highly confident this is legitimate. No review needed.

## What "review needed" means

When the model is unsure (e.g., score near 0.5, or conflicting signals), it flags the transaction for **manual review**. This avoids both false blocks and false approves when the model cannot decide confidently.

## What factors go into an explanation

Each audit report lists:

- **Top drivers**: Features that most influenced the score (e.g., new device, geo distance).
- **Counterfactual**: What would need to change to flip the decision.
- **Similar cases**: How many of the most similar past transactions were fraudulent.

## How personal details are redacted

In audit exports, sensitive fields (user ID, device ID, location coordinates, etc.) are replaced with `[REDACTED]` to support privacy-safe sharing with auditors.

## Sample report

```markdown
# Rift Audit Report

**Decision ID:** a1b2c3d4...
**Generated:** 2024-06-15T14:30:00Z

## Decision
Flagged as fraud (high confidence)

**Calibrated Score:** 0.87
**Confidence Level:** 0.75

## Top Factors
new_device (0.32), geo_jump (0.21), merchant_fraud_rate (0.18)

## Reviewer Recommendation
Recommend blocking and manual investigation.

## Replay
To verify: rift replay a1b2c3d4...
```

---

> Rift records every model decision like a receipt. That receipt can be replayed later to verify the same outcome and explanation. This helps risk teams, auditors, and compliance reviewers understand what happened without needing to read model code.
