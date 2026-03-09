# Rift Audit Guide

This guide explains Rift's decision system in plain language. It is written for risk teams, auditors, compliance reviewers, and anyone who needs to understand how Rift makes and records fraud decisions without reading model code.

---

## What the System Does

Rift analyzes financial transactions to detect potential fraud. For each transaction, it produces:

- A **risk score** between 0 and 1 (higher means more likely fraud)
- A **confidence level** indicating how certain the system is
- A **recommendation** (block, review, or approve)
- A **plain-English explanation** of why the decision was made

## What a Decision ID Is

Every time Rift evaluates a transaction, it creates a unique **Decision ID** (e.g., `DEC_A1B2C3D4E5F6`). This ID is like a receipt number. It permanently links the transaction, the risk score, the explanation, and all the model versions used at that moment.

You can use a Decision ID to:
- Look up what happened
- Replay the decision to verify it produces the same result
- Generate an audit report

## How to Replay a Decision

Replaying means re-running the exact same computation to verify the outcome has not changed. This is important for regulatory compliance and internal reviews.

```bash
rift replay DEC_A1B2C3D4E5F6
```

Or via API:

```
GET /replay/DEC_A1B2C3D4E5F6
```

The replay system will:
1. Retrieve the original transaction data
2. Load the exact model and calibration artifacts that were used
3. Re-run the prediction
4. Compare the result with the stored decision
5. Report whether they match

**A match means the decision is deterministically reproducible.**

## What "Confidence" Means

Rift does not just say "fraud" or "not fraud." It assigns one of three confidence levels:

| Level | What It Means | Typical Action |
|---|---|---|
| **High Confidence Fraud** | The system is very sure this is fraudulent | Block and investigate |
| **Review Needed** | The system is uncertain | Send to an analyst for manual review |
| **High Confidence Legitimate** | The system is very sure this is normal | Approve automatically |

These levels are determined by a statistical method called **conformal prediction** that provides a mathematical guarantee: the system's confidence bands cover the true outcome at least 95% of the time.

## What "Review Needed" Means

"Review needed" means the system detected mixed signals. For example:
- The amount is unusual, but the device and location are normal
- The merchant is high-risk, but the user has shopped there before
- The transaction matches some fraud patterns but not enough for high confidence

When you see "Review needed," it means a human analyst should look at the case and the explanation before making a final decision.

## What Factors Go Into an Explanation

Each explanation is built from several layers:

1. **Feature importance (SHAP):** Which measurable factors most influenced the risk score. For example: "Transaction amount was 5 standard deviations above user average" or "Device is shared by 3 users."

2. **Similar past cases:** The system finds historical transactions that looked similar and reports whether they turned out to be fraudulent or legitimate.

3. **Counterfactual analysis:** What would have needed to be different for the decision to change. For example: "If the amount had been under $200, the decision would have been 'approve.'"

4. **Narrative summary:** All of the above is combined into a plain-English paragraph.

## How Personal Details Are Redacted

When generating reports for external review, Rift can automatically redact personally identifiable information (PII):

- User IDs are replaced with `[REDACTED]`
- Device IDs are replaced with `[REDACTED]`
- Account IDs are replaced with `[REDACTED]`
- Geographic coordinates are replaced with `[REDACTED]`

To generate a redacted report, the audit export system automatically applies redaction rules before output.

## Sample Report

```
# Audit Report: DEC_A1B2C3D4E5F6

**Decision Time:** 2024-03-15T14:30:00
**Outcome:** FLAGGED AS FRAUD
**Confidence Level:** High
**Calibrated Score:** 0.9231

## Summary

This transaction was flagged as likely fraudulent. The calibrated risk
score is 92.31%. Key factors: Location was 487.3 km from the user's
usual area (increased risk); User made 12 transactions in the last
hour (increased risk); Transaction amount was $4,523.00 (increased risk).

## Top Risk Drivers

1. Location was 487.3 km from the user's usual area (increased risk)
2. User made 12 transactions in the last hour (increased risk)
3. Transaction amount was $4,523.00 (increased risk)
4. Device is shared by 3 users (increased risk)
5. Merchant has a 8.5% historical fraud rate (increased risk)

## Similar Historical Cases

- Transaction TX_89A2BC (fraudulent, similarity: 0.94)
- Transaction TX_F123DE (fraudulent, similarity: 0.89)
- Transaction TX_456789 (legitimate, similarity: 0.87)

## Counterfactual Analysis

The decision would change if: dist_from_centroid decreased,
tx_count_1h decreased, amount decreased.

## Recommendation

Block transaction and escalate to fraud investigation team.

## How to Replay This Decision

Run: rift replay DEC_A1B2C3D4E5F6
```

---

Rift records every model decision like a receipt. That receipt can be replayed later to verify the same outcome and explanation. This helps risk teams, auditors, and compliance reviewers understand what happened without needing to read model code.
