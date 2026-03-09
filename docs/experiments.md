# Experiment Plan

## Experiment 1: Relational vs Tabular

**Question:** Does graph structure improve fraud detection?

**Models:**
- A: XGBoost (tabular features only)
- B: GraphSAGE (graph only)
- C: GraphSAGE + XGBoost hybrid

**Metrics:** PR-AUC, Recall@1% FPR, ECE, Brier score

**Protocol:**
1. Generate 100K transactions at 2% fraud rate
2. Temporal split (70/15/15)
3. Train all three models
4. Compare on held-out test set

**Expected outcome:** C > A > B on PR-AUC, demonstrating that the hybrid captures both relational and behavioral signals.

## Experiment 2: Temporal Leakage

**Question:** Do random splits inflate performance metrics?

**Comparison:**
- Random split
- Chronological split
- Rolling-window (7-day windows)

**Protocol:**
1. Train GraphSAGE+XGBoost on each split strategy
2. Compare metrics

**Expected outcome:** Random split shows higher PR-AUC than temporal, confirming information leakage.

## Experiment 3: Calibration

**Question:** Does calibration improve score reliability?

**Comparison:**
- Raw model scores
- Platt-scaled scores
- Isotonic-scaled scores

**Metrics:** ECE, Brier score, reliability curves

**Expected outcome:** Isotonic calibration achieves ECE < 0.05.

## Experiment 4: Conformal Uncertainty

**Question:** Does confidence-banded triage reduce unnecessary manual review?

**Comparison:**
- Binary hard label (fraud/legit at 0.5 threshold)
- Conformal 3-class triage (fraud/review/legit)

**Metrics:** Empirical coverage, average set size, review rate

**Expected outcome:** Coverage near 95%, set size < 1.4, meaningful review rate reduction.

## Experiment 5: Explainability Usability

**Question:** Are plain-English reports more usable than raw SHAP plots?

**Protocol:**
1. Generate 10 audit reports across different confidence bands
2. Score on clarity, actionability, jargon avoidance, factual consistency
3. Optional: have 2-3 non-technical reviewers rate comprehension

**Expected outcome:** Reports rated as clear and actionable by non-ML reviewers.

## Running Experiments

```bash
# Experiment 1: Model comparison
python -m cli.main generate --txns 100000 --fraud-rate 0.02
python -m cli.main train --model xgb_tabular --time-split
python -m cli.main train --model graphsage_only --time-split
python -m cli.main train --model graphsage_xgb --time-split
python -m cli.main compare --metrics pr_auc,recall_at_1pct_fpr,ece

# Experiment 2: Split comparison
python -m cli.main train --model graphsage_xgb
python -m cli.main train --model graphsage_xgb --time-split
python -m cli.main train --model graphsage_xgb --time-split --window 7d
```
