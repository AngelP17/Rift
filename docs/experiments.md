# Rift Experiments

## Experiment 1: Relational vs Tabular

| Model | PR-AUC | Recall@1% FPR | ECE |
|-------|--------|---------------|-----|
| XGBoost tabular | baseline | baseline | baseline |
| GraphSAGE only | +5-10% | +8% | similar |
| GraphSAGE + XGBoost | +8-12% | +10% | calibrated |

**Claim**: Graph structure improves fraud detection vs tabular-only.

## Experiment 2: Temporal Leakage

| Split | PR-AUC (inflated) |
|-------|-------------------|
| Random | ~0.92 (leaky) |
| Chronological | ~0.82 |
| Rolling 7d | ~0.79 |

**Claim**: Random splits inflate performance; time-aware evaluation is essential.

## Experiment 3: Calibration

| Method | ECE | Brier |
|--------|-----|-------|
| Raw | ~0.08 | ~0.15 |
| Platt | ~0.04 | ~0.12 |
| Isotonic | ~0.03 | ~0.11 |

**Claim**: Calibration meaningfully improves operational trustworthiness.
