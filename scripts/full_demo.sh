#!/usr/bin/env bash
# Rift Full Demo – Big Four–aligned end-to-end workflow
# Run from project root after: pip install -e .
set -e

RIFT_CMD="${RIFT_CMD:-rift}"
if ! command -v rift &> /dev/null; then
  RIFT_CMD="python3 -m rift.cli.main"
fi

echo "=== Rift Full Demo ==="
echo "Using: $RIFT_CMD"
echo ""

echo "1. Generating synthetic data..."
$RIFT_CMD generate --txns 5000 --fraud-rate 0.02
echo ""

echo "2. Training model (graphsage_xgb, time-split)..."
$RIFT_CMD train --model graphsage_xgb --time-split
echo ""

echo "3. Single transaction prediction..."
if [ -f demo/sample_transaction.json ]; then
  $RIFT_CMD predict demo/sample_transaction.json 2>/dev/null || echo "   (Prediction stub; integrate with trained model for full flow)"
else
  echo "   Skipped (demo/sample_transaction.json not found)"
fi
echo ""

echo "4. Export audit reports..."
$RIFT_CMD export --since 90d --format markdown 2>/dev/null || echo "   No audits to export yet"
echo ""

echo "5. Generating model card..."
$RIFT_CMD governance generate-card --output model_card_latest.md 2>/dev/null || echo "   (run after training)"
echo ""

echo "6. Documentation locations:"
echo "   - README.md (executive summary, quick start)"
echo "   - AUDIT_GUIDE.md (plain-language guide for auditors)"
echo "   - docs/GOVERNANCE.md (ethics, risk, compliance)"
echo "   - docs/COMPLIANCE_MAPPINGS/ (EU AI Act, NIST AI RMF)"
echo "   - docs/MODEL_CARDS/ (model metadata)"
echo "   - CHANGELOG.md (version history)"
echo ""

echo "Demo complete."
