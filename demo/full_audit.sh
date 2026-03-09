#!/bin/bash
set -e

echo "=== Rift Full Audit Demo ==="
echo ""

export PYTHONPATH=src

echo "Step 1: Generate synthetic transactions..."
python -m cli.main generate --txns 10000 --fraud-rate 0.03 --seed 42

echo ""
echo "Step 2: Train hybrid model..."
python -m cli.main train --model graphsage_xgb --time-split --seed 42

echo ""
echo "Step 3: Run prediction on sample transaction..."
python -m cli.main predict --tx demo/sample_transaction.json

echo ""
echo "Step 4: Export decisions..."
python -m cli.main export --since 90d --format markdown

echo ""
echo "=== Demo Complete ==="
echo "Use 'rift replay <decision_id>' to replay any decision."
echo "Use 'rift audit <decision_id>' to generate an audit report."
