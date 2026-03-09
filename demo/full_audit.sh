#!/bin/bash
# Full Rift demo flow
set -e
echo "=== Rift Demo ==="
echo "1. Generating synthetic data..."
rift generate --txns 5000 --fraud-rate 0.02
echo "2. Training model..."
rift train --model graphsage_xgb --time-split
echo "3. Predicting (requires sample_transaction.json)..."
rift predict demo/sample_transaction.json || true
echo "4. Export audits..."
rift export --since 90d --format markdown || true
echo "Done."
