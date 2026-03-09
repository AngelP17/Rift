#!/usr/bin/env bash
set -euo pipefail

python3 -m pytest -q
python3 -m rift.cli.main generate --txns 1200 --users 120 --merchants 50 --fraud-rate 0.06
python3 -m rift.cli.main train --model graphsage_xgb --time-split --optimize green
python3 -m rift.cli.main fairness audit --sensitive-column channel
python3 -m rift.cli.main monitor drift --reference-path .rift/data/transactions.parquet --current-path .rift/data/transactions.parquet --threshold 0.2
python3 -m rift.cli.main governance generate-card
python3 -m rift.cli.main lakehouse build
python3 -m rift.cli.main query --natural "show recent flagged transactions"
