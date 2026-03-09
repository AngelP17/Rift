#!/usr/bin/env bash
set -euo pipefail

python3 -m pytest -q
python3 -m rift.cli.main pipeline run --txns 1000 --users 100 --merchants 40 --fraud-rate 0.05 --model graphsage_xgb --sample-tx demo/sample_transaction.json
python3 -m rift.cli.main governance generate-card
python3 -m rift.cli.main monitor drift --reference-path .rift/data/transactions.parquet --current-path .rift/data/transactions.parquet
python3 -m rift.cli.main lakehouse build
python3 -m rift.cli.main lakehouse query --sql "select count(*) as transaction_count from transactions"
python3 -m rift.cli.main storage status
python3 -m rift.cli.main fairness status
python3 -m rift.cli.main federated status
python3 -m rift.cli.main query --natural "show recent flagged transactions"

if python3 - <<'PY'
from rift.compute.spark_compat import spark_available
raise SystemExit(0 if spark_available() else 1)
PY
then
  python3 -m rift.cli.main spark summary --data-path .rift/data/transactions.parquet
else
  echo "Spark not available; skipping spark summary."
fi
