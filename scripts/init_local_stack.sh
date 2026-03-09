#!/usr/bin/env bash
set -euo pipefail

echo "Starting local zero-cost stack..."
docker compose up -d

if command -v mc >/dev/null 2>&1; then
  mc alias set local http://localhost:9000 minioadmin minioadmin
  if ! mc ls local/rift-data >/dev/null 2>&1; then
    mc mb local/rift-data
  fi
  echo "MinIO bucket rift-data is ready."
else
  echo "MinIO client 'mc' not found; skipping bucket bootstrap."
fi

echo "Airflow UI: http://localhost:8080"
echo "MinIO console: http://localhost:9001"
echo "MLflow UI: http://localhost:5000"
