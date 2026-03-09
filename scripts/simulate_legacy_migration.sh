#!/usr/bin/env bash
set -euo pipefail

SOURCE_PATH="${1:-demo/government_transactions.csv}"
OUTPUT_PATH="${2:-.rift/reengineered/legacy_output.parquet}"
SECTOR="${3:-fintech}"

python3 -m rift.cli.main reengineer simulate --source "$SOURCE_PATH" --output-path "$OUTPUT_PATH" --sector "$SECTOR"
