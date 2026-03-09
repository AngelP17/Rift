#!/usr/bin/env bash
set -euo pipefail

docker compose -f docker/jupyterhub.yml up -d
echo "JupyterHub UI: http://localhost:8888"
