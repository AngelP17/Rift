FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY demo/ demo/

RUN pip install --no-cache-dir -e ".[dev]"

ENV PYTHONPATH=/app/src
ENV RIFT_AUDIT_DB=/app/data/audit.duckdb
ENV RIFT_MODEL_DIR=/app/artifacts/models
ENV RIFT_DATA_DIR=/app/data

RUN mkdir -p /app/data /app/artifacts/models

EXPOSE 8000

CMD ["python", "-m", "cli.main", "serve", "--host", "0.0.0.0", "--port", "8000"]
