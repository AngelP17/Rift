FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY docs/templates/ docs/templates/
COPY demo/ demo/

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir pyarrow rich python-dotenv structlog shap && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

ENV PYTHONPATH=/app/src
ENV RIFT_HOME=/app/.rift
ENV RIFT_STORAGE_BACKEND=local

RUN mkdir -p /app/.rift/data /app/.rift/runs /app/.rift/governance

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "-m", "rift.cli.main"]
CMD ["serve"]
