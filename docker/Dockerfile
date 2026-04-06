# ── MegatronBridge API Server ─────────────────────────────────────────────────
# CPU-only FastAPI process. No CUDA, no megatron.bridge import.
# The GPU worker runs as a separate container (Dockerfile.worker).
FROM python:3.12-slim AS base

LABEL org.opencontainers.image.title="megatronbridge-api" \
      org.opencontainers.image.description="REST/WebSocket API for Nvidia MegatronBridge" \
      org.opencontainers.image.source="https://github.com/your-org/megatronbridge-api"

# Security: run as non-root
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps — separate layer for caching ─────────────────────────────────
COPY pyproject.toml ./
# Install only the base (no gpu extras) so CUDA libs are never pulled in
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# ── Application code ──────────────────────────────────────────────────────────
COPY app/ ./app/
COPY migrations/ ./migrations/

# Data directories owned by appuser
RUN mkdir -p /data/logs /data/checkpoints /data/hf_cache && \
    chown -R appuser:appgroup /data /app

USER appuser

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/health || exit 1

EXPOSE 8000

ENV DATA_DIR=/data \
    DATABASE_URL=sqlite+aiosqlite:///megatronbridge.db \
    CHECKPOINTS_ROOT=/data/checkpoints \
    LOGS_ROOT=/data/logs \
    HF_CACHE_DIR=/data/hf_cache \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info"]
