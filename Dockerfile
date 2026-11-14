# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# =============================================================================
# Stage 2: Production Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Create non-root system user for security compliance
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser

# Copy application code, configuration, SQL migrations, and trained models
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup configs/ /app/configs/
COPY --chown=appuser:appgroup sql/ /app/sql/
COPY --chown=appuser:appgroup models/ /app/models/
COPY --chown=appuser:appgroup README.md /app/README.md

# Ensure runtime directories exist with appropriate permissions
RUN mkdir -p /app/data /tmp/matplotlib && \
    chown -R appuser:appgroup /app/data /tmp/matplotlib

USER appuser

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--access-log"]

