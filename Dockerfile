# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools, then Python deps into a prefix we can copy cleanly
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder (keeps final image clean)
COPY --from=builder /install /usr/local

# Copy application code
COPY api/       ./api/
COPY src/       ./src/
COPY configs/   ./configs/

# Copy fine-tuned model weights (swap this path for a better checkpoint)
COPY models/final/ ./models/final/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Configuration
ENV MODEL_DIR=/app/models/final/step4-finetune-quick-cpu \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
