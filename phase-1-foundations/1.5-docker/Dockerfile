# .dockerignore (put this content in a file named .dockerignore):
# __pycache__/ *.pyc .venv/ venv/ .env .env.* *.pt *.gguf model_weights/ .git/ tests/ *.log

# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# Install build tools needed for some Python packages (e.g., psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer if requirements.txt unchanged
COPY requirements.txt .

# Install into /install so we can copy just this to the runtime stage
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim

# Install only the runtime libs needed (not build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create a non-root user (never run AI apps as root in production)
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app

USER appuser

# Document which port the app listens on
EXPOSE 8000

# Health check — load balancers use this to know if the container is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default environment variables (override with --env-file or ECS task definition)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Gunicorn (process manager) + Uvicorn (ASGI worker) — the production combination
# -w 2: 2 worker processes (adjust to 2× CPU cores)
# -k uvicorn.workers.UvicornWorker: use Uvicorn's async-capable worker class
CMD ["gunicorn", "main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-"]
