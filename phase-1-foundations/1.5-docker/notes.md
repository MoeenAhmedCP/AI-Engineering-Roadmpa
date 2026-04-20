# 1.5 Docker & Containers

## What Containers Solve

Without containers, deployment means: "it works on my machine" → broken in production because Python version differs, library versions differ, environment variables missing, OS-specific behavior.

A container packages your app + its exact dependencies + its runtime into one portable unit. Same container runs identically on your laptop, a CI server, and AWS.

For AI engineers, this matters more than most because:
- ML libraries (PyTorch, transformers) have complex dependency chains
- Model weights and embedding models need reproducible environments
- AI APIs require secrets that must be managed consistently across environments

---

## Dockerfile — Key Instructions

```dockerfile
FROM python:3.11-slim          # base image (slim = smaller, no extras)
WORKDIR /app                   # all subsequent commands run here
COPY requirements.txt .        # copy first (layer caching — see below)
RUN pip install -r requirements.txt   # install deps
COPY . .                       # copy app code
ENV PORT=8000                  # environment variable with default
EXPOSE 8000                    # document the port (doesn't actually open it)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Layer caching:** Docker caches each instruction. If `requirements.txt` hasn't changed, it reuses the cached pip install layer and only re-runs what changed after it. Always copy requirements.txt before your code.

---

## Multi-Stage Builds

Build dependencies (compilers, build tools) are needed to install packages but not to run them. Multi-stage builds let you install in a "builder" stage and copy only the result into a lean "runtime" image.

```dockerfile
# Stage 1 — builder (temporary, discarded)
FROM python:3.11-slim AS builder
WORKDIR /install
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2 — runtime (final image, shipped)
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local   # copy installed packages only
COPY . .
RUN useradd --no-create-home appuser && chown -R appuser /app
USER appuser                              # never run as root in production
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Result: your final image contains only runtime Python + your app + installed packages. No build tools, no pip cache. Often 50–70% smaller.

---

## .dockerignore

Just like `.gitignore` but for Docker. Prevents unnecessary files from being sent to the build context (makes builds faster, prevents secrets from leaking into the image):

```
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/

# Secrets — CRITICAL: never bake these into an image
.env
.env.*
*.key
credentials.json

# AI model files — too large, mount as volumes instead
*.pt
*.gguf
*.safetensors
model_weights/
checkpoints/

# Dev artifacts
.git/
.pytest_cache/
*.log
tests/

# Data
data/raw/
*.csv
```

---

## Docker Compose for AI Apps

Compose runs multiple containers together with a single command (`docker-compose up`).

```yaml
version: "3.9"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file: .env               # load secrets from .env file
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: ${DB_PASSWORD}   # from .env
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d myapp"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

---

## Environment Variables in Docker

| Method | Use Case | Security |
|---|---|---|
| `ENV KEY=value` in Dockerfile | Non-secret defaults | Not for secrets (baked into image) |
| `--env KEY=value` on `docker run` | Dev/testing | Manual, not scalable |
| `--env-file .env` | Local development | OK for dev, never commit .env |
| ECS task definition env vars | Production | Good, but values visible in task def |
| AWS Secrets Manager + ECS | Production secrets | Best — values never in config files |

**Rule:** Never put API keys in the Dockerfile or docker-compose.yml. Use `.env` locally, Secrets Manager in production.

---

## AI-Specific Gotchas

**Large model files:** Don't COPY model weights into the image. They're gigabytes and would make your image unusable. Instead:
- Download at startup from S3/HuggingFace Hub
- Mount as a Docker volume: `-v /host/models:/app/models`
- Use a sidecar container that pre-downloads and shares a volume

**GPU support:** Requires `nvidia-docker` (NVIDIA Container Toolkit). Run with:
```bash
docker run --gpus all my-ai-app
```

In compose:
```yaml
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Networking between containers:** In a compose network, containers reach each other by **service name**, not `localhost`.
```python
# Wrong in a container:
DB_URL = "postgresql://user:pass@localhost:5432/mydb"
# Right — use service name from docker-compose.yml:
DB_URL = "postgresql://user:pass@db:5432/mydb"
```

**File permissions:** Files created by root inside a container may not be writable by the host. Use `--user $(id -u):$(id -g)` or set a non-root user in your Dockerfile.

---

## Common Commands

```bash
# Build
docker build -t myapp:latest .
docker build --no-cache -t myapp:latest .   # force full rebuild

# Run
docker run -p 8000:8000 --env-file .env myapp:latest
docker run -d --name myapp myapp:latest     # -d = detached (background)

# Debug
docker logs myapp -f                         # stream logs
docker exec -it myapp /bin/bash             # shell inside container
docker inspect myapp                         # detailed container info

# Compose
docker-compose up --build                    # build + start all services
docker-compose down -v                       # stop + remove volumes
docker-compose logs -f app                  # stream app logs

# Cleanup (free disk space)
docker system prune -f                       # remove unused containers, images
docker volume prune -f                       # remove unused volumes
```
