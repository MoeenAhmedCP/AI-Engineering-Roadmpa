# 4.6 Deployment and Serving

## Production Setup: Gunicorn + Uvicorn

FastAPI is an ASGI (Asynchronous Server Gateway Interface) framework. In development, `uvicorn main:app --reload` is fine. In production, you need a process manager.

**Why both Gunicorn and Uvicorn?**

- **Uvicorn** is the ASGI server — it handles async I/O efficiently.
- **Gunicorn** is the process manager — it spawns and monitors multiple Uvicorn worker processes, restarts them on crash, and handles graceful shutdown.

The combination gives you multi-core utilisation (Gunicorn spawns N workers, one per CPU core or more) plus async I/O efficiency within each worker.

```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --keep-alive 5 \
  --access-logfile -
```

Worker count rule of thumb: `(2 × CPU_cores) + 1`. For a 2-vCPU machine, use 5 workers.

For LLM applications specifically, workers are largely I/O-bound (waiting for API responses), so you can often run more workers than CPU cores.

---

## Containerising: Multi-Stage Dockerfile

Multi-stage builds keep the final image small by separating the build environment (which needs compilers and dev headers) from the runtime environment (which only needs the compiled artifacts).

**Stage 1 — builder:**
```dockerfile
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y build-essential libpq-dev
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt
```

**Stage 2 — runtime:**
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y libpq5 curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY . /app
WORKDIR /app
RUN useradd -r -u 1001 appuser && chown -R appuser /app
USER appuser
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
EXPOSE 8000
CMD ["gunicorn", "main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

The final image is typically 150-200MB instead of 600MB+ if you used a full Debian base.

**ECR push workflow:**
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_REGISTRY

docker build -t myapp .
docker tag myapp:latest $ECR_REGISTRY/myapp:$GIT_SHA
docker push $ECR_REGISTRY/myapp:$GIT_SHA
```

---

## Environment Variables and Secrets

**ECS Task Definition environment variables** are fine for non-sensitive config (`LOG_LEVEL`, `ENVIRONMENT`, `MAX_WORKERS`).

**AWS Secrets Manager** is the right home for sensitive values (`ANTHROPIC_API_KEY`, `DATABASE_URL`, `SUPABASE_SERVICE_KEY`).

In the ECS task definition, reference secrets as environment variables:
```json
{
  "secrets": [
    {
      "name": "ANTHROPIC_API_KEY",
      "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:myapp/anthropic-key"
    }
  ]
}
```

ECS injects these into the container at startup, so your application reads them via `os.getenv()` — no code change needed.

Never bake secrets into Docker images. Never commit them to version control. The `--build-arg` mechanism should only be used for non-sensitive build-time config.

---

## Auto-Scaling

ECS Service Auto Scaling adjusts the number of running task replicas based on metrics.

**CPU-based scaling** (most common):
```json
{
  "TargetTrackingScalingPolicy": {
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    }
  }
}
```

Scale out when average CPU > 70%; scale in when it drops below 70% (with a cooldown period).

**Request-based scaling**: Use an ALB metric (`ALBRequestCountPerTarget`) to scale on requests per task. Better for LLM workloads where CPU may not reflect actual load (API calls are I/O-bound).

**Custom metrics**: Emit LLM-specific metrics to CloudWatch (queue depth, p95 latency, token usage). Scale on these.

Minimum: 2 tasks (HA). Maximum: set based on your cost budget and upstream API rate limits.

---

## Compute Platform Comparison

| Option | Best for | Cold start | Cost | Ops overhead |
|---|---|---|---|---|
| ECS Fargate | Long-running API services | 30-60s | Medium | Low |
| App Runner | Simple web apps, fast deploys | 5-15s | Medium-high | Very low |
| Lambda | Short tasks, event-driven | 100ms-3s | Very low at low scale | Low |
| EC2 | GPU workloads, open-source models | None | Low at high scale | High |

**Lambda limitations for LLM applications:**
- 15-minute maximum execution time (fine for most requests, but not streaming).
- 512MB-10GB memory (enough for API client apps, not for running models locally).
- Cold start latency with large packages.

**Recommendation:** ECS Fargate for most LLM API applications. App Runner if you want zero infrastructure management and can tolerate slightly higher cost.

---

## Open-Source Models: vLLM and TGI

When you host your own open-source LLM (Llama 3, Mistral, Qwen), you need a high-performance inference server.

**vLLM** (recommended for most cases):
- Implements PagedAttention for high GPU memory efficiency.
- OpenAI-compatible API (`/v1/chat/completions`) — drop-in replacement.
- Continuous batching: fills GPU with requests from multiple users, maximising throughput.
- Supports tensor parallelism across multiple GPUs.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --tensor-parallel-size 1
```

**TGI** (Text Generation Inference, HuggingFace):
- Similar features to vLLM, tightly integrated with the HuggingFace Hub.
- Better support for some exotic model architectures.
- Available as a pre-built Docker image.

---

## GPU Selection Guide

| GPU | VRAM | Models it fits | Use case |
|---|---|---|---|
| T4 (AWS g4dn) | 16GB | 7B in FP16, 13B in INT4 | Dev/testing, low traffic |
| A10G (AWS g5) | 24GB | 13B in FP16, 70B in INT4 | Production inference |
| A100 (AWS p4d) | 40/80GB | 70B in FP16 | High-throughput, quality |
| H100 (AWS p5) | 80GB HBM3 | 70B fast, 405B in INT4 | Maximum throughput |

Rule of thumb: model requires ~2 bytes per parameter in FP16. A 7B model needs ~14GB of VRAM. Quantisation (INT8 = 1 byte/param, INT4 = 0.5 bytes/param) reduces this at the cost of small quality degradation.

---

## Quantisation for Serving

**INT8 (8-bit):**
- 50% memory reduction vs FP16.
- Quality loss: nearly imperceptible for most tasks.
- Supported by bitsandbytes, vLLM, TGI.

**INT4 (4-bit, e.g. GPTQ, AWQ):**
- 75% memory reduction vs FP16.
- Quality loss: visible on complex reasoning tasks.
- Fits a 70B model on a single A100 (80GB).
- Throughput slightly lower than INT8 due to dequantisation overhead.

**When to quantise:** Always for serving (latency and memory benefits outweigh tiny quality loss). Never for fine-tuning base training (use mixed precision FP16/BF16 instead).

---

## Health Checks

Every production service must expose `/health`:

```python
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}
```

ECS, App Runner, and Kubernetes all use health checks to route traffic and restart unhealthy tasks.

**Liveness probe**: Is the process alive? Restart if not. (`/health` — returns 200 if the process is running)

**Readiness probe**: Is the service ready to accept traffic? Remove from load balancer if not. (`/ready` — returns 200 only when the model is loaded and the DB connection is available)

For LLM API applications, the readiness check should verify the API key is valid (make a cheap test call at startup) and the database is reachable.

---

## Zero-Downtime Deployment

**Rolling update** (ECS default):
- ECS replaces tasks one at a time.
- Minimum healthy percent: 100% (never take any task down before a replacement is healthy).
- Maximum percent: 200% (run double the tasks during the deploy).
- New tasks are health-checked before old ones are terminated.

**Blue-green deployment:**
- Spin up a completely separate "green" ECS service behind a new target group.
- Run your test suite against the green environment.
- Switch the ALB listener rule to point to green.
- Keep blue running for 10-15 minutes as a rollback option.
- Terminate blue.

Blue-green is slower and more expensive but offers instant rollback and zero in-flight request drops. Prefer it for high-stakes production services.
