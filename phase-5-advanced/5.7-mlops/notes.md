# 5.7 MLOps for AI Systems

## Why MLOps Is Different from DevOps

Traditional software either works or it doesn't. AI systems degrade — model quality can drift, prompts that worked last month may perform poorly today, and new model versions may regress on certain inputs. MLOps is the discipline of making AI systems reliable and improvable over time.

---

## CI/CD for ML/LLM Applications

### What Runs on Every PR

```
PR opened
    │
    ├── Unit tests (< 30 seconds)
    │     Logic functions, input validation, schema tests
    │
    ├── Eval suite (1-5 minutes)
    │     Run 50-100 golden test cases through the pipeline
    │     Score with LLM-as-judge or rule-based checks
    │     Block merge if score drops > 5% from baseline
    │
    ├── Lint + type check (ruff, mypy)
    │
    └── Build Docker image (verify it builds)
```

**Block on quality regression** — not just failing tests. If your LLM-as-judge gives your RAG system 4.2/5 on the baseline and your PR drops it to 3.8/5, that's a regression worth blocking.

### Auto-Deploy on Main

```
Merge to main
    │
    ├── Run full eval suite
    ├── Build and push Docker image to ECR
    ├── Update ECS task definition
    └── Rolling deploy (canary → 100%)
```

---

## Model Registry

Tracks which model version is in production, with full metadata.

**What to store:**
- Model name + version (semantic versioning: v1.2.3)
- Artifact location (S3 path or HuggingFace Hub ID)
- Eval metrics at registration time
- Who registered it, when, and why
- Which version is currently in production

**MLflow Model Registry** is the standard open-source tool. Workflow:
1. Train/fine-tune model
2. `mlflow.log_model()` with metrics
3. Register to registry with `MlflowClient().create_registered_model()`
4. Promote to "Staging" → run eval → promote to "Production"
5. Old production version moves to "Archived"

---

## Deployment Strategies

### Shadow Mode

Run the new model on every real request — but don't serve its output to users. Compare offline.

```
Real Request
    │
    ├── Current model → serve response to user
    │
    └── New model → log response to comparison store
                    (never shown to user)
```

Use when: you want to validate a new model on real traffic before exposing it.

### Canary Deployment

Route a small percentage of traffic to the new model. Watch metrics. Expand gradually.

```
5% → new model   │
95% → old model  │  watch for 24h
                 │  no error spike? → 25% new
                 │  no degradation? → 50% → 100%
                 │  any issue? → rollback to 0%
```

### A/B Testing

Randomly assign users to model A or B. Hold the assignment consistent per user (so they don't see different answers to the same question). Measure:
- Task completion rate
- User satisfaction (thumbs up/down)
- Session length
- Regeneration rate (user clicked "try again")

Run until statistical significance (usually 1-2 weeks, depends on volume).

---

## Prompt Versioning

Prompts are code. Version them like code.

**Bad:** Prompt hardcoded in source code, no history.

**Good:**
```python
# prompts.py
PROMPTS = {
    "support_v1": "You are a helpful customer support agent...",
    "support_v2": "You are a precise, empathetic customer support agent. Always...",
    "support_prod": "support_v2",  # pointer to current production version
}
```

**Better:** Store in database with version numbers, deploy tag, eval score, rollback capability.

```sql
CREATE TABLE prompt_versions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    version INTEGER,
    content TEXT,
    eval_score FLOAT,
    is_production BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Data Versioning

**DVC (Data Version Control):** Git for datasets. `dvc add data/train.csv` tracks the file, stores it in remote storage (S3), and keeps a `.dvc` pointer in Git.

**Vector DB versioning:**
- Tag your index with the embedding model version and ingestion date
- Keep old indexes available for rollback
- Never delete indexes until the new one is validated in production

---

## Feedback Loops

Production usage is your best training signal.

| Signal | What it tells you | How to use it |
|---|---|---|
| Thumbs up/down | User satisfaction per response | Add good responses to eval set; bad ones to failure cases |
| Regeneration ("try again") | User didn't like the response | Implicit negative signal |
| Copy button click | Response was useful | Implicit positive signal |
| Session length | Engagement | Proxy for overall quality |
| Escalation rate | AI couldn't handle it | Identify gaps in knowledge base |

Collect these and run a monthly review: which inputs consistently get poor ratings? Update your RAG knowledge base, improve prompts, or flag for fine-tuning.

---

## Monitoring Stack

| Tool | Role |
|---|---|
| **LangSmith / Langfuse** | Trace every LLM call: prompt sent, response, latency, tokens, cost |
| **Grafana** | Dashboards: latency p50/p95/p99, error rates, cost per day |
| **PagerDuty / OpsGenie** | Alert on: error spike, latency degradation, daily cost anomaly |
| **Sentry** | Application errors (not LLM-specific — Python exceptions) |

**Key metrics to track:**
- Request latency (p50, p95, p99)
- Error rate (4xx vs 5xx)
- LLM cost per day / per user
- Cache hit rate (should be 30-60% in steady state)
- Quality score trend (weekly LLM-as-judge on sample)

---

## MLOps Maturity Model

| Level | Description |
|---|---|
| 0 | Manual: run notebook, copy-paste model, no tests |
| 1 | Automated training: CI pipeline, eval suite, model registry |
| 2 | Automated deployment: canary releases, rollback, monitoring |
| 3 | Closed loop: production signals automatically improve models |

Most companies operate at Level 1-2. Level 3 requires significant infrastructure investment.
