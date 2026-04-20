# Phase 4: Production AI Systems

> "This is where most tutorials stop and where real jobs begin."

Every course teaches you to build a chatbot. Almost none teach you what happens the week after you ship it: the hallucinations users find, the cost bill that appears, the latency complaints in Slack, the compliance audit email. This phase covers the operational engineering that separates hobby projects from production systems.

---

## What This Phase Covers

| Section | Topic | Core Skill |
|---|---|---|
| 4.1 | Evaluation | Measuring quality without ground truth |
| 4.2 | Observability | Seeing inside your LLM calls at scale |
| 4.3 | Cost Optimization | Controlling the token bill |
| 4.4 | Latency & Streaming | Making AI feel fast |
| 4.5 | Guardrails & Safety | Blocking bad inputs and outputs |
| 4.6 | Deployment | Shipping reliably to production |
| 4.7 | Caching | Never pay for the same LLM call twice |
| 4.8 | Data Privacy | GDPR, PII, and compliance |

---

## Progress Checklist

### 4.1 Evaluation
- [ ] Read notes.md — understand the eval problem
- [ ] Run examples.py — see a working eval suite
- [ ] Complete exercises.py — build judge prompts and regression checks
- [ ] Milestone: eval suite with regression gate

### 4.2 Observability
- [ ] Read notes.md — understand what to trace and why
- [ ] Run examples.py — structured logger and analytics
- [ ] Milestone: every LLM call emits a structured JSON log line

### 4.3 Cost Optimization
- [ ] Read notes.md — cost math with real numbers
- [ ] Run examples.py — cost calculator and query router
- [ ] Complete exercises.py — caching savings and anomaly detection
- [ ] Milestone: model router that cuts costs 60%+ on mixed workloads

### 4.4 Latency & Streaming
- [ ] Read notes.md — TTFT, async patterns, background jobs
- [ ] Run examples.py — streaming endpoint and concurrent calls
- [ ] Milestone: streaming endpoint with SSE and job queue

### 4.5 Guardrails & Safety
- [ ] Read notes.md — injection, PII, hallucination
- [ ] Run examples.py — full validation pipeline
- [ ] Complete exercises.py — circuit breaker and content policy
- [ ] Milestone: input/output validation layer wrapping every LLM call

### 4.6 Deployment
- [ ] Read notes.md — Gunicorn/Uvicorn, ECS, zero-downtime
- [ ] Study Dockerfile — understand every line
- [ ] Study github-actions-deploy.yml — CI/CD pipeline
- [ ] Milestone: containerized app with automated deploy on push

### 4.7 Caching
- [ ] Read notes.md — two-layer cache strategy
- [ ] Run examples.py — exact + semantic cache simulation
- [ ] Milestone: cache layer reducing LLM calls by 30%+

### 4.8 Data Privacy
- [ ] Read notes.md — GDPR, PII in AI, compliance requirements
- [ ] Run examples.py — GDPR logger and consent tracker
- [ ] Milestone: privacy-safe logging with right-to-erasure support

---

## How to Use This Phase

Each section has three files:
- **notes.md** — Read this first. Concepts, tradeoffs, real numbers.
- **examples.py** — Runnable code. No API keys needed. Run it, read it, modify it.
- **exercises.py** — Blank stubs with docstrings. Implement the solutions, then check against `solutions()`.

Run any examples file with:
```bash
python phase-4-production/4.X-topic/examples.py
```

---

## The Production Mindset

In tutorials, success means "it worked on my machine."
In production, success means:
- Quality is measurable and doesn't regress
- Costs are predictable and controlled
- Failures are visible before users report them
- The system respects user data and legal requirements
- Deploys happen without downtime

Build all of these habits now, before your first production incident.
