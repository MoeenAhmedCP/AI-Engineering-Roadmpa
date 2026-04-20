# 4.2 Observability: Seeing Inside Your LLM System at Scale

## Why Standard Logging Is Not Enough

In a traditional web API, a log line like `INFO 2026-04-19 GET /api/users 200 45ms` tells you almost everything you need. You know the endpoint, the status, and the time.

For an LLM API call, that log line tells you almost nothing. What was the prompt? Did it include a relevant system prompt? Did the user's message trigger something unusual? How many tokens did it use? How much did it cost? Did the model time out or return a hallucination? Was there a retry? Standard HTTP logging does not capture any of this.

Observability for LLM systems means capturing the full context of every AI call: what went in, what came out, how long it took, what it cost, and whether anything went wrong. Without this, you are flying blind. When a user reports "the AI gave me wrong information," you will have no way to investigate. When costs spike 3x, you will not know why. When latency degrades, you will not know which calls are slow.

---

## What to Trace Per LLM Call

Every LLM call should emit a structured record with these fields:

| Field | Type | Notes |
|---|---|---|
| `request_id` | UUID | Unique per call; tie logs across services |
| `user_id` | string | SHA-256 hashed — never store raw user ID in logs |
| `model` | string | `claude-sonnet-4-6`, `gpt-4o`, etc. |
| `prompt_preview` | string | First 200 chars only — enough for debugging, not so much it's a PII risk |
| `response_preview` | string | First 200 chars of response |
| `input_tokens` | int | From the model's usage response |
| `output_tokens` | int | From the model's usage response |
| `cost_usd` | float | Computed from token counts × price table |
| `latency_ms` | int | Wall-clock time from request to first token (TTFT) or full response |
| `error_type` | string | `rate_limit_error`, `timeout`, `context_length_exceeded`, `null` |
| `timestamp` | ISO8601 | UTC always |
| `session_id` | string | Groups a conversation's calls together |

**Critical privacy note:** Never log the full prompt or response. A 200-character preview is sufficient for debugging. Full prompt logs create GDPR compliance risks (they may contain PII the user typed) and increase storage costs significantly.

---

## LangSmith Setup

LangSmith is LangChain's observability platform. If your stack uses LangChain or LangGraph, enabling tracing is one line:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT=my-project
```

That is it. Every LangChain call automatically traces to the LangSmith dashboard. The dashboard shows:
- Full run tree (which chains called which sub-chains)
- Token counts and cost per run
- Latency breakdown per step
- Error visualization
- A/B comparison of prompt versions
- Annotation tools for human review

The limitation: deep LangChain coupling. If you switch frameworks, you lose the integration.

---

## Langfuse

Langfuse is open-source and self-hostable, making it the preferred choice when you cannot send data to a third party (HIPAA, GDPR, enterprise requirements). The SDK works with any Python code — no framework required.

```python
from langfuse import Langfuse
langfuse = Langfuse()

trace = langfuse.trace(name="document-analysis")
span = trace.span(name="llm-call", input={"prompt": prompt})
# ... call your LLM ...
span.end(output={"response": response}, usage={"input": 1200, "output": 400})
```

Self-host on a $20/month VPS with Docker Compose. Your traces stay on your infrastructure.

---

## Helicone

Helicone works differently from LangSmith and Langfuse: it's a proxy. Instead of changing your SDK code, you change one URL:

```python
# Before:
client = anthropic.Anthropic(base_url="https://api.anthropic.com")

# After (all calls now logged through Helicone):
client = anthropic.Anthropic(base_url="https://anthropic.helicone.ai")
```

This is the fastest path to observability on an existing codebase — a one-line change. The tradeoff: your traffic routes through Helicone's servers, which adds ~10-20ms latency and means data goes to their infrastructure.

---

## Structured JSON Logging

Even without a managed platform, structured JSON logs enable powerful analysis:

```json
{
  "timestamp": "2026-04-19T14:32:01Z",
  "request_id": "req_8f3a2c",
  "user_id": "sha256:a3f4b2...",
  "model": "claude-sonnet-4-6",
  "input_tokens": 1847,
  "output_tokens": 312,
  "cost_usd": 0.00638,
  "latency_ms": 2340,
  "error_type": null
}
```

JSON logs are queryable in CloudWatch Insights, Datadog, Splunk, and any ELK stack. This means you can run queries like: "What was the p95 latency for Claude Sonnet calls last 7 days?" or "Which users have spent more than $5 in the last 24 hours?"

---

## Key Dashboards to Build

Once you have structured logs, build these dashboards immediately:

**Latency Dashboard:**
- p50, p95, p99 latency by model
- Latency trend over time (is it getting slower?)
- Slowest 10 calls of the day

**Cost Dashboard:**
- Cost per day (with 7-day moving average)
- Cost per user (identify heavy users)
- Cost breakdown by model
- Projected monthly cost based on today's rate

**Error Dashboard:**
- Error rate by type (rate limit vs timeout vs context length)
- Error trend over time
- Which users or features trigger the most errors

**Quality Dashboard:** (if you have eval scores)
- Mean eval score over time
- Score distribution
- Score by query type

---

## Alerting

Set these alerts immediately when you go to production:

- **Cost spike:** Alert if today's cost > 2x the 7-day average. Prevents surprise bills.
- **Error rate:** Alert if error rate > 5% in any 15-minute window. Usually means a rate limit or service issue.
- **Latency:** Alert if p99 latency > 30 seconds. Users are timing out.
- **Anomalous user:** Alert if any single user generates > $10/day. Probably abuse.

---

## Trace IDs Across Distributed Services

In a real system, an LLM call may be triggered by a chain: API gateway → backend → retrieval service → LLM service. Standard logging makes this invisible. Distributed tracing propagates a single `trace_id` through every service call.

Pattern: generate a UUID at the API boundary, pass it as an HTTP header (`X-Trace-ID`) to every downstream service call, include it in every log line. Now a single search for the trace ID shows the complete request lifecycle across all services.
