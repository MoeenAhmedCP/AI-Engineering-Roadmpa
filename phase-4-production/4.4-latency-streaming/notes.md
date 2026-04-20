# 4.4 Latency and Streaming

## Why LLMs Are Slow

LLMs generate text one token at a time. Each token requires a full forward pass through the model — on a 70B parameter model that means billions of floating-point operations per token. Even on high-end GPU hardware (H100), that yields 30-80 tokens per second. At 50 tokens/second, a 200-token response takes **4 seconds**.

On top of generation time, you pay:
- **Network round-trip**: 20-100ms to reach the API provider.
- **Queueing**: During traffic spikes, requests wait in a queue before generation starts.
- **Prefill time**: Before the first output token, the model processes the entire prompt (the "prefill" or KV-cache population). A 4000-token prompt can take 300-500ms of prefill.

The net result: a user sends a message and stares at a blank screen for several seconds before seeing any response. That is a terrible user experience.

---

## TTFT: Time to First Token

**Time to First Token (TTFT)** is the most important latency metric for interactive LLM applications. It measures the time from sending the request to receiving the first output token.

TTFT determines perceived latency. A streaming response that shows the first word in 200ms feels fast even if total generation takes 4 seconds — because the user immediately sees that something is happening. A non-streaming response with the same 4s total latency feels slow because the user waits 4s for anything.

**Targets:**
- TTFT < 200ms — excellent (feels instant)
- TTFT 200-600ms — acceptable
- TTFT > 1s — noticeable, degrades UX

Factors affecting TTFT:
- Prompt length (longer prompts = more prefill time)
- Model size and hardware at the provider
- Network distance to the API endpoint (use regional endpoints)
- Whether caching is warm

---

## Streaming: Server-Sent Events (SSE)

Streaming delivers tokens to the client as they are generated instead of waiting for the complete response.

**Server-Sent Events (SSE)** is the standard HTTP mechanism for streaming:
- Uses a persistent HTTP connection.
- Server sends `data: {...}\n\n` frames as tokens arrive.
- Browser's `EventSource` API handles reconnection automatically.
- Works through standard HTTP proxies and load balancers (unlike WebSockets).

SSE frame format:
```
data: {"type": "content_block_delta", "delta": {"text": "Hello"}}

data: {"type": "content_block_delta", "delta": {"text": " world"}}

data: [DONE]

```

FastAPI streaming response:
```python
from fastapi.responses import StreamingResponse
import httpx

async def stream_llm(prompt: str):
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", LLM_API_URL, json=payload) as resp:
            async for chunk in resp.aiter_text():
                yield f"data: {chunk}\n\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        stream_llm(request.prompt),
        media_type="text/event-stream"
    )
```

---

## Async Everything

Synchronous code blocks the event loop while waiting for I/O. One blocking LLM call (4s) blocks all other requests from being processed. With asyncio, thousands of concurrent LLM calls can be in-flight simultaneously on a single thread.

Key rules:
- Use `async def` for all route handlers.
- Use `httpx.AsyncClient` (not `requests`) for HTTP calls.
- Use `asyncio.gather()` for concurrent tasks.
- Never use `time.sleep()` in async code — use `await asyncio.sleep()`.
- Never call synchronous blocking code in an async context (use `asyncio.to_thread()` to wrap it).

---

## asyncio.gather: Concurrent LLM Calls

When you need results from multiple independent LLM calls, gather them concurrently:

```python
import asyncio

async def parallel_analysis(documents: list[str]) -> list[str]:
    tasks = [call_llm(doc) for doc in documents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Handle any exceptions in results
    return [r for r in results if not isinstance(r, Exception)]
```

Sequential approach: N calls × 3s each = 3N seconds.
Concurrent approach: max(3s each) = ~3s total (limited by rate limits and throughput).

Error handling with gather: use `return_exceptions=True` so one failure does not cancel all others.

---

## Background Jobs: Celery + Redis

For long-running AI tasks (document ingestion, bulk analysis, fine-tuning runs), do not make the user wait at all. Use a job queue:

1. API receives request → creates a job record → returns `{"job_id": "abc123"}` immediately.
2. Celery worker picks up the job → runs the LLM task → stores result.
3. Client polls `GET /jobs/abc123` → receives `status: "pending"` until done, then `status: "done"` with the result.

Or use WebSockets / SSE to push progress updates to the client.

**Celery setup:**
```bash
pip install celery redis
celery -A tasks worker --concurrency=4
```

The job ID pattern decouples API response time from task execution time entirely.

---

## Speculative Decoding: Concept

Speculative decoding is a hardware-level optimisation used inside model servers. A small "draft" model generates a speculative sequence of K tokens very quickly. The main (large) model then verifies these K tokens in parallel (which is faster than sequential generation). If the draft was correct, you've generated K tokens for roughly the cost of 1 verification pass.

As an API user you do not control this, but understanding it helps you:
- Appreciate why providers can sometimes serve 70B models faster than expected.
- Know that very short outputs benefit less from speculative decoding than long ones.

---

## Connection Pooling

Creating a new HTTP connection for every LLM request wastes 50-100ms on TLS handshake. Use a persistent `httpx.AsyncClient` with connection pooling:

```python
# Module-level: create once, reuse everywhere
http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    timeout=httpx.Timeout(30.0, connect=5.0),
)

# In your FastAPI lifespan:
@asynccontextmanager
async def lifespan(app):
    yield
    await http_client.aclose()
```

The Anthropic Python SDK manages its own client internally; the pool matters most when you are calling the API directly or using custom HTTP layers.

---

## Timeouts: Always Set Them

LLM API calls can hang. A provider outage, network partition, or extremely long prompt can cause a request to block indefinitely. Always set timeouts:

```python
timeout = httpx.Timeout(
    connect=5.0,    # max seconds to establish connection
    read=60.0,      # max seconds to receive a response chunk
    write=10.0,     # max seconds to send the request
    pool=5.0,       # max seconds waiting for a pool connection
)
```

Pair timeouts with exponential backoff retries:

```python
for attempt in range(3):
    try:
        response = await call_llm_with_timeout(prompt)
        break
    except httpx.TimeoutException:
        wait = 2 ** attempt   # 1s, 2s, 4s
        await asyncio.sleep(wait)
```

---

## Latency Budget: Target Numbers

For a production chat application:

| Stage | Target |
|---|---|
| Client → API server | < 50ms |
| Queue + prefill | < 150ms |
| TTFT (end-to-end) | < 200ms |
| Full response (streaming) | < 3s at p50 |
| Full response (streaming) | < 8s at p95 |

Measure TTFT and total latency for every request. Log them. Build a p50/p95/p99 dashboard. A p99 of 15s means 1 in 100 users has a terrible experience — investigate and fix.

```python
import time

start = time.perf_counter()
first_token_time = None

async for chunk in stream_response(prompt):
    if first_token_time is None:
        first_token_time = time.perf_counter() - start
    process(chunk)

total_time = time.perf_counter() - start
log_latency(ttft=first_token_time, total=total_time)
```
