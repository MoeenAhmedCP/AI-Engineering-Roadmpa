"""
4.4 Latency and Streaming — Examples
=======================================
Demonstrates streaming, parallel async LLM calls, latency tracking,
background job queues, and token-bucket rate limiting — no real API calls.

Run: python examples.py
"""

import asyncio
import time
import uuid
import random
from typing import AsyncGenerator, Optional

SIMULATE = True

# ===========================================================================
# 1. simulate_llm_stream — yields words with configurable delay
# ===========================================================================

async def simulate_llm_stream(
    prompt: str,
    delay: float = 0.05,
) -> AsyncGenerator[str, None]:
    """
    Simulates a streaming LLM response by yielding one word at a time
    with `delay` seconds between each token.

    Args:
        prompt: The input prompt (used to determine a canned response).
        delay:  Seconds between tokens (default 0.05 = 50ms, mimics real LLM).

    Yields:
        Individual word tokens.
    """
    response = (
        f"This is a simulated streaming response to your prompt about "
        f"'{prompt[:30]}'. The model generates one token at a time, "
        f"which is why streaming dramatically improves perceived latency "
        f"compared to waiting for the full response."
    )
    words = response.split()
    for word in words:
        await asyncio.sleep(delay)
        yield word + " "


# ===========================================================================
# 2. streaming_endpoint_demo — mimics a FastAPI StreamingResponse
# ===========================================================================

async def streaming_endpoint_demo() -> None:
    """
    Simulates a FastAPI StreamingResponse, printing chunks as they arrive.
    In production this would be a StreamingResponse(media_type="text/event-stream").
    """
    print("  [StreamingResponse] Connection open — tokens arriving:")
    print("  ", end="", flush=True)

    first_token_time: Optional[float] = None
    start = time.perf_counter()
    total_tokens = 0

    async for chunk in simulate_llm_stream("streaming demo", delay=0.03):
        if first_token_time is None:
            first_token_time = time.perf_counter() - start
        print(chunk, end="", flush=True)
        total_tokens += 1

    total_time = time.perf_counter() - start
    print(f"\n\n  TTFT: {first_token_time * 1000:.0f}ms | "
          f"Total: {total_time * 1000:.0f}ms | "
          f"Tokens: {total_tokens}")


# ===========================================================================
# 3. parallel_llm_calls — asyncio.gather pattern
# ===========================================================================

async def _single_llm_call(prompt: str, idx: int, simulate: bool = True) -> str:
    """Simulates a single LLM call with variable latency."""
    if simulate:
        delay = random.uniform(0.1, 0.4)   # 100-400ms simulated latency
        await asyncio.sleep(delay)
        return f"[Response {idx}] Answer to: '{prompt[:40]}' (took {delay*1000:.0f}ms)"
    raise NotImplementedError("Set simulate=True")


async def parallel_llm_calls(prompts: list[str], simulate: bool = True) -> list[str]:
    """
    Runs multiple LLM calls concurrently using asyncio.gather.

    Returns a list of responses in the same order as the input prompts.
    Exceptions are caught per-task so one failure does not cancel the rest.
    """
    tasks = [_single_llm_call(p, i, simulate) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            output.append(f"[ERROR] Prompt {i}: {r}")
        else:
            output.append(r)
    return output


# ===========================================================================
# 4. LatencyTracker — context manager for measuring TTFT and total time
# ===========================================================================

class LatencyTracker:
    """
    Context manager that records TTFT and total response time.

    Usage:
        async with LatencyTracker("chat request") as tracker:
            async for chunk in stream(...):
                tracker.mark_first_token()  # call once on first chunk
        tracker.report()
    """

    def __init__(self, label: str = "request"):
        self.label       = label
        self.start       = 0.0
        self.ttft        = None   # seconds
        self.total       = None   # seconds
        self._first_seen = False

    async def __aenter__(self):
        self.start = time.perf_counter()
        return self

    async def __aexit__(self, *args):
        self.total = time.perf_counter() - self.start

    def mark_first_token(self):
        """Call this when the first token/chunk arrives."""
        if not self._first_seen:
            self.ttft = time.perf_counter() - self.start
            self._first_seen = True

    def report(self) -> dict:
        data = {
            "label":      self.label,
            "ttft_ms":    round(self.ttft * 1000, 1) if self.ttft else None,
            "total_ms":   round(self.total * 1000, 1) if self.total else None,
        }
        status = ""
        if self.ttft:
            if self.ttft < 0.2:
                status = "EXCELLENT"
            elif self.ttft < 0.6:
                status = "ACCEPTABLE"
            else:
                status = "SLOW"
        print(f"  [{data['label']}] TTFT: {data['ttft_ms']}ms | "
              f"Total: {data['total_ms']}ms | {status}")
        return data


# ===========================================================================
# 5. BackgroundJobQueue — submit, poll, and list async jobs
# ===========================================================================

class BackgroundJobQueue:
    """
    Simple in-memory job queue that mimics a Celery/Redis pattern.

    In production: use Celery + Redis or a managed job queue.
    """

    def __init__(self):
        self._jobs: dict[str, dict] = {}

    def submit(self, task_fn, *args, **kwargs) -> str:
        """
        Submit a coroutine task. Returns a job_id immediately.
        The task is scheduled to run in the background.
        """
        job_id = str(uuid.uuid4())[:8]
        self._jobs[job_id] = {"status": "pending", "result": None, "error": None}
        # Schedule the background task without awaiting
        asyncio.ensure_future(self._run(job_id, task_fn, *args, **kwargs))
        return job_id

    async def _run(self, job_id: str, task_fn, *args, **kwargs):
        try:
            self._jobs[job_id]["status"] = "running"
            result = await task_fn(*args, **kwargs)
            self._jobs[job_id]["status"] = "done"
            self._jobs[job_id]["result"] = result
        except Exception as exc:
            self._jobs[job_id]["status"] = "error"
            self._jobs[job_id]["error"]  = str(exc)

    def get_result(self, job_id: str) -> Optional[dict]:
        """Returns the job record, or None if job_id is unknown."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        return [{"id": jid, **info} for jid, info in self._jobs.items()]


async def _heavy_analysis_task(doc: str) -> str:
    """Simulates a long-running document analysis task."""
    await asyncio.sleep(0.3)   # simulate work
    return f"Analysis complete for: '{doc[:30]}'"


# ===========================================================================
# 6. rate_limiter_demo — token bucket implementation
# ===========================================================================

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Tokens refill at `rate` per second up to `capacity`.
    Each request consumes one token; if the bucket is empty the request
    is rejected (or the caller waits).
    """

    def __init__(self, rate: float, capacity: float):
        self.rate     = rate        # tokens added per second
        self.capacity = capacity    # max tokens
        self._tokens  = capacity    # start full
        self._last    = time.perf_counter()

    def _refill(self):
        now = time.perf_counter()
        elapsed = now - self._last
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last = now

    def allow(self, tokens: float = 1.0) -> bool:
        """Returns True if the request is allowed; False if throttled."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


async def rate_limiter_demo() -> None:
    """Shows 20 requests against a limiter that allows 5 req/s."""
    limiter = TokenBucketRateLimiter(rate=5.0, capacity=5.0)
    allowed = 0
    throttled = 0

    print("  Token bucket: 5 req/s, 20 requests fired rapidly")
    for i in range(20):
        if limiter.allow():
            allowed += 1
        else:
            throttled += 1
        await asyncio.sleep(0.08)   # 80ms between requests ≈ 12.5 req/s attempted

    print(f"  Allowed: {allowed} | Throttled: {throttled}")


# ===========================================================================
# Main — async runner
# ===========================================================================

async def _main():
    separator = "=" * 60

    # --- Demo 1: Streaming ---
    print(separator)
    print("DEMO 1: Simulated streaming response")
    print(separator)
    await streaming_endpoint_demo()

    # --- Demo 2: Parallel calls ---
    print(f"\n{separator}")
    print("DEMO 2: Parallel vs Sequential LLM calls")
    print(separator)

    prompts = [
        "Explain gradient descent",
        "What is RAG?",
        "Describe attention mechanisms",
        "How does RLHF work?",
    ]

    # Sequential
    t0 = time.perf_counter()
    seq_results = []
    for i, p in enumerate(prompts):
        r = await _single_llm_call(p, i)
        seq_results.append(r)
    seq_time = time.perf_counter() - t0

    # Parallel
    t0 = time.perf_counter()
    par_results = await parallel_llm_calls(prompts)
    par_time = time.perf_counter() - t0

    print(f"  Sequential: {seq_time * 1000:.0f}ms")
    print(f"  Parallel:   {par_time * 1000:.0f}ms")
    print(f"  Speedup:    {seq_time / par_time:.1f}x")

    # --- Demo 3: LatencyTracker ---
    print(f"\n{separator}")
    print("DEMO 3: LatencyTracker context manager")
    print(separator)
    async with LatencyTracker("streaming chat") as tracker:
        async for chunk in simulate_llm_stream("latency test", delay=0.02):
            tracker.mark_first_token()
            # (process chunk — omitted for demo)
    tracker.report()

    # --- Demo 4: Background jobs ---
    print(f"\n{separator}")
    print("DEMO 4: BackgroundJobQueue")
    print(separator)
    queue = BackgroundJobQueue()

    job1 = queue.submit(_heavy_analysis_task, "Quarterly earnings report Q1 2026")
    job2 = queue.submit(_heavy_analysis_task, "Legal contract — NDA template")
    print(f"  Submitted jobs: {job1}, {job2}")
    print(f"  Returning immediately — jobs run in background...")

    await asyncio.sleep(0.5)   # wait for background tasks to finish

    for job in queue.list_jobs():
        print(f"  Job {job['id']}: {job['status']} — {job.get('result') or job.get('error')}")

    # --- Demo 5: Rate limiter ---
    print(f"\n{separator}")
    print("DEMO 5: Token bucket rate limiter")
    print(separator)
    await rate_limiter_demo()

    print(f"\n{separator}")
    print("All latency/streaming demos complete.")
    print(separator)


if __name__ == "__main__":
    asyncio.run(_main())
