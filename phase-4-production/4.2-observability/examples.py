"""
4.2 Observability — Examples
Run: python examples.py
No API keys required. All LLM calls are simulated.
"""

import hashlib
import json
import random
import statistics
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Optional


# ---------------------------------------------------------------------------
# Pricing Table
# ---------------------------------------------------------------------------

PRICING = {
    "claude-sonnet-4-6": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "claude-haiku-3":    {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
    "gpt-4o":            {"input_per_1k": 0.005, "output_per_1k": 0.015},
    "gpt-4o-mini":       {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICING:
        return 0.0
    p = PRICING[model]
    return (input_tokens / 1000 * p["input_per_1k"]) + (output_tokens / 1000 * p["output_per_1k"])


# ---------------------------------------------------------------------------
# Log Entry
# ---------------------------------------------------------------------------

@dataclass
class LLMLogEntry:
    request_id: str
    user_id_hash: str       # SHA-256 of real user_id — never store raw
    model: str
    prompt_preview: str     # First 200 chars only
    response_preview: str   # First 200 chars only
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    error_type: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = ""


# ---------------------------------------------------------------------------
# Structured Logger
# ---------------------------------------------------------------------------

class StructuredLogger:
    """
    Writes structured JSON log lines to stdout and an in-memory store.
    In production: replace stdout write with CloudWatch/Datadog/Loki sink.
    """

    def __init__(self, silent: bool = False):
        self._log: list[LLMLogEntry] = []
        self.silent = silent

    def _hash_user_id(self, user_id: str) -> str:
        return "sha256:" + hashlib.sha256(user_id.encode()).hexdigest()[:16]

    def log_llm_call(
        self,
        request_id: str,
        user_id: str,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        error: Optional[str] = None,
        session_id: str = "",
    ) -> LLMLogEntry:
        entry = LLMLogEntry(
            request_id=request_id,
            user_id_hash=self._hash_user_id(user_id),
            model=model,
            prompt_preview=prompt[:200],
            response_preview=response[:200],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=compute_cost(model, input_tokens, output_tokens),
            latency_ms=latency_ms,
            error_type=error,
            session_id=session_id,
        )
        self._log.append(entry)
        if not self.silent:
            print(json.dumps(asdict(entry)))
        return entry

    @property
    def entries(self) -> list[LLMLogEntry]:
        return list(self._log)


# ---------------------------------------------------------------------------
# LLM Call Tracer (context manager)
# ---------------------------------------------------------------------------

class LLMCallTracer:
    """
    Context manager that times an LLM call and logs the result.

    Usage:
        with LLMCallTracer(logger, user_id="u123", model="gpt-4o") as tracer:
            response = call_llm(prompt)
            tracer.set_result(prompt, response, input_tokens=500, output_tokens=150)
    """

    def __init__(
        self,
        logger: StructuredLogger,
        user_id: str,
        model: str,
        session_id: str = "",
    ):
        self.logger = logger
        self.user_id = user_id
        self.model = model
        self.session_id = session_id
        self.request_id = str(uuid.uuid4())[:8]
        self._start: float = 0.0
        self._prompt: str = ""
        self._response: str = ""
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._error: Optional[str] = None

    def set_result(
        self,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
    ):
        self._prompt = prompt
        self._response = response
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def set_error(self, error_type: str):
        self._error = error_type

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = int((time.time() - self._start) * 1000)
        if exc_type is not None:
            self._error = exc_type.__name__
        self.logger.log_llm_call(
            request_id=self.request_id,
            user_id=self.user_id,
            model=self.model,
            prompt=self._prompt,
            response=self._response,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=latency_ms,
            error=self._error,
            session_id=self.session_id,
        )
        return False  # Don't suppress exceptions


# ---------------------------------------------------------------------------
# Decorator for async functions
# ---------------------------------------------------------------------------

def trace_llm_call(logger: StructuredLogger, model: str):
    """
    Decorator that wraps an async function and logs its execution.
    The decorated function must return a dict with keys:
    prompt, response, input_tokens, output_tokens, user_id.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())[:8]
            start = time.time()
            error = None
            result = None
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                error = type(e).__name__
                raise
            finally:
                latency_ms = int((time.time() - start) * 1000)
                if result:
                    logger.log_llm_call(
                        request_id=request_id,
                        user_id=result.get("user_id", "unknown"),
                        model=model,
                        prompt=result.get("prompt", ""),
                        response=result.get("response", ""),
                        input_tokens=result.get("input_tokens", 0),
                        output_tokens=result.get("output_tokens", 0),
                        latency_ms=latency_ms,
                        error=error,
                    )
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Usage Analytics
# ---------------------------------------------------------------------------

class UsageAnalytics:
    """Computes analytics from a list of LLMLogEntry objects."""

    def __init__(self, entries: list[LLMLogEntry]):
        self.entries = entries

    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    def calls_by_model(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.entries:
            counts[e.model] = counts.get(e.model, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def percentile_latency(self, p: int = 95) -> float:
        """Return the p-th percentile latency in ms."""
        latencies = sorted(e.latency_ms for e in self.entries)
        if not latencies:
            return 0.0
        idx = max(0, int(len(latencies) * p / 100) - 1)
        return float(latencies[idx])

    def error_rate(self) -> float:
        if not self.entries:
            return 0.0
        errors = sum(1 for e in self.entries if e.error_type is not None)
        return errors / len(self.entries)

    def cost_by_model(self) -> dict[str, float]:
        costs: dict[str, float] = {}
        for e in self.entries:
            costs[e.model] = costs.get(e.model, 0.0) + e.cost_usd
        return dict(sorted(costs.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Simulate 20 realistic log entries
# ---------------------------------------------------------------------------

def simulate_log_entries(logger: StructuredLogger, n: int = 20):
    models = [
        ("claude-sonnet-4-6", 0.5),
        ("claude-haiku-3", 0.3),
        ("gpt-4o-mini", 0.2),
    ]
    users = [f"user_{i}" for i in range(5)]
    errors = [None, None, None, None, None, None, None, None, "rate_limit_error", "timeout"]
    sample_prompts = [
        "Summarize the quarterly report and identify key risks.",
        "Extract all action items from this meeting transcript.",
        "What are the main clauses in this contract?",
        "Classify the sentiment of these customer reviews.",
        "Generate a brief executive summary from this document.",
    ]

    for _ in range(n):
        model = random.choices([m[0] for m in models], weights=[m[1] for m in models])[0]
        user = random.choice(users)
        prompt = random.choice(sample_prompts)
        response = "Here is the analysis based on the provided document..."
        input_tokens = random.randint(500, 3000)
        output_tokens = random.randint(100, 600)
        latency_ms = random.randint(800, 8000)
        error = random.choice(errors)

        logger.log_llm_call(
            request_id=str(uuid.uuid4())[:8],
            user_id=user,
            model=model,
            prompt=prompt,
            response=response if not error else "",
            input_tokens=input_tokens if not error else 0,
            output_tokens=output_tokens if not error else 0,
            latency_ms=latency_ms,
            error=error,
        )


# ---------------------------------------------------------------------------
# Observability Dashboard (printed to console)
# ---------------------------------------------------------------------------

def print_dashboard(analytics: UsageAnalytics):
    entries = analytics.entries
    print()
    print("=" * 60)
    print("  OBSERVABILITY DASHBOARD")
    print("=" * 60)
    print(f"  Total calls:        {len(entries)}")
    print(f"  Total cost:         ${analytics.total_cost():.4f}")
    print(f"  Error rate:         {analytics.error_rate():.1%}")
    print()
    print("  LATENCY (ms):")
    print(f"    p50:  {analytics.percentile_latency(50):>6.0f} ms")
    print(f"    p95:  {analytics.percentile_latency(95):>6.0f} ms")
    print(f"    p99:  {analytics.percentile_latency(99):>6.0f} ms")
    print()
    print("  CALLS BY MODEL:")
    for model, count in analytics.calls_by_model().items():
        print(f"    {model:<25} {count:>3} calls")
    print()
    print("  COST BY MODEL:")
    for model, cost in analytics.cost_by_model().items():
        print(f"    {model:<25} ${cost:.4f}")
    print()
    # Alerting thresholds
    if analytics.error_rate() > 0.05:
        print("  [ALERT] Error rate > 5% — investigate immediately")
    if analytics.percentile_latency(99) > 30000:
        print("  [ALERT] p99 latency > 30s — users are timing out")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("--- Simulating 20 LLM Calls (JSON log lines) ---\n")
    logger = StructuredLogger(silent=False)
    simulate_log_entries(logger, n=20)

    print("\n--- Context Manager Example ---")
    silent_logger = StructuredLogger(silent=True)
    with LLMCallTracer(silent_logger, user_id="alice@example.com", model="gpt-4o") as tracer:
        time.sleep(0.01)  # Simulate a fast LLM call
        tracer.set_result(
            prompt="What are the key risks in this document?",
            response="The key risks identified are: 1) supply chain exposure, 2) regulatory uncertainty.",
            input_tokens=1200,
            output_tokens=45,
        )
    entry = silent_logger.entries[0]
    print(f"Logged call: request_id={entry.request_id}, cost=${entry.cost_usd:.5f}, latency={entry.latency_ms}ms")

    print("\n--- Analytics Dashboard ---")
    analytics = UsageAnalytics(logger.entries)
    print_dashboard(analytics)


if __name__ == "__main__":
    main()
