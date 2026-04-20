"""
1.1 Python Proficiency — Exercises
Attempt each exercise yourself before reading the solution.
Run: python exercises.py
"""

# ═══════════════════════════════════════════════════
# EXERCISE 1 — Data Structures
#
# You have a list of LLM API call logs (dicts).
# Write a function that returns:
#   - total cost (input_tokens * 0.000005 + output_tokens * 0.000015)
#   - a dict mapping each model to its average latency
#   - a set of unique models used
#
# Do it in ONE pass through the list (no calling list twice).
# ═══════════════════════════════════════════════════

api_logs = [
    {"model": "gpt-4o", "input_tokens": 1000, "output_tokens": 300, "latency_ms": 1200},
    {"model": "gpt-4o-mini", "input_tokens": 500, "output_tokens": 150, "latency_ms": 400},
    {"model": "gpt-4o", "input_tokens": 2000, "output_tokens": 600, "latency_ms": 2100},
    {"model": "claude-sonnet", "input_tokens": 800, "output_tokens": 200, "latency_ms": 900},
    {"model": "gpt-4o-mini", "input_tokens": 300, "output_tokens": 100, "latency_ms": 350},
]


def analyze_logs(logs: list[dict]) -> tuple[float, dict, set]:
    """
    Returns: (total_cost, avg_latency_by_model, unique_models)
    """
    # YOUR CODE HERE
    raise NotImplementedError


# ═══════════════════════════════════════════════════
# EXERCISE 2 — Classes
#
# Build a ConversationHistory class that:
#   - stores messages as a list of dicts {"role": str, "content": str}
#   - has add_message(role, content) method
#   - has __len__ returning number of messages
#   - has __iter__ so you can loop over messages
#   - has a property token_estimate that returns approximate tokens
#     (count all characters / 4, rounded to int)
#   - has last_n(n) method returning the last n messages
#   - has clear() method
# ═══════════════════════════════════════════════════

class ConversationHistory:
    # YOUR CODE HERE
    pass


# ═══════════════════════════════════════════════════
# EXERCISE 3 — Decorators
#
# Write a decorator called `log_calls` that:
#   - prints the function name + arguments when called
#   - prints the return value when the function returns
#   - prints the exception type if the function raises
#   - always prints "done" after (success or failure)
#
# Example output:
#   Calling embed(text='hello') ...
#   → returned: [0.1, 0.2]
#   done
# ═══════════════════════════════════════════════════

def log_calls(func):
    # YOUR CODE HERE
    pass


@log_calls
def embed(text: str) -> list:
    if not text:
        raise ValueError("Empty text")
    return [0.1, 0.2, 0.3]


# ═══════════════════════════════════════════════════
# EXERCISE 4 — Async
#
# The function below calls an API synchronously for each document.
# Rewrite it as an async function `embed_all_async` that:
#   - calls `fake_async_embed` concurrently for all texts
#   - returns a list of results in the SAME ORDER as input
#
# Constraint: use asyncio.gather
# ═══════════════════════════════════════════════════

import asyncio
import time


async def fake_async_embed(text: str) -> dict:
    """Simulates an async embedding API call (50ms latency)."""
    await asyncio.sleep(0.05)
    return {"text": text, "embedding": [hash(text) % 100 / 100.0]}


def embed_all_sync(texts: list[str]) -> list[dict]:
    """Sequential — slow baseline."""
    results = []
    for text in texts:
        results.append(asyncio.run(fake_async_embed(text)))
    return results


async def embed_all_async(texts: list[str]) -> list[dict]:
    """YOUR CODE HERE — concurrent version."""
    raise NotImplementedError


# ═══════════════════════════════════════════════════
# EXERCISE 5 — Comprehensions
#
# Given the documents list below, using comprehensions (no for loops with append):
#   a) Filter documents where word_count > 50
#   b) Create a dict mapping doc_id to word_count for all docs
#   c) Find all unique categories (as a set)
#   d) Create a "summary" list: "ID:{id} ({category}): {first 30 chars}..."
#      only for docs with score >= 0.8
# ═══════════════════════════════════════════════════

documents = [
    {"id": 1, "text": "The transformer architecture " * 20, "category": "ML", "score": 0.92},
    {"id": 2, "text": "RAG retrieval augmented", "category": "LLM", "score": 0.78},
    {"id": 3, "text": "Fine-tuning with LoRA " * 15, "category": "ML", "score": 0.85},
    {"id": 4, "text": "Docker containers for AI " * 10, "category": "Infra", "score": 0.91},
    {"id": 5, "text": "Prompt engineering basics", "category": "LLM", "score": 0.65},
]


def comprehension_exercises(docs: list[dict]):
    # a) Filter where word_count > 50
    long_docs = None  # YOUR CODE

    # b) Dict: doc_id → word_count
    word_counts = None  # YOUR CODE

    # c) Unique categories
    categories = None  # YOUR CODE

    # d) Summary for high-score docs
    summaries = None  # YOUR CODE

    return long_docs, word_counts, categories, summaries


# ═══════════════════════════════════════════════════
# EXERCISE 6 — Generator
#
# Write a generator function `sliding_window_chunks` that:
#   - takes a text string, chunk_size (words), and overlap (words)
#   - yields overlapping chunks
#
# Example: text="a b c d e f", size=3, overlap=1
#   yields: "a b c", "c d e", "e f"
#
# This is used in RAG to ensure context isn't lost at chunk boundaries.
# ═══════════════════════════════════════════════════

def sliding_window_chunks(text: str, chunk_size: int = 100, overlap: int = 20):
    """YOUR CODE HERE — yield overlapping text chunks."""
    raise NotImplementedError


# ───────────────────────────────────────
# Solutions (read AFTER attempting)
# ───────────────────────────────────────

def solutions():
    print("\n" + "="*50)
    print("SOLUTIONS")
    print("="*50)

    # Solution 1
    print("\n--- Exercise 1: Log Analysis ---")

    def analyze_logs_solution(logs):
        total_cost = 0.0
        latency_sums = {}
        latency_counts = {}
        unique_models = set()

        for log in logs:
            model = log["model"]
            cost = log["input_tokens"] * 0.000005 + log["output_tokens"] * 0.000015
            total_cost += cost
            unique_models.add(model)
            latency_sums[model] = latency_sums.get(model, 0) + log["latency_ms"]
            latency_counts[model] = latency_counts.get(model, 0) + 1

        avg_latency = {m: latency_sums[m] / latency_counts[m] for m in latency_sums}
        return total_cost, avg_latency, unique_models

    cost, latencies, models = analyze_logs_solution(api_logs)
    print(f"  Total cost: ${cost:.4f}")
    print(f"  Avg latencies: {latencies}")
    print(f"  Models used: {models}")

    # Solution 2
    print("\n--- Exercise 2: ConversationHistory ---")

    class ConversationHistorySolution:
        def __init__(self):
            self._messages: list[dict] = []

        def add_message(self, role: str, content: str):
            self._messages.append({"role": role, "content": content})

        def __len__(self) -> int:
            return len(self._messages)

        def __iter__(self):
            return iter(self._messages)

        @property
        def token_estimate(self) -> int:
            total_chars = sum(len(m["content"]) for m in self._messages)
            return int(total_chars / 4)

        def last_n(self, n: int) -> list[dict]:
            return self._messages[-n:]

        def clear(self):
            self._messages = []

    h = ConversationHistorySolution()
    h.add_message("user", "What is RAG?")
    h.add_message("assistant", "RAG stands for Retrieval-Augmented Generation...")
    print(f"  Messages: {len(h)}, Tokens ≈ {h.token_estimate}")
    print(f"  Last 1: {h.last_n(1)}")

    # Solution 3
    print("\n--- Exercise 3: log_calls decorator ---")
    import functools

    def log_calls_solution(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_str = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
            print(f"  Calling {func.__name__}({args_str}) ...")
            try:
                result = func(*args, **kwargs)
                print(f"  → returned: {result}")
                return result
            except Exception as e:
                print(f"  → raised {type(e).__name__}: {e}")
                raise
            finally:
                print("  done")
        return wrapper

    @log_calls_solution
    def embed_test(text: str) -> list:
        if not text:
            raise ValueError("Empty text")
        return [0.1, 0.2]

    embed_test("hello")
    try:
        embed_test("")
    except ValueError:
        pass

    # Solution 4
    print("\n--- Exercise 4: Async concurrent ---")

    async def embed_all_async_solution(texts):
        tasks = [fake_async_embed(text) for text in texts]
        return await asyncio.gather(*tasks)

    texts = [f"doc {i}" for i in range(5)]
    start = time.perf_counter()
    results = asyncio.run(embed_all_async_solution(texts))
    elapsed = time.perf_counter() - start
    print(f"  {len(results)} results in {elapsed*1000:.0f}ms (concurrent)")

    # Solution 5
    print("\n--- Exercise 5: Comprehensions ---")
    long_docs = [d for d in documents if len(d["text"].split()) > 50]
    word_counts = {d["id"]: len(d["text"].split()) for d in documents}
    categories = {d["category"] for d in documents}
    summaries = [f"ID:{d['id']} ({d['category']}): {d['text'][:30]}..." for d in documents if d["score"] >= 0.8]

    print(f"  Long docs: {[d['id'] for d in long_docs]}")
    print(f"  Word counts: {word_counts}")
    print(f"  Categories: {categories}")
    print(f"  Summaries: {summaries}")

    # Solution 6
    print("\n--- Exercise 6: Sliding window chunks ---")

    def sliding_window_solution(text, chunk_size=4, overlap=1):
        words = text.split()
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = words[i:i + chunk_size]
            if chunk:
                yield " ".join(chunk)

    sample = "a b c d e f g h i j"
    chunks = list(sliding_window_solution(sample, chunk_size=4, overlap=1))
    print(f"  Input: '{sample}'")
    print(f"  Chunks (size=4, overlap=1): {chunks}")


if __name__ == "__main__":
    print("Attempt the exercises, then call solutions() to check your answers.")
    print("To test your implementations, replace the `raise NotImplementedError` lines.")
    solutions()
