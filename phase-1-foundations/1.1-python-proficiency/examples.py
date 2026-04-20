"""
1.1 Python Proficiency — Working Examples
Run: python examples.py
"""

import time
import asyncio
import functools
from typing import Optional, TypedDict
from contextlib import contextmanager


# ─────────────────────────────────────────
# 1. Data Structures
# ─────────────────────────────────────────

def demo_data_structures():
    print("\n=== Data Structures ===")

    # List — ordered, mutable
    tokens = ["The", "cat", "sat"]
    tokens.append("down")
    tokens.insert(0, "START")
    print(f"List: {tokens}")
    print(f"Slice [1:3]: {tokens[1:3]}")

    # Dict — key-value store
    doc = {"id": 42, "text": "Hello world", "tokens": 2}
    doc["embedding"] = [0.1, 0.9, 0.3]
    print(f"\nDict keys: {list(doc.keys())}")
    print(f"Safe get: {doc.get('missing_key', 'default')}")

    # Set — uniqueness + fast membership test
    seen_ids = {"doc_1", "doc_2", "doc_1"}   # duplicates removed
    seen_ids.add("doc_3")
    print(f"\nSet (deduped): {seen_ids}")
    print(f"'doc_1' in set: {'doc_1' in seen_ids}")

    # Tuple — immutable, hashable
    coord = (48.8566, 2.3522)              # Paris lat/lon
    lat, lon = coord                        # unpacking
    print(f"\nTuple unpacked: lat={lat}, lon={lon}")


# ─────────────────────────────────────────
# 2. Classes with dunder methods
# ─────────────────────────────────────────

class DocumentChunk:
    """Represents a text chunk ready for embedding."""

    def __init__(self, text: str, doc_id: int, chunk_index: int = 0):
        self.text = text
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self._embedding: Optional[list[float]] = None

    def __repr__(self) -> str:
        return f"DocumentChunk(doc_id={self.doc_id}, idx={self.chunk_index}, chars={len(self.text)})"

    def __len__(self) -> int:
        return len(self.text)

    def __eq__(self, other) -> bool:
        return self.doc_id == other.doc_id and self.chunk_index == other.chunk_index

    @property
    def embedding(self) -> Optional[list[float]]:
        return self._embedding

    @embedding.setter
    def embedding(self, value: list[float]):
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value)}")
        self._embedding = value

    @property
    def has_embedding(self) -> bool:
        return self._embedding is not None

    @staticmethod
    def from_dict(data: dict) -> "DocumentChunk":
        return DocumentChunk(
            text=data["text"],
            doc_id=data["doc_id"],
            chunk_index=data.get("chunk_index", 0),
        )

    @classmethod
    def split_document(cls, full_text: str, doc_id: int, chunk_size: int = 200) -> list["DocumentChunk"]:
        """Split a long document into chunks."""
        chunks = []
        for i, start in enumerate(range(0, len(full_text), chunk_size)):
            chunk_text = full_text[start:start + chunk_size]
            chunks.append(cls(chunk_text, doc_id, chunk_index=i))
        return chunks


def demo_classes():
    print("\n=== Classes ===")
    chunk = DocumentChunk("The transformer architecture revolutionized NLP.", doc_id=1)
    print(chunk)                        # uses __repr__
    print(f"Length: {len(chunk)}")      # uses __len__

    chunk.embedding = [0.1, 0.5, 0.9]
    print(f"Has embedding: {chunk.has_embedding}")

    # Split a long document
    long_text = "word " * 100
    chunks = DocumentChunk.split_document(long_text, doc_id=2, chunk_size=50)
    print(f"\nSplit into {len(chunks)} chunks")
    print(f"First chunk: {chunks[0]}")


# ─────────────────────────────────────────
# 3. Decorators
# ─────────────────────────────────────────

def timer(func):
    """Log how long a function takes."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  {func.__name__}() → {elapsed*1000:.1f}ms")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 0.1):
    """Retry a function on exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"  Attempt {attempt} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator


attempt_count = 0

@timer
@retry(max_attempts=3, delay=0.05)
def flaky_api_call(succeed_on_attempt: int = 2) -> str:
    global attempt_count
    attempt_count += 1
    if attempt_count < succeed_on_attempt:
        raise ConnectionError("Simulated network error")
    return "API response"


def demo_decorators():
    print("\n=== Decorators ===")
    global attempt_count
    attempt_count = 0
    result = flaky_api_call(succeed_on_attempt=2)
    print(f"  Result: {result}")


# ─────────────────────────────────────────
# 4. Comprehensions and Generators
# ─────────────────────────────────────────

def demo_comprehensions():
    print("\n=== Comprehensions & Generators ===")

    documents = [
        {"id": 1, "text": "Transformers changed AI forever", "score": 0.95},
        {"id": 2, "text": "RAG is the most important AI pattern", "score": 0.88},
        {"id": 3, "text": "Fine-tuning requires good data", "score": 0.72},
    ]

    # List comprehension
    high_score_texts = [d["text"] for d in documents if d["score"] > 0.80]
    print(f"High-score texts: {high_score_texts}")

    # Dict comprehension
    id_to_score = {d["id"]: d["score"] for d in documents}
    print(f"ID → Score map: {id_to_score}")

    # Generator — processes lazily
    def token_chunks(text: str, size: int = 5):
        words = text.split()
        for i in range(0, len(words), size):
            yield " ".join(words[i:i + size])

    long_text = "The quick brown fox jumps over the lazy dog near the river bank"
    for chunk in token_chunks(long_text, size=4):
        print(f"  Chunk: '{chunk}'")


# ─────────────────────────────────────────
# 5. Async / Await
# ─────────────────────────────────────────

async def fake_embed(text: str) -> list[float]:
    """Simulates an async embedding API call."""
    await asyncio.sleep(0.05)           # simulates 50ms network latency
    return [hash(text) % 100 / 100.0]  # fake embedding


async def embed_sequential(texts: list[str]) -> list:
    """Process one at a time — slow."""
    results = []
    for text in texts:
        result = await fake_embed(text)
        results.append(result)
    return results


async def embed_concurrent(texts: list[str]) -> list:
    """Process all at once — fast."""
    tasks = [fake_embed(text) for text in texts]
    return await asyncio.gather(*tasks)


def demo_async():
    print("\n=== Async / Await ===")
    texts = [f"document {i}" for i in range(5)]

    start = time.perf_counter()
    asyncio.run(embed_sequential(texts))
    seq_time = time.perf_counter() - start

    start = time.perf_counter()
    asyncio.run(embed_concurrent(texts))
    conc_time = time.perf_counter() - start

    print(f"Sequential: {seq_time*1000:.0f}ms")
    print(f"Concurrent: {conc_time*1000:.0f}ms")
    print(f"Speedup: {seq_time/conc_time:.1f}x")


# ─────────────────────────────────────────
# 6. Type Hints
# ─────────────────────────────────────────

class SearchResult(TypedDict):
    doc_id: int
    text: str
    score: float
    metadata: dict


def search(query: str, top_k: int = 3) -> list[SearchResult]:
    """Return top-k results for a query."""
    # In reality: embed query → vector search
    return [
        {"doc_id": 1, "text": "Relevant doc 1", "score": 0.95, "metadata": {"source": "wiki"}},
        {"doc_id": 2, "text": "Relevant doc 2", "score": 0.88, "metadata": {"source": "arxiv"}},
    ]


def demo_type_hints():
    print("\n=== Type Hints ===")
    results = search("what is RAG?", top_k=2)
    for r in results:
        print(f"  [{r['score']:.2f}] {r['text']}")


# ─────────────────────────────────────────
# 7. Context Managers
# ─────────────────────────────────────────

@contextmanager
def api_session(service_name: str):
    """Simulates setting up and tearing down an API session."""
    print(f"  Opening connection to {service_name}")
    session = {"service": service_name, "calls": 0}
    try:
        yield session
    finally:
        print(f"  Closing connection. Total calls: {session['calls']}")


def demo_context_managers():
    print("\n=== Context Managers ===")
    with api_session("OpenAI") as session:
        session["calls"] += 1
        print(f"  Call 1 complete")
        session["calls"] += 1
        print(f"  Call 2 complete")


# ─────────────────────────────────────────
# 8. Error Handling
# ─────────────────────────────────────────

class RateLimitError(Exception):
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after {retry_after}s")


class InvalidInputError(ValueError):
    pass


def safe_llm_call(prompt: str) -> str:
    if not prompt.strip():
        raise InvalidInputError("Prompt cannot be empty")
    if len(prompt) > 10000:
        raise InvalidInputError(f"Prompt too long: {len(prompt)} chars (max 10000)")
    # Simulate success
    return f"Response to: {prompt[:50]}..."


def demo_error_handling():
    print("\n=== Error Handling ===")

    test_cases = ["", "  ", "A valid prompt", "x" * 20000]

    for prompt in test_cases:
        try:
            result = safe_llm_call(prompt)
            print(f"  OK: {result[:40]}")
        except InvalidInputError as e:
            print(f"  InvalidInput: {e}")
        except RateLimitError as e:
            print(f"  RateLimit: retry in {e.retry_after}s")
        except Exception as e:
            print(f"  Unexpected: {type(e).__name__}: {e}")
        finally:
            pass  # cleanup would go here


# ─────────────────────────────────────────
# 9. NumPy for Embeddings
# ─────────────────────────────────────────

def demo_numpy():
    print("\n=== NumPy (Embeddings) ===")
    try:
        import numpy as np

        # Simulate embeddings
        query = np.array([0.2, 0.8, 0.4, 0.1])
        documents = np.array([
            [0.2, 0.9, 0.3, 0.1],   # very similar
            [0.9, 0.1, 0.0, 0.5],   # not similar
            [0.3, 0.7, 0.5, 0.2],   # somewhat similar
        ])

        # Cosine similarity — the standard for embeddings
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Batch similarity (all docs vs query at once)
        norms = np.linalg.norm(documents, axis=1) * np.linalg.norm(query)
        similarities = documents @ query / norms

        ranked = np.argsort(similarities)[::-1]    # descending
        for rank, idx in enumerate(ranked):
            print(f"  Rank {rank+1}: doc_{idx} score={similarities[idx]:.3f}")

    except ImportError:
        print("  Run: pip install numpy")


# ─────────────────────────────────────────
# 10. Pandas for Data Analysis
# ─────────────────────────────────────────

def demo_pandas():
    print("\n=== Pandas (Data Processing) ===")
    try:
        import pandas as pd
        import io

        # Simulate CSV data (LLM API usage log)
        csv_data = """date,model,input_tokens,output_tokens,status
2024-01-01,gpt-4o,1200,350,success
2024-01-01,gpt-4o-mini,800,200,success
2024-01-01,gpt-4o,5000,1200,success
2024-01-02,gpt-4o-mini,600,150,error
2024-01-02,gpt-4o,2000,800,success
2024-01-02,gpt-4o,1500,400,success"""

        df = pd.read_csv(io.StringIO(csv_data))

        # Add cost column (approximate pricing)
        pricing = {"gpt-4o": (5e-6, 15e-6), "gpt-4o-mini": (0.15e-6, 0.6e-6)}
        df["cost_usd"] = df.apply(
            lambda r: r["input_tokens"] * pricing[r["model"]][0]
                    + r["output_tokens"] * pricing[r["model"]][1],
            axis=1
        )

        print("  Daily cost by model:")
        summary = df.groupby(["date", "model"])["cost_usd"].sum().round(4)
        print(summary.to_string())

        print(f"\n  Success rate: {(df['status']=='success').mean():.0%}")
        print(f"  Total cost: ${df['cost_usd'].sum():.4f}")

    except ImportError:
        print("  Run: pip install pandas")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    demo_data_structures()
    demo_classes()
    demo_decorators()
    demo_comprehensions()
    demo_async()
    demo_type_hints()
    demo_context_managers()
    demo_error_handling()
    demo_numpy()
    demo_pandas()
    print("\n✓ All demos complete")
