# 1.1 Python Proficiency

## Why This Matters for AI Engineering

Every AI engineering task — calling APIs, processing data, building pipelines — is Python. Weak foundations here slow you down at every later stage.

---

## Data Structures

### Lists — ordered, mutable
```python
items = [1, 2, 3]
items.append(4)        # [1, 2, 3, 4]
items.pop()            # removes last → [1, 2, 3]
items[1:3]             # slicing → [2, 3]
```
**When to use:** ordered sequences, stacks, queues.

### Dictionaries — key-value, insertion-ordered (Python 3.7+)
```python
doc = {"id": 1, "text": "hello", "embedding": [0.1, 0.2]}
doc.get("missing", "default")   # safe access, no KeyError
doc.keys(), doc.values(), doc.items()
```
**When to use:** structured records, counting, memoization/caching.

### Sets — unordered, unique values
```python
seen = set()
seen.add("doc_1")
"doc_1" in seen        # O(1) lookup — much faster than list
```
**When to use:** deduplication, membership testing at scale.

### Tuples — immutable lists
```python
point = (3.14, 2.71)
x, y = point           # unpacking
```
**When to use:** fixed structures (coordinates, return multiple values), dictionary keys.

---

## Classes and OOP

```python
class DocumentChunk:
    def __init__(self, text: str, doc_id: int):
        self.text = text
        self.doc_id = doc_id
        self._embedding = None      # private by convention

    def __repr__(self) -> str:
        return f"DocumentChunk(doc_id={self.doc_id}, len={len(self.text)})"

    def __len__(self) -> int:
        return len(self.text)

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        if not isinstance(value, list):
            raise TypeError("Embedding must be a list")
        self._embedding = value

    @staticmethod
    def from_dict(data: dict) -> "DocumentChunk":
        return DocumentChunk(data["text"], data["doc_id"])
```

**Key dunder methods for AI work:**
- `__init__` — constructor
- `__repr__` — unambiguous string (used in debugging)
- `__len__` — so `len(chunk)` works
- `__iter__` — so you can loop over an object
- `__enter__` / `__exit__` — context manager protocol

---

## Decorators

A decorator is a function that wraps another function to add behaviour.

```python
import time
import functools

def timer(func):
    @functools.wraps(func)      # preserves __name__, __doc__
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

@timer
def embed_document(text: str) -> list:
    # simulates a slow API call
    time.sleep(0.1)
    return [0.1, 0.2, 0.3]

# In AI engineering you'll constantly see:
# @app.get("/")         — FastAPI route registration
# @property             — computed attribute
# @staticmethod         — method that doesn't need self
# @classmethod          — method that gets the class, not instance
# @lru_cache            — memoize expensive function calls
```

---

## Comprehensions and Generators

```python
texts = ["Hello world", "AI is great", "Python rocks"]

# List comprehension — eager, stores all results
lengths = [len(t) for t in texts]

# Dict comprehension
word_counts = {t: len(t.split()) for t in texts}

# Set comprehension
unique_lengths = {len(t) for t in texts}

# Generator expression — lazy, doesn't store all results
# Use when you have millions of items
gen = (len(t) for t in texts)
total = sum(gen)     # consumed once

# Generator function — yields one item at a time
def chunk_text(text: str, size: int = 100):
    for i in range(0, len(text), size):
        yield text[i:i + size]

# This is how you process millions of tokens without OOM
for chunk in chunk_text(very_long_document):
    process(chunk)
```

**Rule of thumb:** if you're iterating once and don't need random access → generator. If you need the list multiple times → list comprehension.

---

## Async / Await

LLM API calls are I/O-bound (you wait for the network). `async` lets you do other work while waiting — critical for building responsive AI apps.

```python
import asyncio
import httpx

# Synchronous — slow (sequential)
def get_embeddings_sync(texts: list[str]) -> list:
    results = []
    for text in texts:
        response = httpx.post("https://api.openai.com/v1/embeddings", ...)
        results.append(response.json())
    return results

# Asynchronous — fast (concurrent)
async def embed_one(client: httpx.AsyncClient, text: str) -> list:
    response = await client.post("https://api.openai.com/v1/embeddings", ...)
    return response.json()

async def get_embeddings_async(texts: list[str]) -> list:
    async with httpx.AsyncClient() as client:
        # Run all requests concurrently
        tasks = [embed_one(client, text) for text in texts]
        return await asyncio.gather(*tasks)

# Run the async function
results = asyncio.run(get_embeddings_async(["hello", "world", "foo"]))
```

**Mental model:** `async def` defines a coroutine. `await` says "pause here and let other tasks run while I wait for this I/O." `asyncio.gather` runs coroutines concurrently.

---

## Type Hints

```python
from typing import Optional, TypedDict

# Basic types
def chunk(text: str, size: int = 500) -> list[str]: ...

# Optional = the value might be None
def get_doc(doc_id: int) -> Optional[dict]: ...

# TypedDict — structured dict with known keys
class ChunkResult(TypedDict):
    text: str
    embedding: list[float]
    doc_id: int
    score: float

# Union types (Python 3.10+)
def process(input: str | list[str]) -> str: ...
```

Type hints don't affect runtime behaviour — they're documentation + tooling support (mypy, IDE autocomplete). Use them everywhere in AI code; it makes debugging much easier.

---

## Context Managers

```python
# Built-in: files, locks, connections
with open("data.txt") as f:
    content = f.read()          # file closes automatically even if error

# Custom context manager
class ModelTimer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start
        print(f"Done in {self.elapsed:.2f}s")
        return False    # don't suppress exceptions

with ModelTimer() as t:
    result = call_expensive_model()
print(t.elapsed)

# Simpler: use @contextmanager
from contextlib import contextmanager

@contextmanager
def temp_directory():
    import tempfile, shutil
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)
```

---

## Error Handling

```python
class APIRateLimitError(Exception):
    """Raised when we hit an API rate limit."""
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after {retry_after}s")

def call_llm(prompt: str) -> str:
    try:
        response = api.complete(prompt)
        return response.text
    except APIRateLimitError as e:
        print(f"Rate limited, sleeping {e.retry_after}s")
        time.sleep(e.retry_after)
        return call_llm(prompt)         # retry
    except ConnectionError:
        raise                           # re-raise, let caller handle
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
    finally:
        # Always runs — cleanup, logging
        log_api_call()
```

---

## NumPy Basics

```python
import numpy as np

# Embeddings are numpy arrays
embedding = np.array([0.1, 0.2, 0.3, 0.4])

# Cosine similarity (fundamental operation in RAG)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Matrix operations — compute similarity of one query vs many docs
query = np.array([0.1, 0.2, 0.3])       # shape (3,)
docs = np.array([[0.1, 0.2, 0.3],        # shape (N, 3)
                 [0.9, 0.1, 0.0],
                 [0.5, 0.5, 0.0]])

# Broadcasting — computes similarity with all docs at once
similarities = docs @ query / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query))
top_doc = np.argmax(similarities)
```

---

## Pandas Basics

```python
import pandas as pd

df = pd.read_csv("conversations.csv")

# Inspect
df.head()
df.info()
df.describe()

# Filter
long_convos = df[df["message_count"] > 10]
errors = df[df["status"] == "error"]

# Transform
df["token_cost"] = df["input_tokens"] * 0.000003 + df["output_tokens"] * 0.000015

# Group and aggregate
daily_cost = df.groupby("date")["token_cost"].sum()
model_usage = df.groupby("model")["input_tokens"].agg(["mean", "sum", "count"])

# Save
df.to_csv("processed.csv", index=False)
df.to_json("processed.json", orient="records")
```
