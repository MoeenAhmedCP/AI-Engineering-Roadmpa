# 4.7 LLM Response Caching

## Why Cache LLM Responses?

LLM API calls are expensive (cost), slow (latency), and often redundant. Many real-world deployments see
the same or very similar queries repeated constantly — a customer support bot gets the same "how do I reset
my password?" question thousands of times a day. Caching lets you answer those repeated queries
instantly and for nearly free.

There are two fundamentally different caching strategies: **exact caching** and **semantic caching**.
Understanding both — and when to combine them — is essential for production LLM systems.

---

## Exact Caching

**Idea:** Hash the complete inputs to produce a deterministic cache key. If the exact same inputs come
in again, return the stored response without calling the LLM.

### Cache Key Design

A good cache key includes everything that would change the model's output:

```
key = SHA-256(model_name + system_prompt + user_message)
```

**Include in the key:**
- `model` — "claude-sonnet-4-6" vs "claude-haiku-3" produce different outputs
- `system_prompt` (hashed) — different system prompts change behavior
- `user_message` — the actual query
- `temperature` — if you vary temperature, include it; if always 0, omit it

**Exclude from the key:**
- `user_id` — the same question from different users should share a cache entry
- `timestamp` — obviously varies; including it breaks caching entirely
- `request_id` / `trace_id` — metadata, not semantically significant
- `session_id` — unless the session history is part of the prompt

If you include conversation history in the prompt, you should hash the full message list. But for stateless
single-turn queries, the above three fields are sufficient.

### Redis Implementation

Redis is the standard backend for exact caching:

```python
import hashlib, json, redis

r = redis.Redis()

def make_cache_key(model, system_prompt, user_message):
    payload = f"{model}||{system_prompt}||{user_message}"
    return "llm:" + hashlib.sha256(payload.encode()).hexdigest()

def cached_call(model, system_prompt, user_message, ttl=3600):
    key = make_cache_key(model, system_prompt, user_message)
    cached = r.get(key)
    if cached:
        return json.loads(cached), True   # (response, cache_hit)
    response = call_llm(model, system_prompt, user_message)
    r.setex(key, ttl, json.dumps(response))
    return response, False
```

---

## TTL Strategy

The right TTL depends on how stable the answer is:

| Query type | Example | TTL |
|---|---|---|
| Factual / stable | "What is the capital of France?" | 7–30 days |
| Documentation Q&A | "How do I use pandas groupby?" | 24–72 hours |
| Business data-driven | "What is our return policy?" | 1–4 hours |
| Dynamic data | "What is today's stock price?" | 60–300 seconds |
| Creative / generative | "Write me a poem about autumn" | No cache (or very short) |

Creative queries should generally not be cached — users expect variation. If they wanted the same poem
every time, they'd bookmark it.

---

## Semantic Caching

Exact caching only helps when queries are byte-for-byte identical. Semantic caching addresses near-duplicates:
"How do I reset my password?" and "I forgot my password, what do I do?" are different strings but
semantically equivalent.

### How It Works

1. Embed the incoming query into a vector using an embedding model
2. Search the vector store for previously seen queries with high cosine similarity
3. If similarity > threshold (e.g., 0.92), return the cached response
4. Otherwise, call the LLM, store the (query_embedding, response) pair

### Similarity Threshold Tuning

This is the hardest part of semantic caching. The threshold controls the false positive/negative tradeoff:

- **Threshold too low (e.g., 0.80):** False positives — semantically different questions get the same
  cached answer. "What is the refund policy?" and "What is the cancellation policy?" might be 0.82 similar
  but deserve different answers.
- **Threshold too high (e.g., 0.99):** Cache miss rate is nearly 100%; semantic cache provides no benefit.
- **Sweet spot (0.90–0.95):** Catches genuine paraphrases while avoiding wrong answers.

Start at 0.92 and tune based on your domain. Legal and medical queries need a higher threshold (0.97+).
FAQ-style queries for consumer apps can tolerate 0.90.

---

## GPTCache

[GPTCache](https://github.com/zilliztech/GPTCache) is an open-source semantic caching library that
integrates directly into OpenAI/LangChain calls. It handles:

- Multiple embedding backends (OpenAI, sentence-transformers, etc.)
- Multiple vector store backends (Milvus, FAISS, Redis)
- Configurable similarity functions
- Cache eviction policies

```python
from gptcache import cache
from gptcache.adapter import openai

cache.init()  # uses default FAISS + OpenAI embeddings
# Now just use openai.ChatCompletion.create() — caching is transparent
```

GPTCache intercepts the OpenAI call, performs the semantic lookup, and only hits the API on a miss.

---

## Momento: Managed Semantic Cache

[Momento](https://www.gomomento.com/) offers a managed semantic cache-as-a-service. Instead of running
your own vector store and embedding pipeline, you make API calls to Momento's cache endpoint. It handles
embedding, similarity search, and storage. Good for teams that want the semantic caching benefit without
the infrastructure overhead.

---

## Two-Layer Cache Architecture

The most robust production pattern combines both approaches:

```
Incoming query
      │
      ▼
[Layer 1: Exact Cache — Redis]
  Hit? → return instantly (microseconds)
      │ Miss
      ▼
[Layer 2: Semantic Cache — Vector store]
  Similarity > 0.92? → return (milliseconds)
      │ Miss
      ▼
[LLM API call]
  → Store in both caches
```

Layer 1 (exact) is nearly free and blazing fast. Layer 2 (semantic) catches paraphrases but costs a
vector search (5–50ms). The LLM API is only called on true misses.

---

## Cache Invalidation

Cache invalidation is famously hard, and it's especially tricky for RAG systems. When your source documents
change (a policy is updated, a product price changes), any cached answers derived from those documents
are now wrong.

**Strategies:**
- **Tag-based invalidation:** Tag each cache entry with the document IDs it depends on. When document X
  is updated, delete all entries tagged with X.
- **Short TTL for RAG answers:** Don't cache for more than a few hours when the underlying data changes
  frequently.
- **Version the cache key:** Include the document corpus version in the key. When docs are updated,
  bump the version; old entries naturally become unreachable (and expire via TTL).

---

## Cache Warming

Before a production launch, pre-populate the cache with known common queries. Collect the top 100 queries
from your beta period or estimate them from your FAQ. Run them through the LLM during off-peak hours and
store the results. On launch day, users hitting those queries get instant responses from cache.

Cache warming is also useful after invalidation — if you know you just updated a document, proactively
re-run the common queries against the new document so the cache is populated before users notice the miss.

---

## Monitoring Cache Performance

Key metrics to track:
- **Exact hit rate** — should be 20–60% for most apps
- **Semantic hit rate** — additional 10–30% on top of exact hits
- **False positive rate** — sample semantic hits for quality; high false positive = lower threshold
- **Cache latency** — Redis p99 should be < 5ms; vector search < 50ms
- **TTL distribution** — are most entries expiring before being hit? TTL may be too short.
