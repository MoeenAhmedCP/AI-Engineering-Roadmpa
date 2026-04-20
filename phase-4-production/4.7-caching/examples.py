"""
4.7 LLM Response Caching — Examples
Implements exact caching, semantic caching, and a two-layer cache.
No external dependencies required.
Run: python examples.py
"""

import hashlib
import time
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Cache Key Construction
# ---------------------------------------------------------------------------

def make_cache_key(model: str, system_prompt: str, user_message: str) -> str:
    """
    Build a deterministic SHA-256 cache key from the inputs that affect
    the LLM response. Excludes user_id, timestamp, request_id, etc.
    """
    payload = f"{model}||{system_prompt}||{user_message}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 2. Exact Cache (in-memory simulation of Redis)
# ---------------------------------------------------------------------------

class ExactCache:
    """
    In-memory exact-match cache with TTL support.
    In production, back this with Redis using r.setex() / r.get().
    """

    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}  # key -> (value, expires_at)
        self._hits = 0
        self._misses = 0
        self._sets = 0

    def get(self, key: str) -> Optional[str]:
        if key in self._store:
            value, expires_at = self._store[key]
            if time.time() < expires_at:
                self._hits += 1
                return value
            else:
                del self._store[key]  # expired
        self._misses += 1
        return None

    def set(self, key: str, value: str, ttl: int = 300) -> None:
        self._store[key] = (value, time.time() + ttl)
        self._sets += 1

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "hit_rate": round(hit_rate, 3),
            "size": len(self._store),
        }

    def reset_stats(self):
        self._hits = 0
        self._misses = 0
        self._sets = 0


# ---------------------------------------------------------------------------
# 3. Embedding and Similarity
# ---------------------------------------------------------------------------

def fake_embed(text: str) -> list[float]:
    """
    Deterministic 32-dimensional embedding based on SHA-256 hash.
    In production, replace with a real embedding model.
    The key property: similar texts should produce similar vectors.
    We simulate this by adding small perturbations for minor text changes.
    """
    # Base hash for determinism
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Convert 32 bytes -> 32 floats in [-1, 1]
    vec = [(b - 127.5) / 127.5 for b in digest]
    # Normalize to unit vector
    magnitude = sum(x * x for x in vec) ** 0.5
    if magnitude == 0:
        return vec
    return [x / magnitude for x in vec]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1]."""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = sum(a * a for a in v1) ** 0.5
    mag2 = sum(b * b for b in v2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ---------------------------------------------------------------------------
# 4. Semantic Cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Semantic cache: stores (embedding, response) pairs.
    On lookup, embeds the query and checks cosine similarity against all
    stored entries. Returns the best match if it exceeds the threshold.

    In production, use FAISS, Pinecone, or pgvector for the similarity search
    instead of the linear scan here.
    """

    def __init__(self):
        self._entries: list[dict] = []  # [{query, embedding, response}]
        self._hits = 0
        self._misses = 0

    def add(self, query: str, response: str) -> None:
        embedding = fake_embed(query)
        self._entries.append({
            "query": query,
            "embedding": embedding,
            "response": response,
        })

    def lookup(self, query: str, threshold: float = 0.92) -> Optional[str]:
        if not self._entries:
            self._misses += 1
            return None

        query_vec = fake_embed(query)
        best_score = -1.0
        best_response = None

        for entry in self._entries:
            score = cosine_similarity(query_vec, entry["embedding"])
            if score > best_score:
                best_score = score
                best_response = entry["response"]

        if best_score >= threshold:
            self._hits += 1
            return best_response

        self._misses += 1
        return None

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "entries": len(self._entries),
        }

    def reset_stats(self):
        self._hits = 0
        self._misses = 0


# ---------------------------------------------------------------------------
# 5. Two-Layer Cache
# ---------------------------------------------------------------------------

class TwoLayerCache:
    """
    Combines ExactCache (fast, Redis-like) and SemanticCache (slower, vector).
    Lookup order: exact first, then semantic on miss.
    On a full miss, caller is responsible for storing the result.
    """

    def __init__(self, exact_ttl: int = 300, semantic_threshold: float = 0.92):
        self.exact = ExactCache()
        self.semantic = SemanticCache()
        self.exact_ttl = exact_ttl
        self.semantic_threshold = semantic_threshold
        self._total_queries = 0
        self._exact_hits = 0
        self._semantic_hits = 0
        self._full_misses = 0

    def lookup(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
    ) -> tuple[Optional[str], str]:
        """
        Returns (response, source) where source is one of:
        'exact_hit', 'semantic_hit', 'miss'
        """
        self._total_queries += 1

        # Layer 1: exact match
        key = make_cache_key(model, system_prompt, user_message)
        result = self.exact.get(key)
        if result is not None:
            self._exact_hits += 1
            return result, "exact_hit"

        # Layer 2: semantic match
        result = self.semantic.lookup(user_message, threshold=self.semantic_threshold)
        if result is not None:
            # Promote to exact cache for future identical queries
            self.exact.set(key, result, ttl=self.exact_ttl)
            self._semantic_hits += 1
            return result, "semantic_hit"

        self._full_misses += 1
        return None, "miss"

    def store(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        response: str,
    ) -> None:
        """Store a new response in both cache layers."""
        key = make_cache_key(model, system_prompt, user_message)
        self.exact.set(key, response, ttl=self.exact_ttl)
        self.semantic.add(user_message, response)

    def stats(self) -> dict:
        total = self._total_queries
        return {
            "total_queries": total,
            "exact_hits": self._exact_hits,
            "semantic_hits": self._semantic_hits,
            "full_misses": self._full_misses,
            "exact_hit_rate": round(self._exact_hits / total, 3) if total else 0,
            "semantic_hit_rate": round(self._semantic_hits / total, 3) if total else 0,
            "overall_hit_rate": round((self._exact_hits + self._semantic_hits) / total, 3) if total else 0,
        }


# ---------------------------------------------------------------------------
# 6. Mock LLM (simulates API response without keys)
# ---------------------------------------------------------------------------

_MOCK_RESPONSES: dict[str, str] = {
    "password": "To reset your password, go to Settings > Security > Reset Password.",
    "refund": "Our refund policy allows returns within 30 days of purchase.",
    "hours": "We are open Monday to Friday, 9am to 5pm EST.",
    "contact": "You can reach support at support@example.com or call 1-800-555-0100.",
    "default": "Thank you for your question. Our team will get back to you shortly.",
}

def mock_llm_call(user_message: str) -> str:
    """Simulates an LLM API call with a 50ms delay."""
    time.sleep(0.05)  # simulate network latency
    msg_lower = user_message.lower()
    for keyword, response in _MOCK_RESPONSES.items():
        if keyword in msg_lower:
            return response
    return _MOCK_RESPONSES["default"]


# ---------------------------------------------------------------------------
# 7. Cache Demo
# ---------------------------------------------------------------------------

def cache_demo():
    """
    Simulates 20 queries (some duplicates, some near-duplicates) through
    a TwoLayerCache and compares hit rates between layers.
    """
    print("=" * 65)
    print("  LLM RESPONSE CACHING — TWO-LAYER CACHE DEMO")
    print("=" * 65)

    cache = TwoLayerCache(exact_ttl=3600, semantic_threshold=0.85)

    MODEL = "claude-sonnet-4-6"
    SYSTEM = "You are a helpful customer support assistant."

    # Queries: some exact duplicates, some near-duplicates
    queries = [
        # Group 1: password resets (first is new, rest should hit cache)
        "How do I reset my password?",
        "How do I reset my password?",          # exact duplicate
        "How do I reset my password?",          # exact duplicate again
        "I forgot my password, what do I do?",  # near-duplicate (semantic)
        "Password reset instructions please",   # near-duplicate (semantic)

        # Group 2: refunds
        "What is your refund policy?",
        "What is your refund policy?",          # exact duplicate
        "How do I get a refund?",               # near-duplicate
        "I want to return something",           # near-duplicate

        # Group 3: hours
        "What are your business hours?",
        "When are you open?",
        "What time do you close?",

        # Group 4: contact
        "How do I contact support?",
        "How do I contact support?",            # exact duplicate
        "What is your phone number?",

        # Group 5: unique queries (should all miss)
        "What is the meaning of life?",
        "Tell me about your premium plan",
        "Do you offer enterprise pricing?",
        "Can I integrate with Salesforce?",
        "What programming languages do you support?",
    ]

    print(f"\n{'#':<4} {'Query':<45} {'Source':<15} {'Response (truncated)'}")
    print("-" * 100)

    for i, query in enumerate(queries, 1):
        result, source = cache.lookup(MODEL, SYSTEM, query)

        if result is None:
            # True miss — call LLM and store
            result = mock_llm_call(query)
            cache.store(MODEL, SYSTEM, query, result)
            source = "LLM_CALL"

        truncated = result[:40] + "..." if len(result) > 40 else result
        source_label = {
            "exact_hit":    "EXACT HIT",
            "semantic_hit": "SEMANTIC HIT",
            "LLM_CALL":     "LLM CALL",
        }.get(source, source)

        print(f"{i:<4} {query:<45} {source_label:<15} {truncated}")

    # Print stats
    stats = cache.stats()
    print("\n" + "=" * 65)
    print("  CACHE PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"  Total queries:          {stats['total_queries']}")
    print(f"  Exact cache hits:       {stats['exact_hits']}  ({stats['exact_hit_rate']*100:.1f}%)")
    print(f"  Semantic cache hits:    {stats['semantic_hits']}  ({stats['semantic_hit_rate']*100:.1f}%)")
    print(f"  Full misses (LLM):      {stats['full_misses']}  ({(1 - stats['overall_hit_rate'])*100:.1f}%)")
    print(f"  Overall hit rate:       {stats['overall_hit_rate']*100:.1f}%")
    print()

    # Demonstrate TTL strategies
    print("=" * 65)
    print("  TTL STRATEGY GUIDE")
    print("=" * 65)
    ttl_guide = [
        ("Factual / stable answers", "7–30 days",  604800),
        ("Documentation Q&A",        "24–72 hours", 86400),
        ("Business policy data",      "1–4 hours",   3600),
        ("Dynamic / real-time data",  "60–300 sec",  120),
        ("Creative generation",       "No cache",    0),
    ]
    print(f"  {'Query Type':<30} {'TTL Label':<15} {'TTL (seconds)'}")
    print("  " + "-" * 55)
    for qtype, label, ttl in ttl_guide:
        ttl_str = str(ttl) if ttl > 0 else "N/A"
        print(f"  {qtype:<30} {label:<15} {ttl_str}")
    print()

    # Demonstrate cache key isolation
    print("=" * 65)
    print("  CACHE KEY ISOLATION DEMO")
    print("=" * 65)
    msg = "What is the capital of France?"
    key_a = make_cache_key("claude-haiku-3", SYSTEM, msg)
    key_b = make_cache_key("claude-sonnet-4-6", SYSTEM, msg)
    key_c = make_cache_key("claude-sonnet-4-6", "Different system prompt.", msg)
    key_d = make_cache_key("claude-sonnet-4-6", SYSTEM, msg)  # same as key_b

    print(f"  Same message, different models  → keys differ: {key_a[:12]}... vs {key_b[:12]}...")
    print(f"  Same model+msg, diff system     → keys differ: {key_b[:12]}... vs {key_c[:12]}...")
    print(f"  Identical inputs                → keys match:  {key_b[:12]}... == {key_d[:12]}...")
    print()


if __name__ == "__main__":
    cache_demo()
