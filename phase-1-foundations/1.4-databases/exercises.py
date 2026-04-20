"""
1.4 Databases — Exercises
Attempt each before reading solutions.
Run: python exercises.py
"""

# ═══════════════════════════════════════════════
# EXERCISE 1 — SQL Query Writing
# Write a SQL query (as a string) that finds the top 3
# most expensive models by total token cost in the last 30 days.
# Table: api_logs(id, model, input_tokens, output_tokens, created_at)
# Cost: input_tokens * 0.000005 + output_tokens * 0.000015
# ═══════════════════════════════════════════════

TOP_MODELS_QUERY = """
-- YOUR SQL HERE
"""

# ═══════════════════════════════════════════════
# EXERCISE 2 — Conversation within Token Budget
# Given a list of messages (each with a 'token_count' field),
# write a function that returns the most recent messages
# that fit within max_tokens total.
# ═══════════════════════════════════════════════

def trim_to_budget(messages: list[dict], max_tokens: int) -> list[dict]:
    """Return the most recent messages fitting within max_tokens."""
    raise NotImplementedError


# ═══════════════════════════════════════════════
# EXERCISE 3 — Daily Model Call Counter (Redis pattern)
# Using the InMemoryRedis class from examples.py (copied below),
# implement track_model_call(model) that increments a counter
# for model+today. Keys expire after 24h.
# Also implement get_today_stats() returning {model: count}.
# ═══════════════════════════════════════════════

import time
import json
from datetime import datetime


class InMemoryRedis:
    def __init__(self):
        self._store = {}

    def set(self, key, value, ex=None):
        expires_at = time.time() + ex if ex else float("inf")
        self._store[key] = (value, expires_at)

    def get(self, key):
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def incr(self, key):
        current = int(self.get(key) or 0) + 1
        self.set(key, str(current))
        return current

    def expire(self, key, seconds):
        if key in self._store:
            value, _ = self._store[key]
            self._store[key] = (value, time.time() + seconds)

    def keys(self, pattern="*"):
        import fnmatch
        now = time.time()
        return [k for k, (_, exp) in self._store.items() if now < exp and fnmatch.fnmatch(k, pattern)]


redis = InMemoryRedis()


def track_model_call(model: str):
    raise NotImplementedError


def get_today_stats() -> dict:
    raise NotImplementedError


# ═══════════════════════════════════════════════
# EXERCISE 4 — RAG Retrieval Log Schema
# Write CREATE TABLE SQL statements for storing RAG retrieval logs:
# - query_id (PK), user_id, query_text, created_at
# - retrieved_chunks table: chunk_id, query_id (FK), doc_id, chunk_text, score
# Include appropriate indexes.
# ═══════════════════════════════════════════════

RAG_LOG_SCHEMA = """
-- YOUR SQL HERE
"""

# ═══════════════════════════════════════════════
# EXERCISE 5 — Cache-Aside Pattern
# Implement cache_aside(key, fetch_fn, ttl):
# 1. Check redis for key
# 2. If miss, call fetch_fn() to get the value
# 3. Store result in redis with ttl
# 4. Return value
# fetch_fn is a callable that returns a dict
# ═══════════════════════════════════════════════

def cache_aside(key: str, fetch_fn, ttl: int = 300) -> dict:
    raise NotImplementedError


# ═══════════════════════════════════════════════
# EXERCISE 6 — Conversation Summary
# Write a function that detects when a conversation exceeds
# a token budget and returns:
# - the messages to keep (most recent, under budget)
# - a "summary placeholder" message to prepend
# ═══════════════════════════════════════════════

def compress_conversation(messages: list[dict], max_tokens: int, keep_last: int = 4) -> list[dict]:
    """
    If total token_count > max_tokens:
    - Keep the last `keep_last` messages unchanged
    - Replace all older messages with a single {"role": "system", "content": "[Summary of earlier conversation]"}
    Otherwise return messages unchanged.
    """
    raise NotImplementedError


# ─────────────────────────────────────────
# Solutions
# ─────────────────────────────────────────

def solutions():
    print("\n" + "="*50 + "\nSOLUTIONS\n" + "="*50)

    # 1 — SQL
    print("\n--- Exercise 1: Top 3 expensive models ---")
    sol = """
SELECT
    model,
    SUM(input_tokens * 0.000005 + output_tokens * 0.000015) AS total_cost,
    COUNT(*) AS call_count
FROM api_logs
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY model
ORDER BY total_cost DESC
LIMIT 3;
"""
    print(sol)

    # 2 — Token budget
    print("--- Exercise 2: Trim to budget ---")
    def trim_solution(messages, max_tokens):
        result, total = [], 0
        for msg in reversed(messages):
            tc = msg.get("token_count", len(msg["content"]) // 4)
            if total + tc > max_tokens:
                break
            result.append(msg)
            total += tc
        return list(reversed(result))

    msgs = [
        {"role": "user", "content": "First message", "token_count": 10},
        {"role": "assistant", "content": "First reply", "token_count": 20},
        {"role": "user", "content": "Second message", "token_count": 10},
        {"role": "assistant", "content": "Second reply", "token_count": 20},
    ]
    trimmed = trim_solution(msgs, max_tokens=35)
    print(f"  Input: {len(msgs)} messages. Budget: 35 tokens.")
    print(f"  Output: {len(trimmed)} messages kept")

    # 3 — Daily counter
    print("\n--- Exercise 3: Daily model counter ---")
    today = datetime.utcnow().strftime("%Y-%m-%d")

    def track_solution(model):
        key = f"model_calls:{model}:{today}"
        count = redis.incr(key)
        if count == 1:
            redis.expire(key, 86400)

    def stats_solution():
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        pattern = f"model_calls:*:{today_str}"
        result = {}
        for key in redis.keys(pattern):
            model = key.split(":")[1]
            result[model] = int(redis.get(key) or 0)
        return result

    for _ in range(3): track_solution("gpt-4o")
    for _ in range(5): track_solution("claude-sonnet")
    print(f"  Today's stats: {stats_solution()}")

    # 4 — RAG schema
    print("\n--- Exercise 4: RAG Log Schema ---")
    print("""
CREATE TABLE rag_queries (
    query_id   BIGSERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL,
    query_text TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_rag_queries_user ON rag_queries(user_id, created_at DESC);

CREATE TABLE rag_retrieved_chunks (
    id         BIGSERIAL PRIMARY KEY,
    query_id   BIGINT NOT NULL REFERENCES rag_queries(query_id) ON DELETE CASCADE,
    doc_id     BIGINT NOT NULL,
    chunk_text TEXT,
    score      FLOAT NOT NULL,
    rank       INTEGER NOT NULL
);
CREATE INDEX idx_rag_chunks_query ON rag_retrieved_chunks(query_id);
""")

    # 5 — Cache-aside
    print("--- Exercise 5: Cache-aside ---")
    def cache_aside_solution(key, fetch_fn, ttl=300):
        cached = redis.get(key)
        if cached:
            print(f"  Cache HIT for key={key}")
            return json.loads(cached)
        print(f"  Cache MISS for key={key}")
        value = fetch_fn()
        redis.set(key, json.dumps(value), ex=ttl)
        return value

    call_count = 0
    def fake_db_query():
        nonlocal call_count
        call_count += 1
        return {"result": "document content", "id": 42}

    r1 = cache_aside_solution("doc:42", fake_db_query, ttl=60)
    r2 = cache_aside_solution("doc:42", fake_db_query, ttl=60)
    print(f"  DB called {call_count} time(s). Both results equal: {r1 == r2}")

    # 6 — Compress conversation
    print("\n--- Exercise 6: Compress conversation ---")
    def compress_solution(messages, max_tokens, keep_last=4):
        total = sum(m.get("token_count", 10) for m in messages)
        if total <= max_tokens:
            return messages
        recent = messages[-keep_last:]
        summary = {"role": "system", "content": "[Summary of earlier conversation]", "token_count": 50}
        return [summary] + recent

    long_msgs = [{"role": "user" if i%2==0 else "assistant", "content": f"msg {i}", "token_count": 20} for i in range(10)]
    compressed = compress_solution(long_msgs, max_tokens=100, keep_last=4)
    print(f"  Input: {len(long_msgs)} messages ({sum(m['token_count'] for m in long_msgs)} tokens)")
    print(f"  Output: {len(compressed)} messages (summary + last 4)")


if __name__ == "__main__":
    print("Attempt exercises before calling solutions()")
    solutions()
