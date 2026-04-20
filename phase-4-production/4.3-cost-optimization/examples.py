"""
4.3 Cost Optimization — Examples
===================================
Implements query routing, semantic caching, and cost monitoring
without any real API calls.

Run: python examples.py
"""

import math
import random
import time
from typing import Optional

SIMULATE = True

# ===========================================================================
# 1. Cost table (realistic April 2026 prices, per 1M tokens)
# ===========================================================================

COST_TABLE: dict[str, dict[str, float]] = {
    "claude-haiku-3-5":  {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-opus-4":     {"input": 15.00, "output": 75.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "gpt-4o":            {"input": 2.50,  "output": 10.00},
    "gemini-2.0-flash":  {"input": 0.10,  "output": 0.40},
}


def calculate_request_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Returns the USD cost for a single API request.

    Args:
        model:         model name (must be in COST_TABLE)
        input_tokens:  number of input tokens
        output_tokens: number of output tokens

    Returns:
        Cost in USD (float).
    """
    if model not in COST_TABLE:
        raise ValueError(f"Unknown model '{model}'. Known: {list(COST_TABLE)}")
    prices = COST_TABLE[model]
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


# ===========================================================================
# 2. QueryRouter — keyword-based tier classification
# ===========================================================================

class QueryRouter:
    """
    Routes queries to the cheapest model tier capable of handling them.

    Tiers are defined by the caller; classification uses keyword heuristics
    (in production: train an ML classifier on labelled historical queries).
    """

    # Keywords that suggest a simple / fast-tier query
    _FAST_PATTERNS = [
        "what is", "define", "hello", "hi there", "thanks", "thank you",
        "how do i", "when was", "who is", "faq", "price of", "hours",
    ]
    # Keywords that suggest a complex / power-tier query
    _POWER_PATTERNS = [
        "analyse", "analyze", "compare and contrast", "write a detailed",
        "explain why", "reason through", "critique", "evaluate the tradeoffs",
        "multi-step", "research", "comprehensive",
    ]

    def __init__(self, tiers: Optional[dict] = None):
        """
        tiers: maps tier name → model name.
        Defaults to a sensible three-tier Anthropic setup.
        """
        self.tiers = tiers or {
            "fast":     "claude-haiku-3-5",
            "standard": "claude-sonnet-4-6",
            "power":    "claude-opus-4",
        }

    def classify(self, query: str) -> str:
        """Returns tier name: 'fast', 'standard', or 'power'."""
        q = query.lower()
        if any(p in q for p in self._FAST_PATTERNS):
            return "fast"
        if any(p in q for p in self._POWER_PATTERNS):
            return "power"
        return "standard"

    def get_model(self, query: str) -> str:
        """Returns the model name for this query."""
        tier = self.classify(query)
        return self.tiers[tier]

    def route_cost_estimate(self, queries: list[str],
                            avg_input_tokens: int = 500,
                            avg_output_tokens: int = 250) -> dict:
        """
        Estimates cost breakdown for a list of queries.

        Returns a dict with per-tier counts, estimated cost, and comparison
        cost if everything had been sent to the power model.
        """
        breakdown: dict[str, dict] = {t: {"count": 0, "cost": 0.0} for t in self.tiers}

        for q in queries:
            tier  = self.classify(q)
            model = self.tiers[tier]
            cost  = calculate_request_cost(model, avg_input_tokens, avg_output_tokens)
            breakdown[tier]["count"] += 1
            breakdown[tier]["cost"]  += cost

        total_routed = sum(v["cost"] for v in breakdown.values())

        power_model = self.tiers["power"]
        cost_if_all_power = len(queries) * calculate_request_cost(
            power_model, avg_input_tokens, avg_output_tokens
        )

        return {
            "per_tier":        breakdown,
            "total_routed":    total_routed,
            "total_if_power":  cost_if_all_power,
            "savings_usd":     cost_if_all_power - total_routed,
            "savings_pct":     (1 - total_routed / cost_if_all_power) * 100 if cost_if_all_power else 0,
        }


# ===========================================================================
# 3. SemanticCache — bag-of-words cosine similarity (no ML deps needed)
# ===========================================================================

class SemanticCache:
    """
    Caches (query, response) pairs and returns cached responses for
    semantically similar queries using bag-of-words cosine similarity.

    In production, replace _embed with a real embedding model.
    """

    def __init__(self):
        self._cache: list[dict] = []   # [{query, response, embedding}]
        self.hits  = 0
        self.misses = 0

    # --- Embedding stub: bag-of-words TF vector -------------------------

    def _embed(self, text: str) -> dict[str, float]:
        """Returns a word-frequency dict as a sparse vector."""
        words = text.lower().split()
        vec: dict[str, float] = {}
        for w in words:
            # Strip punctuation
            w = w.strip(".,!?;:'\"()")
            if w:
                vec[w] = vec.get(w, 0) + 1
        # L2-normalise
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def _cosine(self, a: dict, b: dict) -> float:
        """Cosine similarity between two sparse vectors."""
        shared = set(a) & set(b)
        return sum(a[k] * b[k] for k in shared)

    # --- Public API --------------------------------------------------------

    def add(self, query: str, response: str) -> None:
        """Store a new (query, response) pair."""
        self._cache.append({
            "query":     query,
            "response":  response,
            "embedding": self._embed(query),
        })

    def lookup(self, query: str, threshold: float = 0.95) -> Optional[str]:
        """
        Return a cached response if a similar query exists, else None.
        Similarity >= threshold triggers a cache hit.
        """
        if not self._cache:
            self.misses += 1
            return None

        q_emb = self._embed(query)
        best_score = 0.0
        best_resp  = None

        for entry in self._cache:
            score = self._cosine(q_emb, entry["embedding"])
            if score > best_score:
                best_score = score
                best_resp  = entry["response"]

        if best_score >= threshold:
            self.hits += 1
            return best_resp

        self.misses += 1
        return None

    def stats(self) -> dict:
        return {
            "entries": len(self._cache),
            "hits":    self.hits,
            "misses":  self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) else 0.0,
        }


# ===========================================================================
# 4. CostMonitor — per-request logging + daily summary + budget alert
# ===========================================================================

class CostMonitor:
    """
    Logs every LLM request and provides daily rollups and budget alerting.
    In production, back this with a time-series DB (ClickHouse, TimescaleDB).
    """

    def __init__(self):
        self._log: list[dict] = []

    def log_request(self, model: str, in_tok: int, out_tok: int,
                    user_id: str = "anonymous") -> float:
        """Record a request. Returns the USD cost."""
        cost = calculate_request_cost(model, in_tok, out_tok)
        self._log.append({
            "model":   model,
            "in_tok":  in_tok,
            "out_tok": out_tok,
            "user_id": user_id,
            "cost":    cost,
            "ts":      time.time(),
        })
        return cost

    def daily_summary(self) -> dict:
        """Aggregate today's logged requests by model and user."""
        summary: dict[str, dict] = {}
        total = 0.0
        for entry in self._log:
            m = entry["model"]
            summary.setdefault(m, {"requests": 0, "cost": 0.0, "users": set()})
            summary[m]["requests"] += 1
            summary[m]["cost"]     += entry["cost"]
            summary[m]["users"].add(entry["user_id"])
            total += entry["cost"]

        # Convert sets to counts for serialisability
        for m in summary:
            summary[m]["unique_users"] = len(summary[m].pop("users"))

        return {
            "total_cost_usd": total,
            "total_requests": len(self._log),
            "by_model":       summary,
        }

    def alert_if_over(self, budget: float) -> bool:
        """Returns True and prints an alert if total cost exceeds budget."""
        total = sum(e["cost"] for e in self._log)
        if total > budget:
            print(f"  [ALERT] Daily cost ${total:.4f} exceeds budget ${budget:.4f}!")
            return True
        return False

    def top_users(self, n: int = 5) -> list[tuple[str, float]]:
        """Returns top N users by cost."""
        user_costs: dict[str, float] = {}
        for entry in self._log:
            uid = entry["user_id"]
            user_costs[uid] = user_costs.get(uid, 0) + entry["cost"]
        return sorted(user_costs.items(), key=lambda x: x[1], reverse=True)[:n]


# ===========================================================================
# 5. simulate_cost_savings — full end-to-end demo
# ===========================================================================

# 100 sample queries across complexity levels
_SAMPLE_QUERIES = (
    # Fast queries (40%)
    ["What is your return policy?"] * 8 +
    ["Hi, how are you?"] * 5 +
    ["What is machine learning?"] * 7 +
    ["What are your hours?"] * 5 +
    ["Define RAG."] * 5 +
    # Standard queries (40%)
    ["Summarise this contract for me."] * 10 +
    ["Translate this paragraph to Spanish."] * 8 +
    ["Generate 5 email subject lines for this campaign."] * 7 +
    ["Rewrite this paragraph to be more formal."] * 7 +
    ["Extract key dates from this document."] * 8 +
    # Power queries (20%)
    ["Analyse the tradeoffs between microservices and monoliths."] * 7 +
    ["Write a detailed research summary on transformer architectures."] * 7 +
    ["Evaluate the tradeoffs of three database options for this use case."] * 6
)

random.shuffle(_SAMPLE_QUERIES)
_SAMPLE_QUERIES = _SAMPLE_QUERIES[:100]


def simulate_cost_savings() -> None:
    router  = QueryRouter()
    cache   = SemanticCache()
    monitor = CostMonitor()

    AVG_IN  = 600
    AVG_OUT = 300

    # --- Baseline: all queries to power model, no cache ------------------
    baseline_cost = len(_SAMPLE_QUERIES) * calculate_request_cost(
        "claude-opus-4", AVG_IN, AVG_OUT
    )

    # --- Optimised: routing + semantic cache -----------------------------
    optimised_cost = 0.0
    cache_hits     = 0

    for i, query in enumerate(_SAMPLE_QUERIES):
        # Check semantic cache first
        cached = cache.lookup(query, threshold=0.95)
        if cached:
            cache_hits += 1
            continue  # free — no API call

        # Route to appropriate model
        model = router.get_model(query)
        cost  = calculate_request_cost(model, AVG_IN, AVG_OUT)
        optimised_cost += cost

        # Simulate response and cache it
        fake_response = f"[Simulated response #{i} from {model}]"
        cache.add(query, fake_response)

        # Log to cost monitor
        uid = f"user_{(i % 10) + 1:02d}"
        monitor.log_request(model, AVG_IN, AVG_OUT, user_id=uid)

    # --- Report -----------------------------------------------------------
    savings_usd = baseline_cost - optimised_cost
    savings_pct = (savings_usd / baseline_cost) * 100 if baseline_cost else 0

    print("\n" + "=" * 60)
    print("COST SAVINGS SIMULATION (100 queries)")
    print("=" * 60)
    print(f"  Baseline (all claude-opus-4, no cache): ${baseline_cost:.4f}")
    print(f"  Optimised (routing + cache):            ${optimised_cost:.4f}")
    print(f"  Savings:                                ${savings_usd:.4f}  ({savings_pct:.1f}%)")
    print(f"  Cache hits: {cache_hits}/100")

    route_report = router.route_cost_estimate(_SAMPLE_QUERIES, AVG_IN, AVG_OUT)
    print("\n  Routing breakdown (without cache):")
    for tier, info in route_report["per_tier"].items():
        model = router.tiers[tier]
        print(f"    {tier:10s} ({model:25s}): {info['count']:3d} queries | ${info['cost']:.4f}")

    daily = monitor.daily_summary()
    print(f"\n  Daily monitor — total: ${daily['total_cost_usd']:.4f} | "
          f"requests: {daily['total_requests']}")
    print("  Top 3 users by cost:")
    for uid, cost in monitor.top_users(3):
        print(f"    {uid}: ${cost:.4f}")

    monitor.alert_if_over(budget=0.05)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    # Quick unit checks
    print("--- Unit checks ---")
    cost = calculate_request_cost("claude-sonnet-4-6", 1000, 500)
    print(f"  1000 in + 500 out @ claude-sonnet-4-6 = ${cost:.6f}")

    router = QueryRouter()
    for q in ["What is Python?", "Summarise this report.", "Analyse the tradeoffs of X vs Y."]:
        tier = router.classify(q)
        print(f"  '{q[:40]}' → {tier}")

    simulate_cost_savings()
    print("\nDone.")
