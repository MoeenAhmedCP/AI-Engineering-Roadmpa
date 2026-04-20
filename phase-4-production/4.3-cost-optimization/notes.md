# 4.3 Cost Optimization

## Why Costs Spiral

LLM inference costs are priced per token. That sounds manageable until you multiply by scale.

Consider a modest product: 1 million daily active users, each sending one message and receiving one response averaging 1000 tokens combined. At $3 per million tokens (a mid-tier model), that is **$3,000 per day** — $1.1 million per year. Add retrieval context (RAG often triples prompt length), system prompts, and retry logic, and you are looking at $3-5 million annually before any feature growth.

Cost compounds in four ways:

1. **Context length**: Every token in the prompt (including history, retrieved docs, system prompt) is billed.
2. **Retries**: Failed or invalid responses mean you pay for both the failure and the correction.
3. **Feature creep**: Adding more retrieved chunks, richer system prompts, and multi-turn memory all silently increase average token counts.
4. **Output verbosity**: Models default to verbose responses. Unconstrained `max_tokens` lets the model charge you for unnecessary prose.

Sustainable LLM products are built with cost as a first-class engineering constraint from day one.

---

## Prompt Caching: 90% Discount on Repeated Context

Anthropic's prompt caching feature lets you designate portions of your prompt as cacheable. When the same cached prefix is reused within the cache lifetime (5 minutes for standard cache), you pay 10% of the normal input token price for those tokens.

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": very_long_system_context,   # 2000+ tokens
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": user_question
            }
        ]
    }
]
```

Requirements and mechanics:
- Minimum cacheable block: **1024 tokens** (otherwise caching is not triggered).
- Cache writes cost 25% more than normal input tokens (amortised over reads).
- Cache reads cost 10% of normal input token price.
- Cache lifetime: 5 minutes (ephemeral). Extended caching (1 hour) is available on some plans.
- Best for: long system prompts, large retrieved documents, few-shot example blocks that are reused across many requests.

**Break-even**: If the same cached block is read more than ~1.3 times, caching saves money.

OpenAI's equivalent is the Cached Input Tokens system (automatic, no explicit marking needed — it caches the prefix of identical prompts).

---

## Model Tiering: Route to the Cheapest Capable Model

Not every query needs your most powerful (and most expensive) model. A simple FAQ lookup costs the same as a complex multi-step reasoning task if you route everything to the same model.

Typical tier structure:

| Tier | Model | Cost (input/1M) | Use case |
|---|---|---|---|
| Fast | claude-haiku-3-5 | ~$0.80 | Simple Q&A, classification, extraction |
| Standard | claude-sonnet-4-6 | ~$3.00 | General reasoning, summarisation |
| Power | claude-opus-4 | ~$15.00 | Complex analysis, long-form generation |

A routing classifier (even a simple keyword-based or logistic regression model) assigns each query to a tier before the LLM call. Routing 70% of traffic to the fast tier and 20% to standard yields approximately 60-70% cost reduction versus routing everything to the power tier.

The classifier itself should be:
- **Fast**: adds <10ms latency.
- **Cheap**: a local model or a simple ML classifier, not another LLM call.
- **Conservative**: when in doubt, route up (cost > quality failure).

---

## Query Classification for Routing

A simple logistic regression or keyword-based classifier is often sufficient:

```python
FAST_KEYWORDS = ["what is", "define", "hello", "hi", "thanks", "faq"]
POWER_KEYWORDS = ["analyse", "compare", "write a detailed", "explain why", "reason through"]

def classify_query(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in FAST_KEYWORDS):
        return "fast"
    if any(kw in q for kw in POWER_KEYWORDS):
        return "power"
    return "standard"
```

For production, train a proper classifier on your historical queries labelled with the tier that produced satisfactory results.

---

## Semantic Caching: Return Cached Answers for Similar Queries

Many LLM applications receive semantically equivalent queries repeatedly. "What is your return policy?" and "How do I return a product?" have different wording but the same answer.

Semantic caching:
1. Embed each incoming query.
2. Check similarity against a store of previously answered queries.
3. If a cached entry has cosine similarity > threshold (typically 0.92-0.95), return the cached response.
4. Otherwise, call the LLM, cache the (query, response) pair.

This is distinct from exact-match caching (useful for identical queries). Semantic caching handles paraphrase.

Tools: GPtcache, Redis with vector extension, or a simple in-memory store for small scale.

The threshold matters: too low and you return wrong cached answers; too high and you miss obvious paraphrases. 0.95 is a safe default; tune on your traffic.

---

## Batch API: 50% Cheaper for Non-Real-Time Work

Both Anthropic and OpenAI offer asynchronous batch APIs that process requests in batches (up to 24h turnaround). The price is typically **50% of standard API pricing**.

When to use:
- Overnight document processing
- Bulk data extraction and transformation
- Generating training data
- Evaluating model outputs at scale
- Any task where users do not need a synchronous response

When NOT to use:
- Real-time chat
- Interactive Q&A
- Any user-facing feature requiring <3s response

The workflow: submit a batch → poll for completion → retrieve results. Use job IDs to track status.

---

## Output Length Control

The model's default behaviour is to be thorough and explanatory. Unconstrained, it will explain its reasoning, hedge with caveats, and summarise what it said. All of that costs tokens.

Controls:
- **`max_tokens`**: Always set this. Even a generous limit (e.g., 512 tokens) prevents runaway responses.
- **Prompt instruction**: "Respond in at most 3 sentences." or "Return only the JSON object, no explanation."
- **System prompt framing**: "You are a concise assistant. Never repeat the question. Use bullet points."

Measure your p50 and p95 output token counts. If p95 is 3x your p50, the model is being verbose on some queries — investigate and add constraints.

---

## Token Counting Before Send

Count tokens before submitting to catch oversized inputs early.

```python
# Rough estimate (no tokenizer dependency)
def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)

# With anthropic SDK
client.count_tokens(messages=messages)  # exact count
```

Set a hard limit and either reject or truncate inputs that exceed it. Truncation should be intelligent: trim the middle of retrieved context, not the user's question.

---

## Cost Monitoring

Track cost per request from day one. You cannot optimise what you cannot measure.

Minimum viable cost monitoring:
- **Per-request**: log `model`, `input_tokens`, `output_tokens`, `user_id`, `feature_name`, `cost_usd`.
- **Daily rollups**: sum by `model`, `feature_name`, `user_id`. Alert when daily cost exceeds budget.
- **Per-user cost**: identify power users and consider rate limits or tiering.

```python
COST_TABLE = {
    "claude-haiku-3-5":  {"input": 0.80,  "output": 4.00},   # per 1M tokens
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-opus-4":     {"input": 15.00, "output": 75.00},
}
```

Daily budget alerts via a cron job or webhook prevent nasty surprises at month-end.

---

## Real Cost Numbers (April 2026, approximate)

| Model | Input $/1M | Output $/1M | Notes |
|---|---|---|---|
| claude-haiku-3-5 | $0.80 | $4.00 | Fast tier, great for simple tasks |
| claude-sonnet-4-6 | $3.00 | $15.00 | Default workhorse |
| claude-opus-4 | $15.00 | $75.00 | Most capable, use sparingly |
| gpt-4o-mini | $0.15 | $0.60 | OpenAI fast tier |
| gpt-4o | $2.50 | $10.00 | OpenAI standard |
| gpt-4.1 | $2.00 | $8.00 | OpenAI power |
| gemini-2.0-flash | $0.10 | $0.40 | Google fast tier |

Cache reads are typically 90% cheaper (Anthropic) or included at reduced rates (OpenAI). Batch API is 50% off across all tiers.
