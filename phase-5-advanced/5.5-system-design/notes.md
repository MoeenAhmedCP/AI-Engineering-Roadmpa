# 5.5 AI System Design

## Why System Design Is the Senior Interview Skill

System design separates junior from senior AI engineers. Anyone can build a demo. The question is: can you build it to be fast, reliable, cheap, and safe for 100,000 users — while your team can still maintain it?

Interviewers aren't looking for one "correct" answer. They're evaluating how you think: do you clarify before designing? Do you identify the right tradeoffs? Can you back your choices with numbers?

---

## The Framework — Use This for Every Question

### 1. Clarify Requirements (2 minutes, non-negotiable)

Before drawing anything, ask:
- **Scale:** How many users/requests/documents?
- **Latency SLA:** < 500ms? < 2s? Real-time or async?
- **Accuracy requirements:** Can it occasionally be wrong, or is this medical/legal?
- **Cost budget:** $100/month or $10,000/month?
- **Data constraints:** Can we send data to OpenAI/Anthropic? GDPR?
- **Team size:** 2 engineers or 20?

A system for 100 users is totally different from one for 10M.

### 2. Data Layer

- What data exists? Where does it live? How fresh must it be?
- Ingestion pipeline: how does data get in?
- Storage: SQL (structured), vector DB (embeddings), object store (files)
- Indexing: how will data be retrieved efficiently?

### 3. AI/ML Layer

- Model choice: which LLM? Which embedding model? Open-source or API?
- Retrieval: RAG? Hybrid search? Reranking?
- Prompt strategy: system prompt, few-shot, CoT, structured output?
- Fallbacks: what happens when the AI is wrong or unavailable?

### 4. API Layer

- Endpoints: what does the API expose?
- Auth: API keys, JWT, OAuth?
- Rate limiting: per user, per IP?
- Async or synchronous responses?

### 5. Evaluation

- Offline: golden test set, LLM-as-judge, RAGAS metrics
- Online: user feedback (thumbs up/down), task completion rate, session length
- Monitoring: latency percentiles, error rates, cost per request

### 6. Scale

- How many requests/second at peak?
- Horizontal scaling: stateless services behind load balancer
- Caching: semantic cache for repeated queries
- Queue: async background jobs for slow operations
- Database: read replicas, connection pooling

### 7. Cost

Back your choices with numbers:
- Estimate tokens per request × model price × requests/day = daily cost
- Show where you can save: caching (40-60% reduction), model tiering (60-70%), batch API (50%)

---

## Common Architecture Patterns

### Pattern 1: RAG Service

```
User Query
    │
    ▼
Embed query (embedding model)
    │
    ▼
Vector DB similarity_search(query, k=10)
    │
    ▼
Optional: rerank with cross-encoder
    │
    ▼
Build prompt: system + context + question
    │
    ▼
LLM generate (streamed)
    │
    ▼
Return response + source citations
```

### Pattern 2: Multi-Tier Cost Optimization

```
Incoming query
    │
    ├─ Semantic cache hit? → return cached (free)
    │
    ├─ Simple query (classifier)? → cheap model (Haiku/mini)
    │
    └─ Complex query → full model (Sonnet/GPT-4o)
```

### Pattern 3: Async Background Processing

```
User uploads document
    │
    ▼
Return job_id immediately (202 Accepted)
    │
    ▼
Worker processes in background:
  extract text → chunk → embed → store
    │
    ▼
User polls GET /jobs/{job_id} or gets webhook
```

---

## Cost Estimation Table

| Model | Input per 1M tokens | Output per 1M tokens |
|---|---|---|
| Claude Haiku 4.5 | $0.80 | $4.00 |
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Opus 4.7 | $15.00 | $75.00 |
| GPT-4o mini | $0.15 | $0.60 |
| GPT-4o | $2.50 | $10.00 |

**Quick estimate formula:**
```
cost/request = (input_tokens × input_price + output_tokens × output_price) / 1,000,000
```

Example: 500 input + 300 output on Sonnet:
= (500 × $3 + 300 × $15) / 1,000,000 = ($1500 + $4500) / 1,000,000 = $0.006/request

At 10,000 requests/day = $60/day = $1,800/month.

---

## What Interviewers Look For

| Behavior | Junior | Senior |
|---|---|---|
| Starts with | Drawing architecture immediately | Asking clarifying questions |
| Cost awareness | "We'll use GPT-4" | "GPT-4 is $X/day at scale, let's tier by complexity" |
| Failure modes | Ignores them | Identifies top 3, proposes mitigations |
| Evaluation | "It works on my test cases" | Offline + online eval strategy with specific metrics |
| Scale | Doesn't mention | Estimates RPS, identifies bottlenecks, proposes fixes |
| Data privacy | Ignores | Asks about compliance requirements up front |

---

## Practice Tips

1. **Time yourself** — real interviews are 45 minutes. Allocate: 5 min clarify, 25 min design, 10 min deep dive, 5 min wrap.
2. **Draw as you talk** — use ASCII or whiteboard, don't just describe verbally.
3. **Back claims with numbers** — "that's fast enough" is weak. "$0.003/request × 1M/day = $3,000/day" is strong.
4. **Volunteer tradeoffs** — "I chose pgvector over Pinecone here because we're already on Postgres, but Pinecone would be better if we need multi-region replication."
5. **Read `practice_questions.md`** for 5 worked examples with full architectures.
