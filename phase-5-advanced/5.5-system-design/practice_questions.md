# 5.5 AI System Design — Practice Questions

Work through each question before reading the answer. For each: clarify requirements, sketch the data layer, AI layer, API layer, evaluation, and scaling. Time yourself (20 minutes each).

---

## Question 1: Customer Support AI

**Prompt:** Design a customer support AI for 1 million users. Requirements: latency < 2 seconds, cost < $0.01 per query.

### Answer

**Clarify first:**
- What domains? (billing, returns, product questions) — scope the knowledge base
- What can it escalate? — human handoff protocol
- Languages? — multilingual support adds complexity

**Architecture:**

```
User (web/mobile)
       │
       ▼
   API Gateway (rate limiting, auth)
       │
       ▼
 FastAPI Service (async, Uvicorn workers)
       │
   ┌───┴───────────────┐
   │                   │
   ▼                   ▼
Query Router      Semantic Cache
(classify:        (Redis + embeddings)
 simple/complex)  hit → return cached
   │                   │
   ├─ simple ──► Claude Haiku ($0.00025/1K)
   └─ complex ─► Claude Sonnet ($0.003/1K)
       │
       ▼
  RAG Retrieval (pgvector)
  ← embed query → top-5 chunks
       │
       ▼
  Response + sources
       │
       ▼
  LangSmith (trace every call)
```

**Cost math:**
- 70% simple queries → Haiku: 500 tokens × $0.0000003 = $0.00015
- 30% complex queries → Sonnet: 1000 tokens × $0.000003 = $0.003
- Blended: 0.7×$0.00015 + 0.3×$0.003 = $0.00105/query ✓ (under $0.01)
- Semantic cache hits 40% → effective cost $0.00063/query

**Scaling:**
- ECS Fargate auto-scaling (target CPU 60%)
- Read replica for pgvector (read-heavy)
- Redis Cluster for cache (3 nodes)

**Evaluation:**
- LLM-as-judge: 5% of queries scored daily (helpfulness, accuracy, tone)
- Deflection rate: % resolved without human escalation
- CSAT: thumbs up/down inline

---

## Question 2: Law Firm Document Q&A

**Prompt:** Design a document Q&A system for a law firm. 100,000 case documents. Cannot send data to external cloud APIs. 99.9% uptime required.

### Answer

**Key constraints:**
- Data sovereignty: no OpenAI/Anthropic — must use on-premises models
- 99.9% uptime = 8.7 hours downtime/year — needs redundancy
- Legal accuracy critical — hallucinations are liability

**Architecture:**

```
Lawyer Browser / Desktop App
          │
          ▼
     Load Balancer (nginx, active-active)
     ┌────┴────┐
     │         │
  App Node 1  App Node 2   (hot standby)
     │
     ▼
FastAPI + LangChain
     │
  ┌──┴──────────────┐
  │                 │
  ▼                 ▼
Qdrant            Llama 3 70B
(self-hosted,    (vLLM on A100 GPU)
 2-node cluster)  + INT8 quantization
  │
  ▼
Hybrid search
(BM25 + vector + rerank)
     │
     ▼
Cross-encoder reranker
(BAAI/bge-reranker-large)
```

**Ingestion pipeline:**
1. PDF/DOCX → Apache Tika text extraction
2. Recursive text splitting (512 tokens, 50 overlap)
3. Contextual chunk prefixing (case name, date, document type)
4. Embed with `bge-large-en-v1.5` (self-hosted)
5. Store in Qdrant with metadata (case_id, date, doc_type, attorney)

**Uptime strategy:**
- 2 app nodes behind load balancer
- Qdrant 2-node replication
- GPU inference: primary + warm standby (costly but required)
- Automated health checks every 30s, auto-failover

**Accuracy (critical):**
- Always cite source chunks with document name + page
- "I found these relevant passages — I cannot find a definitive answer in the documents" pattern
- Attorney review workflow: flag low-confidence answers
- No summarization without explicit user request

---

## Question 3: GitHub Code Review Assistant

**Prompt:** Design an AI code review assistant that integrates with GitHub.

### Answer

**Requirements to clarify:**
- Review all PRs or opt-in? (opt-in to avoid noise)
- Languages? (affects which models/linters)
- What kind of feedback? (bugs, style, security, best practices)

**Architecture:**

```
GitHub PR opened/updated
         │
    GitHub Webhook (HMAC verified)
         │
         ▼
    Review Service (FastAPI)
         │
    Background Job Queue (Redis + Celery)
         │
    ┌────┴──────────────────┐
    │                       │
    ▼                       ▼
Static Analysis       LLM Review
(ruff, semgrep,      (Claude Sonnet)
 bandit)              ← diff + context
    │                       │
    └──────┬────────────────┘
           │
           ▼
    Merge feedback
    (deduplicate, rank by severity)
           │
           ▼
    GitHub API → PR comment
    (line-level annotations)
```

**Prompt strategy:**
- Send only the diff (not whole repo) — limit tokens
- Include surrounding context (10 lines before/after)
- System prompt: "You are a senior engineer. Review for: bugs, security, performance, readability. Be concise. Format as JSON array of {line, severity, comment}."
- Few-shot: 5 example high-quality review comments

**Cost control:**
- Only review diffs ≤ 500 lines (skip huge refactors)
- Cache reviews for identical diffs (rare but possible in rebases)
- Batch API for non-blocking PRs (50% cheaper)

**Evaluation:**
- Developer acceptance rate (did they act on the comment?)
- False positive rate (dismissals with "not applicable")
- Time-to-merge (does review help or slow down?)

---

## Question 4: Content Moderation at 1M Posts/Day

**Prompt:** Design an AI content moderation system for a social platform with 1 million posts per day.

### Answer

**1M posts/day = 11.6 posts/second. Peaks 3-5×: need to handle ~50/sec.**

**Two-stage pipeline (cost-critical):**

```
Post Created
    │
    ▼
Stage 1: Fast classifier (< 10ms)
  ─ Rule-based: known bad words, URLs on blocklist
  ─ Fine-tuned distilBERT (CPU inference, cheap)
  → Score 0-1
    │
    ├─ Score < 0.1 → Auto-approve (70% of traffic)
    ├─ Score > 0.9 → Auto-reject (10% of traffic)
    └─ Score 0.1-0.9 → Stage 2 (20% of traffic)
         │
         ▼
Stage 2: LLM review (Claude Haiku, fast + cheap)
  ─ Full context: post + user history + thread
  → {decision: approve/reject/escalate, categories: [], confidence}
         │
         ├─ approve/reject → automated action
         └─ escalate → Human Review Queue
                  │
                  ▼
            Human moderators
            (UI with context, 1-click decisions)
            → feedback loops back to fine-tune Stage 1
```

**Scale math:**
- Stage 1 (80% traffic): distilBERT, $0.000001/post = $0.80/day ✓
- Stage 2 (20% traffic): Haiku, 200 tokens × $0.00000025 = $0.00005/post → $10/day ✓
- Human review (2% traffic): 20,000 posts/day, manageable team

**Latency:**
- Stage 1: async Kafka consumer, < 100ms
- Stage 2: async worker pool, < 2s (not blocking post creation)
- Posts shown immediately, removed if rejected (post-hoc moderation)

**Evaluation:**
- Precision/recall on labeled test set (weekly)
- Appeal rate (users contesting decisions)
- False positive rate by demographic (bias monitoring)

---

## Question 5: E-Commerce AI Search for 10M Products

**Prompt:** Design an AI-powered search for an e-commerce site with 10 million products.

### Answer

**Challenges:** 10M products, query latency < 200ms, personalization, multi-modal (text + images).

**Architecture:**

```
User types query
       │
       ▼
Query Understanding Service
  ─ Spell correction
  ─ Query expansion (synonyms: "couch" → "sofa")
  ─ Intent classification (product search vs. category browse)
       │
       ▼
  ┌────┴─────────────────────┐
  │                          │
  ▼                          ▼
Elasticsearch             pgvector / Qdrant
(keyword: BM25)           (semantic: embeddings)
  └──────────────┬───────────┘
                 │
                 ▼
          RRF Fusion (combine rankings)
                 │
                 ▼
          Business Rules Layer
          (boost in-stock, penalize low-rated,
           apply user's size/preference filters)
                 │
                 ▼
          Personalization Re-Ranker
          (user's purchase/click history
           → trained ranking model)
                 │
                 ▼
          Return top 20 results → UI
```

**Embedding strategy for 10M products:**
- Embed at index time: title + bullet points + category (not reviews)
- Model: `text-embedding-3-small` (cheap) or `bge-large` (self-hosted)
- Batch embed new products via Batch API
- Incremental updates: embed delta daily, full re-index weekly

**Latency budget (< 200ms):**
- Query understanding: 10ms (in-memory models)
- Elasticsearch + pgvector parallel: 50ms
- RRF fusion: 5ms
- Business rules: 5ms
- Personalization: 30ms (pre-computed user vectors)
- Total: ~100ms ✓

**AI-assisted features:**
- "Complete the look" — visual embeddings (CLIP) for image similarity
- Semantic query understanding: "affordable laptop for college" → filters price range + category
- LLM-generated product descriptions for SEO (batch, offline)

**Evaluation:**
- NDCG@10 (offline, labeled test set)
- Click-through rate (online A/B test)
- Add-to-cart rate (conversion)
- Zero-result rate (queries that return nothing)
