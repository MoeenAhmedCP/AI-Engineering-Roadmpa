# 5.4 Advanced RAG

## Why Naive RAG Fails

Naive RAG is simple: chunk the documents, embed everything, retrieve the top-k chunks by cosine
similarity, and pass them to the LLM. This works well on demos. In production, it fails in
predictable ways:

**Problem 1: Chunk context loss.** A chunk saying "It increased by 47% in the following quarter"
has lost its subject. The chunk has no idea what "it" refers to because the context was in the
previous chunk. The LLM cannot answer the question.

**Problem 2: Keyword gaps.** A user asks about "heart attack" but your documents use "myocardial
infarction." Embedding similarity helps but doesn't fully bridge medical/legal/technical jargon gaps.
BM25 keyword search fails completely; semantic search partially recovers.

**Problem 3: Top-k quality issues.** With top-k=5, you might retrieve 3 relevant chunks and 2
noise chunks. The LLM might hallucinate by anchoring to the noisy chunks rather than the relevant ones.

**Problem 4: Retrieval-generation mismatch.** A small chunk might rank well for retrieval but lack
the surrounding context needed for a good generation. The model knows the fact but can't explain it
properly.

These problems have specific solutions that form the advanced RAG toolkit.

---

## Hybrid Search: BM25 + Vector

The two main search paradigms complement each other:

**BM25 (Okapi BM25):** A probabilistic keyword-based ranking function. Descendant of TF-IDF.
Excellent at exact keyword matching — if the user says "Article 15 GDPR," BM25 will find that
exact phrase. Completely fails on semantic meaning: "What are my data rights?" won't match
"GDPR Article 17 Right to Erasure."

**Vector search:** Dense semantic embeddings capture meaning. "Heart attack" and "myocardial
infarction" have high similarity. Weak at exact entity names, IDs, codes, and rare terms.

**Hybrid:** Combine both. Run BM25 and vector search independently, then merge results using
Reciprocal Rank Fusion.

---

## Reciprocal Rank Fusion (RRF)

RRF is a simple, parameter-light method for combining multiple ranked lists:

```
RRF_score(doc) = Σ 1 / (k + rank_in_list_i)
```

Where `k` is a constant (typically 60) that dampens the impact of very high-ranked results.

**Why it works:** A document that ranks 3rd in BM25 and 5th in vector search is probably very
relevant. A document that ranks 1st in BM25 but doesn't appear in vector search is less certain.
RRF captures this intuition without needing to calibrate score scales between the two systems
(BM25 scores and cosine distances are on completely different scales and cannot be simply added).

```python
def rrf(bm25_ranking: list[int], vector_ranking: list[int], k: int = 60) -> list[int]:
    scores = {}
    for rank, doc_id in enumerate(bm25_ranking):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(vector_ranking):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

---

## Reranking

After initial retrieval (fast but approximate), apply a more expensive reranker to the top-N
candidates to get better final ordering.

**Bi-encoder (embedding model):** Embeds query and document independently, computes cosine similarity.
Fast because document embeddings are pre-computed. Approximate.

**Cross-encoder:** Takes (query, document) as a single input and outputs a relevance score. Much
more accurate because it can model interactions between query and document tokens. 10–100x slower.
Cannot pre-compute because it needs the query.

**Workflow:** Retrieve top-50 with bi-encoder (fast), rerank with cross-encoder, use top-5.

Popular rerankers:
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (Hugging Face)
- Cohere Rerank API
- BGE-reranker-large

---

## HyDE: Hypothetical Document Embeddings

**Problem:** The query "What causes myocardial infarction?" is short and vague from an embedding
perspective. Your documents contain long, detailed explanations. The short query embedding may not
closely match the detailed document embedding even though they're semantically related.

**HyDE solution:**
1. Ask the LLM to generate a hypothetical answer to the query (without retrieving anything)
2. Embed the hypothetical answer (which is now long and detailed)
3. Use this embedding to search for similar real documents

The hypothetical answer uses domain-appropriate language and structure, so it matches real documents
better than the raw question would.

```python
hypothetical = llm("Answer this briefly: " + query)
results = vector_store.search(embed(hypothetical), k=5)
```

**Tradeoff:** Extra LLM call per query (cost + latency). Worth it for high-stakes retrieval.

---

## Parent-Child Chunking

**Problem:** Small chunks retrieve precisely but lack context for generation. Large chunks provide
context but retrieve noisily.

**Solution:** Index small child chunks for retrieval, return the larger parent chunk for generation.

```
Document
├── Parent chunk 1 (500 words)
│   ├── Child chunk 1a (100 words) ← indexed for retrieval
│   ├── Child chunk 1b (100 words) ← indexed for retrieval
│   └── Child chunk 1c (100 words) ← indexed for retrieval
└── Parent chunk 2 (500 words)
    ├── Child chunk 2a (100 words)
    └── Child chunk 2b (100 words)
```

When child chunk 1b is retrieved, return all of parent chunk 1 to the LLM. The LLM gets full
context, but retrieval precision is maintained by the small child chunks.

---

## Multi-Query Retrieval

**Problem:** A single query formulation may miss relevant documents that are more closely aligned
with alternative phrasings.

**Solution:** Generate multiple query variants and take the union of results.

```python
variants = llm(f"""
Generate 3 different search queries to find information about: {original_query}
Return as a JSON list.
""")
# variants = ["query A", "query B", "query C"]
all_results = set()
for v in variants:
    all_results.update(vector_store.search(v))
```

Deduplication is by document ID. This improves recall at the cost of more embedding computations
and slightly more noisy candidates going to the reranker.

---

## Corrective RAG

**Problem:** Sometimes the retrieved documents are simply not relevant to the query. Passing garbage
context to the LLM produces garbage answers (or worse, hallucinations dressed up in real text).

**Solution:** After retrieval, evaluate relevance. If below a threshold, re-query with a modified
strategy (different query, different retrieval method, web search).

```python
for doc in retrieved_docs:
    relevance = evaluator.score(query, doc)  # LLM-based 0-1 score
    if relevance < 0.5:
        # Discard or flag; re-retrieve with modified query
        ...
```

This adds latency but significantly improves answer quality on out-of-distribution queries.

---

## Contextual Retrieval (Anthropic)

Published by Anthropic in 2024, contextual retrieval improves embedding quality by prepending
document context to each chunk before embedding:

**Without context:**
> "The company reported quarterly revenue of $2.3 billion."

**With context:**
> "This chunk is from Acme Corp's Q3 2024 earnings report, discussing financial results.
> The company reported quarterly revenue of $2.3 billion."

The contextualized chunk produces an embedding that captures both the specific fact and the broader
document context, dramatically improving retrieval precision.

Anthropic reported this technique reduces retrieval failures by 49% on their test datasets, and
by up to 67% when combined with reranking.

**Implementation note:** Use prompt caching (Anthropic's beta feature) when prepending the same
document summary to hundreds of chunks — the document summary is identical for every chunk and
will be served from cache after the first call.

---

## RAPTOR: Recursive Abstractive Processing

RAPTOR builds a hierarchical index by recursively summarizing clusters of documents:

1. Embed all leaf chunks
2. Cluster similar chunks (k-means or GMM)
3. Summarize each cluster → creates parent nodes
4. Embed and cluster the summaries → higher-level nodes
5. Repeat until a single root summary

At query time, search at multiple levels. For broad questions, match against high-level summaries.
For specific questions, match against leaf chunks.

RAPTOR is particularly effective for question-answering over long documents where relevant
information is scattered across many sections.

---

## Summary: Choosing Techniques

| Problem | Technique |
|---|---|
| Keyword vs semantic mismatch | Hybrid search (BM25 + vector) + RRF |
| Top-k has noise | Reranking (cross-encoder) |
| Short query, long docs | HyDE |
| Context loss at chunk boundaries | Parent-child chunking |
| Single query misses variants | Multi-query retrieval |
| Retrieved docs are irrelevant | Corrective RAG |
| Embedding quality is poor | Contextual retrieval |
| Long documents, scattered info | RAPTOR |
