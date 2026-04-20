# 3.4 RAG — Retrieval-Augmented Generation

## The Problem RAG Solves

Large language models have three fundamental limitations that RAG directly addresses:

**Knowledge cutoff.** Models are trained on data up to a fixed date. Ask GPT-4 about an event from last month and it either hallucinates or admits ignorance. RAG lets you attach a live knowledge source — your database, your docs, your API — to any frozen model.

**Private data.** You cannot fine-tune a public model on every company's internal wiki, contract library, or support ticket history — nor would you want to. RAG keeps your data in your own vector store and retrieves only what is needed per query.

**Hallucination.** Without grounding, a model invents plausible-sounding facts. When you inject the actual relevant passages into the prompt and instruct the model to cite them, grounding improves dramatically. The model can still hallucinate, but the bar is much higher because it has real text to work from.

RAG is now the default architecture for any LLM application that needs to reason over a body of knowledge larger than fits in one context window.

---

## The RAG Flow

RAG has two separate pipelines that run at different times.

### Ingest Pipeline (runs once, or on updates)

```
Raw documents
  → Load  (read bytes from disk / S3 / DB)
  → Chunk (split into overlapping pieces)
  → Embed (convert each chunk to a vector)
  → Store (persist to a vector database)
```

This is a batch job. You run it when your knowledge base changes.

### Query Pipeline (runs on every user request)

```
User question
  → Embed question
  → Retrieve top-k similar chunks from vector store
  → Augment: build a prompt = system instructions + retrieved chunks + question
  → Generate: LLM produces an answer grounded in the retrieved context
  → Return answer (with optional citations)
```

The key insight: you never send the whole document to the LLM. You send only the few chunks most relevant to the question.

---

## Chunking Strategies

Chunking is where most RAG systems fail or succeed. The right strategy depends on your document type and query patterns.

### Fixed-Size Chunking
Split on character count, every N characters with an overlap of M.

- Simple to implement. Works for homogeneous text.
- Ignores sentence and paragraph boundaries — can split mid-sentence.
- Best for: log files, code, highly uniform data.

### Recursive Character Splitting
Tries to split on `\n\n`, then `\n`, then `. `, then ` `, falling back to characters only when needed.

- Respects semantic structure as much as possible given a size budget.
- The default choice for most text documents.
- LangChain's `RecursiveCharacterTextSplitter` implements this.

### Semantic Chunking
Embeds every sentence, computes similarity between adjacent sentences, and splits when similarity drops below a threshold — i.e., when the topic shifts.

- Produces chunks that are semantically coherent.
- More expensive (requires embedding during ingestion).
- Best for: long, topic-dense documents like research papers or legal contracts.

### Parent-Child Chunking
Index small chunks (100–200 tokens) for retrieval precision; but when a chunk matches, return its larger parent (500–1000 tokens) to the LLM for richer context.

- Combines precision of small chunks with context of large ones.
- Requires storing relationships between chunks.
- Best for: documents where full sections matter for answering but matching is precise.

### Tradeoffs at a Glance

| Strategy       | Complexity | Retrieval precision | Context quality | Best for               |
|----------------|------------|---------------------|-----------------|------------------------|
| Fixed-size     | Low        | Medium              | Low             | Uniform data, logs     |
| Recursive      | Low        | Good                | Good            | General docs, default  |
| Semantic       | High       | High                | High            | Dense academic text    |
| Parent-child   | High       | High                | High            | Structured documents   |

---

## Chunk Size and Overlap Tradeoffs

**Too small (< 100 tokens):** Chunks lose enough context that individual chunks are ambiguous. "The deadline is June 15th" with no surrounding context is nearly useless.

**Too large (> 1000 tokens):** The embedding vector tries to represent too much content. When you retrieve this chunk for a narrow question, most of the chunk is irrelevant — diluting the signal and wasting context window space.

**Sweet spot:** 200–500 tokens for most use cases.

**Overlap:** Typically 10–20% of chunk size. Without overlap, a key sentence at the boundary of two chunks might be split and appear in neither. Overlap ensures boundary content is always represented in at least one chunk. More overlap means more storage and more redundancy during retrieval.

---

## Embedding Models

The embedding model converts text to a dense vector. Similar texts get similar vectors (cosine distance near 0).

| Model | Dims | Notes |
|---|---|---|
| `text-embedding-3-small` (OpenAI) | 1536 | Fast, cheap, solid baseline |
| `text-embedding-3-large` (OpenAI) | 3072 | Best OpenAI quality, 2× cost |
| `all-MiniLM-L6-v2` (sentence-transformers) | 384 | Free, runs locally, good for English |
| `bge-large-en-v1.5` (BAAI) | 1024 | State-of-the-art open model, higher RAM |
| `mxbai-embed-large` | 1024 | Strong open-source alternative |

Critical rule: use the same embedding model at ingest and at query time. Mixing models produces meaningless similarity scores.

---

## Vector Stores

| Store | Hosting | Persistence | Best for |
|---|---|---|---|
| FAISS | In-process | No (save to disk manually) | Prototyping, no infra |
| Chroma | In-process or server | Yes (SQLite) | Local dev, small projects |
| pgvector | Postgres extension | Yes | Projects already on Postgres |
| Pinecone | Managed cloud | Yes | Production, scale, no ops |
| Weaviate | Self-hosted or cloud | Yes | Hybrid search needs |
| Qdrant | Self-hosted or cloud | Yes | High-performance, Rust-based |

For a new project: start with Chroma locally, migrate to pgvector or Pinecone when you need production persistence or scale.

---

## Retrieval

**similarity_search(query, k=4):** Returns the k most similar chunks. Most systems use k between 3 and 8. Larger k gives the LLM more information but risks including irrelevant chunks.

**Threshold filtering:** Only return chunks with similarity score above a threshold (e.g., 0.75). Prevents the LLM from getting forced to answer with unrelated context.

**MMR (Maximal Marginal Relevance):** Balances relevance with diversity — avoids returning 5 nearly identical chunks when one would do.

**Hybrid search:** Combine dense vector search with BM25 keyword search. Often beats pure vector search for precise term queries (e.g., product codes, names).

---

## Prompt Construction

The augmented prompt follows this structure:

```
System: You are a helpful assistant. Answer using only the provided context.
If the context does not contain the answer, say "I don't have enough information."
Cite the source of each claim using [Source: <filename>].

Context:
[Source: company_policy.pdf, chunk 3]
"Annual leave must be requested 5 days in advance..."

[Source: employee_handbook.pdf, chunk 12]
"Remote work is permitted up to 3 days per week..."

User: How much notice do I need to give for annual leave?
```

Key instructions to include: (1) answer only from context, (2) cite sources, (3) say "I don't know" if context is insufficient. Without instruction 3, the model will happily fill gaps with hallucination.

---

## Evaluation with RAGAS

RAGAS is the standard framework for evaluating RAG pipelines. Four metrics:

- **Context Precision:** Of the retrieved chunks, what fraction were actually relevant?
- **Context Recall:** Of the chunks needed to answer, what fraction were retrieved?
- **Answer Faithfulness:** Is every claim in the answer supported by the retrieved context? (0 = pure hallucination, 1 = fully grounded)
- **Answer Relevancy:** Does the answer actually address the question?

A production RAG system should be evaluated on all four. Faithfulness is the most critical — if it drops, your retrieval or prompt construction is broken.

---

## Common Failure Modes

| Failure | Symptoms | Fix |
|---|---|---|
| Wrong chunks retrieved | Answer is off-topic or misses obvious facts | Adjust chunk size; improve embedding model; add metadata filters |
| Answer not grounded | Model ignores context, uses training knowledge | Strengthen system prompt; lower temperature |
| No relevant chunks found | "I don't have information" on answerable questions | Check embedding quality; lower similarity threshold; verify data was ingested |
| Chunking breaks context | Answers miss facts that span chunk boundaries | Increase overlap; use parent-child chunking |
| Too much context | Slow, expensive, model misses needle in haystack | Reduce k; add threshold filtering; use MMR |

---

## Mental Model

Think of RAG as giving the LLM a "just-in-time textbook." You don't stuff the whole book in the prompt — you look up the relevant pages and hand them over. The LLM's job is to synthesize an answer from those pages, not to recall from memory.
