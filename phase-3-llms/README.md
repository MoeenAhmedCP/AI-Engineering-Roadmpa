# Phase 3: LLMs & Modern AI APIs

> "80% of AI engineering jobs are here."

This phase covers the core knowledge stack for working professionally with large language models — from how transformers work internally, to building production RAG systems, to prompt engineering, structured outputs, and conversation memory. These are the skills that appear in virtually every AI engineering role.

---

## Overview Table

| Section | Topic | Key Concept | Hands-On Project |
|---------|-------|-------------|-----------------|
| 3.1 | Transformers & Attention | Self-attention, Q/K/V, multi-head, context windows | Implement attention from scratch in numpy |
| 3.2 | Embeddings & Vector Search | Cosine similarity, vector DBs, ANN indexing | Custom VectorStore with semantic search |
| 3.3 | Prompt Engineering | System prompts, few-shot, CoT, temperature | PromptTestHarness with ranked variants |
| 3.4 | Retrieval-Augmented Generation | Chunking, retrieval, augmentation, generation | Full RAG pipeline — no LangChain |
| 3.5 | OpenAI & Anthropic APIs | Messages, streaming, tool use, cost estimation | Simulated full API workflow |
| 3.6 | LangChain & LlamaIndex | LCEL, chains, retrievers, memory | Simulated RAG chain with real patterns |
| 3.7 | Open-Source Models | Ollama, HuggingFace, GGUF, model families | Model comparison + decision flowchart |
| 3.8 | Structured Output | Pydantic, JSON mode, Instructor, retries | Extraction pipeline with self-healing parser |
| 3.9 | Conversation Memory | Buffer, window, summary, SQLite, token budget | Multi-strategy memory system |

---

## Progress Checklist

### 3.1 Transformers & Attention
- [ ] Read notes.md — understand tokens, embeddings, self-attention
- [ ] Run examples.py — watch attention computed from scratch
- [ ] Explain Q/K/V intuition to yourself without notes
- [ ] Understand why transformers beat RNNs

### 3.2 Embeddings & Vector Search
- [ ] Read notes.md — cosine vs Euclidean, vector DB comparison
- [ ] Run examples.py — semantic search on 15 sentences
- [ ] Complete exercises.py — cosine from scratch, BM25, MMR
- [ ] Know when to use FAISS vs Chroma vs pgvector

### 3.3 Prompt Engineering
- [ ] Read notes.md — system prompt, few-shot, CoT, temperature
- [ ] Run examples.py — PromptTestHarness with mock scores
- [ ] Complete exercises.py — legal summarizer system prompt
- [ ] Write a prompt, then improve it using at least 2 techniques

### 3.4 RAG (Most Important Section)
- [ ] Read notes.md — understand all 4 chunking strategies
- [ ] Run rag_pipeline.py — ingest 5 docs, answer 3 questions
- [ ] Complete exercises.py — recursive chunker, MMR, faithfulness
- [ ] Be able to describe full RAG flow from memory

### 3.5 OpenAI & Anthropic APIs
- [ ] Read notes.md — messages array, streaming, tool use, costs
- [ ] Run examples.py — simulated chat, streaming, tool loop
- [ ] Memorize the cost table (Haiku/mini vs Sonnet/4o vs Opus)
- [ ] Understand when to use batch API and prompt caching

### 3.6 LangChain & LlamaIndex
- [ ] Read notes.md — LCEL pipe syntax, when to use framework vs scratch
- [ ] Run examples.py — simulated RAG chain
- [ ] Know the LangChain v0.1 vs v0.3 differences
- [ ] Decide: for your next project, framework or scratch?

### 3.7 Open-Source Models
- [ ] Read notes.md — model families, Ollama, GGUF
- [ ] Run examples.py — decision flowchart, model comparison table
- [ ] Install Ollama and pull llama3 (optional but recommended)
- [ ] Know the decision framework: API vs open-source

### 3.8 Structured Output
- [ ] Read notes.md — JSON mode vs structured outputs vs Instructor
- [ ] Run examples.py — extraction pipeline, retry on ValidationError
- [ ] Complete exercises.py — job posting schema, invoice pipeline
- [ ] Know when to use tool calling vs JSON mode vs Instructor

### 3.9 Conversation Memory
- [ ] Read notes.md — 5 memory strategies and tradeoffs
- [ ] Run examples.py — buffer, window, summary, SQLite
- [ ] Understand token budget management
- [ ] Know how to isolate memory per user in a multi-user system

---

## Key Mental Models for This Phase

**The RAG mantra:** Load → Chunk → Embed → Store → Retrieve → Augment → Generate

**Model selection:** Haiku/mini for cheap+fast, Sonnet/4o for balanced, Opus/o1 for hard reasoning

**Prompt order of impact:** System prompt > Few-shot examples > CoT instruction > Temperature

**Open-source decision:** Privacy required OR cost at scale OR need fine-tuning → open-source

**Memory strategy:** Short session → buffer; Long session → window or summary; Many users → SQLite + session IDs

---

## Prerequisites

- Phase 1 (Python fundamentals)
- Phase 2 (APIs and data)
- Basic comfort with classes and async Python

## What Comes After

- Phase 4: AI Agents & Tool Use (builds directly on 3.4 RAG and 3.5 APIs)
- Phase 5: Fine-tuning & Model Adaptation
- Phase 6: Production Deployment & MLOps
