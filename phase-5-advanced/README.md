# Phase 5 — Advanced Topics

> "This is what separates senior AI engineers from everyone else."

Phase 5 covers the topics that don't show up in tutorials but dominate production AI systems and system design interviews. You'll move from "I can call an API" to "I can design, evaluate, deploy, and maintain AI systems at scale."

---

## Overview

| Section | Topic | Core Skill |
|---------|-------|-----------|
| 5.1 | Fine-Tuning | Adapt models to your domain without retraining from scratch |
| 5.2 | Agents | Build LLMs that take actions in the world via tool use |
| 5.3 | Multimodal | Process images, audio, and video alongside text |
| 5.4 | Advanced RAG | Hybrid search, reranking, HyDE, corrective retrieval |
| 5.5 | System Design | Design AI systems end-to-end for interviews and production |
| 5.6 | Quantization | Run large models cheaply on constrained hardware |
| 5.7 | MLOps | CI/CD for ML, model registries, A/B testing, canary deploys |

---

## Why These Topics Matter

Most engineers can wire together an OpenAI API call. Senior engineers understand:

- **When NOT to fine-tune** (it's usually the wrong answer)
- **Why agents fail in production** and how to design around it
- **How to make RAG actually work** beyond the "stuff chunks in a vector DB" tutorial approach
- **How to deploy safely** using shadow traffic, canary releases, and automatic rollback
- **How to talk through a system design** in an interview with a structured, production-aware framework

These skills are the difference between a $120k role and a $200k+ role.

---

## Progress Checklist

### 5.1 Fine-Tuning
- [ ] Understand the prompting → RAG → fine-tuning decision framework
- [ ] Know what LoRA is and why it's orders of magnitude cheaper than full fine-tuning
- [ ] Know what QLoRA adds on top of LoRA
- [ ] Understand the ChatML and Alpaca dataset formats
- [ ] Be able to estimate fine-tuning cost for a given dataset
- [ ] Have run the LoRA training skeleton (`examples.py`)

### 5.2 Agents
- [ ] Explain the ReAct pattern (Thought → Action → Observation loop)
- [ ] Understand why tool descriptions are critical to agent reliability
- [ ] Build a working agent from scratch without LangChain
- [ ] Implement loop detection and budget caps as safety mechanisms
- [ ] Sketch a multi-agent orchestrator architecture

### 5.3 Multimodal
- [ ] Know the difference between base64 and URL image passing, and when to use each
- [ ] Write the correct API message format for vision (OpenAI and Anthropic)
- [ ] Know how to extract structured data from invoice/receipt images
- [ ] Understand how Whisper works at a conceptual level
- [ ] Explain how CLIP enables text-to-image search

### 5.4 Advanced RAG
- [ ] Explain why naive top-k vector search fails and what each technique fixes
- [ ] Implement BM25 from scratch in Python
- [ ] Understand Reciprocal Rank Fusion (RRF) and implement it
- [ ] Explain HyDE and when it helps vs hurts
- [ ] Know the difference between bi-encoder retrieval and cross-encoder reranking
- [ ] Implement parent-child chunking

### 5.5 System Design
- [ ] Walk through the 7-step framework without notes
- [ ] Calculate per-request cost and monthly total for a given system
- [ ] Know what to ask in the requirements phase (the 6 clarifying questions)
- [ ] Understand the 45-minute time allocation
- [ ] Have practiced at least 3 of the 5 design questions

### 5.6 Quantization
- [ ] Know memory requirements for common model sizes in float32/float16/INT8/INT4
- [ ] Know which GGUF variant to use for a given RAM constraint
- [ ] Understand what Flash Attention 2 does (memory-efficient attention computation)
- [ ] Know what the KV cache is and why it matters for inference speed
- [ ] Be able to run a quantized model locally with llama.cpp

### 5.7 MLOps
- [ ] Implement a model registry that tracks version + prompt + eval score
- [ ] Understand shadow mode and why it's safer than direct A/B testing
- [ ] Implement deterministic A/B routing via user_id hash
- [ ] Understand canary deployment (5% → 25% → 100%) with automatic rollback
- [ ] Be able to version prompts in git and explain rollback procedure

---

## Prerequisites

Before starting Phase 5, you should have completed:
- Phase 3: LLMs & APIs (embeddings, RAG, prompt engineering)
- Phase 4: Production (FastAPI, async, caching, databases)

You should be comfortable with:
- Python async/await
- Pydantic models
- Basic numpy operations
- REST API design

---

## How to Use This Phase

Each section has:
- **`notes.md`** — Read this first. Dense, reference-quality notes with every concept explained.
- **`examples.py`** — Run this. Runnable code demonstrating every concept.
- **`exercises.py`** (where provided) — Attempt before reading solutions.

Recommended order: read notes.md → run examples.py → attempt exercises before looking at solutions.

Time estimate: 3-4 weeks at 2 hours/day, or 1 week full-time.
