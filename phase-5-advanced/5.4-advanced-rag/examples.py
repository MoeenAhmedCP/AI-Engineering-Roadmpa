"""
5.4 Advanced RAG Techniques — Examples
Demonstrates: BM25, hybrid search, RRF, HyDE, multi-query, parent-child chunking.
Run: python examples.py
"""

import math
import re
import hashlib
from collections import defaultdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. BM25 Index (keyword search)
# ─────────────────────────────────────────────────────────────────────────────

class BM25Index:
    """BM25 (Okapi BM25) keyword-based ranking."""
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        self._tokenize_all()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-z]+\b', text.lower())

    def _tokenize_all(self):
        self.tokenized = [self._tokenize(d) for d in self.docs]
        self.avgdl = sum(len(t) for t in self.tokenized) / max(self.N, 1)
        self.df: dict[str, int] = defaultdict(int)
        for tokens in self.tokenized:
            for term in set(tokens):
                self.df[term] += 1

    def score(self, query: str, doc_idx: int) -> float:
        query_terms = self._tokenize(query)
        doc_tokens = self.tokenized[doc_idx]
        dl = len(doc_tokens)
        tf_map: dict[str, int] = defaultdict(int)
        for t in doc_tokens:
            tf_map[t] += 1

        total = 0.0
        for term in query_terms:
            if term not in self.df:
                continue
            tf = tf_map[term]
            idf = math.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += idf * numerator / denominator
        return total

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(idx, s) for idx, s in scores[:k] if s > 0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reciprocal Rank Fusion (RRF)
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """
    Combine multiple ranked lists using RRF.
    Each ranking is a list of doc indices from best to worst.
    Returns fused ranking (best first).
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fake vector embeddings (deterministic, hash-based)
# ─────────────────────────────────────────────────────────────────────────────

def fake_embed(text: str, dims: int = 32) -> list[float]:
    h = hashlib.md5(text.lower().encode()).digest()
    vec = []
    for i in range(dims):
        byte_val = h[i % 16]
        vec.append((byte_val / 127.5) - 1.0)
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2 + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hybrid Searcher (BM25 + vector + RRF)
# ─────────────────────────────────────────────────────────────────────────────

class HybridSearcher:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.bm25 = BM25Index(documents)
        self.embeddings = [fake_embed(d) for d in documents]

    def vector_search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        qvec = fake_embed(query)
        sims = [(i, cosine_similarity(qvec, emb)) for i, emb in enumerate(self.embeddings)]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def search(self, query: str, k: int = 5) -> list[dict]:
        bm25_results = self.bm25.search(query, k=10)
        vec_results = self.vector_search(query, k=10)

        bm25_ranking = [idx for idx, _ in bm25_results]
        vec_ranking = [idx for idx, _ in vec_results]

        fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking])[:k]
        return [{"idx": i, "text": self.documents[i][:80] + "..."} for i in fused]


# ─────────────────────────────────────────────────────────────────────────────
# 5. HyDE (Hypothetical Document Embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def generate_hypothetical_answer(question: str) -> str:
    """In production: call LLM to generate a hypothetical answer. Here: rule-based mock."""
    templates = {
        "what": f"A document explaining {question.lower().replace('what is', '').strip()} in detail.",
        "how": f"Step-by-step guide: {question.lower().replace('how to', '').strip()}.",
        "why": f"The reason is that {question.lower().replace('why', '').strip()}.",
    }
    for prefix, template in templates.items():
        if question.lower().startswith(prefix):
            return template
    return f"Relevant information about: {question}"


def hyde_retrieval(question: str, searcher: HybridSearcher, k: int = 5) -> list[dict]:
    """Retrieve by embedding a hypothetical answer instead of the raw question."""
    hypothetical = generate_hypothetical_answer(question)
    qvec = fake_embed(hypothetical)
    sims = [(i, cosine_similarity(qvec, emb)) for i, emb in enumerate(searcher.embeddings)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [{"idx": i, "score": round(s, 4), "text": searcher.documents[i][:80] + "..."} for i, s in sims[:k]]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multi-Query Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def generate_query_variants(question: str) -> list[str]:
    """In production: call LLM to rephrase. Here: simple transformations."""
    variants = [question]
    # Rephrase as "explain X"
    variants.append("explain " + question.lower().lstrip("what is ").lstrip("how does ").strip())
    # Rephrase as keyword search
    words = re.findall(r'\b[a-z]{4,}\b', question.lower())
    if words:
        variants.append(" ".join(words[:4]))
    return variants[:3]


def multi_query_retrieval(question: str, searcher: HybridSearcher, k: int = 5) -> list[dict]:
    """Generate multiple query variants, retrieve for each, deduplicate by RRF."""
    variants = generate_query_variants(question)
    all_rankings: list[list[int]] = []
    for variant in variants:
        results = searcher.bm25.search(variant, k=10)
        all_rankings.append([idx for idx, _ in results])

    fused = reciprocal_rank_fusion(all_rankings)[:k]
    return [{"idx": i, "text": searcher.documents[i][:80] + "..."} for i in fused]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Contextual Chunk (Anthropic technique)
# ─────────────────────────────────────────────────────────────────────────────

def contextual_chunk(chunk: str, doc_title: str, doc_summary: str) -> str:
    """Prepend document context to each chunk before embedding. Improves retrieval ~50%."""
    context = f"[From: {doc_title}. Context: {doc_summary}]\n\n"
    return context + chunk


# ─────────────────────────────────────────────────────────────────────────────
# 8. Parent-Child Chunker
# ─────────────────────────────────────────────────────────────────────────────

class ParentChildChunker:
    """
    Index small chunks (for precise retrieval).
    Retrieve large parent chunks (for full context).
    """
    def __init__(self, child_size: int = 100, parent_size: int = 400, overlap: int = 20):
        self.child_size = child_size
        self.parent_size = parent_size
        self.overlap = overlap

    def _split(self, text: str, size: int, overlap: int) -> list[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i:i + size])
            chunks.append(chunk)
            i += size - overlap
        return chunks

    def chunk_document(self, text: str, doc_id: str = "doc") -> list[dict]:
        parents = self._split(text, self.parent_size, self.overlap)
        result = []
        for p_idx, parent in enumerate(parents):
            children = self._split(parent, self.child_size, self.overlap)
            for c_idx, child in enumerate(children):
                result.append({
                    "child_id": f"{doc_id}_p{p_idx}_c{c_idx}",
                    "parent_id": f"{doc_id}_p{p_idx}",
                    "child_text": child,
                    "parent_text": parent,
                })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_DOCS = [
    "Python is a high-level programming language known for its simplicity and readability. It supports multiple paradigms including object-oriented, functional, and procedural programming.",
    "FastAPI is a modern web framework for building APIs with Python. It is based on standard Python type hints and provides automatic documentation with OpenAPI.",
    "RAG stands for Retrieval-Augmented Generation. It combines a retrieval system with a language model to answer questions using external knowledge.",
    "Vector databases store embeddings and enable fast similarity search. Popular options include Pinecone, Qdrant, Weaviate, and pgvector.",
    "Docker containers package applications with their dependencies, ensuring consistent environments across development, testing, and production.",
    "Transformers use self-attention mechanisms to model relationships between tokens in a sequence. The attention score is computed as softmax(QK^T / sqrt(d_k)).",
    "LoRA (Low-Rank Adaptation) enables fine-tuning large language models by training small adapter matrices instead of updating all model weights.",
    "Redis is an in-memory data structure store used as a database, cache, and message broker. It supports strings, hashes, lists, sets, and sorted sets.",
    "Kubernetes orchestrates containerized applications, handling deployment, scaling, and management of container workloads across clusters.",
    "Prompt engineering involves designing input prompts to guide language models toward desired outputs. Techniques include few-shot examples and chain-of-thought reasoning.",
]


def run_demo():
    print("=" * 60)
    print("Advanced RAG Techniques Demo")
    print("=" * 60)

    searcher = HybridSearcher(SAMPLE_DOCS)
    query = "How do vector databases work for similarity search?"

    print(f"\nQuery: {query!r}\n")

    # Naive BM25
    print("── Naive BM25 Search ──")
    bm25_results = searcher.bm25.search(query, k=3)
    for idx, score in bm25_results:
        print(f"  [{score:.3f}] {SAMPLE_DOCS[idx][:70]}...")

    # Hybrid
    print("\n── Hybrid Search (BM25 + Vector + RRF) ──")
    hybrid_results = searcher.search(query, k=3)
    for r in hybrid_results:
        print(f"  [{r['idx']}] {r['text']}")

    # HyDE
    print("\n── HyDE Retrieval ──")
    hypo = generate_hypothetical_answer(query)
    print(f"  Hypothetical answer: {hypo}")
    hyde_results = hyde_retrieval(query, searcher, k=3)
    for r in hyde_results:
        print(f"  [score={r['score']}] {r['text']}")

    # Multi-query
    print("\n── Multi-Query Retrieval ──")
    variants = generate_query_variants(query)
    print(f"  Variants generated: {variants}")
    mq_results = multi_query_retrieval(query, searcher, k=3)
    for r in mq_results:
        print(f"  [{r['idx']}] {r['text']}")

    # Contextual chunk
    print("\n── Contextual Chunk Example ──")
    raw_chunk = "It stores vectors and enables fast approximate nearest-neighbour search."
    ctx_chunk = contextual_chunk(raw_chunk, "Vector Databases Guide", "Overview of vector storage and retrieval systems")
    print(f"  Original: {raw_chunk}")
    print(f"  Contextual:\n{ctx_chunk}")

    # Parent-child chunking
    print("\n── Parent-Child Chunking ──")
    chunker = ParentChildChunker(child_size=15, parent_size=40, overlap=5)
    doc_text = " ".join(SAMPLE_DOCS[:3])
    chunks = chunker.chunk_document(doc_text, doc_id="sample")
    print(f"  Document split into {len(chunks)} child chunks across {len(set(c['parent_id'] for c in chunks))} parents")
    print(f"  Sample child: {chunks[0]['child_text'][:60]}...")
    print(f"  Its parent:   {chunks[0]['parent_text'][:80]}...")


if __name__ == "__main__":
    run_demo()
