"""
3.2 Embeddings & Vector Search — Examples
Run: python examples.py
No API keys needed. Optional: sentence-transformers, faiss-cpu
Install optionals: pip install sentence-transformers faiss-cpu
"""

import math
import hashlib
import struct
from typing import Any, Callable, Optional

# ─────────────────────────────────────────────
# OPTIONAL DEPENDENCY IMPORTS
# ─────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("[info] sentence-transformers not installed — using hash-based fake embeddings")
    print("       Install: pip install sentence-transformers")

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("[info] faiss not installed — FAISS demo will be skipped")
    print("       Install: pip install faiss-cpu numpy")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ─────────────────────────────────────────────
# FAKE EMBEDDINGS (stdlib only — deterministic)
# ─────────────────────────────────────────────

def fake_embed(text: str, dims: int = 64) -> list[float]:
    """
    Deterministic fake embedding using hash functions.
    Different texts → different vectors.
    Semantically similar texts → NOT necessarily similar vectors (this is the limitation).
    Use this only to test infrastructure, not to test retrieval quality.
    """
    # Use multiple hash seeds to fill all dimensions
    vector = []
    for i in range(0, dims, 8):
        seed = f"{i}:{text}"
        h = hashlib.sha256(seed.encode()).digest()
        # Unpack 8 floats from 32 bytes (4 bytes each)
        chunk = struct.unpack("8f", h[:32])
        # Normalize each float to [-1, 1]
        for val in chunk[:min(8, dims - i)]:
            # Map arbitrary float to [-1, 1]
            normalized = (val % 2.0) - 1.0
            vector.append(normalized)
    vec = vector[:dims]
    # L2 normalize
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude > 0:
        vec = [v / magnitude for v in vec]
    return vec


def get_embedder(dims: int = 64):
    """Returns (embed_fn, dims) — real or fake depending on what's installed."""
    if HAS_SENTENCE_TRANSFORMERS:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        real_dims = model.get_sentence_embedding_dimension()
        def embed_fn(text: str) -> list[float]:
            return model.encode(text).tolist()
        print(f"[embedder] Using all-MiniLM-L6-v2 ({real_dims} dims)")
        return embed_fn, real_dims
    else:
        def embed_fn(text: str) -> list[float]:
            return fake_embed(text, dims=dims)
        print(f"[embedder] Using hash-based fake embeddings ({dims} dims)")
        print("           NOTE: fake embeddings have NO semantic meaning.")
        print("           Retrieval results will be random, not semantic.")
        return embed_fn, dims


# ─────────────────────────────────────────────
# CUSTOM VECTORSTORE CLASS (stdlib only)
# ─────────────────────────────────────────────

class VectorStore:
    """
    Pure Python in-memory vector store.
    No external dependencies required.
    """

    def __init__(self):
        self._store: dict[str, dict] = {}
        # Each entry: { "text": str, "embedding": list[float], "metadata": dict }

    def add(self, doc_id: str, text: str, embedding: list[float], metadata: dict = None) -> None:
        """Add a document with its precomputed embedding."""
        self._store[doc_id] = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict = None
    ) -> list[dict]:
        """
        Search for top-k most similar documents.

        Args:
            query_embedding: the query vector
            top_k: number of results to return
            filter_metadata: optional dict of {key: value} — only return docs where
                             metadata[key] == value for all keys in filter_metadata

        Returns:
            List of dicts: [{id, text, score, metadata}, ...] sorted by score desc
        """
        results = []
        for doc_id, doc in self._store.items():
            # Metadata filter
            if filter_metadata:
                skip = False
                for key, value in filter_metadata.items():
                    if doc["metadata"].get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            score = self._cosine_similarity(query_embedding, doc["embedding"])
            results.append({
                "id": doc_id,
                "text": doc["text"],
                "score": score,
                "metadata": doc["metadata"],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f"VectorStore(documents={len(self._store)})"


# ─────────────────────────────────────────────
# COSINE VS EUCLIDEAN DISTANCE DEMO
# ─────────────────────────────────────────────

def demo_cosine_vs_euclidean():
    """
    Show why cosine similarity outperforms Euclidean distance for text embeddings.
    Uses simple 2D vectors for clarity — easy to visualize.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Cosine vs Euclidean Distance")
    print("=" * 60)

    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0.0

    def euclidean_dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    # Imagine 2D embedding space
    # The direction (angle) encodes semantic meaning
    # The magnitude might differ based on training, but is less meaningful

    query = [1.0, 1.0]          # "happy" — points in NE direction
    similar_short = [0.5, 0.5]  # "glad" — same direction, shorter vector
    similar_long = [3.0, 3.1]   # "joyful" — same direction roughly, longer vector
    dissimilar = [1.0, -1.0]    # "sad" — different direction

    vectors = [
        ("query (happy)", query),
        ("similar-short (glad)", similar_short),
        ("similar-long (joyful)", similar_long),
        ("dissimilar (sad)", dissimilar),
    ]

    print("\nVectors (2D for illustration):")
    for name, v in vectors:
        print(f"  {name:<25} {v}")

    print(f"\n{'Pair':<40} {'Cosine Sim':>12} {'Euclidean':>12} {'Cosine Rank':>12}")
    print("-" * 80)

    comparisons = [
        ("query vs similar-short", query, similar_short),
        ("query vs similar-long", query, similar_long),
        ("query vs dissimilar", query, dissimilar),
    ]

    results = []
    for name, a, b in comparisons:
        cos = cosine_sim(a, b)
        euc = euclidean_dist(a, b)
        results.append((name, cos, euc))

    # Sort by cosine similarity (descending) and euclidean (ascending)
    cos_ranked = sorted(results, key=lambda x: x[1], reverse=True)
    euc_ranked = sorted(results, key=lambda x: x[2])

    for i, (name, cos, euc) in enumerate(results):
        cos_rank = next(j + 1 for j, r in enumerate(cos_ranked) if r[0] == name)
        euc_rank = next(j + 1 for j, r in enumerate(euc_ranked) if r[0] == name)
        print(f"  {name:<38} {cos:>12.4f} {euc:>12.4f} {f'#{cos_rank}':>12}")

    print("\nKey insight:")
    print("  - Cosine correctly ranks similar-short and similar-long as BOTH similar to query")
    print("  - Euclidean penalizes similar-long (different magnitude) — ranks it as far away")
    print("  - For text embeddings: meaning = direction, not magnitude")
    print("  - Always use cosine similarity (or normalize then use dot product)")


# ─────────────────────────────────────────────
# SEMANTIC SEARCH DEMO
# ─────────────────────────────────────────────

SAMPLE_SENTENCES = [
    # AI/ML topics
    ("doc_01", "Transformers use self-attention to model relationships between tokens.", {"category": "architecture"}),
    ("doc_02", "The BERT model uses bidirectional attention for language understanding.", {"category": "architecture"}),
    ("doc_03", "GPT models generate text by predicting the next token autoregressively.", {"category": "architecture"}),
    ("doc_04", "Embeddings encode semantic meaning as dense vectors in high-dimensional space.", {"category": "embeddings"}),
    ("doc_05", "Cosine similarity measures the angle between two vectors in embedding space.", {"category": "embeddings"}),
    ("doc_06", "Vector databases enable efficient similarity search over millions of embeddings.", {"category": "infrastructure"}),
    ("doc_07", "FAISS is a library for efficient similarity search developed by Meta.", {"category": "infrastructure"}),
    ("doc_08", "RAG systems retrieve relevant context before generating LLM responses.", {"category": "rag"}),
    ("doc_09", "Chunking splits documents into smaller pieces before embedding them.", {"category": "rag"}),
    ("doc_10", "Fine-tuning adapts a pretrained model to a specific task or domain.", {"category": "training"}),
    ("doc_11", "RLHF trains models to follow instructions and be helpful using human feedback.", {"category": "training"}),
    ("doc_12", "Prompt engineering involves crafting inputs to elicit better LLM outputs.", {"category": "prompting"}),
    ("doc_13", "Chain-of-thought prompting encourages models to reason step by step.", {"category": "prompting"}),
    ("doc_14", "LangChain provides abstractions for building LLM-powered applications.", {"category": "frameworks"}),
    ("doc_15", "Ollama lets you run open-source LLMs locally on your own hardware.", {"category": "infrastructure"}),
]

SEARCH_QUERIES = [
    "how do language models understand text structure",
    "storing and searching vector representations efficiently",
    "making models follow instructions from human feedback",
]


def demo_semantic_search():
    print("\n" + "=" * 60)
    print("DEMO 2: Semantic Search")
    print("=" * 60)

    embed_fn, dims = get_embedder(dims=64)

    # Build the index
    store = VectorStore()
    print(f"\nIndexing {len(SAMPLE_SENTENCES)} sentences...")
    for doc_id, text, metadata in SAMPLE_SENTENCES:
        embedding = embed_fn(text)
        store.add(doc_id, text, embedding, metadata)
    print(f"Index built: {store}")

    # Run searches
    for query in SEARCH_QUERIES:
        print(f"\n{'─' * 55}")
        print(f"Query: '{query}'")
        print(f"{'─' * 55}")

        query_emb = embed_fn(query)
        results = store.search(query_emb, top_k=3)

        for i, result in enumerate(results):
            score = result["score"]
            text = result["text"]
            cat = result["metadata"].get("category", "?")
            bar = "█" * int(score * 20 + 0.5)
            print(f"  #{i+1} [{score:.3f}] {bar}")
            print(f"       {text}")
            print(f"       category: {cat}")

    if not HAS_SENTENCE_TRANSFORMERS:
        print("\n[note] Results above are random (fake embeddings).")
        print("       Install sentence-transformers to see real semantic ranking.")


# ─────────────────────────────────────────────
# METADATA FILTERING DEMO
# ─────────────────────────────────────────────

def demo_metadata_filtering():
    print("\n" + "=" * 60)
    print("DEMO 3: Metadata Filtering")
    print("=" * 60)

    embed_fn, dims = get_embedder(dims=64)

    store = VectorStore()
    for doc_id, text, metadata in SAMPLE_SENTENCES:
        embedding = embed_fn(text)
        store.add(doc_id, text, embedding, metadata)

    query = "how models learn from data"
    query_emb = embed_fn(query)

    print(f"\nQuery: '{query}'")

    # Search all
    all_results = store.search(query_emb, top_k=3)
    print(f"\nTop 3 (no filter):")
    for r in all_results:
        print(f"  [{r['score']:.3f}] [{r['metadata']['category']}] {r['text'][:60]}")

    # Filter by category
    for category in ["training", "architecture", "rag"]:
        filtered = store.search(query_emb, top_k=3, filter_metadata={"category": category})
        print(f"\nTop 3 (filtered: category='{category}'):")
        if filtered:
            for r in filtered:
                print(f"  [{r['score']:.3f}] {r['text'][:70]}")
        else:
            print("  No results after filtering.")

    print("\nKey insight: metadata filtering lets you scope search to subsets of your data.")
    print("Use case: search only documents belonging to current user, or from last 30 days.")


# ─────────────────────────────────────────────
# FAISS DEMO (optional)
# ─────────────────────────────────────────────

def demo_faiss():
    if not HAS_FAISS:
        print("\n[skipping] FAISS demo — install faiss-cpu numpy")
        return

    print("\n" + "=" * 60)
    print("DEMO 4: FAISS Index (Approximate Nearest Neighbor)")
    print("=" * 60)

    import numpy as np

    dims = 128
    n_vectors = 50

    # Generate random vectors (in real use, these are your embeddings)
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dims).astype("float32")

    # L2 normalize (convert to unit vectors so inner product = cosine similarity)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Build FAISS index
    # IndexFlatIP = flat (exact) index using inner product (= cosine for unit vectors)
    index = faiss.IndexFlatIP(dims)
    index.add(vectors)

    print(f"\nBuilt FAISS index:")
    print(f"  Vectors: {n_vectors}")
    print(f"  Dims: {dims}")
    print(f"  Index type: IndexFlatIP (exact inner product)")

    # Search
    query = np.random.randn(1, dims).astype("float32")
    query = query / np.linalg.norm(query)

    k = 5
    distances, indices = index.search(query, k)

    print(f"\nTop {k} results for random query:")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"  #{rank+1} vector_id={idx:3d}  cosine_sim={dist:.4f}")

    print(f"\nFAISS advantages:")
    print(f"  - Handles millions of vectors efficiently")
    print(f"  - HNSW index: O(log n) search instead of O(n)")
    print(f"  - Supports GPU acceleration")
    print(f"  - Used inside Chroma, many production systems")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("3.2 Embeddings & Vector Search — Live Demos")
    print("=" * 60)

    demo_cosine_vs_euclidean()
    demo_semantic_search()
    demo_metadata_filtering()
    demo_faiss()

    print("\n" + "=" * 60)
    print("All demos complete.")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Cosine similarity only cares about direction — correct for text embeddings")
    print("  2. VectorStore is just a dict + cosine similarity — surprisingly simple")
    print("  3. Metadata filtering narrows search scope — essential for multi-user systems")
    print("  4. FAISS gives you O(log n) ANN search — use it when n > 100k")
    print("  5. Without sentence-transformers, results are meaningless — semantic quality matters")
