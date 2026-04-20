"""
3.2 Embeddings & Vector Search — Exercises
Complete each function, then run the solution() to verify.

Run: python exercises.py
"""

import math
import hashlib
import struct
from typing import Any, Callable, Optional

# ─────────────────────────────────────────────
# HELPER: Fake embedder (no dependencies)
# ─────────────────────────────────────────────

def _fake_embed(text: str, dims: int = 64) -> list[float]:
    """Deterministic hash-based fake embedding."""
    vector = []
    for i in range(0, dims, 8):
        seed = f"{i}:{text}"
        h = hashlib.sha256(seed.encode()).digest()
        chunk = struct.unpack("8f", h[:32])
        for val in chunk[:min(8, dims - i)]:
            vector.append((val % 2.0) - 1.0)
    vec = vector[:dims]
    mag = math.sqrt(sum(v * v for v in vec))
    return [v / mag for v in vec] if mag > 0 else vec


# ─────────────────────────────────────────────
# EXERCISE 1: cosine_similarity from scratch
# ─────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Formula: (a · b) / (|a| * |b|)

    Args:
        a: first vector (list of floats)
        b: second vector (list of floats, same length as a)

    Returns:
        float in [-1, 1] where 1 = identical direction

    Raises:
        ValueError: if vectors have different lengths
        ZeroDivisionError: if either vector is all zeros

    TODO: implement this function using only stdlib math operations.
          No numpy. One line for dot product, one line per magnitude, one for result.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Exercise 1: implement cosine_similarity")


def test_exercise_1():
    """Solution and verification for Exercise 1."""
    print("\n" + "=" * 55)
    print("Exercise 1: cosine_similarity from scratch")
    print("=" * 55)

    def solution_cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            raise ZeroDivisionError("Cannot compute cosine similarity for zero vector")
        return dot / (mag_a * mag_b)

    # Test cases
    test_cases = [
        ([1.0, 0.0], [1.0, 0.0], 1.0, "identical vectors"),
        ([1.0, 0.0], [0.0, 1.0], 0.0, "perpendicular vectors"),
        ([1.0, 0.0], [-1.0, 0.0], -1.0, "opposite vectors"),
        ([1.0, 1.0], [2.0, 2.0], 1.0, "same direction, different magnitude"),
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], None, "higher-dimensional"),
    ]

    all_pass = True
    for a, b, expected, desc in test_cases:
        result = solution_cosine_similarity(a, b)
        if expected is not None:
            close = abs(result - expected) < 1e-9
            status = "PASS" if close else "FAIL"
            if not close:
                all_pass = False
        else:
            status = "PASS"
            expected = result

        print(f"  {status}  {desc}")
        print(f"         a={a}, b={b}")
        print(f"         result={result:.6f}, expected≈{expected:.6f}")

    print(f"\n{'All tests passed!' if all_pass else 'Some tests failed — check your implementation'}")

    print("\nSOLUTION:")
    print("""
    def cosine_similarity(a, b):
        if len(a) != len(b):
            raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            raise ZeroDivisionError("Cannot compute cosine similarity for zero vector")
        return dot / (mag_a * mag_b)
    """)


# ─────────────────────────────────────────────
# EXERCISE 2: find_duplicates
# ─────────────────────────────────────────────

def find_duplicates(
    texts: list[str],
    embed_fn: Callable[[str], list[float]],
    threshold: float = 0.95
) -> list[tuple[int, int, float]]:
    """
    Find pairs of texts that are likely duplicates (cosine similarity >= threshold).

    Args:
        texts: list of text strings
        embed_fn: function that takes a string and returns a list[float] embedding
        threshold: minimum cosine similarity to flag as duplicate (0.95 = very similar)

    Returns:
        List of (i, j, similarity) tuples where i < j and similarity >= threshold
        Sorted by similarity descending.

    Example:
        texts = ["The cat sat.", "The cat sat down.", "Python is great.", "Python is amazing."]
        find_duplicates(texts, embed_fn, threshold=0.90)
        # might return [(0, 1, 0.98), (2, 3, 0.91)]

    TODO: embed all texts, then compare every pair (O(n^2) is fine for this exercise).
          Use your cosine_similarity implementation above.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Exercise 2: implement find_duplicates")


def test_exercise_2():
    """Solution and verification for Exercise 2."""
    print("\n" + "=" * 55)
    print("Exercise 2: find_duplicates")
    print("=" * 55)

    def _sol_cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0.0

    def solution_find_duplicates(texts, embed_fn, threshold=0.95):
        embeddings = [embed_fn(t) for t in texts]
        pairs = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = _sol_cosine(embeddings[i], embeddings[j])
                if sim >= threshold:
                    pairs.append((i, j, sim))
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    # Demo with fake embeddings
    # (With real embeddings, truly similar texts would be flagged)
    texts = [
        "The transformer architecture uses self-attention.",
        "Self-attention is the core of transformer models.",
        "Python is a programming language.",
        "Python is used for data science.",
        "RAG systems retrieve context before generation.",
        "RAG retrieves documents to augment the LLM prompt.",
    ]

    print(f"\nTexts to check ({len(texts)} texts):")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t}")

    pairs = solution_find_duplicates(texts, _fake_embed, threshold=0.80)
    print(f"\nDuplicate pairs (threshold=0.80, using FAKE embeddings):")
    if pairs:
        for i, j, sim in pairs:
            print(f"  [{i}] vs [{j}]  sim={sim:.4f}")
            print(f"       '{texts[i]}'")
            print(f"       '{texts[j]}'")
    else:
        print("  None found (fake embeddings have no semantic meaning)")

    print("\nSOLUTION:")
    print("""
    def find_duplicates(texts, embed_fn, threshold=0.95):
        embeddings = [embed_fn(t) for t in texts]
        pairs = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    pairs.append((i, j, sim))
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    NOTE: With real sentence-transformers embeddings:
        - Truly similar sentences would score 0.90+
        - This function would correctly flag them
        - Fake embeddings produce random scores
    """)


# ─────────────────────────────────────────────
# EXERCISE 3: BM25 vs Cosine Similarity
# ─────────────────────────────────────────────

def bm25_score(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """
    Compute BM25 scores for each document given a query.

    BM25 is a keyword-based ranking function (TF-IDF family).
    Better than raw TF-IDF for handling term frequency saturation.

    Formula per term t in query, for document d:
        IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    Where:
        tf(t, d) = count of term t in document d
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)  [smooth IDF]
        N = total number of documents
        df(t) = number of documents containing term t
        |d| = length of document d in tokens
        avgdl = average document length

    Args:
        query: search query string
        documents: list of document strings
        k1: term frequency saturation parameter (1.2-2.0 typical)
        b: length normalization parameter (0=no normalization, 1=full)

    Returns:
        List of float scores, one per document (higher = more relevant)

    TODO: implement BM25 from scratch using stdlib only.
          Tokenize by splitting on whitespace and lowercasing.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Exercise 3: implement bm25_score")


def test_exercise_3():
    """Solution and verification for Exercise 3."""
    print("\n" + "=" * 55)
    print("Exercise 3: BM25 vs Cosine Similarity")
    print("=" * 55)

    import math
    from collections import Counter

    def solution_bm25_score(query, documents, k1=1.5, b=0.75):
        query_terms = query.lower().split()
        tokenized_docs = [doc.lower().split() for doc in documents]
        N = len(documents)
        avgdl = sum(len(d) for d in tokenized_docs) / N if N > 0 else 1

        scores = []
        for doc_tokens in tokenized_docs:
            doc_len = len(doc_tokens)
            tf = Counter(doc_tokens)
            score = 0.0
            for term in query_terms:
                tf_term = tf.get(term, 0)
                df_term = sum(1 for d in tokenized_docs if term in d)
                idf = math.log((N - df_term + 0.5) / (df_term + 0.5) + 1)
                numerator = tf_term * (k1 + 1)
                denominator = tf_term + k1 * (1 - b + b * doc_len / avgdl)
                score += idf * (numerator / denominator if denominator > 0 else 0)
            scores.append(score)
        return scores

    def _sol_cosine(a, b_):
        dot = sum(x * y for x, y in zip(a, b_))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b_))
        return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0.0

    documents = [
        "transformers use attention mechanisms for sequence modeling",
        "attention is all you need is the famous transformer paper",
        "recurrent neural networks process sequences step by step",
        "bert uses bidirectional attention for language understanding",
        "gpt generates text with autoregressive language modeling",
        "convolutional neural networks excel at image classification",
        "the attention mechanism assigns weights to different tokens",
        "llms are large language models trained on massive datasets",
        "fine-tuning adapts pretrained models to specific tasks",
        "prompt engineering improves llm outputs without training",
    ]

    queries = [
        "attention mechanism transformers",
        "language model training",
        "neural network sequence",
    ]

    print(f"\nDocuments ({len(documents)}):")
    for i, doc in enumerate(documents):
        print(f"  [{i}] {doc}")

    print(f"\n{'Query':<35} {'BM25 top-3':>25} {'Cosine top-3 (fake)':>25}")
    print("-" * 90)

    for query in queries:
        bm25_scores = solution_bm25_score(query, documents)
        bm25_top3 = sorted(range(len(documents)), key=lambda i: bm25_scores[i], reverse=True)[:3]

        query_emb = _fake_embed(query)
        doc_embs = [_fake_embed(doc) for doc in documents]
        cos_scores = [_sol_cosine(query_emb, d) for d in doc_embs]
        cos_top3 = sorted(range(len(documents)), key=lambda i: cos_scores[i], reverse=True)[:3]

        bm25_str = str(bm25_top3)
        cos_str = str(cos_top3)
        print(f"  '{query}'")
        print(f"    BM25 top-3 indices: {bm25_str}  (keyword match — accurate for short queries)")
        print(f"    Cosine top-3 idx:   {cos_str}  (fake — random without real embeddings)")

    print("""
SOLUTION:
    def bm25_score(query, documents, k1=1.5, b=0.75):
        query_terms = query.lower().split()
        tokenized_docs = [doc.lower().split() for doc in documents]
        N = len(documents)
        avgdl = sum(len(d) for d in tokenized_docs) / N

        scores = []
        for doc_tokens in tokenized_docs:
            doc_len = len(doc_tokens)
            tf = Counter(doc_tokens)
            score = 0.0
            for term in query_terms:
                tf_term = tf.get(term, 0)
                df_term = sum(1 for d in tokenized_docs if term in d)
                idf = math.log((N - df_term + 0.5) / (df_term + 0.5) + 1)
                numerator = tf_term * (k1 + 1)
                denominator = tf_term + k1 * (1 - b + b * doc_len / avgdl)
                score += idf * (numerator / denominator if denominator > 0 else 0)
            scores.append(score)
        return scores
    """)


# ─────────────────────────────────────────────
# EXERCISE 4: hybrid_search
# ─────────────────────────────────────────────

def hybrid_search(
    bm25_scores: list[float],
    vector_scores: list[float],
    alpha: float = 0.5
) -> list[tuple[int, float]]:
    """
    Combine BM25 and vector similarity scores into a hybrid ranking.

    Both score lists must be normalized to [0, 1] before combining.
    Formula: hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score

    Args:
        bm25_scores: raw BM25 scores for each document (higher = more relevant)
        vector_scores: cosine similarity scores for each document (higher = more relevant)
        alpha: weight for vector scores (0 = pure BM25, 1 = pure vector, 0.5 = equal blend)

    Returns:
        List of (doc_index, hybrid_score) sorted by score descending.

    TODO:
        1. Normalize each score list to [0, 1] using min-max normalization
        2. Compute weighted combination
        3. Return sorted results

    Hint: min-max normalization: (x - min) / (max - min), handle division by zero (all same score)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Exercise 4: implement hybrid_search")


def test_exercise_4():
    """Solution and verification for Exercise 4."""
    print("\n" + "=" * 55)
    print("Exercise 4: hybrid_search")
    print("=" * 55)

    def solution_hybrid_search(bm25_scores, vector_scores, alpha=0.5):
        def normalize(scores):
            min_s = min(scores)
            max_s = max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        norm_bm25 = normalize(bm25_scores)
        norm_vec = normalize(vector_scores)

        hybrid = [
            (i, alpha * v + (1 - alpha) * b)
            for i, (v, b) in enumerate(zip(norm_vec, norm_bm25))
        ]
        return sorted(hybrid, key=lambda x: x[1], reverse=True)

    # Simulate scores from a 5-document search
    bm25_s = [10.5, 2.1, 0.0, 8.3, 4.7]    # BM25 favors doc 0 and 3
    vec_s  = [0.65, 0.82, 0.71, 0.55, 0.90]  # Vector favors doc 4 and 1

    print("\nDocument scores:")
    print(f"{'Doc':>5} {'BM25':>8} {'Vector':>8}")
    print("-" * 25)
    for i in range(len(bm25_s)):
        print(f"  [{i}]  {bm25_s[i]:>6.2f}  {vec_s[i]:>7.3f}")

    for alpha in [0.0, 0.5, 1.0]:
        results = solution_hybrid_search(bm25_s, vec_s, alpha=alpha)
        ranking = [idx for idx, _ in results]
        print(f"\n  alpha={alpha:.1f} ({'pure BM25' if alpha==0 else 'pure vector' if alpha==1 else 'equal blend'})")
        print(f"  Ranking: {ranking}")
        for idx, score in results:
            print(f"    [{idx}] {score:.4f}")

    print("""
SOLUTION:
    def hybrid_search(bm25_scores, vector_scores, alpha=0.5):
        def normalize(scores):
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        norm_bm25 = normalize(bm25_scores)
        norm_vec = normalize(vector_scores)
        hybrid = [
            (i, alpha * v + (1 - alpha) * b)
            for i, (v, b) in enumerate(zip(norm_vec, norm_bm25))
        ]
        return sorted(hybrid, key=lambda x: x[1], reverse=True)
    """)


# ─────────────────────────────────────────────
# EXERCISE 5: maximal_marginal_relevance
# ─────────────────────────────────────────────

def maximal_marginal_relevance(
    query_emb: list[float],
    candidate_embs: list[list[float]],
    candidate_ids: list[str],
    k: int = 5,
    lambda_: float = 0.5
) -> list[str]:
    """
    Maximal Marginal Relevance (MMR): select k documents balancing relevance AND diversity.

    The problem with top-k by similarity alone: you often get k very similar documents.
    If the top 5 chunks are all from the same paragraph, your context is redundant.
    MMR iteratively selects documents that are relevant to the query but dissimilar to
    already-selected documents.

    Algorithm (greedy):
        1. Start with empty selected set S
        2. Repeat k times:
           - For each candidate not yet in S:
             score = lambda * sim(candidate, query) - (1-lambda) * max(sim(candidate, s) for s in S)
           - Select the candidate with the highest score
           - Add it to S

    Args:
        query_emb: query embedding
        candidate_embs: list of candidate embeddings
        candidate_ids: corresponding IDs for each candidate
        k: number of documents to select
        lambda_: trade-off. 1.0 = pure relevance (regular top-k). 0.0 = pure diversity.

    Returns:
        List of k selected candidate IDs in selection order.

    TODO: implement the greedy MMR algorithm above.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Exercise 5: implement maximal_marginal_relevance")


def test_exercise_5():
    """Solution and verification for Exercise 5."""
    print("\n" + "=" * 55)
    print("Exercise 5: Maximal Marginal Relevance (MMR)")
    print("=" * 55)

    def _sol_cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0.0

    def solution_mmr(query_emb, candidate_embs, candidate_ids, k=5, lambda_=0.5):
        selected = []
        remaining = list(range(len(candidate_embs)))

        for _ in range(min(k, len(candidate_embs))):
            best_idx = None
            best_score = float("-inf")

            for i in remaining:
                relevance = _sol_cos(query_emb, candidate_embs[i])
                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        _sol_cos(candidate_embs[i], candidate_embs[j])
                        for j in selected
                    )
                score = lambda_ * relevance - (1 - lambda_) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = i

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidate_ids[i] for i in selected]

    # Create candidates with some very similar ones
    # In a real scenario these are real embeddings — here we craft them manually
    # Using simple 3D vectors for clarity
    query_emb = [1.0, 0.0, 0.0]

    candidates = {
        "doc_A": [0.99, 0.1, 0.0],   # very relevant, close to query
        "doc_B": [0.98, 0.15, 0.0],  # very relevant, very similar to A
        "doc_C": [0.97, 0.2, 0.0],   # relevant, similar to A and B
        "doc_D": [0.7, 0.7, 0.0],    # moderately relevant, different angle
        "doc_E": [0.5, 0.0, 0.85],   # less relevant, very diverse
    }

    ids = list(candidates.keys())
    embs = list(candidates.values())

    # Normalize embeddings
    def norm(v):
        m = math.sqrt(sum(x*x for x in v))
        return [x/m for x in v] if m > 0 else v
    embs = [norm(e) for e in embs]

    print("\nCandidates:")
    for doc_id, emb in zip(ids, embs):
        rel = _sol_cos(query_emb, emb)
        print(f"  {doc_id}  relevance={rel:.3f}  emb={[f'{x:.2f}' for x in emb]}")

    print(f"\n--- Top-3 by pure relevance (lambda_=1.0) ---")
    result_pure = solution_mmr(query_emb, embs, ids, k=3, lambda_=1.0)
    print(f"Selected: {result_pure}")
    print("(Picks the 3 most relevant — likely A, B, C which are very similar to each other)")

    print(f"\n--- MMR with diversity (lambda_=0.5) ---")
    result_mmr = solution_mmr(query_emb, embs, ids, k=3, lambda_=0.5)
    print(f"Selected: {result_mmr}")
    print("(Balances relevance and diversity — avoids picking near-duplicates)")

    print("""
SOLUTION:
    def maximal_marginal_relevance(query_emb, candidate_embs, candidate_ids, k=5, lambda_=0.5):
        selected = []
        remaining = list(range(len(candidate_embs)))

        for _ in range(min(k, len(candidate_embs))):
            best_idx, best_score = None, float("-inf")

            for i in remaining:
                relevance = cosine_similarity(query_emb, candidate_embs[i])
                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        cosine_similarity(candidate_embs[i], candidate_embs[j])
                        for j in selected
                    )
                score = lambda_ * relevance - (1 - lambda_) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = i

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidate_ids[i] for i in selected]

    KEY INSIGHT: MMR prevents your retrieved context from being 5 copies of the same paragraph.
    Use it in RAG when top-k retrieval gives you redundant chunks.
    """)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("3.2 Embeddings & Vector Search — Exercises")
    print("=" * 60)
    print("Each test_exercise_N() shows the solution and explanation.")
    print("Try implementing the functions first, then run to verify.\n")

    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    test_exercise_4()
    test_exercise_5()

    print("\n" + "=" * 60)
    print("All exercises complete.")
    print("=" * 60)
