"""
2.4 NLP Basics — Examples
Uses stdlib only (tiktoken optional).

Covers:
- TF-IDF from scratch with cosine similarity search
- N-gram extraction (bigrams and trigrams)
- Stop word removal effect on TF-IDF
- BPE demonstration (character-level merging)
- tiktoken token counting (fallback to word count)
"""

import math
import re
from collections import Counter, defaultdict


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# PART 1: TF-IDF from Scratch
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks with many layers",
    "natural language processing enables computers to understand text",
    "transformers are the foundation of modern language models",
    "gradient descent optimizes neural network weights during training",
    "attention mechanisms allow models to focus on relevant tokens",
]

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "if", "not", "that", "this", "it", "its",
    "as", "at", "be", "so", "we", "our", "they", "their", "you",
}


def tokenize(text, remove_stop_words=False):
    """Lowercase, split on non-alpha, optionally remove stop words."""
    tokens = re.findall(r'[a-z]+', text.lower())
    if remove_stop_words:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def compute_tf(tokens):
    """Term frequency: proportion of this term in the document."""
    total = len(tokens)
    counts = Counter(tokens)
    return {term: count / total for term, count in counts.items()}


def compute_idf(docs_tokenized):
    """Inverse document frequency for all terms across all documents."""
    n_docs = len(docs_tokenized)
    doc_freq = defaultdict(int)  # how many docs contain each term
    for doc_tokens in docs_tokenized:
        for term in set(doc_tokens):
            doc_freq[term] += 1
    idf = {
        term: math.log(n_docs / (1 + freq))
        for term, freq in doc_freq.items()
    }
    return idf


def compute_tfidf_matrix(docs, remove_stop_words=False):
    """Compute TF-IDF vectors for all documents."""
    docs_tokenized = [tokenize(doc, remove_stop_words) for doc in docs]
    idf = compute_idf(docs_tokenized)

    tfidf_matrix = []
    for tokens in docs_tokenized:
        tf = compute_tf(tokens)
        tfidf = {term: tf_val * idf.get(term, 0)
                 for term, tf_val in tf.items()}
        tfidf_matrix.append(tfidf)
    return tfidf_matrix, idf


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two TF-IDF dicts."""
    all_terms = set(vec_a) | set(vec_b)
    dot_product = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in all_terms)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def search_documents(query, docs, remove_stop_words=False):
    """Find most relevant document for a query using TF-IDF + cosine similarity."""
    tfidf_matrix, idf = compute_tfidf_matrix(docs, remove_stop_words)

    # TF-IDF for query
    query_tokens = tokenize(query, remove_stop_words)
    query_tf = compute_tf(query_tokens)
    query_tfidf = {term: tf_val * idf.get(term, 0)
                   for term, tf_val in query_tf.items()}

    scores = []
    for i, doc_tfidf in enumerate(tfidf_matrix):
        score = cosine_similarity(query_tfidf, doc_tfidf)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def demo_tfidf():
    query = "neural network training optimization"
    print(f"  Query: '{query}'")
    print()
    print(f"  {'Doc':<5} {'Score':>8} {'Document (truncated)'}")
    print(f"  {'-'*5} {'-'*8} {'-'*40}")

    scores = search_documents(query, SAMPLE_DOCS)
    for rank, (doc_idx, score) in enumerate(scores):
        snippet = SAMPLE_DOCS[doc_idx][:45] + "..."
        marker = " <-- BEST" if rank == 0 else ""
        print(f"  {doc_idx:<5} {score:>8.4f} {snippet}{marker}")


# ---------------------------------------------------------------------------
# PART 2: N-gram Extraction
# ---------------------------------------------------------------------------

def extract_ngrams(text, n):
    """Extract all n-grams from text as tuples."""
    tokens = tokenize(text)
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def demo_ngrams():
    sentence = "deep learning models use attention mechanisms to process text"
    print(f"  Sentence: '{sentence}'")
    print()

    for n, name in [(2, "Bigrams"), (3, "Trigrams")]:
        ngrams = extract_ngrams(sentence, n)
        print(f"  {name} ({len(ngrams)} total):")
        for gram in ngrams:
            print(f"    {' '.join(gram)}")
        print()


# ---------------------------------------------------------------------------
# PART 3: Stop Word Removal Effect on TF-IDF
# ---------------------------------------------------------------------------

def demo_stop_word_effect():
    query = "neural network training"
    print(f"  Query: '{query}'")
    print()

    for remove_sw in [False, True]:
        label = "WITH stop words removed" if remove_sw else "WITHOUT stop word removal"
        scores = search_documents(query, SAMPLE_DOCS, remove_stop_words=remove_sw)
        top_doc_idx, top_score = scores[0]
        print(f"  {label}:")
        print(f"    Top doc: [{top_doc_idx}] '{SAMPLE_DOCS[top_doc_idx][:50]}...'")
        print(f"    Score: {top_score:.4f}")
        print()


# ---------------------------------------------------------------------------
# PART 4: BPE Demonstration
# ---------------------------------------------------------------------------

def get_vocab(corpus_tokens):
    """Build initial character-level vocabulary with end-of-word marker."""
    vocab = Counter()
    for word in corpus_tokens:
        # Represent each word as a tuple of characters + </w> end marker
        char_seq = tuple(list(word) + ['</w>'])
        vocab[char_seq] += 1
    return vocab


def get_pairs(vocab):
    """Count all adjacent symbol pairs in the vocabulary."""
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += freq
    return pairs


def merge_pair(pair, vocab):
    """Merge the most frequent pair in all vocabulary entries."""
    new_vocab = {}
    bigram = pair[0] + pair[1]
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(bigram)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


def demo_bpe():
    # Tiny corpus — real BPE runs on billions of words
    corpus = [
        "low", "low", "low", "low", "low",
        "lower", "lower",
        "newest", "newest", "newest", "newest", "newest", "newest",
        "widest", "widest", "widest",
    ]

    vocab = get_vocab(corpus)

    print("  Initial character-level vocab:")
    for word, freq in sorted(vocab.items(), key=lambda x: -x[1])[:5]:
        print(f"    {' '.join(word):<25} (freq={freq})")

    print()
    print("  BPE merge steps:")
    for step in range(5):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best_pair = pairs.most_common(1)[0][0]
        best_freq = pairs[best_pair]
        vocab = merge_pair(best_pair, vocab)
        merged_symbol = best_pair[0] + best_pair[1]
        print(f"  Step {step+1}: merge '{best_pair[0]}' + '{best_pair[1]}' "
              f"-> '{merged_symbol}' (appeared {best_freq} times)")

    print()
    print("  Vocab after 5 merges:")
    for word, freq in sorted(vocab.items(), key=lambda x: -x[1]):
        print(f"    {' '.join(word):<25} (freq={freq})")


# ---------------------------------------------------------------------------
# PART 5: Token Counting (tiktoken with fallback)
# ---------------------------------------------------------------------------

def demo_token_counting():
    sample_texts = [
        "Hello, world!",
        "The transformer architecture revolutionized NLP in 2017.",
        "Tokenization: photosynthesis, 1234567, 🤔, 中文",
        "The quick brown fox jumps over the lazy dog.",
    ]

    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        print("  Using tiktoken (actual token counts):")
        print(f"  {'Text':<55} {'Tokens':>7} {'Words':>7}")
        print(f"  {'-'*55} {'-'*7} {'-'*7}")
        for text in sample_texts:
            tokens = enc.encode(text)
            words = len(text.split())
            short = (text[:50] + "...") if len(text) > 50 else text
            print(f"  {short:<55} {len(tokens):>7} {words:>7}")
    except ImportError:
        print("  tiktoken not installed. Falling back to word count.")
        print("  Install with: pip install tiktoken")
        print()
        print(f"  {'Text':<55} {'Word Count':>12}")
        print(f"  {'-'*55} {'-'*12}")
        for text in sample_texts:
            words = len(text.split())
            short = (text[:50] + "...") if len(text) > 50 else text
            print(f"  {short:<55} {words:>12}")
        print()
        print("  Note: actual token count is typically 25-33% more than word count")
        print("  for English text (BPE splits punctuation and subwords separately).")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_section("PART 1: TF-IDF Search Engine (6 documents)")
    demo_tfidf()

    print_section("PART 2: N-gram Extraction (bigrams and trigrams)")
    demo_ngrams()

    print_section("PART 3: Stop Word Removal Effect")
    demo_stop_word_effect()

    print_section("PART 4: BPE (Byte Pair Encoding) — 5 merge steps")
    demo_bpe()

    print_section("PART 5: Token Counting")
    demo_token_counting()

    print_section("SUMMARY")
    print("  1. TF-IDF: term frequency × inverse document frequency")
    print("     high score = important to this doc, rare across corpus")
    print("  2. Cosine similarity: measures angle between vectors (ignores magnitude)")
    print("  3. N-grams capture local word order that bag-of-words misses")
    print("  4. BPE: merge most frequent character pairs iteratively")
    print("     Result: common words = 1 token, rare words = subword pieces")
    print("  5. Token count != word count — matters for API cost + context limits")
