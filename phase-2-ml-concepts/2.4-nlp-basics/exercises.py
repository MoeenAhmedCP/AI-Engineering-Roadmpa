"""
2.4 NLP Basics — Exercises
Attempt each before reading solutions.
Run: python exercises.py
"""

import math
import re
import string
from collections import defaultdict

# ---------------------------------------------------------------------------
# EXERCISE 1 — Tokenize text into words (split on whitespace + strip punctuation)
# Input: "Hello, world! How are you?"
# Output: ["Hello", "world", "How", "are", "you"]
# ---------------------------------------------------------------------------

def simple_tokenize(text: str) -> list:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 2 — Build word frequency dict from a list of sentences
# ---------------------------------------------------------------------------

def word_frequency(sentences: list) -> dict:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 3 — TF-IDF (no libraries)
# Given a list of documents (strings), compute TF-IDF for each word in each doc.
# Return list of dicts: [{word: tfidf_score}, ...]
# ---------------------------------------------------------------------------

def compute_tfidf(documents: list) -> list:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 4 — Cosine similarity between two word-frequency vectors (dicts)
# ---------------------------------------------------------------------------

def cosine_similarity(vec1: dict, vec2: dict) -> float:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 5 — N-gram generator
# ---------------------------------------------------------------------------

def ngrams(tokens: list, n: int) -> list:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 6 — Simple Naive Bayes text classifier (positive/negative)
# Train on labeled examples, predict on new text.
# ---------------------------------------------------------------------------

class NaiveBayesClassifier:
    def fit(self, texts: list, labels: list):
        raise NotImplementedError

    def predict(self, text: str) -> str:
        raise NotImplementedError


# ===========================================================================
# SOLUTIONS
# ===========================================================================

def _sol_simple_tokenize(text: str) -> list:
    """Strip punctuation, split on whitespace, drop empty tokens."""
    # Remove punctuation using translate
    translator = str.maketrans("", "", string.punctuation)
    cleaned = text.translate(translator)
    return [tok for tok in cleaned.split() if tok]


def _sol_word_frequency(sentences: list) -> dict:
    """Count word occurrences across all sentences."""
    freq = defaultdict(int)
    for sentence in sentences:
        for word in _sol_simple_tokenize(sentence.lower()):
            freq[word] += 1
    return dict(freq)


def _sol_compute_tfidf(documents: list) -> list:
    """
    TF(t, d)  = count(t in d) / len(d)
    IDF(t)    = log(N / (1 + df(t)))   [smoothed]
    TFIDF     = TF * IDF
    """
    tokenized = [_sol_simple_tokenize(doc.lower()) for doc in documents]
    N = len(documents)

    # Document frequency: how many docs contain each word
    df = defaultdict(int)
    for tokens in tokenized:
        for word in set(tokens):
            df[word] += 1

    results = []
    for tokens in tokenized:
        tf = defaultdict(float)
        for word in tokens:
            tf[word] += 1
        doc_len = len(tokens) if tokens else 1
        tfidf = {}
        for word, count in tf.items():
            term_freq = count / doc_len
            inv_doc_freq = math.log(N / (1 + df[word]))
            tfidf[word] = round(term_freq * inv_doc_freq, 6)
        results.append(tfidf)
    return results


def _sol_cosine_similarity(vec1: dict, vec2: dict) -> float:
    """Dot product divided by product of magnitudes."""
    all_keys = set(vec1) | set(vec2)
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in all_keys)
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return round(dot / (mag1 * mag2), 6)


def _sol_ngrams(tokens: list, n: int) -> list:
    """Return all n-grams as tuples."""
    if n <= 0 or n > len(tokens):
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class _SolNaiveBayesClassifier:
    """
    Multinomial Naive Bayes with Laplace smoothing.
    P(label | text) ∝ P(label) * prod P(word | label)
    Work in log space to avoid underflow.
    """

    def fit(self, texts: list, labels: list):
        self._classes = list(set(labels))
        self._log_prior = {}
        self._log_likelihood = {}
        self._vocab = set()
        n = len(labels)

        # Gather word counts per class
        class_word_counts = {c: defaultdict(int) for c in self._classes}
        class_total = defaultdict(int)

        for text, label in zip(texts, labels):
            for word in _sol_simple_tokenize(text.lower()):
                class_word_counts[label][word] += 1
                class_total[label] += 1
                self._vocab.add(word)

        vocab_size = len(self._vocab)

        for c in self._classes:
            # Prior
            self._log_prior[c] = math.log(labels.count(c) / n)
            # Likelihoods with Laplace smoothing
            self._log_likelihood[c] = {}
            total = class_total[c] + vocab_size  # smoothed denominator
            for word in self._vocab:
                count = class_word_counts[c].get(word, 0) + 1  # +1 smoothing
                self._log_likelihood[c][word] = math.log(count / total)

    def predict(self, text: str) -> str:
        words = _sol_simple_tokenize(text.lower())
        scores = {}
        for c in self._classes:
            score = self._log_prior[c]
            for word in words:
                if word in self._log_likelihood[c]:
                    score += self._log_likelihood[c][word]
                # Unknown words: skip (could use UNK smoothing)
            scores[c] = score
        return max(scores, key=scores.get)


# ---------------------------------------------------------------------------

def solutions():
    sep = "=" * 55

    print(sep)
    print("SOLUTIONS — 2.4 NLP Basics Exercises")
    print(sep)

    # --- Exercise 1 ---
    print("\n[Exercise 1] simple_tokenize")
    text = "Hello, world! How are you?"
    result = _sol_simple_tokenize(text)
    print(f"  Input : {text!r}")
    print(f"  Output: {result}")

    # --- Exercise 2 ---
    print("\n[Exercise 2] word_frequency")
    sentences = [
        "The cat sat on the mat",
        "The cat ate the rat",
        "The mat was flat",
    ]
    freq = _sol_word_frequency(sentences)
    sorted_freq = sorted(freq.items(), key=lambda x: -x[1])[:8]
    print(f"  Top 8 words: {sorted_freq}")

    # --- Exercise 3 ---
    print("\n[Exercise 3] compute_tfidf")
    docs = [
        "the cat sat on the mat",
        "the dog barked at the cat",
        "a completely unrelated document about ships",
    ]
    tfidf_results = _sol_compute_tfidf(docs)
    for i, scores in enumerate(tfidf_results):
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        print(f"  Doc {i}: top-3 TF-IDF words = {top3}")

    # --- Exercise 4 ---
    print("\n[Exercise 4] cosine_similarity")
    v1 = {"cat": 2, "mat": 1, "sat": 1}
    v2 = {"cat": 1, "mat": 2, "dog": 1}
    v3 = {"ship": 3, "ocean": 2}
    print(f"  sim(v1, v2) = {_sol_cosine_similarity(v1, v2):.4f}  (expect: high)")
    print(f"  sim(v1, v3) = {_sol_cosine_similarity(v1, v3):.4f}  (expect: 0.0)")

    # --- Exercise 5 ---
    print("\n[Exercise 5] ngrams")
    tokens = ["the", "quick", "brown", "fox"]
    print(f"  tokens   : {tokens}")
    print(f"  bigrams  : {_sol_ngrams(tokens, 2)}")
    print(f"  trigrams : {_sol_ngrams(tokens, 3)}")

    # --- Exercise 6 ---
    print("\n[Exercise 6] NaiveBayesClassifier")
    train_texts = [
        "I love this product",
        "Amazing quality great value",
        "Excellent service highly recommend",
        "Wonderful experience very happy",
        "Terrible product broke immediately",
        "Awful quality waste of money",
        "Very disappointed poor service",
        "Horrible never buying again",
    ]
    train_labels = ["positive"] * 4 + ["negative"] * 4

    clf = _SolNaiveBayesClassifier()
    clf.fit(train_texts, train_labels)

    test_cases = [
        "I love the quality of this item",
        "Terrible waste of money very bad",
        "Highly recommend amazing experience",
    ]
    for tc in test_cases:
        pred = clf.predict(tc)
        print(f"  {pred:10s} | {tc!r}")

    print(f"\n{sep}")
    print("All solutions complete.")
    print(sep)


if __name__ == "__main__":
    print("Attempt the exercises above, then call solutions() to compare.\n")
    solutions()
