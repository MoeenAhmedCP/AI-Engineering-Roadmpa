"""
2.7 Data Pipelines — Examples
No external dependencies required (stdlib only).
Run: python examples.py
"""

import math
import random
import re
import string
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. BatchDataLoader
# ---------------------------------------------------------------------------

class BatchDataLoader:
    """
    Minimal dataloader that iterates over data in batches.
    Supports optional shuffling each epoch.
    """

    def __init__(self, data: list, batch_size: int = 32, shuffle: bool = True):
        self.data       = data
        self.batch_size = batch_size
        self.shuffle    = shuffle

    def __len__(self) -> int:
        """Number of batches (ceiling division)."""
        return math.ceil(len(self.data) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(self.data[idx])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:          # last partial batch
            yield batch


# ---------------------------------------------------------------------------
# 2. TextPreprocessor
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """
    Builds a word-level vocabulary and encodes/decodes text as token IDs.
    Special tokens: <PAD>=0, <UNK>=1.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self._word2id: dict = {}
        self._id2word: dict = {}

    # ---- static helpers ----

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenize(text: str) -> list:
        return TextPreprocessor.normalize_text(text).split()

    # ---- vocabulary ----

    def build_vocab(self, texts: list) -> dict:
        """
        Fit vocabulary from a list of raw strings.
        Returns the word→id mapping.
        """
        # Reserve IDs 0 and 1
        self._word2id = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self._id2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

        word_counts: dict = defaultdict(int)
        for text in texts:
            for word in self._tokenize(text):
                word_counts[word] += 1

        # Sort by frequency (most common first) for stability
        for word in sorted(word_counts, key=lambda w: -word_counts[w]):
            if word not in self._word2id:
                idx = len(self._word2id)
                self._word2id[word] = idx
                self._id2word[idx] = word

        return dict(self._word2id)

    def encode(self, text: str) -> list:
        """Convert raw text to a list of integer token IDs."""
        if not self._word2id:
            raise RuntimeError("Call build_vocab() before encode().")
        unk_id = self._word2id[self.UNK_TOKEN]
        return [self._word2id.get(w, unk_id) for w in self._tokenize(text)]

    def decode(self, ids: list) -> str:
        """Convert a list of token IDs back to a string."""
        if not self._id2word:
            raise RuntimeError("Call build_vocab() before decode().")
        return " ".join(self._id2word.get(i, self.UNK_TOKEN) for i in ids)


# ---------------------------------------------------------------------------
# 3. Streaming loader
# ---------------------------------------------------------------------------

def streaming_load(text_lines, batch_size: int = 4):
    """
    Generator that yields batches from any iterable of strings.
    Simulates reading a large file line-by-line without loading all at once.
    Memory usage: O(batch_size), not O(total data).
    """
    batch = []
    for line in text_lines:
        batch.append(line)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# 4. Train / Validation / Test Split
# ---------------------------------------------------------------------------

def train_val_test_split(
    data: list,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Split data into (train, val, test) lists.
    Ratios must sum to 1.0 (within floating-point tolerance).
    """
    assert abs(train + val + test - 1.0) < 1e-9, "Ratios must sum to 1.0"

    data = list(data)
    rng = random.Random(seed)
    rng.shuffle(data)

    n = len(data)
    n_train = int(n * train)
    n_val   = int(n * val)

    train_split = data[:n_train]
    val_split   = data[n_train : n_train + n_val]
    test_split  = data[n_train + n_val :]

    return train_split, val_split, test_split


# ---------------------------------------------------------------------------
# 5. Oversample Minority Class
# ---------------------------------------------------------------------------

def oversample_minority(X: list, y: list) -> tuple:
    """
    Duplicate minority-class samples until all classes have equal representation.
    Works for any number of classes.
    Returns (X_balanced, y_balanced).
    """
    # Group indices by class
    class_indices: dict = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[label].append(i)

    max_count = max(len(v) for v in class_indices.values())

    X_out, y_out = [], []
    rng = random.Random(42)

    for label, indices in class_indices.items():
        # Keep all original samples
        X_out.extend(X[i] for i in indices)
        y_out.extend([label] * len(indices))

        # Add duplicates until we reach max_count
        shortfall = max_count - len(indices)
        extras = rng.choices(indices, k=shortfall)
        X_out.extend(X[i] for i in extras)
        y_out.extend([label] * shortfall)

    # Shuffle the result
    combined = list(zip(X_out, y_out))
    rng.shuffle(combined)
    X_bal, y_bal = zip(*combined) if combined else ([], [])

    return list(X_bal), list(y_bal)


# ---------------------------------------------------------------------------
# 6. Chunking text with overlap
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list:
    """
    Split text into fixed-size character chunks with overlap.
    Each chunk is at most `chunk_size` characters.
    Consecutive chunks share `overlap` characters at the boundary.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sep = "=" * 55

    # ---- BatchDataLoader ----
    print(sep)
    print("1. BatchDataLoader")
    print(sep)
    data = list(range(17))
    loader = BatchDataLoader(data, batch_size=5, shuffle=False)
    print(f"  Data     : {data}")
    print(f"  Batches  : {len(loader)}")
    for i, batch in enumerate(loader):
        print(f"  Batch {i}  : {batch}")

    # ---- TextPreprocessor ----
    print(f"\n{sep}")
    print("2. TextPreprocessor")
    print(sep)
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Data pipelines transform raw data into useful features",
        "The fox was quick and the dog was lazy",
    ]
    proc = TextPreprocessor()
    vocab = proc.build_vocab(corpus)
    print(f"  Vocab size: {len(vocab)} tokens (incl. PAD, UNK)")
    print(f"  Sample vocab: { {k: v for k, v in list(vocab.items())[:8]} }")

    sample = "The lazy fox reads machine learning papers"
    encoded = proc.encode(sample)
    decoded = proc.decode(encoded)
    print(f"\n  Original : {sample!r}")
    print(f"  Encoded  : {encoded}")
    print(f"  Decoded  : {decoded!r}")

    # ---- Streaming load ----
    print(f"\n{sep}")
    print("3. streaming_load (simulated file lines)")
    print(sep)
    lines = [f"line {i}: some text content here" for i in range(13)]
    for batch_num, batch in enumerate(streaming_load(lines, batch_size=4)):
        print(f"  Batch {batch_num}: {len(batch)} items | first='{batch[0]}'")

    # ---- Train/Val/Test Split ----
    print(f"\n{sep}")
    print("4. train_val_test_split")
    print(sep)
    dataset = list(range(100))
    train_d, val_d, test_d = train_val_test_split(dataset, seed=42)
    print(f"  Total   : {len(dataset)}")
    print(f"  Train   : {len(train_d)}  ({len(train_d)/len(dataset)*100:.0f}%)")
    print(f"  Val     : {len(val_d)}   ({len(val_d)/len(dataset)*100:.0f}%)")
    print(f"  Test    : {len(test_d)}  ({len(test_d)/len(dataset)*100:.0f}%)")
    print(f"  No overlap: {len(set(train_d) & set(val_d) & set(test_d)) == 0}")

    # ---- Oversample ----
    print(f"\n{sep}")
    print("5. oversample_minority")
    print(sep)
    # 80 class-0, 20 class-1 — imbalanced
    X_imb = [f"sample_{i}" for i in range(100)]
    y_imb = [0] * 80 + [1] * 20
    from collections import Counter
    print(f"  Before: {dict(Counter(y_imb))}")
    X_bal, y_bal = oversample_minority(X_imb, y_imb)
    print(f"  After : {dict(Counter(y_bal))}")
    print(f"  Total samples: {len(y_bal)}")

    # ---- Chunk text ----
    print(f"\n{sep}")
    print("6. chunk_text (fixed-size with overlap)")
    print(sep)
    long_text = (
        "In the beginning of AI engineering, data pipelines were often overlooked. "
        "But practitioners quickly learned that garbage in means garbage out. "
        "Careful preprocessing, chunking, and batching strategies can make or break "
        "the performance of even the most sophisticated model architecture. "
        "Reproducibility, class balance, and streaming all matter at scale."
    )
    chunks = chunk_text(long_text, chunk_size=100, overlap=20)
    print(f"  Text length  : {len(long_text)} chars")
    print(f"  Chunk size   : 100 | Overlap: 20")
    print(f"  Num chunks   : {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: [{len(c):3d} chars] '{c[:60]}...'")

    print(f"\n{sep}")
    print("All demos complete.")
    print(sep)
