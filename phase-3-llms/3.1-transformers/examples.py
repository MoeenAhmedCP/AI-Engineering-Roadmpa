"""
3.1 Transformers & Attention — Examples
Run: python examples.py
No API keys needed. Optional deps: numpy, tiktoken, transformers, torch
"""

import math
import json
from typing import Optional

# ─────────────────────────────────────────────
# OPTIONAL DEPENDENCY IMPORTS
# ─────────────────────────────────────────────

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[warning] numpy not installed — attention demos will be skipped")
    print("          Install: pip install numpy")

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("[warning] tiktoken not installed — using character fallback tokenizer")
    print("          Install: pip install tiktoken")

try:
    from transformers import pipeline, AutoTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[warning] transformers/torch not installed — HuggingFace demo will be skipped")
    print("          Install: pip install transformers torch")


# ─────────────────────────────────────────────
# 1. SCALED DOT-PRODUCT ATTENTION FROM SCRATCH
# ─────────────────────────────────────────────

def softmax_numpy(x: "np.ndarray") -> "np.ndarray":
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(
    Q: "np.ndarray",
    K: "np.ndarray",
    V: "np.ndarray",
    mask: Optional["np.ndarray"] = None
) -> tuple:
    """
    Scaled dot-product attention.

    Args:
        Q: queries  shape (seq_len, d_k)
        K: keys     shape (seq_len, d_k)
        V: values   shape (seq_len, d_v)
        mask: optional causal mask (seq_len, seq_len) — True means mask out

    Returns:
        output: (seq_len, d_v)
        weights: (seq_len, seq_len) attention weight matrix
    """
    d_k = Q.shape[-1]

    # Step 1: compute raw attention scores
    # QK^T has shape (seq_len, seq_len) — each row is "how much does this token attend to all others"
    scores = Q @ K.T  # shape: (seq_len, seq_len)

    # Step 2: scale to prevent softmax saturation in high dimensions
    # Without this, large d_k pushes scores to extremes where softmax gradient vanishes
    scores = scores / math.sqrt(d_k)

    # Step 3: apply causal mask (for decoder — token i cannot see token j if j > i)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)  # -inf before softmax → 0 after

    # Step 4: softmax to get attention weights (probability distribution per row)
    weights = softmax_numpy(scores)  # shape: (seq_len, seq_len)

    # Step 5: weighted sum of values
    output = weights @ V  # shape: (seq_len, d_v)

    return output, weights


def demo_single_head_attention():
    """Demonstrate single-head attention with a 4-token sequence."""
    if not HAS_NUMPY:
        return

    print("\n" + "=" * 60)
    print("DEMO 1: Single-Head Scaled Dot-Product Attention")
    print("=" * 60)

    np.random.seed(42)
    seq_len = 4   # 4 tokens: ["The", "cat", "sat", "down"]
    d_model = 8   # embedding dimension
    d_k = 4       # key/query dimension

    # Simulate token embeddings (in a real model, these come from the embedding table)
    token_embeddings = np.random.randn(seq_len, d_model)

    # Learned projection matrices (these are what training optimizes)
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_k) * 0.1

    # Project embeddings to Q, K, V
    Q = token_embeddings @ W_Q  # (4, 4) — queries
    K = token_embeddings @ W_K  # (4, 4) — keys
    V = token_embeddings @ W_V  # (4, 4) — values

    print(f"\nSequence: ['The', 'cat', 'sat', 'down']")
    print(f"Shape — Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

    # Compute attention (bidirectional — encoder-style, can see all tokens)
    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"\nAttention weight matrix (rows=query token, cols=key token):")
    print("Each row shows how much that token attends to each position.")
    print(f"Rows sum to 1.0 (softmax).\n")
    tokens = ["The ", "cat ", "sat ", "down"]
    header = "         " + "  ".join(f"{t:6}" for t in tokens)
    print(header)
    for i, row in enumerate(weights):
        row_str = "  ".join(f"{v:.4f}" for v in row)
        print(f"  {tokens[i]:<6}  {row_str}")

    print(f"\nOutput shape: {output.shape}")
    print("Each output token is now a weighted mix of all value vectors.")

    # Causal (decoder) attention — token i can only see tokens 0..i
    print(f"\n--- Causal attention (decoder-style) ---")
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print("Causal mask (True = blocked):")
    print(causal_mask.astype(int))
    print("\nCausal attention weights (lower-triangular — future is masked):")
    for i, row in enumerate(weights_causal):
        row_str = "  ".join(f"{v:.4f}" for v in row)
        print(f"  {tokens[i]:<6}  {row_str}")


# ─────────────────────────────────────────────
# 2. MULTI-HEAD ATTENTION
# ─────────────────────────────────────────────

def multi_head_attention(
    x: "np.ndarray",
    num_heads: int = 4,
    d_model: int = 16
) -> "np.ndarray":
    """
    Multi-head attention: split into heads, compute per head, concatenate.

    Args:
        x: input embeddings (seq_len, d_model)
        num_heads: number of parallel attention heads
        d_model: total model dimension (must be divisible by num_heads)

    Returns:
        output: (seq_len, d_model)
    """
    if not HAS_NUMPY:
        return None

    seq_len = x.shape[0]
    d_head = d_model // num_heads  # dimension per head

    np.random.seed(7)
    # Each head has its own projection matrices
    W_Qs = [np.random.randn(d_model, d_head) * 0.1 for _ in range(num_heads)]
    W_Ks = [np.random.randn(d_model, d_head) * 0.1 for _ in range(num_heads)]
    W_Vs = [np.random.randn(d_model, d_head) * 0.1 for _ in range(num_heads)]
    W_O = np.random.randn(d_model, d_model) * 0.1  # output projection

    head_outputs = []
    for h in range(num_heads):
        Q_h = x @ W_Qs[h]  # (seq_len, d_head)
        K_h = x @ W_Ks[h]
        V_h = x @ W_Vs[h]
        out_h, _ = scaled_dot_product_attention(Q_h, K_h, V_h)
        head_outputs.append(out_h)

    # Concatenate all heads along the last dimension
    concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)

    # Final linear projection
    output = concat @ W_O  # (seq_len, d_model)
    return output


def demo_multi_head_attention():
    if not HAS_NUMPY:
        return

    print("\n" + "=" * 60)
    print("DEMO 2: Multi-Head Attention")
    print("=" * 60)

    seq_len, d_model, num_heads = 6, 16, 4
    d_head = d_model // num_heads

    print(f"\nConfig: seq_len={seq_len}, d_model={d_model}, num_heads={num_heads}")
    print(f"Each head operates in d_head={d_head} dimensions")
    print(f"Intuition: each head can learn a different relationship type")
    print(f"  Head 0 might learn: syntactic dependencies")
    print(f"  Head 1 might learn: coreference (pronoun → noun)")
    print(f"  Head 2 might learn: positional proximity")
    print(f"  Head 3 might learn: semantic similarity")

    np.random.seed(42)
    x = np.random.randn(seq_len, d_model)
    output = multi_head_attention(x, num_heads=num_heads, d_model=d_model)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Output has same shape as input — ready for next transformer layer.")


# ─────────────────────────────────────────────
# 3. TOKENIZATION DEMO
# ─────────────────────────────────────────────

class CharacterTokenizer:
    """Fallback tokenizer when tiktoken is unavailable."""

    def __init__(self):
        self.name = "character-split (fallback)"

    def encode(self, text: str) -> list:
        # Rough approximation: split on spaces, then count characters/4
        words = text.split()
        # Simulate ~0.75 words per token by splitting some words
        tokens = []
        for word in words:
            if len(word) > 5:
                tokens.append(word[:3])
                tokens.append(word[3:])
            else:
                tokens.append(word)
        return tokens

    def decode(self, tokens: list) -> str:
        return " ".join(str(t) for t in tokens)


def demo_tokenization():
    print("\n" + "=" * 60)
    print("DEMO 3: Tokenization")
    print("=" * 60)

    texts_to_tokenize = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "1000000 dollars in 2024-01-15 format",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Emoji test: 🤖 🦙 ✨",
        "GPT-4o has a 128,000 token context window.",
        "Tokenization is surprisingly non-obvious for numbers like 42, 100, 1024, 65536.",
    ]

    if HAS_TIKTOKEN:
        enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        tokenizer_name = "cl100k_base (GPT-4 encoding)"
    else:
        enc = CharacterTokenizer()
        tokenizer_name = enc.name

    print(f"\nTokenizer: {tokenizer_name}")
    print(f"{'Text':<55} {'Tokens':>6} {'Chars':>6} {'Ratio':>7}")
    print("-" * 80)

    for text in texts_to_tokenize:
        tokens = enc.encode(text)
        n_tokens = len(tokens)
        n_chars = len(text)
        ratio = n_chars / n_tokens if n_tokens > 0 else 0
        display_text = text[:52] + "..." if len(text) > 52 else text
        print(f"{display_text:<55} {n_tokens:>6} {n_chars:>6} {ratio:>6.1f}c/t")

    print("\nKey observations:")
    print("  - English averages ~4 chars/token (~0.75 words/token)")
    print("  - Numbers like 1000000 may split unexpectedly (try it with tiktoken!)")
    print("  - Emoji often take 2-3 tokens despite being 1 character")
    print("  - Code is often more token-efficient than prose")

    # Cost estimation
    print("\n--- Cost Estimation ---")
    cost_texts = [
        ("Short message", "Summarize this contract in 3 bullet points."),
        ("Medium document", "Annual report analysis " * 50),
        ("Large document", "Full legal contract text " * 500),
    ]

    cost_per_1k = 0.003  # $0.003 per 1k tokens (approximate GPT-4o-mini input)
    print(f"\nCost model: ${cost_per_1k}/1k tokens (GPT-4o-mini approximate)")
    print(f"\n{'Description':<20} {'Tokens':>8} {'Est. Cost':>12}")
    print("-" * 45)
    for description, text in cost_texts:
        tokens = enc.encode(text)
        n_tokens = len(tokens)
        cost = (n_tokens / 1000) * cost_per_1k
        print(f"{description:<20} {n_tokens:>8} ${cost:>10.6f}")

    print("\nNote: Output tokens cost 2-4x more than input tokens for most models.")


# ─────────────────────────────────────────────
# 4. HUGGINGFACE FORWARD PASS (optional)
# ─────────────────────────────────────────────

def demo_huggingface_forward_pass():
    if not HAS_TRANSFORMERS:
        print("\n[skipping] HuggingFace demo — transformers not installed")
        return

    print("\n" + "=" * 60)
    print("DEMO 4: HuggingFace Forward Pass (distilgpt2)")
    print("=" * 60)
    print("Loading distilgpt2 (82M params, tiny, CPU-friendly)...")

    try:
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        generator = pipeline("text-generation", model="distilgpt2", tokenizer=tokenizer)

        prompt = "The transformer architecture works by"
        print(f"\nPrompt: '{prompt}'")

        # Tokenize and show token IDs
        token_ids = tokenizer.encode(prompt)
        print(f"\nToken IDs: {token_ids}")
        print(f"Tokens:    {[tokenizer.decode([tid]) for tid in token_ids]}")

        # Run forward pass and get logits for next token prediction
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        model.eval()

        input_ids = torch.tensor([token_ids])
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # shape: (1, seq_len, vocab_size)

        # Get next-token logits (last position)
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # Convert to probabilities
        probs = torch.softmax(next_token_logits, dim=-1)

        # Top 5 predictions
        top5_probs, top5_ids = torch.topk(probs, 5)
        print(f"\nTop 5 next token predictions:")
        for i, (prob, token_id) in enumerate(zip(top5_probs.tolist(), top5_ids.tolist())):
            token_str = repr(tokenizer.decode([token_id]))
            print(f"  {i+1}. {token_str:<20} prob: {prob:.4f}  ({prob*100:.2f}%)")

        print(f"\nVocabulary size: {len(probs):,} tokens")
        print(f"Logits shape: (batch=1, seq_len={len(token_ids)}, vocab={len(probs):,})")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("This may happen if you're offline. Run once with internet access to cache the model.")


# ─────────────────────────────────────────────
# 5. POSITIONAL ENCODING VISUALIZATION
# ─────────────────────────────────────────────

def demo_positional_encoding():
    if not HAS_NUMPY:
        return

    print("\n" + "=" * 60)
    print("DEMO 5: Sinusoidal Positional Encoding")
    print("=" * 60)

    def sinusoidal_pe(seq_len: int, d_model: int) -> "np.ndarray":
        """Original transformer positional encoding."""
        pe = np.zeros((seq_len, d_model))
        positions = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
        dims = np.arange(0, d_model, 2)  # even dimensions

        # Different frequencies for different dimensions
        freqs = 1.0 / (10000 ** (dims / d_model))

        pe[:, 0::2] = np.sin(positions * freqs)  # even dims
        pe[:, 1::2] = np.cos(positions * freqs)  # odd dims
        return pe

    seq_len, d_model = 8, 16
    pe = sinusoidal_pe(seq_len, d_model)

    print(f"\nPositional encoding shape: {pe.shape}")
    print(f"Each of the {seq_len} positions gets a unique {d_model}-dim vector.")
    print("These are ADDED to token embeddings before the first attention layer.")
    print("\nFirst 4 dimensions for each position (showing sinusoidal patterns):")
    print(f"\n{'Pos':>4}  {'dim0':>8}  {'dim1':>8}  {'dim2':>8}  {'dim3':>8}")
    print("-" * 50)
    for i in range(seq_len):
        vals = "  ".join(f"{pe[i, d]:>8.4f}" for d in range(4))
        print(f"{i:>4}  {vals}")

    print("\nKey insight: different dimensions oscillate at different frequencies.")
    print("Low dimensions (slow oscillation) encode coarse position.")
    print("High dimensions (fast oscillation) encode fine-grained position.")
    print("The model learns to read position from this signal.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("3.1 Transformers & Attention — Live Demos")
    print("=" * 60)

    if HAS_NUMPY:
        demo_single_head_attention()
        demo_multi_head_attention()
        demo_positional_encoding()
    else:
        print("\n[skipping attention and positional encoding demos — install numpy]")

    demo_tokenization()
    demo_huggingface_forward_pass()

    print("\n" + "=" * 60)
    print("All demos complete.")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Attention is just weighted sum — the magic is in what the weights learn")
    print("  2. Multi-head = run attention N times with different projections, concatenate")
    print("  3. Tokens ≠ words — numbers, emoji, non-English are often more expensive")
    print("  4. Positional encoding is added to embeddings so the model knows word order")
    print("  5. The output of every layer has the same shape as the input — stackable")
