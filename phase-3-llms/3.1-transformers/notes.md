# 3.1 Transformers & Attention — Notes

Transformers are the foundation of every major AI system you will work with. Understanding them deeply — even without math derivations — lets you make better engineering decisions: why context windows matter, why some prompts confuse models, why certain tasks are hard for LLMs. This section builds that intuition.

---

## Tokens: The Atomic Unit of LLMs

Before a model sees any text, it converts characters into **tokens**. A token is roughly 0.75 words on average for English text, but the relationship is not fixed. Common short words are often a single token. Rare words get split into multiple tokens. Numbers and code behave unexpectedly.

**Byte-Pair Encoding (BPE)** is the dominant tokenization method. The algorithm starts with individual characters, then iteratively merges the most frequently co-occurring pairs into new tokens. It runs over a large corpus at training time and produces a fixed vocabulary (GPT-4 uses 100k tokens). At inference time, new text is split greedily using this vocabulary.

Practical consequences for engineers:
- Pricing is per token, not per word. 1 million tokens ≈ 750k words.
- Numbers like `1,000,000` may tokenize as 4-6 tokens. `1000000` is different from `1,000,000`.
- Emoji are often 2-3 tokens each.
- Code tokens are denser — Python is more token-efficient than some natural languages.
- Non-English languages are often less token-efficient than English, meaning higher cost for same content.

---

## Embeddings: Meaning as Position in Space

Once text is tokenized, each token is converted to an **embedding** — a dense vector of floating-point numbers (e.g., 4096 dimensions for Llama 3 8B). These vectors are learned during training.

The key insight: **semantically similar concepts end up close together in this vector space.** The classic example is that `king - man + woman ≈ queen` in embedding space. This is not hand-coded — it emerges from training on vast text by predicting next tokens.

Embeddings are looked up from a learned table (the embedding matrix). Position 0 might be the token for "the", and its row in the matrix is its embedding vector. This lookup is the first operation in a transformer.

---

## Self-Attention: How Tokens Talk to Each Other

This is the core innovation of transformers. Before self-attention, models processed sequences left-to-right with no way to directly connect token 1 to token 100. Self-attention lets every token attend to every other token simultaneously.

**Intuition with Q/K/V:**

Think of a library lookup system:
- **Query (Q):** "What am I looking for?" — each token asks a question about what information it needs
- **Key (K):** "What do I contain?" — each token broadcasts what it has to offer
- **Value (V):** "What I actually contribute if selected" — the content that flows forward

For each token, you compute how well its Query matches every other token's Key. High match score means "pay a lot of attention to that token's Value." The result is a weighted sum of all Values, where the weights come from Query-Key matches.

**The formula:** `Attention(Q, K, V) = softmax(QK^T / √d_k) V`

- `QK^T` computes dot products between all query-key pairs — a big matrix of compatibility scores
- `/ √d_k` scales down the scores (without this, large dimensions cause very peaked softmax, killing gradients)
- `softmax(...)` converts scores to a probability distribution (all positive, sums to 1)
- Multiplying by `V` computes the weighted sum of values

---

## Multi-Head Attention: Multiple Relationship Types Simultaneously

A single attention head might learn to track grammatical agreement. Another might learn coreference (connecting "it" to the noun it refers to). Another might learn positional proximity. Multi-head attention runs several attention computations in parallel with different learned weight matrices.

With 8 heads, the model projects Q/K/V into 8 smaller subspaces, computes attention in each, then concatenates the results and projects back to the original dimension. This lets the model capture multiple different relationship types at each layer, which is essential for understanding complex language.

---

## Positional Encoding: Why Order Matters

Pure attention has no notion of position — it treats the input as an unordered set. "Dog bites man" and "Man bites dog" would be identical without positional information. Positional encodings fix this by adding a position-dependent signal to each token's embedding before attention is computed.

The original transformer used sinusoidal functions (different frequencies for different dimensions). Modern models use **RoPE** (Rotary Position Embedding), which encodes position as a rotation applied to Q and K before their dot product — this is why models like Llama 3 handle longer contexts better than earlier designs.

---

## Feed-Forward Layers: Per-Token Processing

After attention mixes information across tokens, each token goes through an identical feed-forward network applied **independently**. This is where much of the model's "knowledge" is stored — factual associations, grammar rules, world knowledge. It's typically 4x wider than the model dimension, uses a nonlinear activation (ReLU or GeLU or SwiGLU), then projects back down.

The pattern is: attend (mix across positions) → feed-forward (process each position). Stacked many times (32 layers for Llama 3 8B, 96 layers for GPT-4 class models).

---

## Layer Norm and Residual Connections

Two stability tricks that make training deep transformers possible:

**Residual connections:** Each sublayer adds its output to its input: `output = sublayer(x) + x`. This means gradients can flow directly back through the addition, bypassing the sublayer entirely. Without residuals, gradients vanish in 32+ layer networks and training fails.

**Layer normalization:** Normalizes the activations across the feature dimension at each layer, stabilizing training. Applied before each sublayer (pre-norm) in modern models. Without this, activations grow or shrink exponentially through layers.

---

## Context Windows: The Hard Limit

Every transformer has a maximum sequence length — the **context window**. Beyond this limit, the model cannot process additional tokens:
- GPT-4o: 128k tokens (~96k words)
- Claude 3.5 Sonnet: 200k tokens (~150k words)
- Llama 3 8B: 8k tokens

When a conversation or document exceeds the context window, you must truncate. Common strategies: drop oldest messages (for conversation), use RAG to retrieve only relevant chunks (for documents), or use a summarization step to compress history.

The **attention matrix** scales as O(n²) with sequence length — doubling context quadruples compute. This is why extending context windows requires architectural innovations like sparse attention or sliding window attention.

---

## Decoder-Only vs Encoder-Decoder

**Decoder-only** (GPT, Llama, Claude, Mistral): Each token can only attend to itself and previous tokens (causal masking). Trained to predict next tokens. Best for generation tasks.

**Encoder-decoder** (T5, BART): Encoder sees the full input bidirectionally, decoder generates output one token at a time while attending to encoder output. Best for translation, summarization where you have a full input to compress before generating output.

GPT-4, Claude, and most modern chat models are decoder-only.

---

## Self-Supervised Pretraining

Transformers learn by **predicting the next token** on massive text corpora (the internet, books, code). No human labels needed — the training signal is free and abundant. This is called self-supervised learning. The model sees trillions of tokens and learns language structure, facts, reasoning patterns, and code by compression.

This is why GPT-4 knows about Paris being in France, can write Python, and can reason about causality — all from next-token prediction on raw text.

---

## RLHF: Why Claude and ChatGPT Are Helpful

Pretrained models predict next tokens — they don't inherently "want" to be helpful. They might complete your prompt with the next statistically likely text, which could be harmful, verbose, or off-topic.

**Reinforcement Learning from Human Feedback (RLHF)** fine-tunes the model to produce outputs that humans prefer:
1. Collect human preference data (humans compare two model responses, pick the better one)
2. Train a reward model to predict human preferences
3. Use RL (PPO) to optimize the LLM's policy to maximize the reward model's score

This is why Claude follows instructions, stays on topic, declines harmful requests, and formats responses clearly. The helpfulness is trained in post-pretraining.

---

## Why Transformers Beat RNNs

Recurrent Neural Networks (LSTMs, GRUs) process sequences one token at a time, left to right. Two critical weaknesses:
1. **No parallelization during training:** each step depends on the previous hidden state, so you can't use GPUs efficiently
2. **Long-range dependency problem:** information from token 1 must pass through every intermediate hidden state to reach token 100 — it gets diluted and forgotten

Transformers solve both:
1. **Full parallelization:** all attention computations across all positions happen simultaneously
2. **Direct long-range attention:** token 1 can directly attend to token 100 with a single dot product — no information highway through intermediate states

This enabled training on the internet scale (trillions of tokens) and made modern LLMs possible.
