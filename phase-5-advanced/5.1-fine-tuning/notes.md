# 5.1 Fine-Tuning

## The Most Important Question: Should You Fine-Tune at All?

Fine-tuning is seductive. It sounds powerful, and it is — but it's also expensive, slow to iterate on, and often the wrong tool. Before writing a single line of training code, work through this decision framework.

### Decision Framework: Prompting → RAG → Fine-Tune

Always exhaust cheaper options first. The correct order is:

**Step 1 — Try prompting.** Give the model a detailed system prompt, few-shot examples, and a good output format. For 80% of use cases, a well-engineered prompt is sufficient. Cost: hours of your time. Iteration cycle: minutes.

**Step 2 — Add RAG.** If the model needs factual knowledge it doesn't have, retrieve it. RAG gives the model access to your private data, recent information, and cited sources. Cost: embedding + vector DB infrastructure. Iteration cycle: hours.

**Step 3 — Fine-tune only if both are insufficient.** Fine-tuning changes the model's weights permanently for your use case. It's the right answer in specific circumstances, described below.

---

## When to Fine-Tune

Fine-tuning is the right choice when you have a specific **style, format, or reasoning pattern** that you cannot achieve through prompting alone.

### Fine-tune when:

**1. Specific output style/format that prompting can't reliably achieve.**
If you're generating SQL queries in your company's specific style, outputting JSON in an unusual schema, or writing customer emails that match a very specific brand voice — and your few-shot examples still produce inconsistent results — fine-tuning bakes that style into the weights. Every output will match the pattern without needing examples in every prompt.

**2. Compressing many few-shot examples into weights.**
If you need 10 examples in every prompt just to get acceptable output (at 2000 tokens per prompt × millions of requests = enormous cost), fine-tuning converts those examples into model weights. Your production prompt can be much shorter. This is a cost optimization play.

**3. Domain-specific reasoning patterns, not just domain knowledge.**
This is the subtle distinction: RAG gives the model domain **knowledge** (facts, documents, data). Fine-tuning gives the model domain **reasoning patterns** — how an expert in your field thinks through problems. A medical diagnosis model fine-tuned on physician notes doesn't just know medical facts; it applies medical reasoning patterns. RAG cannot do this.

### RAG when:

- Your knowledge base changes frequently (new documents, updated prices, recent news)
- Users need to see citations and sources
- You have large amounts of private data you can't include in training
- The problem is "model doesn't know X" (a knowledge gap, not a reasoning gap)

### Prompting when:

- You're still in early product development and requirements will change
- Few-shot examples in the prompt already work acceptably
- You have no labeled training data
- The task involves general knowledge and reasoning

---

## Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning (PEFT)

### Full Fine-Tuning

Update all model parameters on your dataset. For a 7B parameter model at float32, that's 28GB just for the model — plus optimizer states (Adam needs 2x parameters = 56GB more), plus gradients (another 28GB) = ~112GB minimum GPU memory for a 7B model. This is why full fine-tuning requires A100/H100 clusters. Iteration cycles are slow and expensive.

### Parameter-Efficient Fine-Tuning (PEFT)

Only update a small fraction of parameters. The big winner here is LoRA.

---

## LoRA: Low-Rank Adaptation

**The core insight:** When you fine-tune a model, the weight updates (delta W) tend to be low-rank. Instead of computing and storing the full high-dimensional weight update, LoRA approximates it with two small matrices.

### How LoRA Works

For each weight matrix W (dimensions d × d), instead of computing the full update ΔW, LoRA adds:

```
W' = W + ΔW = W + B × A
```

Where:
- A is a d × r matrix (randomly initialized with normal distribution)
- B is a r × d matrix (initialized to zero, so ΔW = 0 at the start of training)
- r is the rank (hyperparameter, typically 4–64)

**Parameter count:** If d = 4096 and r = 8, then A has 32,768 parameters and B has 32,768 parameters = 65,536 total. The original weight matrix had 4096 × 4096 = 16,777,216 parameters. LoRA uses ~0.4% of the original parameters for that layer.

A 7B model with LoRA (r=8) typically trains ~40M parameters instead of 7B — that's 175x fewer parameters to train and store.

**At inference:** You either keep the adapter separate (fast to swap between tasks) or merge it into the base weights (no inference overhead).

### LoRA Applied to Attention

LoRA is typically applied to the attention weight matrices Q, K, V, and O (query, key, value, output projections). You can also apply it to the feed-forward layers, which improves quality at the cost of more trainable parameters.

---

## QLoRA: Quantized LoRA

QLoRA = LoRA + 4-bit quantization of the base model.

The base model weights are stored in 4-bit (NF4 quantization, a special format for normally-distributed weights). During the forward pass, weights are dequantized to bfloat16 on the fly, computation happens, then weights are discarded. Only the LoRA adapter matrices A and B are kept in full precision.

**Result:** A 7B model that required 112GB GPU memory for full fine-tuning now fits in 6-8GB. You can fine-tune a 7B model on a single consumer GPU (RTX 3090, RTX 4090). You can fine-tune a 13B model on a 24GB GPU.

This democratized fine-tuning. Before QLoRA (2023), fine-tuning required enterprise GPU clusters.

---

## Training Approaches

### Instruction Tuning

The most common form of fine-tuning. Train on (instruction, response) pairs so the model learns to follow natural language instructions in your domain.

Dataset format (ChatML):
```
<|im_start|>system
You are a medical coding assistant...<|im_end|>
<|im_start|>user
What ICD-10 code applies to type 2 diabetes with chronic kidney disease stage 3?<|im_end|>
<|im_start|>assistant
E11.22 — Type 2 diabetes mellitus with diabetic chronic kidney disease, stage 3...<|im_end|>
```

### RLHF (Reinforcement Learning from Human Feedback) — Conceptually

RLHF is how ChatGPT was trained to be helpful and harmless. Three stages:
1. Supervised fine-tuning on demonstrations (instruction tuning)
2. Train a reward model on human preference data (which response is better?)
3. Optimize the policy model against the reward model using PPO (Proximal Policy Optimization)

In practice: very complex, requires large amounts of human preference data, and is mainly done by AI labs.

### DPO: Direct Preference Optimization

DPO is a simpler alternative to RLHF that skips the reward model entirely. You train directly on pairs of (chosen response, rejected response) for each prompt. The model learns to increase the probability of the chosen response relative to the rejected response.

Much simpler to implement than RLHF, comparable quality, and becoming the standard for preference fine-tuning in production.

---

## Dataset Requirements

**Minimum:** 1,000 high-quality examples. Quality beats quantity. 100 perfect examples > 10,000 mediocre ones.

**Format: ChatML**
The standard format for instruction fine-tuning. Uses special tokens `<|im_start|>` and `<|im_end|>` with role prefixes (system, user, assistant).

**Format: Alpaca**
Simpler format with `instruction`, `input`, `output` fields. Used by many early open-source fine-tuning projects. LLaMA Factory and other tools can convert between formats.

**Data cleaning matters more than you think:**
- Remove near-duplicate examples (ROUGE-1 dedup)
- Filter responses shorter than 10 tokens or longer than 2048 tokens
- Check for formatting inconsistencies
- If using GPT-4 to generate training data, check OpenAI's terms of service (distillation restrictions)

---

## Key Hyperparameters

| Hyperparameter | Typical Value | Notes |
|----------------|---------------|-------|
| Learning rate | 2e-4 | For LoRA. Full fine-tuning is 1e-5 to 5e-5. |
| Epochs | 1–3 | More epochs → overfitting on small datasets |
| LoRA rank (r) | 8–64 | Higher r = more capacity but more parameters |
| LoRA alpha | 2× rank | Scaling factor for LoRA update (alpha/r) |
| LoRA dropout | 0.05 | Regularization for LoRA layers |
| Batch size | 4–16 | Use gradient accumulation if memory limited |
| Warmup ratio | 0.03 | Fraction of steps to warm up learning rate |
| Max sequence length | 2048–4096 | Longer = more memory |

**Rule of thumb:** Start with r=8. If your task is complex (requires a lot of new behavior), try r=16 or r=32.

---

## Tools

**HuggingFace PEFT:** The standard library. Works with any HuggingFace model. Well-documented, production-ready.

**Unsloth:** 2x faster than standard PEFT, less memory, identical results. Uses hand-written CUDA kernels. Best choice for training on a single GPU.

**LLaMA Factory:** Excellent UI + config-based fine-tuning. Good for teams who want a non-code interface.

**OpenAI Fine-Tuning API:** Easiest path if you're already on GPT-3.5 or GPT-4o mini. You provide JSONL training data, they handle the rest. ~$8/1M training tokens. No infrastructure to manage. Limited to OpenAI models.

**Axolotl:** Config-file-based fine-tuning with many advanced features. Popular in the open-source community for serious training runs.

---

## Common Mistakes

1. **Fine-tuning before trying prompting** — Always try prompting first. It takes hours, not days.
2. **Using too little data** — 50 examples will overfit. You need diversity.
3. **Forgetting to evaluate** — Always hold out 10–20% of data for evaluation. Track eval loss across epochs.
4. **Not checkpointing** — Save checkpoints every 500 steps. Training can crash.
5. **Fine-tuning when you need RAG** — If the model needs updated facts, use RAG. Fine-tuning doesn't update factual knowledge well.
