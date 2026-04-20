# 3.7 Open-Source Models

## Why Open-Source?

When you call the Anthropic or OpenAI API, your data leaves your environment and travels to a third-party server. For many use cases — consumer apps, internal tools, academic research — this is perfectly acceptable. But certain situations demand local inference:

**Data privacy.** Medical records, legal documents, financial data, PII. Some industries have regulatory requirements (HIPAA, GDPR) that prevent data from being sent to external processors without explicit agreements.

**No per-token cost.** Cloud APIs charge per token. At high volume (millions of requests per day) the economics shift in favor of running your own model on owned or reserved compute.

**Customization.** You can fine-tune open weights on your proprietary data in ways that are impossible with closed models. You can also modify the system prompt, change sampling parameters, and inspect model internals.

**Offline use.** Edge devices, air-gapped environments, mobile applications. A 4-bit quantized 7B model fits in 4–5 GB and runs on a laptop CPU.

**Reproducibility.** Model providers can deprecate models with 3 months' notice. Open weights are yours forever.

---

## Key Model Families

### Llama 3 (Meta)
The most widely used open-source model family. Llama 3.1 and 3.2 are available in 1B, 3B, 8B, 70B, and 405B parameter sizes. The 8B model runs comfortably on consumer GPUs (8GB VRAM) and delivers quality competitive with earlier GPT-3.5. The 70B model requires 48GB+ VRAM but reaches near GPT-4 level on many benchmarks. Llama models are licensed for commercial use with restrictions.

### Mistral / Mixtral (Mistral AI)
Mistral 7B punches above its weight — consistently outperforms Llama 2 13B on most benchmarks at half the parameter count. Mixtral 8×7B is a Mixture-of-Experts (MoE) architecture: 8 expert networks of 7B each, routing each token to 2 experts. Effective quality of a 40B+ model with inference cost of a 13B. Both are Apache 2.0 licensed — fully open commercial use.

### Qwen 2.5 (Alibaba)
Particularly strong on multilingual tasks and code. Qwen 2.5-Coder is a code-specialized variant. Available in 0.5B to 72B sizes. Strong Chinese/English bilingual performance.

### Phi-3 / Phi-4 (Microsoft)
The "small but capable" family. Phi-3 Mini (3.8B) matches or exceeds larger models on reasoning tasks. The key insight: models trained on very high-quality synthetic data can be much smaller than models trained on internet crawls. Phi runs on CPU-only machines and even some mobile devices.

### Gemma (Google)
Based on the same research as Gemini. Gemma 2 (2B, 9B, 27B) is released under a permissive license. Strong reasoning, instruction following, and safety. Good choice when you want a Google-backed open model.

---

## Running with Ollama

Ollama is the easiest way to run open-source models locally. It handles model download, quantization, and serving with an OpenAI-compatible API.

```bash
# Install (macOS)
brew install ollama

# Start the Ollama server
ollama serve

# Pull and run a model (interactive chat)
ollama run llama3

# Pull a model for API use
ollama pull llama3
ollama pull mistral
ollama pull phi3
```

Ollama exposes `http://localhost:11434/api/chat` — compatible with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "What is RAG?"}],
)
print(response.choices[0].message.content)
```

---

## Hugging Face Hub

The Hugging Face Hub hosts over 500,000 models. Every model has a model card describing training data, intended use, limitations, and benchmarks. Read the model card before deploying any model.

```python
from transformers import pipeline

# Text generation
generator = pipeline("text-generation", model="microsoft/phi-3-mini-4k-instruct")
result = generator("Explain attention mechanisms:", max_new_tokens=100)
print(result[0]["generated_text"])
```

The `pipeline` API abstracts tokenization, model loading, and decoding. It supports:
- `text-generation` — LLM completion
- `text-classification` — sentiment, intent
- `summarization` — document summarization
- `zero-shot-classification` — classify into labels without training examples
- `feature-extraction` — get embeddings
- `question-answering` — extractive QA
- `translation` — machine translation

For production, prefer explicit `AutoModelForCausalLM` + `AutoTokenizer` over pipeline — gives you more control over batching, quantization, and device placement.

---

## GGUF Format and Quantization

The original model weights are 32-bit floats or 16-bit floats — each parameter takes 2–4 bytes. A 7B model at float16 requires ~14 GB of RAM.

**Quantization** reduces the precision of weights (e.g., from 16-bit to 4-bit integers), shrinking RAM requirements by 4× at some quality cost.

**GGUF** is the file format used by `llama.cpp` — a highly optimized C++ inference engine. GGUF models run entirely on CPU (though GPU offloading is supported).

Common quantization levels:
- `Q8_0`: 8-bit, ~8 GB for 7B. Near lossless quality.
- `Q4_K_M`: 4-bit mixed, ~4.5 GB for 7B. Good quality/size tradeoff, most popular.
- `Q2_K`: 2-bit, ~2.5 GB for 7B. Noticeable quality loss.

Ollama uses GGUF internally and handles quantization automatically.

---

## Model Selection Guide

| Size | VRAM | Use case | Example models |
|---|---|---|---|
| 1B–3B | 2–4 GB (CPU) | Mobile, edge, single-task classifiers | Phi-3 Mini, Llama 3.2 3B |
| 7B–8B | 6–8 GB GPU or fast CPU | General chat, coding, summarization | Llama 3 8B, Mistral 7B |
| 13B | 12–16 GB GPU | Better reasoning, balanced quality | Llama 2 13B |
| 70B | 48 GB GPU (2× 24 GB) | Near frontier quality, complex tasks | Llama 3 70B, Qwen 2.5 72B |
| 400B+ | Multi-GPU cluster | Frontier quality, open weights | Llama 3.1 405B |

Start with the smallest model that passes your eval suite. Bigger is not always better if your task is simple.

---

## Benchmarks to Know

**MMLU (Massive Multitask Language Understanding)** — 57 academic subjects, multiple choice. Tests broad knowledge. Score = accuracy %. GPT-4o: ~88%, Llama 3 8B: ~68%.

**HumanEval** — 164 Python coding problems. Tests code generation. Score = pass@1 (fraction solved on first try). GPT-4o: ~90%, Llama 3 8B: ~62%.

**MT-Bench** — Human-judged multi-turn conversation quality. Score 1–10 across categories. Measures instruction following, reasoning, coding, math.

**MATH / GSM8K** — Mathematical reasoning benchmarks. Critical for use cases involving calculations.

Caveat: benchmark numbers are widely gamed. A model that is #1 on MMLU may underperform on your specific domain. Always run evals on your own data before committing to a model.

---

## Open-Source vs Closed: Decision Framework

| Factor | Prefer Open-Source | Prefer Closed API |
|---|---|---|
| Data privacy | Sensitive / regulated data | Public or low-sensitivity data |
| Volume | >10M tokens/day | <10M tokens/day |
| Quality needed | Good enough at 70B | Best available, no compromises |
| Infra resources | GPU available or willing to invest | No GPU, no ops team |
| Customization | Fine-tuning required | General purpose works |
| Latency target | < 50ms on fast GPU | Acceptable to wait for cloud |
| Maintenance | Team can run ML infra | No ML ops capacity |
| Startup phase | Later, when scale justifies | Early, move fast |

Most production systems end up using both: a closed API for high-stakes or complex tasks, an open-source model for high-volume, lower-complexity requests.
