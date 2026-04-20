# 5.6 Model Quantization & Optimization

## Why Quantization Matters

A 70B parameter LLM in float32 requires 280GB of GPU memory — that's more than a single H100 (80GB). Quantization compresses model weights from 32-bit floats to 8-bit or 4-bit integers, dramatically reducing memory requirements at a small quality cost.

Without quantization, running open-source LLMs at scale is prohibitively expensive. With quantization, a 7B model runs on a $300 consumer GPU.

---

## Numeric Precision Basics

| Format | Bits | Bytes/param | 7B model memory | Quality |
|---|---|---|---|---|
| float32 | 32 | 4 | 28 GB | Reference |
| bfloat16 | 16 | 2 | 14 GB | ~same as float32 |
| int8 | 8 | 1 | 7 GB | -1 to -3% perplexity |
| int4 | 4 | 0.5 | 3.5 GB | -5 to -10% perplexity |
| int2 | 2 | 0.25 | 1.75 GB | Significant degradation |

**Perplexity** measures how "surprised" a model is by text — lower is better. A 5% perplexity increase is often imperceptible to users. A 20% increase is clearly noticeable.

---

## GGUF Format and llama.cpp

GGUF is a binary format for quantized models, used with **llama.cpp** — a CPU-first inference engine. This lets you run LLMs on a laptop without a GPU.

### Naming conventions

| Name | Bits | Description |
|---|---|---|
| Q4_K_M | 4-bit | K-quantization, medium — best all-round choice |
| Q4_K_S | 4-bit | K-quantization, small — slightly worse than M |
| Q5_K_M | 5-bit | Higher quality, more memory |
| Q8_0 | 8-bit | Near-float16 quality, 2× memory of Q4 |
| F16 | 16-bit | Full half-precision, same quality as bfloat16 |

**Rule of thumb:** Start with Q4_K_M. If quality is noticeably poor, move to Q5_K_M or Q8_0.

### Running with llama.cpp Python bindings

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./llama-3-8b-instruct.Q4_K_M.gguf",
    n_ctx=4096,        # context window
    n_gpu_layers=-1,   # -1 = offload all layers to GPU (if available)
    verbose=False,
)

output = llm("Explain RAG in one sentence:", max_tokens=100)
print(output["choices"][0]["text"])
```

---

## bitsandbytes (PyTorch)

For HuggingFace models, `bitsandbytes` enables in-memory quantization:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit quantization (QLoRA style)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # NormalFloat4 — better for normal distributions
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # nested quantization for extra memory savings
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

This loads the 8B model in ~5GB instead of 16GB.

---

## Flash Attention 2

Flash Attention 2 is a GPU kernel optimization for the attention computation — not a quantization technique. It produces **identical outputs** to standard attention but uses 2-4× less memory and runs 2-4× faster by restructuring memory access patterns.

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # just this line
)
```

Requires: Ampere+ GPU (A10, A100, H100) and `pip install flash-attn`.

---

## KV Cache

During autoregressive generation, the model recomputes Key and Value matrices for all previous tokens at every step. The **KV cache** stores these, eliminating redundant computation.

- Without cache: O(n²) computation per generation step
- With cache: O(n) — only compute for the new token

KV cache grows with sequence length and batch size. For a 70B model with a 100K context window, the KV cache alone can require 80GB+. This is why long-context models are memory-intensive even when quantized.

---

## Continuous Batching (vLLM)

Traditional static batching: wait until batch is full, run all together. Problem: requests have different lengths — short requests wait for long ones.

**Continuous batching** (vLLM's technique): process tokens across multiple requests in a single forward pass, retiring finished requests and adding new ones dynamically. This achieves 10-50× higher throughput than naive batching.

```bash
# Start vLLM server (OpenAI-compatible API)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --quantization awq \
    --max-model-len 8192 \
    --port 8000
```

---

## Speculative Decoding

A small draft model generates k candidate tokens quickly. The main model then verifies them all in parallel (one forward pass). If the draft was correct, you get k tokens for the cost of 1 main-model pass.

Speedup: 2-3× on CPU-bound workloads. Best when draft and main model are the same family (e.g., Llama 3 8B drafts for Llama 3 70B).

---

## ONNX — Portable Model Format

ONNX (Open Neural Network Exchange) exports PyTorch models to a standard format that runs on TensorRT, OpenVINO, CoreML, and other backends:

```python
torch.onnx.export(
    model,
    (input_ids,),
    "model.onnx",
    opset_version=17,
    input_names=["input_ids"],
    output_names=["logits"],
)
```

Use case: deploy a model to edge devices, mobile, or specialized inference hardware without PyTorch.

---

## When to Quantize (Decision Guide)

| Scenario | Recommendation |
|---|---|
| Production inference, cost-sensitive | INT8 (near-lossless, 2× speedup) |
| Consumer GPU (< 16GB VRAM) | INT4/Q4_K_M |
| Fine-tuning (QLoRA) | NF4 (bitsandbytes) |
| CPU-only deployment | GGUF Q4_K_M |
| Highest possible quality | bfloat16 |
| Cross-platform deployment | ONNX |

**Never quantize during fine-tuning** if you want the best model — quantize after.
