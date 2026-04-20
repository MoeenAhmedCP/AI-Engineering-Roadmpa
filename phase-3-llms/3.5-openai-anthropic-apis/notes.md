# 3.5 OpenAI and Anthropic APIs — Production Usage Guide

## Chat Completions API

Both OpenAI and Anthropic use a messages array as the core abstraction. Every API call is stateless — you send the full conversation history each time.

### Message Roles

- **system** — Sets the model's behavior, persona, and constraints. Only one system message, at the start. Anthropic passes this as a top-level `system` parameter; OpenAI puts it in the messages array.
- **user** — The human's input.
- **assistant** — The model's previous response. Include past assistant turns to maintain conversation context.

```python
# Anthropic
messages = [
    {"role": "user", "content": "What is a transformer?"},
    {"role": "assistant", "content": "A transformer is an architecture..."},
    {"role": "user", "content": "Can you give me an example?"},
]

# OpenAI (same structure, system goes inside messages)
messages = [
    {"role": "system", "content": "You are an expert ML engineer."},
    {"role": "user", "content": "What is a transformer?"},
]
```

The model does not have persistent memory — you are responsible for maintaining history and truncating it when it grows too long.

---

## Streaming

Without streaming, the user stares at a blank screen for 10–30 seconds while the model generates a long response. With streaming, tokens appear as they are produced — far better UX.

```python
# Anthropic streaming
with client.messages.stream(model="claude-sonnet-4-6", max_tokens=512, messages=messages) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# OpenAI streaming
stream = client.chat.completions.create(model="gpt-4o", messages=messages, stream=True)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

Key points: always `flush=True` when printing streamed output. Streaming does not change total token cost — just delivery timing.

---

## Function Calling and Tool Use

Tool use lets the model request that your code execute a function and return results. The model does not call the function — it signals intent and you run the code.

### Defining a Tool (Anthropic)

```python
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}]
```

### The Tool Use Loop

1. Send user message + tools to model.
2. Model returns `stop_reason="tool_use"` with a tool name + arguments.
3. Your code executes the function with those arguments.
4. You send the result back as a `tool_result` message.
5. Model generates its final answer using the function result.

This loop can repeat if the model needs multiple tool calls. This is the foundation of agentic systems.

---

## Structured Outputs

Getting reliable JSON from an LLM is critical for production use.

**OpenAI JSON mode:** Set `response_format={"type": "json_object"}`. The model is forced to output valid JSON, but the schema is not enforced — you still need to validate with Pydantic.

**OpenAI Structured Outputs (GPT-4o and later):** Provide a JSON Schema and the model is guaranteed to match it exactly via constrained decoding.

**Anthropic prefill:** Start the `assistant` message with `{` to force JSON output. Less robust than constrained decoding but widely used.

**Best practice for both:** Ask for JSON in your prompt, specify the exact schema, and parse with Pydantic. Always wrap parsing in try/except.

---

## Vision

Both APIs accept image content in messages as base64 or URL.

```python
# Anthropic
{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_str}},
    {"type": "text", "text": "What is in this image?"},
]}

# OpenAI
{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
    {"type": "text", "text": "What is in this image?"},
]}
```

Images consume tokens based on resolution. Always resize large images before sending — a 4K screenshot costs 10× a thumbnail.

---

## Token Counting

Tokens are the billing unit. 1 token ≈ 4 characters or ¾ of a word in English.

```python
# OpenAI — tiktoken library
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, how are you?")
print(len(tokens))  # 6

# Anthropic — SDK method
client = anthropic.Anthropic()
response = client.messages.count_tokens(model="claude-sonnet-4-6", messages=messages)
print(response.input_tokens)
```

For quick estimates without libraries: `len(text) // 4` is accurate within ~15%.

---

## Cost Math

Total cost = (input_tokens × input_price) + (output_tokens × output_price)

Prices as of early 2025 (per million tokens):

| Model | Input | Output | Best for |
|---|---|---|---|
| claude-haiku-3-5 | $0.80 | $4.00 | Simple tasks, high volume |
| claude-sonnet-4-6 | $3.00 | $15.00 | Balanced quality + cost |
| claude-opus-4 | $15.00 | $75.00 | Complex reasoning |
| gpt-4o-mini | $0.15 | $0.60 | Simple tasks, cheapest |
| gpt-4o | $2.50 | $10.00 | Strong general purpose |
| o1 | $15.00 | $60.00 | Deep reasoning |

Rule of thumb: use the cheapest model that passes your eval suite. Upgrade when quality is the bottleneck, not by default.

---

## Model Selection Guide

- **Haiku / gpt-4o-mini:** Classification, extraction, simple Q&A, high-volume pipelines where cost matters.
- **Sonnet / gpt-4o:** The default for most features. Good reasoning, good code, multimodal.
- **Opus / o1:** Long, complex reasoning chains; tasks where quality errors are very costly.

Start with Sonnet, run evals, downgrade to Haiku if results hold, upgrade to Opus only if Sonnet fails.

---

## Batch API

Both providers offer batch processing at roughly 50% discount.

- Submit a JSONL file of requests.
- Results are available asynchronously (minutes to hours).
- Use for: bulk evaluation, offline processing, dataset generation, nightly analysis jobs.
- Not for: real-time user-facing requests.

```python
# Anthropic Batch
batch = client.messages.batches.create(requests=[...])
# Poll until batch.processing_status == "ended"
for result in client.messages.batches.results(batch.id):
    print(result.result.message.content)
```

---

## Prompt Caching (Anthropic)

Cache frequently repeated prompt prefixes to reduce cost and latency. Cached tokens cost 10% of normal input price on re-use.

Requirements to use caching:
- The cacheable block must be at least 1,024 tokens.
- Mark it with `"cache_control": {"type": "ephemeral"}`.
- Cache persists for approximately 5 minutes (ephemeral) or longer (TTL varies).

Best candidates: large system prompts, long reference documents, few-shot examples. Not worth caching for small prompts or single-use requests.

```python
{"role": "user", "content": [
    {"type": "text", "text": very_long_doc, "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "Summarize the above."},
]}
```

---

## Error Handling and Retry Logic

| HTTP Status | Meaning | Action |
|---|---|---|
| 400 | Bad request (invalid params, content policy) | Fix the request; do not retry |
| 401 | Invalid API key | Check key; do not retry |
| 429 | Rate limit exceeded | Exponential backoff with jitter |
| 500 | Server error | Retry with backoff (up to 3×) |
| 529 | Overloaded (Anthropic-specific) | Retry with backoff |

```python
import time, random

def call_with_retry(fn, max_retries=3):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "500" in str(e) or "529" in str(e):
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait:.1f}s (attempt {attempt + 1})")
                time.sleep(wait)
            else:
                raise  # Don't retry 400/401
    raise RuntimeError("Max retries exceeded")
```

Always set `max_tokens` explicitly — the default is low and will truncate long responses mid-sentence.
