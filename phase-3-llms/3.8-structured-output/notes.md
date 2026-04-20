# 3.8 Structured Output

## The Problem: Unstructured LLM Output Breaks Pipelines

LLMs are trained to produce human-readable text. When you ask a model to "return JSON," it does its best — but it is still generating tokens probabilistically, and nothing in the base generation process enforces that the output will be valid JSON, let alone valid for your specific schema.

The failure modes are painful:

- **Markdown code fences**: The model wraps JSON in ` ```json ... ``` `, so `json.loads()` raises an exception.
- **Trailing commas**: Valid in JavaScript, invalid in JSON — the model learned from JS code too.
- **Missing or extra keys**: The model adds helpful commentary fields you didn't ask for, or drops optional ones.
- **Wrong types**: You asked for an integer count, the model returns the string `"42"`.
- **Truncation**: Long responses get cut off mid-JSON, producing an unclosed object.
- **Prose mixed in**: "Here is the JSON you requested: {...} Let me know if you need anything else!"

The instinct to handle this with regex is understandable but dangerous. Regex-based JSON extraction (`re.search(r'\{.*\}', response, re.DOTALL)`) fails on nested objects, breaks on escaped quotes, and creates a maintenance nightmare as prompt wording evolves.

---

## JSON Mode: A Floor, Not a Ceiling

Major providers added "JSON mode" (OpenAI `response_format={"type": "json_object"}`, Anthropic using prompt prefilling with `{`). This guarantees:

- The output is **syntactically valid JSON** — `json.loads()` will not raise.

What it does NOT guarantee:

- That the JSON matches your schema.
- That the keys you need are present.
- That values have the right types.

JSON mode eliminates parse errors but still requires you to validate the structure manually. You've moved the problem from "can I parse this?" to "does this have what I need?"

---

## Structured Outputs: Constrained Decoding

OpenAI's Structured Outputs feature (and similar approaches in other runtimes) solves the schema problem by changing how generation works. Instead of sampling freely from the vocabulary at each token position, the runtime uses your JSON Schema to **constrain which tokens are valid** at each step.

If your schema says `"status"` must be one of `["active", "inactive", "pending"]`, then after generating `"status": "` the model can only emit `active`, `inactive`, or `pending` — no other token sequence is possible.

This means:
- The output is guaranteed schema-valid (required fields present, types correct, enum values respected).
- You pay no inference speed penalty in practice.
- Nullable/Optional fields are handled by including `null` in the union type.

The tradeoff: constrained decoding requires the provider to compile a finite-state machine from your schema. Very deep recursive schemas may not be supported. And this is a provider-side feature — not available with all models or self-hosted setups.

---

## Pydantic: Schema as Code

Pydantic is a Python library that lets you define data schemas as classes. It is the standard way to declare what you expect from an LLM.

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class ActionItem(BaseModel):
    owner: str = Field(description="Person responsible")
    task: str = Field(description="What needs to be done")
    deadline: Optional[str] = Field(None, description="ISO date or None")

class MeetingNotes(BaseModel):
    title: str
    date: str
    attendees: List[str]
    action_items: List[ActionItem]
    summary: Optional[str] = None
```

Key Pydantic concepts:

- **`Field()`**: Adds metadata (description, default, constraints like `min_length`, `ge`/`le` for numbers). Descriptions flow into the JSON Schema and help the model understand intent.
- **`Optional[T]`**: Equivalent to `Union[T, None]`. Field can be absent or null.
- **Validators**: `@field_validator('email')` lets you run custom logic (e.g., check email format).
- **Nested models**: A field whose type is another `BaseModel` creates nested JSON objects.
- **`model_json_schema()`**: Produces the JSON Schema dict you can pass to APIs or include in prompts.

Pydantic v2 (the current version) is significantly faster than v1 and uses `model_validate()` instead of `parse_obj()`.

---

## The Instructor Library

Instructor wraps OpenAI and Anthropic clients to make structured extraction ergonomic:

```python
import instructor
from anthropic import Anthropic

client = instructor.from_anthropic(Anthropic())

result = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Extract: " + text}],
    response_model=ContactInfo,  # Pydantic model here
)
# result is a ContactInfo instance — no JSON parsing needed
```

What Instructor does internally:

1. Calls `model_json_schema()` on your Pydantic model and injects it into the prompt.
2. Sets JSON mode (or uses tool calling, which also enforces structure).
3. Calls `model_validate()` on the response.
4. If validation fails, it **automatically retries** — sending the validation error back to the model as context ("You returned X, but field Y must be a string, not an integer. Try again.").

The retry loop is configurable (`max_retries=3`). Instructor tracks usage across retries so you can monitor cost accurately.

---

## Validation Errors and the Retry Pattern

When a model returns output that fails Pydantic validation, you have two choices:

1. **Raise the error**: Fail fast, surface the issue to the caller.
2. **Retry with feedback**: Send the error message back to the model: "Your previous response had this validation error: [error]. Please return corrected JSON."

The retry-with-feedback pattern is powerful because the error message itself is often enough context for the model to self-correct. In practice, 95%+ of failures are corrected in one retry.

The failure budget matters: if a model consistently fails validation on a particular schema, that's a signal to simplify the schema, add more examples to the prompt, or switch to a more capable model.

---

## Nested Schemas and Discriminated Unions

Nested models are handled naturally:

```python
class Report(BaseModel):
    sections: List[Section]   # Section is another BaseModel
    metadata: DocumentMeta
```

Discriminated unions let you handle polymorphic output:

```python
from typing import Literal, Union

class EmailContact(BaseModel):
    type: Literal["email"]
    address: str

class PhoneContact(BaseModel):
    type: Literal["phone"]
    number: str

Contact = Union[EmailContact, PhoneContact]
```

Pydantic uses the `type` field as the discriminator to choose which model to validate against.

---

## Prompt Strategies for Structured Output

Even with constrained decoding or Instructor, prompts matter:

1. **Include the schema**: Paste the JSON Schema or a Pydantic-style pseudocode block into the system prompt. Models trained on code recognize schema notation.
2. **Provide a filled example**: Show one complete example of valid output. Few-shot beats description for format tasks.
3. **Name fields descriptively**: `"confidence_score_0_to_100"` is clearer than `"score"`.
4. **Instruct about missing data**: "If a field cannot be determined from the text, use null. Do not guess."
5. **Keep schemas as flat as possible**: Every nesting level increases the chance of structural error.

---

## Scale Patterns: 1000 Documents

When extracting structured data at scale:

- **Batching**: Group documents into batches of 10-50. Use `asyncio.gather()` for concurrent extraction. The Batch API (OpenAI/Anthropic) gives 50% cost reduction for async workloads.
- **Error handling**: Track failures per document. Write successful extractions to DB immediately. Re-queue failures with exponential backoff.
- **Schema versioning**: Add a `schema_version` field to your output. When you update the schema, old records stay parseable.
- **Partial extraction**: If a document yields a partial result (some required fields present), decide upfront whether to store partial or discard.
- **Cost estimation**: `len(text.split()) * 1.3` gives a rough token estimate. Multiply by price before submitting large batches.
