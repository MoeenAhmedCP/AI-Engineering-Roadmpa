"""
3.3 Prompt Engineering — Exercises
Attempt each before reading solutions.
No API calls required — all exercises work with strings and mock responses.
Run: python exercises.py
"""

import hashlib
import re


# ---------------------------------------------------------------------------
# EXERCISE 1 — Write a system prompt for a customer support bot
# Return as a plain string. No API call needed.
# The bot should: be helpful and empathetic, never promise refunds without
# authorization, always ask for an order number when relevant, stay on-topic.
# ---------------------------------------------------------------------------

def customer_support_system_prompt() -> str:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 2 — Build a function that adds few-shot examples to any prompt
# examples: list of dicts {"input": str, "output": str}
# base_prompt: the task instruction
# query: the new input to answer
# Return: complete prompt string
# ---------------------------------------------------------------------------

def add_few_shot_examples(base_prompt: str, examples: list, query: str) -> str:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 3 — Chain-of-thought wrapper
# Takes any question string, wraps it with CoT instruction.
# Should encourage the model to reason before giving a final answer.
# ---------------------------------------------------------------------------

def chain_of_thought_wrapper(question: str) -> str:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 4 — Force JSON output
# Add a JSON instruction and a schema example to a prompt.
# json_schema: dict describing the expected keys/types (as a comment/example)
# Returns the modified prompt string.
# ---------------------------------------------------------------------------

def force_json_output(prompt: str, json_schema: dict) -> str:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 5 — PromptVersionRegistry
# Store named prompt versions, retrieve by name, diff two versions.
# ---------------------------------------------------------------------------

class PromptVersionRegistry:
    def save(self, name: str, prompt: str, version: str = "1.0"):
        raise NotImplementedError

    def get(self, name: str, version: str = None) -> str:
        """Return the prompt for the given name (latest version if version=None)."""
        raise NotImplementedError

    def list_versions(self, name: str) -> list:
        """Return list of version strings for a given prompt name."""
        raise NotImplementedError

    def diff(self, name: str, version_a: str, version_b: str) -> str:
        """
        Return a simple line-level diff between two versions.
        Lines only in A are prefixed with '-', only in B with '+'.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 6 — Prompt testing harness
# variants:   list of (name: str, prompt_template: str) — template has {input}
# test_cases: list of input strings
# scorer:     callable(response: str) -> float
# Returns: dict with ranking list sorted by avg score
# ---------------------------------------------------------------------------

def prompt_test_harness(
    variants: list,
    test_cases: list,
    scorer,
    llm_fn=None,
) -> dict:
    raise NotImplementedError


# ===========================================================================
# SOLUTIONS
# ===========================================================================

def _sol_customer_support_system_prompt() -> str:
    return """## Role
You are Aria, a friendly and empathetic customer support specialist for ShopEasy, \
an online retail platform.

## Behaviour
- Greet customers warmly and acknowledge their concern before troubleshooting.
- Always ask for the customer's order number when the issue relates to a specific order.
- If you cannot resolve the issue yourself, escalate politely: \
"Let me connect you with a specialist who can help further."
- Never promise a refund, replacement, or credit without explicit authorisation \
from a supervisor or the system.
- Keep responses focused on ShopEasy products and services. \
Politely decline off-topic requests.

## Tone
Professional, warm, concise. Avoid jargon. Use plain language.

## Output Format
Respond in plain prose. If listing steps, use a numbered list. \
Do not output JSON or markdown headers."""


def _sol_add_few_shot_examples(base_prompt: str, examples: list, query: str) -> str:
    lines = [base_prompt, ""]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Input: {ex['input']}")
        lines.append(f"Output: {ex['output']}")
        lines.append("")
    lines.append("Now answer this:")
    lines.append(f"Input: {query}")
    lines.append("Output:")
    return "\n".join(lines)


def _sol_chain_of_thought_wrapper(question: str) -> str:
    return (
        "Think through the following question step-by-step. "
        "Write out each reasoning step explicitly before giving your final answer.\n\n"
        f"Question: {question}\n\n"
        "Step-by-step reasoning:\n"
        "1."
    )


def _sol_force_json_output(prompt: str, json_schema: dict) -> str:
    import json
    schema_str = json.dumps(json_schema, indent=2)
    json_instruction = (
        "\n\nIMPORTANT: You MUST respond with valid JSON only. "
        "Do not include any text outside the JSON object. "
        "Do not wrap the JSON in markdown code fences.\n\n"
        "Your response must match this schema:\n"
        f"{schema_str}"
    )
    return prompt + json_instruction


class _SolPromptVersionRegistry:
    def __init__(self):
        # {name: {version: prompt_str}}
        self._store: dict = {}

    def save(self, name: str, prompt: str, version: str = "1.0"):
        if name not in self._store:
            self._store[name] = {}
        self._store[name][version] = prompt

    def get(self, name: str, version: str = None) -> str:
        if name not in self._store:
            raise KeyError(f"No prompt named {name!r}")
        versions = self._store[name]
        if version is None:
            # Return latest by string sort (works for "1.0", "1.1", "2.0" etc.)
            version = sorted(versions.keys())[-1]
        if version not in versions:
            raise KeyError(f"Version {version!r} not found for {name!r}")
        return versions[version]

    def list_versions(self, name: str) -> list:
        if name not in self._store:
            return []
        return sorted(self._store[name].keys())

    def diff(self, name: str, version_a: str, version_b: str) -> str:
        text_a = self.get(name, version_a).splitlines()
        text_b = self.get(name, version_b).splitlines()
        lines = []
        set_a, set_b = set(text_a), set(text_b)
        for line in text_a:
            if line not in set_b:
                lines.append(f"- {line}")
        for line in text_b:
            if line not in set_a:
                lines.append(f"+ {line}")
        return "\n".join(lines) if lines else "(no differences)"


# Mock LLM for the harness
def _mock_llm(prompt: str) -> str:
    responses = [
        "The solution involves careful analysis of all available data.",
        "I recommend a structured, step-by-step approach to resolve this.",
        "Based on the context provided, the answer is straightforward.",
        "There are multiple valid perspectives worth considering here.",
        "The key insight is to focus on the most impactful variable first.",
    ]
    idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(responses)
    return responses[idx]


def _sol_prompt_test_harness(
    variants: list,
    test_cases: list,
    scorer,
    llm_fn=None,
) -> dict:
    if llm_fn is None:
        llm_fn = _mock_llm

    results = {}
    for name, template in variants:
        scores = []
        for inp in test_cases:
            prompt = template.replace("{input}", inp)
            response = llm_fn(prompt)
            score = scorer(response)
            scores.append(round(score, 4))
        avg = round(sum(scores) / len(scores), 4) if scores else 0.0
        results[name] = {"avg_score": avg, "scores": scores}

    ranking = sorted(results.items(), key=lambda x: -x[1]["avg_score"])
    return {
        "ranking": [{"name": k, **v} for k, v in ranking],
        "raw": results,
    }


# ---------------------------------------------------------------------------

def solutions():
    sep = "=" * 58
    print(sep)
    print("SOLUTIONS — 3.3 Prompt Engineering Exercises")
    print(sep)

    # --- Exercise 1 ---
    print("\n[Exercise 1] customer_support_system_prompt")
    prompt = _sol_customer_support_system_prompt()
    lines = prompt.splitlines()
    print(f"  Lines: {len(lines)}")
    print(f"  First line: {lines[0]!r}")
    print(f"  Last line : {lines[-1]!r}")
    print(f"  Contains order number rule: {'order number' in prompt.lower()}")
    print(f"  Contains refund restriction: {'refund' in prompt.lower()}")

    # --- Exercise 2 ---
    print("\n[Exercise 2] add_few_shot_examples")
    examples = [
        {"input": "sky",   "output": "blue"},
        {"input": "grass", "output": "green"},
    ]
    result = _sol_add_few_shot_examples(
        base_prompt="Respond with the typical color of the object.",
        examples=examples,
        query="fire engine",
    )
    print("  Generated prompt:")
    for line in result.splitlines():
        print(f"    {line}")

    # --- Exercise 3 ---
    print("\n[Exercise 3] chain_of_thought_wrapper")
    cot = _sol_chain_of_thought_wrapper(
        "A store sells 3 apples for $1. How much do 9 apples cost?"
    )
    print("  Wrapped prompt:")
    for line in cot.splitlines():
        print(f"    {line}")

    # --- Exercise 4 ---
    print("\n[Exercise 4] force_json_output")
    schema = {
        "sentiment": "positive | negative | neutral",
        "confidence": "float 0.0–1.0",
        "reason": "string",
    }
    base = "Classify the sentiment of this text: 'The product was amazing!'"
    json_prompt = _sol_force_json_output(base, schema)
    print(f"  Original length : {len(base)} chars")
    print(f"  With JSON instr : {len(json_prompt)} chars")
    print(f"  Contains 'valid JSON': {'valid JSON' in json_prompt}")
    print(f"  Contains schema keys: {all(k in json_prompt for k in schema)}")

    # --- Exercise 5 ---
    print("\n[Exercise 5] PromptVersionRegistry")
    reg = _SolPromptVersionRegistry()
    reg.save("support_bot", "You are a helpful assistant.", version="1.0")
    reg.save("support_bot", "You are a helpful and empathetic assistant.", version="1.1")
    reg.save("support_bot", "You are Aria, a helpful, empathetic support agent.", version="2.0")

    print(f"  Versions for 'support_bot': {reg.list_versions('support_bot')}")
    print(f"  Latest (v2.0): {reg.get('support_bot')!r}")
    print(f"  v1.0: {reg.get('support_bot', '1.0')!r}")
    print("\n  Diff v1.0 vs v2.0:")
    for line in reg.diff("support_bot", "1.0", "2.0").splitlines():
        print(f"    {line}")

    # --- Exercise 6 ---
    print("\n[Exercise 6] prompt_test_harness")
    variants = [
        ("minimal",  "Answer briefly: {input}"),
        ("detailed", "Provide a thorough, well-structured answer to: {input}"),
        ("expert",   "As a world-class expert, answer: {input}"),
    ]
    test_cases = [
        "What is overfitting in machine learning?",
        "Explain attention mechanisms.",
        "What is a vector database used for?",
        "How does fine-tuning differ from prompting?",
        "What is the purpose of a learning rate scheduler?",
    ]

    # Score by word count (longer = higher, capped at 1.0)
    def word_count_score(response: str) -> float:
        return min(1.0, len(response.split()) / 25.0)

    harness_result = _sol_prompt_test_harness(variants, test_cases, word_count_score)

    print(f"\n  {'Rank':<5} {'Variant':<12} {'Avg Score':<12} {'Per-input scores'}")
    print("  " + "-" * 58)
    for rank, entry in enumerate(harness_result["ranking"], 1):
        scores_str = "  ".join(f"{s:.2f}" for s in entry["scores"])
        print(f"  {rank:<5} {entry['name']:<12} {entry['avg_score']:<12} {scores_str}")

    winner = harness_result["ranking"][0]
    print(f"\n  Winner: '{winner['name']}' (avg score: {winner['avg_score']})")

    print(f"\n{sep}")
    print("All solutions complete.")
    print(sep)


if __name__ == "__main__":
    print("Attempt the exercises above, then call solutions() to compare.\n")
    solutions()
