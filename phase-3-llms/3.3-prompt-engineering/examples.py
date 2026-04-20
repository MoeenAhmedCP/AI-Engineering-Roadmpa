"""
3.3 Prompt Engineering — Examples
No API calls at top level — uses deterministic mock responses.
Set SIMULATE=False and provide ANTHROPIC_API_KEY to use real Claude.
Run: python examples.py
"""

import hashlib
import re
import string
from collections import defaultdict

SIMULATE = True   # Set False + set ANTHROPIC_API_KEY env var for live calls


# ---------------------------------------------------------------------------
# 1. PromptTemplate
# ---------------------------------------------------------------------------

class PromptTemplate:
    """
    Simple string template with {variable} placeholders.

    Usage:
        t = PromptTemplate("Summarise the following in {style}: {text}")
        t.validate_vars(style="bullet points", text="...")
        prompt = t.format(style="bullet points", text="...")
    """

    def __init__(self, template: str):
        self.template = template
        # Extract variable names from {placeholders}
        self._vars = set(re.findall(r"\{(\w+)\}", template))

    def format(self, **kwargs) -> str:
        """
        Fill in the template. Raises ValueError for missing variables.
        Extra kwargs are silently ignored.
        """
        missing = self._vars - set(kwargs)
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.template.format(**{k: kwargs[k] for k in self._vars})

    def validate_vars(self, **kwargs):
        """Check all required variables are present. Raises ValueError if not."""
        missing = self._vars - set(kwargs)
        extra   = set(kwargs) - self._vars
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        if extra:
            print(f"  [PromptTemplate] Warning: unused variables: {extra}")
        return True

    def __repr__(self):
        return f"PromptTemplate(vars={sorted(self._vars)})"


# ---------------------------------------------------------------------------
# 2. Few-shot prompt builder
# ---------------------------------------------------------------------------

def few_shot_prompt(task: str, examples: list, query: str) -> str:
    """
    Build a few-shot prompt string.

    Args:
        task:     Task description / instruction.
        examples: List of dicts with "input" and "output" keys.
        query:    The actual input to solve.

    Returns:
        A formatted prompt string.
    """
    lines = [task, ""]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Input: {ex['input']}")
        lines.append(f"Output: {ex['output']}")
        lines.append("")
    lines.append("Now solve this:")
    lines.append(f"Input: {query}")
    lines.append("Output:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Chain-of-thought prompt
# ---------------------------------------------------------------------------

def chain_of_thought_prompt(question: str) -> str:
    """
    Wrap a question with a chain-of-thought instruction.
    Encourages the model to reason step-by-step before answering.
    """
    return (
        "Answer the following question by thinking step-by-step. "
        "Show your reasoning before giving the final answer.\n\n"
        f"Question: {question}\n\n"
        "Let's think step by step:\n"
    )


# ---------------------------------------------------------------------------
# 4. System prompt builder
# ---------------------------------------------------------------------------

def system_prompt_builder(
    persona: str,
    rules: list,
    output_format: str,
) -> str:
    """
    Construct a structured system prompt.

    Args:
        persona:       Who the assistant is (role + style).
        rules:         List of behavioural constraints.
        output_format: Expected output structure.
    """
    rules_block = "\n".join(f"- {r}" for r in rules)
    return (
        f"## Persona\n{persona}\n\n"
        f"## Rules\n{rules_block}\n\n"
        f"## Output Format\n{output_format}"
    )


# ---------------------------------------------------------------------------
# 5. Mock LLM (deterministic, no API calls)
# ---------------------------------------------------------------------------

_MOCK_RESPONSES = [
    "The answer is clearly yes, based on the available evidence.",
    "I would recommend starting with the simplest approach first.",
    "There are three key considerations: clarity, consistency, and conciseness.",
    "This is a nuanced topic with multiple valid perspectives.",
    "Based on the examples provided, the pattern suggests an affirmative outcome.",
    "The data supports the hypothesis, though further validation is recommended.",
    "In summary: keep it simple, test early, iterate often.",
    "A step-by-step breakdown reveals the solution is straightforward.",
]


def mock_llm(prompt: str, temperature: float = 0.7) -> str:
    """
    Deterministic mock LLM. Returns a response based on a hash of the prompt.
    Same prompt always returns the same response — useful for testing.
    """
    prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    idx = prompt_hash % len(_MOCK_RESPONSES)
    response = _MOCK_RESPONSES[idx]

    # Simulate temperature effect: high temp -> append a caveat
    if temperature > 0.8:
        response += " (Note: high temperature may introduce variability.)"
    return response


def real_llm(prompt: str, temperature: float = 0.7) -> str:
    """
    Real Claude API call. Only used when SIMULATE=False.
    Requires: pip install anthropic; ANTHROPIC_API_KEY in environment.
    """
    import os
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def llm(prompt: str, temperature: float = 0.7) -> str:
    """Route to mock or real LLM based on SIMULATE flag."""
    if SIMULATE:
        return mock_llm(prompt, temperature)
    return real_llm(prompt, temperature)


# ---------------------------------------------------------------------------
# 6. Prompt A/B test
# ---------------------------------------------------------------------------

def prompt_ab_test(
    prompt_variants: list,
    test_inputs: list,
    judge_fn,
) -> dict:
    """
    Run A/B test across prompt variants.

    Args:
        prompt_variants: List of (name, template_str) tuples.
                         Template may contain {input} placeholder.
        test_inputs:     List of input strings to test each variant on.
        judge_fn:        callable(response: str) -> float  (0.0 to 1.0 score)

    Returns:
        dict with keys: "ranking" (list of dicts sorted by score),
                        "raw" (all individual scores per variant per input)
    """
    results = {}

    for name, template in prompt_variants:
        scores = []
        for inp in test_inputs:
            prompt = template.replace("{input}", inp)
            response = llm(prompt)
            score = judge_fn(response)
            scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0.0
        results[name] = {"avg_score": round(avg, 4), "scores": scores}

    ranking = sorted(results.items(), key=lambda x: -x[1]["avg_score"])
    return {
        "ranking": [{"name": k, **v} for k, v in ranking],
        "raw": results,
    }


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def _demo_prompt_template():
    print("=" * 55)
    print("1. PromptTemplate")
    print("=" * 55)

    t = PromptTemplate(
        "You are a {role}. Answer the following question in {style}:\n\n{question}"
    )
    print(f"  Template vars: {sorted(t._vars)}")

    prompt = t.format(
        role="senior data scientist",
        style="plain English with bullet points",
        question="What is gradient descent?",
    )
    print(f"\n  Formatted prompt:\n  ---\n{_indent(prompt)}\n  ---")

    # Validate
    try:
        t.validate_vars(role="assistant")   # missing style and question
    except ValueError as e:
        print(f"\n  Missing var error caught: {e}")


def _demo_few_shot():
    print("\n" + "=" * 55)
    print("2. few_shot_prompt")
    print("=" * 55)

    examples = [
        {"input": "The movie was fantastic!",     "output": "positive"},
        {"input": "Terrible experience, awful.",  "output": "negative"},
        {"input": "It was okay, nothing special.", "output": "neutral"},
    ]

    prompt = few_shot_prompt(
        task="Classify the sentiment of each text as positive, negative, or neutral.",
        examples=examples,
        query="I absolutely loved the ending, it was breathtaking.",
    )
    print(_indent(prompt))
    print(f"\n  Mock LLM response: {mock_llm(prompt)!r}")


def _demo_cot():
    print("\n" + "=" * 55)
    print("3. chain_of_thought_prompt")
    print("=" * 55)

    prompt = chain_of_thought_prompt(
        "If a train travels 120 km in 1.5 hours, what is its average speed in km/h?"
    )
    print(_indent(prompt))
    print(f"\n  Mock LLM response: {mock_llm(prompt)!r}")


def _demo_system_prompt():
    print("\n" + "=" * 55)
    print("4. system_prompt_builder")
    print("=" * 55)

    sys_prompt = system_prompt_builder(
        persona=(
            "You are DocSense Assistant, an expert at analysing business documents. "
            "You are concise, professional, and cite specific sections."
        ),
        rules=[
            "Never fabricate facts not present in the document.",
            "Always highlight the top 3 risks explicitly.",
            "Respond in the user's language.",
            "If the document is unclear, ask for clarification rather than guessing.",
        ],
        output_format=(
            '{"summary": "...", "risks": [...], "action_items": [...], '
            '"confidence": 0.0..1.0}'
        ),
    )
    print(_indent(sys_prompt))


def _demo_ab_test():
    print("\n" + "=" * 55)
    print("5. prompt_ab_test — 3 variants x 5 inputs")
    print("=" * 55)

    variants = [
        (
            "direct",
            "Answer this question directly and concisely: {input}",
        ),
        (
            "cot",
            "Think step-by-step, then answer: {input}",
        ),
        (
            "expert",
            "You are a world-class expert. Provide a detailed answer to: {input}",
        ),
    ]

    test_inputs = [
        "What is the capital of France?",
        "Explain backpropagation in one sentence.",
        "What causes transformer attention to scale quadratically?",
        "Name three benefits of experiment tracking.",
        "When should you use AdamW instead of Adam?",
    ]

    # Judge: longer responses score slightly higher (as a simple proxy)
    def length_judge(response: str) -> float:
        words = len(response.split())
        return min(1.0, words / 30.0)

    result = prompt_ab_test(variants, test_inputs, judge_fn=length_judge)

    print(f"\n  {'Rank':<5} {'Variant':<12} {'Avg Score':<12} {'Individual Scores'}")
    print("  " + "-" * 55)
    for rank, entry in enumerate(result["ranking"], 1):
        scores_str = "  ".join(f"{s:.2f}" for s in entry["scores"])
        print(f"  {rank:<5} {entry['name']:<12} {entry['avg_score']:<12} {scores_str}")

    winner = result["ranking"][0]
    print(f"\n  Winner: '{winner['name']}' with avg score {winner['avg_score']}")


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


if __name__ == "__main__":
    print("=" * 55)
    print("3.3 Prompt Engineering — Examples")
    print(f"Mode: {'SIMULATE (mock LLM)' if SIMULATE else 'LIVE (Claude API)'}")
    print("=" * 55)

    _demo_prompt_template()
    _demo_few_shot()
    _demo_cot()
    _demo_system_prompt()
    _demo_ab_test()

    print("\nDone.")
