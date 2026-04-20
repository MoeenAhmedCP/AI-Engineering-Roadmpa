"""
3.5 OpenAI & Anthropic APIs — Examples

Set SIMULATE=True (default) to run without API keys.
Set SIMULATE=False with ANTHROPIC_API_KEY / OPENAI_API_KEY for real calls.

Run: python examples.py
"""

import os
import json
import time
import random

SIMULATE = os.getenv("SIMULATE", "true").lower() != "false"

# ---------------------------------------------------------------------------
# Mock clients (simulate API behavior without network calls)
# ---------------------------------------------------------------------------

class _MockMessage:
    """Mimics anthropic.types.Message"""
    def __init__(self, text: str, input_tokens: int = 50, output_tokens: int = 80):
        self.content = [type("Block", (), {"type": "text", "text": text})()]
        self.usage = type("Usage", (), {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })()
        self.stop_reason = "end_turn"
        self.model = "claude-sonnet-4-6-simulated"


class _MockMessages:
    def create(self, model="", max_tokens=256, messages=None, system="", tools=None, **kwargs):
        last_user = ""
        if messages:
            for m in reversed(messages):
                if m.get("role") == "user":
                    content = m.get("content", "")
                    if isinstance(content, str):
                        last_user = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                last_user = block.get("text", "")
                    break

        # Simulate tool use if tools are provided and query matches
        if tools and "weather" in last_user.lower():
            mock_tool = type("ToolUse", (), {
                "type": "tool_use",
                "id": "mock_tool_001",
                "name": "get_weather",
                "input": {"location": "San Francisco", "unit": "celsius"},
            })()
            msg = _MockMessage("")
            msg.content = [mock_tool]
            msg.stop_reason = "tool_use"
            return msg

        response_text = (
            f"[SIMULATED] I understand your question about: '{last_user[:60]}'. "
            "In a real API call, Claude would provide a thoughtful, detailed response here."
        )
        return _MockMessage(response_text)

    def stream(self, **kwargs):
        """Context manager that yields fake streaming chunks."""
        class _MockStream:
            def __init__(self, text):
                self.text_stream = iter(text.split())
                self._words = text.split()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        words = "[SIMULATED-STREAM] This is a streaming response token by token.".split()
        return _FakeStreamContext(words)


class _FakeStreamContext:
    def __init__(self, words):
        self._words = words

    def __enter__(self):
        self.text_stream = (w + " " for w in self._words)
        return self

    def __exit__(self, *args):
        pass


class MockAnthropicClient:
    """Mimics the anthropic.Anthropic client interface."""

    def __init__(self, api_key: str = "mock"):
        self.messages = _MockMessages()

    def count_tokens_estimate(self, text: str) -> int:
        return len(text) // 4


class _MockChoice:
    def __init__(self, text):
        self.message = type("Msg", (), {"content": text, "role": "assistant"})()
        self.delta = type("Delta", (), {"content": text})()


class _MockCompletion:
    def __init__(self, text):
        self.choices = [_MockChoice(text)]
        self.usage = type("Usage", (), {"prompt_tokens": 40, "completion_tokens": 60})()


class MockOpenAIClient:
    """Mimics the openai.OpenAI client interface."""

    def __init__(self, api_key: str = "mock"):
        self.chat = type("Chat", (), {"completions": _MockCompletions()})()


class _MockCompletions:
    def create(self, model="", messages=None, stream=False, **kwargs):
        last_user = ""
        if messages:
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = m.get("content", "")[:60]
                    break
        text = (
            f"[SIMULATED-OpenAI] Response to: '{last_user}'. "
            "Real GPT-4o would answer here."
        )
        if stream:
            return _mock_stream_generator(text)
        return _MockCompletion(text)


def _mock_stream_generator(text: str):
    """Yields fake OpenAI streaming chunks."""
    for word in text.split():
        chunk = type("Chunk", (), {
            "choices": [type("Choice", (), {
                "delta": type("Delta", (), {"content": word + " "})()
            })()]
        })()
        yield chunk


# ---------------------------------------------------------------------------
# Real clients (loaded only when SIMULATE=False)
# ---------------------------------------------------------------------------

def _get_anthropic_client():
    if SIMULATE:
        return MockAnthropicClient()
    try:
        import anthropic  # type: ignore
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except ImportError:
        print("[WARN] anthropic not installed; using mock.")
        return MockAnthropicClient()
    except KeyError:
        print("[WARN] ANTHROPIC_API_KEY not set; using mock.")
        return MockAnthropicClient()


def _get_openai_client():
    if SIMULATE:
        return MockOpenAIClient()
    try:
        import openai  # type: ignore
        return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    except ImportError:
        print("[WARN] openai not installed; using mock.")
        return MockOpenAIClient()
    except KeyError:
        print("[WARN] OPENAI_API_KEY not set; using mock.")
        return MockOpenAIClient()


# ---------------------------------------------------------------------------
# 1. Basic chat completion
# ---------------------------------------------------------------------------

def chat_completion(
    messages: list[dict],
    model: str = "claude-sonnet-4-6",
    simulate: bool = SIMULATE,
) -> str:
    """
    Send a messages array to Claude and return the text response.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str}
        model:    Model identifier.
        simulate: If True, uses MockAnthropicClient.

    Returns:
        Response text string.
    """
    client = MockAnthropicClient() if simulate else _get_anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=messages,
    )
    return response.content[0].text


def demo_chat_completion():
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Chat Completion")
    print("=" * 60)

    messages = [
        {"role": "user", "content": "What are the three pillars of RAG systems?"},
    ]
    answer = chat_completion(messages)
    print(f"User:      {messages[0]['content']}")
    print(f"Assistant: {answer}")

    # Multi-turn conversation
    messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": "Can you expand on the retrieval step?"})
    follow_up = chat_completion(messages)
    print(f"\nFollow-up: {messages[-1]['content']}")
    print(f"Assistant: {follow_up}")


# ---------------------------------------------------------------------------
# 2. Streaming
# ---------------------------------------------------------------------------

def streaming_chat(
    messages: list[dict],
    model: str = "claude-sonnet-4-6",
    simulate: bool = SIMULATE,
):
    """
    Generator that yields text chunks from a streaming API call.
    Prints tokens as they arrive (simulating real-time output).
    """
    client = MockAnthropicClient() if simulate else _get_anthropic_client()

    if simulate:
        # Simulate streaming by yielding words with a tiny delay
        full = (
            "[SIMULATED STREAM] Streaming responses improve user experience because "
            "the user sees tokens appear immediately rather than waiting for the full response. "
            "Each word here represents one chunk arriving from the API."
        )
        for word in full.split():
            yield word + " "
    else:
        with client.messages.stream(
            model=model,
            max_tokens=256,
            messages=messages,
        ) as stream:
            yield from stream.text_stream


def demo_streaming():
    print("\n" + "=" * 60)
    print("DEMO 2: Streaming Response")
    print("=" * 60)

    messages = [{"role": "user", "content": "Explain streaming APIs in two sentences."}]
    print("Streaming output: ", end="")
    for chunk in streaming_chat(messages):
        print(chunk, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 3. Tool use
# ---------------------------------------------------------------------------

def _execute_get_weather(location: str, unit: str = "celsius") -> dict:
    """Mock weather function — would call a real weather API in production."""
    return {
        "location": location,
        "temperature": 18 if unit == "celsius" else 64,
        "unit": unit,
        "condition": "partly cloudy",
        "humidity": "72%",
    }


def tool_use_demo(simulate: bool = SIMULATE):
    """
    Demonstrates the full tool use loop:
    1. Send user message + tool definitions to model
    2. Model returns tool_use block
    3. Execute the function locally
    4. Send result back to model
    5. Model responds with final answer
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Tool Use (Function Calling)")
    print("=" * 60)

    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a specific location.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location string",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        }
    ]

    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    client = MockAnthropicClient() if simulate else _get_anthropic_client()

    print(f"User: {messages[0]['content']}")

    # Step 1: Call model with tools
    if simulate:
        # Simulate tool_use response
        tool_call = {
            "type": "tool_use",
            "id": "mock_tool_001",
            "name": "get_weather",
            "input": {"location": "San Francisco", "unit": "celsius"},
        }
        print(f"\n[Model requests tool]: {tool_call['name']}({tool_call['input']})")
        stop_reason = "tool_use"
    else:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            tools=tools,
            messages=messages,
        )
        stop_reason = response.stop_reason
        if stop_reason == "tool_use":
            tool_block = next(b for b in response.content if b.type == "tool_use")
            tool_call = {"type": "tool_use", "id": tool_block.id, "name": tool_block.name, "input": tool_block.input}
            print(f"\n[Model requests tool]: {tool_call['name']}({tool_call['input']})")

    # Step 2: Execute the tool
    if stop_reason == "tool_use":
        result = _execute_get_weather(**tool_call["input"])
        print(f"[Tool result]: {result}")

        # Step 3: Send result back to model
        messages.append({"role": "assistant", "content": [tool_call]})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call["id"],
                "content": json.dumps(result),
            }]
        })

        final = chat_completion(messages, simulate=simulate)
        print(f"\n[Final answer]: {final}")
    else:
        print("[Model answered directly without tool use]")


# ---------------------------------------------------------------------------
# 4. Token counting and cost estimation
# ---------------------------------------------------------------------------

def count_tokens_estimate(text: str) -> int:
    """Quick token count approximation: ~4 chars per token."""
    return max(1, len(text) // 4)


# Prices per million tokens (as of early 2025)
MODEL_PRICES = {
    "claude-haiku-3-5":   {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":  {"input": 3.00,  "output": 15.00},
    "claude-opus-4":      {"input": 15.00, "output": 75.00},
    "gpt-4o-mini":        {"input": 0.15,  "output": 0.60},
    "gpt-4o":             {"input": 2.50,  "output": 10.00},
    "o1":                 {"input": 15.00, "output": 60.00},
}


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-sonnet-4-6",
) -> float:
    """
    Estimate API call cost in USD.

    Returns:
        Cost in dollars (e.g., 0.00045)
    """
    prices = MODEL_PRICES.get(model, MODEL_PRICES["claude-sonnet-4-6"])
    cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
    return round(cost, 6)


def demo_token_counting():
    print("\n" + "=" * 60)
    print("DEMO 4: Token Counting and Cost Estimation")
    print("=" * 60)

    sample_text = (
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "information retrieval with language model generation to produce more "
        "accurate and contextually relevant responses."
    )

    token_count = count_tokens_estimate(sample_text)
    print(f"Text:         {sample_text[:60]}...")
    print(f"Char count:   {len(sample_text)}")
    print(f"Token est.:   {token_count}")

    print("\nCost estimates for a 500 input / 200 output token call:")
    print(f"{'Model':<22} {'Input $/M':<12} {'Output $/M':<12} {'Call cost'}")
    print("-" * 55)
    for model, prices in MODEL_PRICES.items():
        cost = estimate_cost(500, 200, model)
        print(f"{model:<22} ${prices['input']:<11.2f} ${prices['output']:<11.2f} ${cost:.6f}")


# ---------------------------------------------------------------------------
# 5. Structured output extraction
# ---------------------------------------------------------------------------

def structured_output_demo(simulate: bool = SIMULATE):
    """
    Extract structured data from unstructured text.
    Returns a dict with {name, email, company}.
    """
    print("\n" + "=" * 60)
    print("DEMO 5: Structured Output Extraction")
    print("=" * 60)

    paragraph = (
        "Hi, I'm Sarah Chen and I work as a senior engineer at Vertex Systems. "
        "The best way to reach me is at sarah.chen@vertexsystems.io — "
        "I check my email every morning."
    )

    schema = {"name": "string", "email": "string", "company": "string"}

    system_prompt = (
        "You are a data extraction assistant. Extract the requested fields from the "
        "user's text and respond ONLY with a valid JSON object matching this schema: "
        f"{json.dumps(schema)}. No explanation, no markdown, just the JSON object."
    )

    messages = [{"role": "user", "content": paragraph}]

    print(f"Input text: {paragraph}")
    print(f"Target schema: {schema}")

    if simulate:
        # Deterministic mock extraction
        result = {"name": "Sarah Chen", "email": "sarah.chen@vertexsystems.io", "company": "Vertex Systems"}
        result_str = json.dumps(result, indent=2)
        print(f"\n[SIMULATED] Extracted JSON:\n{result_str}")
    else:
        client = _get_anthropic_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=system_prompt,
            messages=messages,
        )
        raw = response.content[0].text.strip()
        try:
            result = json.loads(raw)
            print(f"\nExtracted JSON:\n{json.dumps(result, indent=2)}")
        except json.JSONDecodeError:
            print(f"\n[WARN] Model returned non-JSON: {raw}")
            result = {}

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("3.5 OpenAI & Anthropic APIs — Examples")
    print(f"Mode: {'SIMULATE (no API keys needed)' if SIMULATE else 'REAL API calls'}")
    print("=" * 60)

    demo_chat_completion()
    demo_streaming()
    tool_use_demo()
    demo_token_counting()
    structured_output_demo()

    print("\n" + "=" * 60)
    print("All demos complete.")
    if SIMULATE:
        print("To use real APIs:")
        print("  SIMULATE=false ANTHROPIC_API_KEY=sk-ant-... python examples.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
