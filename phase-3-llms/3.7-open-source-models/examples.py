"""
3.7 Open-Source Models — Examples

SIMULATE=True (default): runs without Ollama or transformers.
SIMULATE=False: tries Ollama first, then HuggingFace transformers.

Run: python examples.py
"""

import os
import time
import json
import statistics
from typing import Any

SIMULATE = os.getenv("SIMULATE", "true").lower() != "false"

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------

OLLAMA_SDK_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

if not SIMULATE:
    try:
        import urllib.request
        import urllib.error
        # We'll use urllib (stdlib) for Ollama HTTP calls — no extra package needed
        OLLAMA_SDK_AVAILABLE = True  # indicates we can try the HTTP API
    except ImportError:
        pass

    try:
        from transformers import pipeline  # type: ignore
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        print("[INFO] transformers not installed. pip install transformers torch")


# ---------------------------------------------------------------------------
# 1. Ollama client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Minimal Ollama client using only stdlib (urllib).
    Communicates with the local Ollama server at http://localhost:11434.
    """

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        """Check if the Ollama server is running and the model is available."""
        try:
            import urllib.request
            url = f"{self.base_url}/api/tags"
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                models = [m["name"].split(":")[0] for m in data.get("models", [])]
                return self.model.split(":")[0] in models
        except Exception:
            return False

    def chat(self, messages: list[dict]) -> str:
        """
        Send a messages list to Ollama and return the response text.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}

        Returns:
            Model response as a string.
        """
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return data["message"]["content"]


# ---------------------------------------------------------------------------
# 2. HuggingFace pipeline client
# ---------------------------------------------------------------------------

class HFPipelineClient:
    """
    Wraps a HuggingFace transformers pipeline.
    Downloads the model on first use (may be slow).
    """

    def __init__(self, task: str = "text-generation", model_name: str = "microsoft/phi-2"):
        self.task = task
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("transformers not installed. pip install transformers torch")
            from transformers import pipeline  # type: ignore
            print(f"[HF] Loading {self.model_name}... (first load may download model)")
            self._pipeline = pipeline(self.task, model=self.model_name)

    def run(self, text: str, max_new_tokens: int = 100) -> str:
        """
        Run the pipeline on input text.

        Returns:
            Generated or classified output as a string.
        """
        self._load()
        result = self._pipeline(text, max_new_tokens=max_new_tokens)

        if self.task == "text-generation":
            return result[0]["generated_text"]
        elif self.task == "summarization":
            return result[0]["summary_text"]
        elif self.task == "text-classification":
            return f"{result[0]['label']} (score: {result[0]['score']:.3f})"
        elif self.task == "zero-shot-classification":
            return result["labels"][0]
        else:
            return str(result)


# ---------------------------------------------------------------------------
# 3. Mock client (fallback for SIMULATE mode)
# ---------------------------------------------------------------------------

class MockLocalClient:
    """Simulates a local model response without any inference."""

    MODEL_RESPONSES = {
        "what is rag": (
            "RAG (Retrieval-Augmented Generation) combines a retrieval system with an LLM. "
            "The retriever finds relevant documents; the LLM synthesizes an answer from them."
        ),
        "explain transformers": (
            "Transformers use self-attention to process sequences in parallel. "
            "Each token attends to all other tokens, capturing long-range dependencies efficiently."
        ),
        "what is quantization": (
            "Quantization reduces model weight precision (e.g., from 16-bit to 4-bit floats), "
            "shrinking memory requirements by 4x with minimal quality loss."
        ),
    }

    def chat(self, messages: list[dict]) -> str:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "").lower()
                break
        for key, response in self.MODEL_RESPONSES.items():
            if any(word in last_user for word in key.split()):
                return f"[MOCK LOCAL MODEL] {response}"
        return (
            f"[MOCK LOCAL MODEL] I understand your question about '{last_user[:40]}'. "
            "In a real setup, Ollama would serve Llama 3 here and produce a contextual response."
        )


# ---------------------------------------------------------------------------
# 4. run_local_chat
# ---------------------------------------------------------------------------

def run_local_chat(prompt: str, simulate: bool = SIMULATE) -> str:
    """
    Send a prompt to a local model.
    Priority: Ollama (if available) → HuggingFace → Mock

    Args:
        prompt:   User message string.
        simulate: If True, always uses MockLocalClient.

    Returns:
        Model response string.
    """
    messages = [{"role": "user", "content": prompt}]

    if simulate:
        return MockLocalClient().chat(messages)

    # Try Ollama first
    ollama = OllamaClient(model="llama3")
    if ollama.is_available():
        print("[run_local_chat] Using Ollama (llama3)")
        return ollama.chat(messages)

    # Try HuggingFace
    if TRANSFORMERS_AVAILABLE:
        print("[run_local_chat] Ollama not available; using HuggingFace pipeline")
        client = HFPipelineClient(task="text-generation", model_name="microsoft/phi-2")
        return client.run(prompt)

    print("[run_local_chat] No local model available; using mock.")
    return MockLocalClient().chat(messages)


# ---------------------------------------------------------------------------
# 5. Latency benchmarking
# ---------------------------------------------------------------------------

def benchmark_latency(
    prompts: list[str],
    client: Any,
    n_runs: int = 3,
) -> dict:
    """
    Measure average, min, and max latency for a list of prompts.

    Args:
        prompts:  List of prompt strings to send.
        client:   Any object with a .chat(messages) method.
        n_runs:   Number of times to repeat each prompt for stability.

    Returns:
        dict with keys: avg_ms, min_ms, max_ms, total_requests
    """
    latencies = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        for _ in range(n_runs):
            start = time.perf_counter()
            client.chat(messages)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

    return {
        "avg_ms": round(statistics.mean(latencies), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "p50_ms": round(statistics.median(latencies), 2),
        "total_requests": len(latencies),
    }


def demo_benchmark():
    print("\n" + "=" * 60)
    print("DEMO: Latency Benchmark")
    print("=" * 60)

    prompts = [
        "What is a neural network?",
        "Explain gradient descent in one sentence.",
        "What is the difference between BERT and GPT?",
    ]

    client = MockLocalClient()  # Always mock in this demo for speed

    print(f"Benchmarking {len(prompts)} prompts × 3 runs...")
    results = benchmark_latency(prompts, client, n_runs=3)

    print(f"\nResults (SIMULATED — mock responses are instant):")
    for k, v in results.items():
        print(f"  {k:<20} {v}")

    print("\n[NOTE] Real latency depends on model size, quantization, and hardware.")
    print("  Typical values:")
    print("  - Mock/Simulate:     < 1 ms")
    print("  - Llama 3 8B (CPU):  3,000–8,000 ms / token (slow)")
    print("  - Llama 3 8B (GPU):  30–80 ms first token, fast streaming")
    print("  - Llama 3 70B (GPU): 150–400 ms first token")


# ---------------------------------------------------------------------------
# 6. Model comparison table
# ---------------------------------------------------------------------------

def model_comparison_table():
    print("\n" + "=" * 60)
    print("Open-Source Model Comparison")
    print("=" * 60)

    headers = ["Model", "Params", "VRAM (Q4)", "MMLU", "HumanEval", "Best for"]
    rows = [
        ("Llama 3.2 3B",   "3B",   "2 GB",   "58%",  "41%",  "Edge, mobile"),
        ("Phi-3 Mini",     "3.8B", "3 GB",   "68%",  "58%",  "CPU inference, reasoning"),
        ("Llama 3 8B",     "8B",   "5 GB",   "68%",  "62%",  "General chat, default"),
        ("Mistral 7B",     "7B",   "5 GB",   "63%",  "37%",  "Fast, commercial use"),
        ("Gemma 2 9B",     "9B",   "6 GB",   "72%",  "51%",  "Instruction following"),
        ("Mixtral 8×7B",  "47B*", "26 GB",  "71%",  "41%",  "MoE, quality at lower cost"),
        ("Llama 3 70B",   "70B",  "42 GB",  "82%",  "81%",  "Near-frontier quality"),
        ("Qwen 2.5 72B",  "72B",  "42 GB",  "85%",  "86%",  "Multilingual, coding"),
    ]

    widths = [16, 7, 11, 6, 11, 30]

    def fmt_row(row):
        return "  " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths))

    print(fmt_row(headers))
    print("  " + "-" * sum(w + 2 for w in widths))
    for row in rows:
        print(fmt_row(row))

    print("\n  * Mixtral activates 2 of 8 experts per token (effective ~13B active params)")
    print("\nKey insight: Phi-3 Mini (3.8B) matches Llama 2 13B quality at 3× smaller size.")
    print("Quality gains from data quality often exceed gains from pure parameter scaling.")


# ---------------------------------------------------------------------------
# 7. Model card evaluation checklist
# ---------------------------------------------------------------------------

def read_model_card_checklist():
    print("\n" + "=" * 60)
    print("Model Card Evaluation Checklist")
    print("=" * 60)

    checklist = [
        ("Training data",         "What data was the model trained on? Any known quality issues?"),
        ("Intended uses",         "What tasks is the model designed for? What is out-of-scope?"),
        ("Language coverage",     "What languages are supported? Is your primary language included?"),
        ("License",               "Apache 2.0 (free commercial)? Llama license (restricted)? CC-BY?"),
        ("Evaluation results",    "What benchmarks were run? Are they relevant to your use case?"),
        ("Limitations",           "Known failure modes, biases, or hallucination tendencies?"),
        ("Quantization options",  "Are GGUF/GPTQ/AWQ versions available for your hardware?"),
        ("Context window",        "Max tokens supported? Is it enough for your documents?"),
        ("Fine-tune base",        "Is it a base model or instruction-tuned? Right one for your task?"),
        ("Recency",               "When was it released/updated? Is there a newer version?"),
        ("Community activity",    "Downloads, likes, recent issues on HuggingFace — is it maintained?"),
        ("Safety alignment",      "Does it refuse harmful content? Too restrictive for your use case?"),
    ]

    for i, (category, question) in enumerate(checklist, start=1):
        print(f"\n  {i:2}. [{category}]")
        print(f"      {question}")

    print("\n  Tip: Always run evals on your own task-specific data before committing.")
    print("  Benchmark leaderboard rankings rarely translate directly to real-world performance.")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("3.7 Open-Source Models — Examples")
    print(f"Mode:         {'SIMULATE (no local model needed)' if SIMULATE else 'REAL'}")
    print(f"Ollama:       {'available' if (not SIMULATE and OllamaClient().is_available()) else 'not running (using mock)'}")
    print(f"Transformers: {'available' if TRANSFORMERS_AVAILABLE else 'not installed (using mock)'}")
    print("=" * 60)

    # Demo 1: Local chat
    print("\n" + "=" * 60)
    print("DEMO 1: Local Chat")
    print("=" * 60)

    questions = [
        "What is RAG?",
        "What is quantization?",
        "Explain transformers briefly.",
    ]
    for q in questions:
        answer = run_local_chat(q, simulate=SIMULATE)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    # Demo 2: Benchmark latency
    demo_benchmark()

    # Demo 3: Model comparison
    model_comparison_table()

    # Demo 4: Model card checklist
    read_model_card_checklist()

    print("\n" + "=" * 60)
    print("All demos complete.")
    if SIMULATE:
        print("\nTo use a real local model:")
        print("  1. Install Ollama: brew install ollama")
        print("  2. Start server:   ollama serve")
        print("  3. Pull model:     ollama pull llama3")
        print("  4. Run:            SIMULATE=false python examples.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
