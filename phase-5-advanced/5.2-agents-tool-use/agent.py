"""
5.2 AI Agents — Complete ReAct Agent Implementation
SIMULATE=True: uses mock LLM, no API keys needed
SIMULATE=False: requires ANTHROPIC_API_KEY
Run: python agent.py
"""

import os
import json
import re
import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SIMULATE = os.getenv("SIMULATE", "true").lower() != "false"
MAX_ITERATIONS = 10


# ---------------------------------------------------------------------------
# 1. Tool Definition & Decorator
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema for the parameters object
    fn: Callable[..., str]


_REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict):
    """Decorator to register a function as an agent tool."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            fn=fn,
        )
        return fn
    return decorator


# ---------------------------------------------------------------------------
# 2. Built-in Tool Implementations
# ---------------------------------------------------------------------------

@tool(
    name="search_web",
    description=(
        "Search the internet for current information. Use this when you need facts "
        "about real-world events, prices, people, or anything that might have changed "
        "recently. Returns a brief summary of search results."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and concise.",
            }
        },
        "required": ["query"],
    },
)
def search_web(query: str) -> str:
    """Mock web search. Returns deterministic results based on keywords."""
    q = query.lower()
    if "rag" in q or "retrieval augmented" in q:
        return (
            "RAG (Retrieval-Augmented Generation) is a technique that combines information "
            "retrieval with language model generation. It retrieves relevant documents from "
            "a knowledge base and provides them as context to the LLM, improving factual "
            "accuracy and reducing hallucinations. Key components: embedding model, vector "
            "store, retriever, and generator. Popularized by Meta AI research paper (2020)."
        )
    if "weather" in q:
        return "Current weather: Partly cloudy, 22°C (72°F). Humidity 58%. Wind 12 km/h."
    if "python" in q:
        return "Python 3.13 is the latest stable release. Python is widely used for AI/ML."
    if "tip" in q or "gratuity" in q:
        return "Standard tipping etiquette: 15-20% for restaurants, 10-15% for takeout."
    return f"Search results for '{query}': No specific results found in mock database."


@tool(
    name="calculate",
    description=(
        "Evaluate a mathematical expression and return the result. "
        "Use for arithmetic, percentages, and basic math. "
        "Supports: +, -, *, /, **, %, parentheses, and basic math functions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A Python-compatible math expression. E.g., '47.50 * 0.15'",
            }
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    # Whitelist: only allow digits, operators, spaces, dots, parentheses
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%\*\*]+$', expression):
        return f"Error: Expression contains unsupported characters: {expression}"
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool(
    name="get_time",
    description="Get the current date and time. Use when the task requires knowing today's date or the current time.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def get_time() -> str:
    """Return current timestamp."""
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} (local time)"


@tool(
    name="read_file",
    description=(
        "Read the contents of a file. Only files in the ./sandbox/ directory are accessible. "
        "Use this to read configuration files, data files, or notes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to current directory. Must start with ./sandbox/",
            }
        },
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    """Read a file, restricted to the ./sandbox/ directory."""
    # Safety: only allow sandbox directory
    if not path.startswith("./sandbox/") and not path.startswith("sandbox/"):
        return f"Error: Access denied. Only files in ./sandbox/ are readable. Got: {path}"
    # Normalize path to prevent traversal
    clean_path = os.path.normpath(path)
    if ".." in clean_path:
        return "Error: Path traversal detected."
    try:
        with open(clean_path, "r") as f:
            content = f.read()
        return f"Contents of {path}:\n{content}"
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"


# ---------------------------------------------------------------------------
# 3. Prompt Formatting
# ---------------------------------------------------------------------------

def format_tools_for_prompt(tools: dict[str, Tool]) -> str:
    """
    Generate a natural-language description of all available tools
    for inclusion in the system prompt.
    """
    lines = ["Available tools:\n"]
    for t in tools.values():
        lines.append(f"Tool: {t.name}")
        lines.append(f"Description: {t.description}")
        props = t.parameters.get("properties", {})
        if props:
            lines.append("Parameters:")
            for param_name, param_info in props.items():
                required = param_name in t.parameters.get("required", [])
                req_str = " (required)" if required else " (optional)"
                lines.append(f"  - {param_name}{req_str}: {param_info.get('description', '')}")
        lines.append("")
    return "\n".join(lines)


REACT_SYSTEM_PROMPT = """You are a helpful AI assistant that solves tasks using tools.

{tools_description}

To use a tool, respond in this EXACT format:
Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"key": "value"}}

When you have the final answer, respond in this EXACT format:
Thought: [your reasoning]
Final Answer: [your complete answer to the user]

Rules:
- Always start with a Thought
- After every Action, wait for the Observation before proceeding
- Use Final Answer only when you have enough information to answer the task
- Be concise and accurate
"""


# ---------------------------------------------------------------------------
# 4. LLM Response Parsing
# ---------------------------------------------------------------------------

def parse_llm_response(text: str) -> dict:
    """
    Parse a ReAct-format LLM response into structured components.

    Returns a dict with keys:
      - 'thought': str
      - 'action': str | None
      - 'action_input': dict | None
      - 'final_answer': str | None
    """
    result = {"thought": "", "action": None, "action_input": None, "final_answer": None}

    # Extract Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|$)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # Extract Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result

    # Extract Action
    action_match = re.search(r"Action:\s*(\w+)", text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    # Extract Action Input (JSON)
    input_match = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)
    if input_match:
        try:
            result["action_input"] = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            # Try to salvage a simple string value
            result["action_input"] = {"query": input_match.group(1).strip()}

    return result


# ---------------------------------------------------------------------------
# 5. Mock LLM
# ---------------------------------------------------------------------------

def mock_llm(messages: list[dict], tools: dict[str, Tool]) -> str:
    """
    Deterministic mock LLM that follows the ReAct format.
    Produces appropriate tool calls based on task keywords.
    Used when SIMULATE=True.
    """
    # Reconstruct conversation to understand current state
    full_text = " ".join(m.get("content", "") for m in messages).lower()

    # Has there already been an observation? If so, give final answer.
    observations = [m for m in messages if m.get("role") == "tool"]

    # Task: calculate tip
    if "tip" in full_text and "47.50" in full_text:
        if observations:
            # We have the calculation result, give final answer
            last_obs = observations[-1]["content"]
            result = last_obs.split("=")[-1].strip() if "=" in last_obs else "7.13"
            return (
                f"Thought: I now have the calculated tip amount.\n"
                f"Final Answer: A 15% tip on $47.50 is ${result}. "
                f"Total with tip would be ${47.50 + float(result.replace('$','')):.2f}."
            )
        return (
            "Thought: I need to calculate 15% of $47.50. I'll use the calculate tool.\n"
            "Action: calculate\n"
            'Action Input: {"expression": "47.50 * 0.15"}'
        )

    # Task: what time is it?
    if "time" in full_text or "date" in full_text:
        if observations:
            last_obs = observations[-1]["content"]
            return (
                f"Thought: I have the current time from the tool.\n"
                f"Final Answer: {last_obs}"
            )
        return (
            "Thought: The user wants to know the current time. I'll use the get_time tool.\n"
            "Action: get_time\n"
            "Action Input: {}"
        )

    # Task: search for RAG information
    if "rag" in full_text or "retrieval" in full_text:
        if observations:
            last_obs = observations[-1]["content"]
            return (
                f"Thought: I have retrieved information about RAG from the search.\n"
                f"Final Answer: {last_obs}"
            )
        return (
            "Thought: The user wants information about RAG. I'll search the web for this.\n"
            "Action: search_web\n"
            'Action Input: {"query": "RAG retrieval augmented generation"}'
        )

    # Default fallback
    return (
        "Thought: I can answer this from my knowledge.\n"
        "Final Answer: I'm a mock LLM and don't have a specific answer for this query. "
        "In production mode, this would use the actual Claude API."
    )


def real_llm(messages: list[dict], tools: dict[str, Tool]) -> str:
    """
    Real LLM call using Anthropic API.
    Only used when SIMULATE=False.
    """
    try:
        import anthropic
    except ImportError:
        return "Error: anthropic package not installed. Run: pip install anthropic"

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Build tools list for API
    api_tools = [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools.values()
    ]

    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    conversation = [m for m in messages if m["role"] != "system"]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=conversation,
        tools=api_tools,
    )

    # Convert API response to ReAct text format
    content = response.content[0]
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "type") and content.type == "tool_use":
        return (
            f"Thought: I should use the {content.name} tool.\n"
            f"Action: {content.name}\n"
            f"Action Input: {json.dumps(content.input)}"
        )
    return str(content)


# ---------------------------------------------------------------------------
# 6. ReAct Agent
# ---------------------------------------------------------------------------

class ReActAgent:
    """
    A ReAct (Reasoning + Acting) agent that iteratively calls tools
    until it arrives at a final answer.
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        llm_fn: Callable[[list[dict], dict], str],
        max_iter: int = MAX_ITERATIONS,
        verbose: bool = True,
    ):
        self.tools = tools
        self.llm_fn = llm_fn
        self.max_iter = max_iter
        self.verbose = verbose
        self.trace: list[dict] = []

    def _log(self, role: str, content: str) -> None:
        self.trace.append({"role": role, "content": content})
        if self.verbose:
            separator = "─" * 55
            print(f"\n{separator}")
            print(f"  [{role.upper()}]")
            print(separator)
            print(content)

    def run(self, task: str) -> str:
        """
        Run the agent on a task. Returns the final answer string.
        """
        if self.verbose:
            print("\n" + "=" * 55)
            print(f"  TASK: {task}")
            print("=" * 55)

        self.trace = []
        tools_desc = format_tools_for_prompt(self.tools)
        system_content = REACT_SYSTEM_PROMPT.format(tools_description=tools_desc)

        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": task},
        ]

        for iteration in range(self.max_iter):
            # Call LLM
            response_text = self.llm_fn(messages, self.tools)
            self._log("assistant", response_text)

            # Parse response
            parsed = parse_llm_response(response_text)

            # Check for final answer
            if parsed["final_answer"]:
                self._log("result", parsed["final_answer"])
                return parsed["final_answer"]

            # Execute tool
            if parsed["action"]:
                tool_name = parsed["action"]
                tool_input = parsed["action_input"] or {}

                if tool_name not in self.tools:
                    observation = f"Error: Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"
                else:
                    try:
                        observation = self.tools[tool_name].fn(**tool_input)
                    except TypeError as e:
                        observation = f"Error calling {tool_name}: {e}"

                self._log("observation", observation)

                # Append to conversation
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "tool", "content": f"Observation: {observation}"})
            else:
                # No action and no final answer — something went wrong
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "tool",
                    "content": "Observation: No action was taken. Please either call a tool or provide a Final Answer.",
                })

        return "Error: Agent exceeded maximum iterations without reaching a final answer."


# ---------------------------------------------------------------------------
# 7. Demo
# ---------------------------------------------------------------------------

def run_demo():
    print("=" * 65)
    print("  5.2 AI AGENTS — ReAct AGENT DEMO")
    mode = "SIMULATE (no API key needed)" if SIMULATE else "LIVE (using Anthropic API)"
    print(f"  Mode: {mode}")
    print("=" * 65)

    llm_fn = mock_llm if SIMULATE else real_llm
    agent = ReActAgent(tools=_REGISTRY, llm_fn=llm_fn, max_iter=MAX_ITERATIONS, verbose=True)

    tasks = [
        "Calculate 15% tip on $47.50",
        "What time is it?",
        "Search for information about RAG",
    ]

    results = []
    for task in tasks:
        answer = agent.run(task)
        results.append((task, answer))
        agent.trace = []  # Reset trace for next task

    print("\n\n" + "=" * 65)
    print("  DEMO SUMMARY — ALL TASKS")
    print("=" * 65)
    for i, (task, answer) in enumerate(results, 1):
        print(f"\n  Task {i}: {task}")
        print(f"  Answer: {answer}")

    print("\n")
    print("=" * 65)
    print("  AGENT ARCHITECTURE OVERVIEW")
    print("=" * 65)
    print("""
  Task Input
      │
      ▼
  [System Prompt: tools + ReAct instructions]
      │
      ▼
  ┌─────────────────────────────────────────┐
  │            REACT LOOP                   │
  │                                         │
  │  LLM → Thought + Action + Action Input  │
  │           │                             │
  │           ▼                             │
  │  Tool Execution (your code runs this)   │
  │           │                             │
  │           ▼                             │
  │  Observation injected into context      │
  │           │                             │
  │           └──── repeat ─────────────── │
  │                                         │
  │  LLM → Thought + Final Answer           │
  └─────────────────────────────────────────┘
      │
      ▼
  Final Answer returned to user

  Safety: MAX_ITERATIONS={max_iter}, sandbox-only file access
  """.format(max_iter=MAX_ITERATIONS))


if __name__ == "__main__":
    run_demo()
