# 3.3 Prompt Engineering — Notes

Prompt engineering is the skill of communicating effectively with LLMs. Despite being called "just prompting," it has a dramatic impact on output quality — the difference between a carefully designed prompt and a casual one can change accuracy from 60% to 90% on structured tasks. This section covers every major technique with the engineering discipline it deserves.

---

## The System Prompt: The Constitution

The system prompt is the highest-priority instruction in an LLM conversation. It runs before any user message and defines:
- **Persona:** Who the model is ("You are an expert legal document reviewer...")
- **Rules:** What the model must and must not do
- **Format:** How outputs should be structured (JSON, markdown, bullet points)
- **Tone:** Formal, casual, concise, verbose

Think of the system prompt as the constitution of the conversation — user messages are laws, but the system prompt is the founding document that constrains everything. In Claude's terms, the system prompt sets the "context window constitution."

**System prompt best practices:**
- State persona before rules ("You are X. Your rules are Y.")
- Use explicit positive framing ("Always cite sources" vs "Don't forget to cite sources")
- Define the output format explicitly — never leave it ambiguous
- Keep it as concise as possible — bloated system prompts can dilute attention

---

## The User Prompt: The Task

The user prompt is what the user sends. In production systems, you rarely receive this directly — you construct it from a template that includes user input, retrieved context (RAG), conversation history, and instructions.

Good user prompts are specific about:
- The exact task ("Summarize this contract" is weaker than "Extract the 3 most important obligations of Party A from this contract")
- The expected output format (if not in system prompt)
- Any constraints ("In 200 words or less", "Focus only on risks")

---

## Few-Shot Examples: The Fastest Quality Boost

Providing 2-5 input/output examples in your prompt dramatically improves consistency. The model learns the exact format, tone, and reasoning style you want from the examples — no additional training required.

**Structure:**
```
Input: <example input 1>
Output: <example output 1>

Input: <example input 2>
Output: <example output 2>

Input: <actual task input>
Output:
```

**When to use few-shot:**
- Tasks with unusual output formats
- Classification tasks with custom categories
- Extraction tasks where you need consistent field names
- Any task where zero-shot gives inconsistent results

**Choosing examples:** use examples that cover edge cases and the variety of inputs you expect. Avoid examples that are too similar to each other. Order matters slightly — put the most representative example last.

---

## Chain-of-Thought: Force Explicit Reasoning

Chain-of-thought (CoT) prompting tells the model to reason through a problem step by step before giving its final answer. This dramatically improves performance on math, logic, multi-step reasoning, and complex analysis.

**Zero-shot CoT:** simply append "Let's think step by step." to the user prompt. Surprisingly effective.

**Explicit CoT:** provide a few-shot example where the output includes reasoning steps, then the answer. The model follows the pattern.

**Why CoT works:**
- Forces the model to "show its work" in the context window
- Each reasoning step conditions the next one, keeping the model on track
- Errors in intermediate steps can be spotted and corrected (including by the model itself)
- The model's final answer is conditioned on its own correct reasoning

**When NOT to use CoT:** simple factual lookups, classification, short-form generation. CoT adds tokens and cost without benefit on tasks that don't require multi-step reasoning.

---

## Output Formatting: Always Be Explicit

Never leave output format to chance in a production system. Specify it exactly.

**JSON output:** describe the exact schema. Include field names, types, and examples. Wrap in a code block instruction: "Return a JSON object with exactly these fields: ..."

**Markdown:** specify heading levels, whether to use bullet points or numbered lists.

**Length:** "In exactly 3 bullet points" or "In 100-150 words" gives much more consistent results than "briefly."

**Negative space:** if the model might add preamble ("Sure! Here's..."), explicitly forbid it: "Return only the JSON, no preamble or explanation."

---

## Prompt Templates with Variables

In production, prompts are rarely static. Build them from templates:

```python
template = """You are a {persona}.

Task: {task}

Context:
{context}

Output format: {format_spec}"""

prompt = template.format(
    persona="financial analyst",
    task="identify the top 3 risks",
    context=document_text,
    format_spec="JSON list of {risk, severity, mitigation}"
)
```

Keep templates in separate files, version-controlled. Treat prompt templates like code — they ARE code.

---

## Temperature and Top-p: Controlling Randomness

**Temperature** controls how peaked the probability distribution is over the vocabulary:
- **Temperature = 0:** always pick the highest-probability token (greedy decoding). Deterministic, most conservative.
- **Temperature = 0.7:** sample from a slightly smoothed distribution. Good balance for chat.
- **Temperature = 1.0:** sample from the raw distribution as trained.
- **Temperature > 1.5:** very random — words become unpredictable. Useful for creative brainstorming, dangerous for factual tasks.

Mechanically: logits are divided by temperature before softmax. High temperature flattens the distribution (more random). Low temperature sharpens it (more deterministic).

**Top-p (nucleus sampling):** instead of sampling from the full vocabulary, only sample from the smallest set of tokens whose cumulative probability exceeds p.
- `top_p=0.9`: sample from the tokens that account for 90% of probability mass
- Cuts off unlikely "tail" tokens, reducing garbage outputs
- Often used together with temperature

**Practical settings:**
| Task | Temperature | Top-p |
|------|-------------|-------|
| Code generation | 0.0-0.2 | 0.95 |
| Factual extraction | 0.0-0.3 | 1.0 |
| Chat / Q&A | 0.7 | 0.9 |
| Creative writing | 1.0-1.3 | 0.9 |
| Brainstorming | 1.2-1.5 | 0.95 |

---

## Positive vs Negative Framing

Negative instructions ("do not X") are consistently weaker than positive instructions ("only do Y").

**Why:** attention mechanisms weight positive content more strongly. "Do not include disclaimers" can be overridden by training to include safety text. "Return only the JSON, no other text" is a stronger constraint.

**Reframe everything positively:**
- "Don't be verbose" → "Use exactly 2 sentences per point"
- "Don't make up information" → "Only use information explicitly stated in the provided text"
- "Don't include irrelevant content" → "Include only information directly relevant to the query"

---

## Systematic Prompt Testing

Treat prompts like code — test them systematically.

**Build a test harness:**
1. Define a set of representative test inputs (10-20 examples)
2. Define what a good output looks like (rubric or reference outputs)
3. Run multiple prompt variants against all test inputs
4. Score each output (human or LLM-as-judge)
5. Rank prompt variants by average score

**LLM-as-judge:** use a separate LLM call to score each output on a 1-5 scale. Faster than human review. Provide a clear rubric: "Score 1-5 where 5=perfectly extracts all required fields with correct values, 1=fails to extract any required fields."

---

## Prompt Versioning

Treat prompts like code. They belong in version control.

**Git tag deployed versions:**
```bash
git tag prompt-v1.2-legal-summarizer
```

**In code, load prompts from files:**
```python
# prompts/legal_summarizer_v1.2.txt
system_prompt = Path("prompts/legal_summarizer_v1.2.txt").read_text()
```

**Never embed long prompts as inline strings** in application code — they become impossible to review, test, or iterate on. Keep prompts in separate `.txt` or `.jinja2` files, committed to git, with changelog notes.

**Prompt registry pattern:** maintain a `prompts/` directory with versioned files and a `registry.json` that maps prompt names to their current version file. Your application code loads prompts by name, making version upgrades a one-line registry change with easy rollback.
