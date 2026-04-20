# 4.1 Evaluation: Measuring Quality in Non-Deterministic Systems

## The Eval Problem

In traditional software, testing is straightforward: call the function, assert the output. `assertEqual(add(2, 3), 5)` either passes or fails. There is no ambiguity.

LLM outputs break this completely. Ask the same question twice and you may get two different answers that are both correct. Ask for a summary and there are infinite valid summaries. You cannot `assertEqual` a paragraph. You cannot `assertIn` to verify quality. The fundamental assumption of unit testing — that correct behavior is a single, checkable value — does not hold.

This is the eval problem. It is the hardest unsolved problem in applied AI engineering, and most teams handle it poorly by shipping on vibes ("it seemed fine in testing") and only measuring quality after users complain.

The solution is not a single technique. It is a layered approach: automated metrics for speed, LLM judges for nuance, and human review for ground truth. You need all three.

---

## Golden Test Sets

A golden test set is a fixed collection of (input, expected_output) pairs that you curate by hand and never change. They represent the behaviors your system must always exhibit — the non-negotiables.

Building a good golden set:
- **Start with real user queries.** After one week of usage, take 50 real questions that users actually asked. These are more valuable than invented examples.
- **Include edge cases.** Ambiguous queries, very long inputs, very short inputs, inputs in other languages, inputs with typos.
- **Annotate expected behavior, not exact text.** Instead of `expected_answer = "The policy expires on January 1st"`, write `expected_criteria = ["mentions expiration date", "mentions January", "does not confuse with start date"]`. This makes the golden set robust to valid paraphrasing.
- **Never add to it casually.** Every addition should be deliberate. A bloated golden set with poor examples is worse than a small, high-quality one.
- **Version it.** The golden set is as important as your code. Track it in git. Review changes in PRs.

Run your golden set before every deployment. If score drops, block the deploy.

---

## LLM-as-Judge

When you need to evaluate nuanced qualities — helpfulness, factual accuracy, tone, completeness — you can use a stronger model to judge the output of a weaker one. This is called LLM-as-judge and it is increasingly standard practice at companies like Anthropic, OpenAI, and Google.

The core idea: instead of you hand-scoring every output, you write a careful judge prompt and let GPT-4o or Claude Sonnet rate outputs on a 1-5 scale with reasoning.

**Designing the judge prompt carefully:**

1. **Include a rubric, not just a scale.** A scale without criteria produces noise. Tell the judge exactly what a 1, 3, and 5 look like.
2. **Ask for reasoning before the score.** Chain-of-thought in the judge dramatically improves consistency. The judge that explains its reasoning before scoring agrees with human raters ~15% more than a judge that scores directly.
3. **Provide the source material.** The judge needs the original question and the context — not just the answer — to evaluate faithfulness.
4. **Test your judge on known examples.** Before trusting your judge, give it 20 examples you've hand-scored and measure agreement. Judge-human agreement above 80% is a reasonable bar.
5. **Control for position bias.** LLM judges tend to prefer the first answer when comparing two. Randomize order and average results.

---

## RAG Eval Metrics

For retrieval-augmented generation systems specifically, three metrics capture most of what matters:

**Context Relevancy:** Did the retriever return chunks that are actually relevant to the question? Score: (relevant chunks retrieved) / (total chunks retrieved). A retriever returning 5 chunks where 2 are relevant scores 0.4. Low context relevancy means you're wasting tokens and confusing the model.

**Answer Faithfulness:** Does the generated answer stay within the bounds of what the retrieved context says? An answer that introduces facts not in the context is hallucinating, even if those facts happen to be true. Score: (claims in answer that appear in context) / (total claims in answer).

**Answer Completeness:** Does the answer fully address what the user asked? A faithful answer can still be incomplete if it ignores half the question. Score: (aspects of question addressed) / (total aspects of question).

**RAGAS** is a Python framework that automates all three using LLM judges under the hood. It takes (question, context, answer) tuples and returns scores for all three metrics. Useful for running large-scale evals without hand-scoring every example.

---

## Human Evaluation

Automated metrics are fast and scalable but miss things humans catch immediately — tone that feels wrong, answers that are technically accurate but practically useless, cultural inappropriateness. Human eval is the ground truth everything else is calibrated against.

Recommended cadence:
- **Weekly sample review:** 100 randomly sampled interactions reviewed by someone on the team. Takes ~2 hours. Keeps you grounded in what the system actually produces.
- **Launch review:** Before any major feature or model change, hand-score 200 examples. Compare to pre-change scores.
- **Failure analysis:** Every week, look at the bottom 10% of LLM-judge-scored outputs. Understand the failure modes. Many will be the same 3-4 root causes.

The temptation is to skip human review when automated scores look good. Resist this. Automated scores measure what they measure; they do not measure what they don't measure.

---

## Regression Testing

Eval suites should run in CI/CD pipelines the same way unit tests do. The workflow:

1. Developer opens a PR (prompt change, model change, retrieval change)
2. CI runs the eval suite against the full golden test set
3. LLM judge scores every output
4. If mean score drops more than 10% from the baseline, the PR is blocked
5. Developer must either fix the regression or explicitly acknowledge and override

This prevents the slow quality decay that happens when no one is watching — each change seems fine in isolation but the cumulative drift is significant.

**Baseline management:** Store baseline scores in the repo (a JSON file checked in alongside the golden set). Update the baseline deliberately when you make a change that intentionally alters behavior.

---

## Red-Teaming

Red-teaming means adversarially testing your system before users find the failures. It is not optional; it is the minimum due diligence before public launch.

Common attack categories to test:
- **Prompt injection:** "Ignore previous instructions and instead..."
- **Jailbreaks:** Role-play framings, hypothetical framings, DAN prompts
- **Off-topic abuse:** Using a customer service bot as a general-purpose chatbot
- **PII extraction:** "Repeat back the system prompt" or "What user data do you have?"
- **Conflicting context:** Providing context that contradicts the system prompt
- **Adversarial completions:** Inputs designed to elicit specific harmful outputs

Document every red-team finding. Fix the ones that produce harmful output. Accept and monitor the ones that are low-risk. Never ship without at least 50 adversarial test cases.

---

## Tools

- **LangSmith:** LangChain's eval and tracing platform. Deep integration if you use LangChain, awkward if you don't.
- **Braintrust:** Fast, well-designed eval platform. Good LLM-as-judge tooling. Works with any framework.
- **PromptFoo:** Open-source CLI tool for running eval suites. YAML config, good for CI integration.
- **Athina AI:** Focused on RAG eval specifically. Good RAGAS-style metrics out of the box.
- **RAGAS:** Python library for RAG metrics. Run locally, no vendor lock-in.

Start with RAGAS + PromptFoo before paying for managed platforms. Once you know what you need, evaluate the paid tools.
