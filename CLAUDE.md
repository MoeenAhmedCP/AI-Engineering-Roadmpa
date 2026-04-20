# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What This Repo Is

A structured, self-paced AI engineering curriculum — from Python foundations to production RAG, agents, and fine-tuning. Built to be followed with Claude Code as a pair-programming tutor.

**Roadmap:** See `roadmap.md` for the full 22-week plan with checkboxes, project ideas, and suggested prompts.

---

## Directory Structure

```
learning/
├── roadmap.md                        ← Master plan — tick off topics as you finish
├── phase-1-foundations/              ← Weeks 1–3: Python, APIs, Git, DB, Docker
├── phase-2-ml-concepts/              ← Weeks 4–7: ML theory, PyTorch, NLP basics
├── phase-3-llms/                     ← Weeks 8–12: Transformers, RAG, APIs, prompting
├── phase-4-production/               ← Weeks 13–17: Eval, latency, cost, safety
└── phase-5-advanced/                 ← Weeks 18–22: Fine-tuning, agents, multimodal
```

Each section folder contains:
- `notes.md` — concept explanations and mental models
- `examples.py` (or relevant code files) — working, runnable code
- `exercises.py` — practice problems (attempt before reading solutions)

---

## How to Use This With Claude Code

Work one section at a time. For each section:
1. Read `notes.md` in that section's folder
2. Run and study `examples.py`
3. Attempt `exercises.py` yourself, then ask Claude Code to review

**Useful prompts while studying any section:**
```
"Quiz me on <topic> with 5 questions at increasing difficulty"
"Review my solution to exercise N — what would you improve?"
"Explain <concept> differently, I didn't fully get the notes"
"Build me a mini-project that uses everything from section <X.Y>"
```

---

## Running Code

Each section is self-contained. Setup per section:

```bash
# Create and activate a virtual environment (do this once per phase)
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install dependencies for a section
pip install -r requirements.txt   # if present in that section folder

# Run an example
python examples.py

# Run exercises
python exercises.py
```

---

## Environment Variables

Some sections (Phase 3+) need API keys. Never commit these.

```bash
# Create a .env in the section folder
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Load with:
```python
from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv("ANTHROPIC_API_KEY")
```

---

## Phase Entry Points

| Phase | Start Here | Key Build |
|---|---|---|
| Phase 1 | `phase-1-foundations/1.1-python-proficiency/` | FastAPI + tests |
| Phase 2 | `phase-2-ml-concepts/2.1-how-ml-works/` | sklearn model served via API |
| Phase 3 | `phase-3-llms/3.1-transformers/` | RAG pipeline |
| Phase 4 | `phase-4-production/4.1-evaluation/` | Eval + monitoring stack |
| Phase 5 | `phase-5-advanced/5.1-fine-tuning/` | Fine-tuned model + full agent |
