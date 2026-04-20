# AI Engineering Learning Roadmap

A structured, self-paced curriculum to build practical AI engineering skills, from Python fundamentals to production-grade LLM systems.

## What This Project Covers

This repository is organized into five progressive phases:

- `phase-1-foundations` - Python, APIs, Git, databases, Docker
- `phase-2-ml-concepts` - Core ML theory, PyTorch, NLP basics
- `phase-3-llms` - Transformers, prompting, RAG pipelines
- `phase-4-production` - Evaluation, latency, cost, safety
- `phase-5-advanced` - Fine-tuning, agents, multimodal systems

The full week-by-week plan lives in `roadmap.md`.

## How To Study

For each section:

1. Read `notes.md`
2. Run `examples.py`
3. Attempt `exercises.py`
4. Check your work and iterate

## Quick Start

From this folder:

```bash
python -m venv venv
source venv/bin/activate
```

Then install dependencies for the section you are working in (if present):

```bash
pip install -r requirements.txt
```

Run learning files:

```bash
python examples.py
python exercises.py
```

## Suggested Workflow

- Follow `roadmap.md` in order
- Complete one section at a time
- Build a small project at the end of each phase
- Keep notes on what was confusing and revisit weak areas

## API Keys and Environment Variables

Some LLM sections require keys (for example, Anthropic or OpenAI). Use local `.env` files and never commit secrets.
