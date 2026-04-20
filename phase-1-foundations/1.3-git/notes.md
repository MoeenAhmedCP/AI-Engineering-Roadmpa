# 1.3 Git & Version Control

## Why Git Matters for AI Engineers

AI projects have more things to version than normal code:
- **Code** — models, pipelines, evaluators
- **Prompts** — treat these like code; bad prompt = bad product
- **Configs** — model selection, hyperparameters, chunk sizes
- **Experiment results** — which prompt + model combination scored best

---

## Core Workflow

```bash
# One-time setup
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# Start a project
git init
git remote add origin https://github.com/you/repo.git

# Daily workflow
git status                          # what changed?
git diff                            # see exact changes
git add path/to/file.py             # stage specific files (never `git add .` blindly)
git commit -m "feat: add RAG retrieval with reranking"
git push origin main
```

---

## Branching Strategy

```
main ──────────────────────────────────────────── production
  │
  ├── feat/rag-pipeline ──── PR ──▶ main
  ├── feat/streaming-api ─── PR ──▶ main
  ├── fix/rate-limit-bug ─── PR ──▶ main
  └── experiment/hyde-retrieval     (experimental, may never merge)
```

```bash
# Create a feature branch
git checkout -b feat/add-reranking

# Work, commit, push
git add reranker.py
git commit -m "feat: add cross-encoder reranking step to RAG pipeline"
git push -u origin feat/add-reranking

# Keep up to date with main
git fetch origin
git rebase origin/main              # preferred over merge for cleaner history

# After PR is merged
git checkout main
git pull
git branch -d feat/add-reranking   # delete local branch
```

---

## Merge vs Rebase

**Merge** — creates a merge commit, preserves branch history
```
main:   A──B──────M
              \  /
feature:       C──D
```

**Rebase** — replays your commits on top of main, linear history
```
main:   A──B──C──D
```

**Rule:** rebase your feature branch onto main before PR. Never rebase shared branches.

---

## .gitignore for AI Projects

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/

# Secrets (CRITICAL)
.env
.env.*
*.key
secrets.json
credentials.json

# AI-specific
*.pkl               # serialized models
*.pt                # PyTorch model weights
*.onnx              # ONNX models
model_weights/
checkpoints/
*.gguf              # quantized LLM weights

# Data (often too large for git)
data/raw/
data/processed/
*.csv               # usually too large
*.parquet

# Notebooks (optional — outputs can be noisy)
.ipynb_checkpoints/
# Uncomment to ignore notebook outputs:
# *.ipynb

# Vector indexes (large, binary)
*.faiss
*.index
chroma_db/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/settings.json
.idea/
*.swp
```

---

## Conventional Commits

Structured commit messages make history scannable and enable changelog generation:

```
<type>(<scope>): <short description>

feat(rag): add hybrid BM25 + vector search
fix(api): handle 429 rate limit with exponential backoff
docs(readme): add deployment instructions
test(eval): add RAGAS evaluation test suite
refactor(embeddings): extract embedding logic into service class
perf(retrieval): cache embeddings to avoid redundant API calls
chore(deps): upgrade langchain to 0.3.0
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`

---

## Resolving Merge Conflicts

```bash
# You're rebasing and hit a conflict
git rebase origin/main
# CONFLICT in rag_pipeline.py

# Open the file — Git marks conflicts like this:
# <<<<<<< HEAD (your changes)
# def retrieve(query, top_k=5):
# =======
# def retrieve(query, top_k=10):
# >>>>>>> origin/main (incoming changes)

# Edit the file to keep what you want
# Remove the conflict markers

git add rag_pipeline.py
git rebase --continue

# If things go wrong
git rebase --abort               # go back to before you started
```

---

## Git for Prompts (Prompt Versioning)

Prompts are code. Version them properly:

```
prompts/
├── system_prompt_v1.md          ← tagged, in production
├── system_prompt_v2.md          ← in testing
├── summarize_v3.md
└── CHANGELOG.md                 ← what changed and why
```

```bash
# Tag a prompt version when deploying to production
git tag v1.2.0-prompt-rewrite
git push origin --tags

# Find when a prompt changed
git log --follow -p prompts/system_prompt_v1.md
```

---

## Useful Commands

```bash
# See full history with branching
git log --oneline --graph --all

# Find when a bug was introduced
git bisect start
git bisect bad                   # current commit is broken
git bisect good v1.0.0           # this commit was fine
# git will binary search — test each commit, mark good/bad

# See who changed a line
git blame path/to/file.py

# Temporarily save work without committing
git stash
git stash pop

# Undo the last commit but keep changes staged
git reset --soft HEAD~1

# See what's different from main
git diff origin/main...HEAD
```

---

## Pre-commit Hooks (Catch Mistakes Automatically)

```bash
pip install pre-commit
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: detect-private-key        # blocks committing private keys
      - id: check-added-large-files   # blocks files > 500kb
        args: ['--maxkb=500']
      - id: check-json
      - id: end-of-file-fixer
      - id: trailing-whitespace

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff                      # lint
      - id: ruff-format               # format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
```

```bash
pre-commit install                   # runs hooks before every commit
pre-commit run --all-files           # run manually on everything
```
