# 1.6 Environment Management

## Virtual Environments: venv vs conda vs poetry

Choosing the right environment tool is the first decision in any Python project. Each tool solves a slightly different problem.

### Comparison Table

| Feature | venv | conda | poetry |
|---|---|---|---|
| Built into Python | Yes (3.3+) | No | No |
| Manages Python version | No | Yes | Via pyenv integration |
| Manages non-Python deps | No | Yes (C libs, CUDA) | No |
| Dependency resolution | No (pip handles it) | Yes | Yes (smart resolver) |
| Lock file | No (requirements.txt) | environment.yml | poetry.lock |
| Publish to PyPI | No | No | Yes |
| Best for | Simple scripts, APIs | Data science, ML | Libraries, complex apps |
| Speed | Fast | Slow to solve | Medium |
| Learning curve | Low | Medium | Medium-High |

### When to Use Each

**venv** — Use this for FastAPI backends, simple AI apps, anything you deploy to a server. It ships with Python, it's fast, and it's what most production tutorials assume.

```bash
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows
pip install -r requirements.txt
deactivate
```

**conda** — Use this when you need GPU/CUDA libraries (PyTorch, TensorFlow) or when mixing Python and non-Python packages. Common in ML research. Not ideal for web APIs because conda environments are large and slow to build in Docker.

```bash
conda create -n myproject python=3.11
conda activate myproject
conda install pytorch torchvision -c pytorch
conda deactivate
```

**poetry** — Use this when building a Python library you want to publish, or when you need precise dependency locking across a team. The `pyproject.toml` is the modern standard. Slower to learn but the dependency resolver prevents "works on my machine" issues.

```bash
pip install poetry
poetry new myproject
poetry add anthropic fastapi uvicorn
poetry add --group dev pytest ruff
poetry install
poetry run uvicorn main:app --reload
```

---

## requirements.txt vs pyproject.toml

`requirements.txt` is the old way — flat, simple, but no dependency groups, no metadata, no build configuration.

`pyproject.toml` is the modern Python standard (PEP 517/518/621). It replaces `setup.py`, `setup.cfg`, and `requirements.txt` in one file.

### Sample pyproject.toml for an AI Project

```toml
[tool.poetry]
name = "docsense-backend"
version = "0.1.0"
description = "AI document intelligence backend"
authors = ["Your Name <you@example.com>"]
python = "^3.11"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.115.0"
uvicorn = {extras = ["standard"], version = "^0.30.0"}
anthropic = "^0.40.0"
pydantic-settings = "^2.3.0"
python-dotenv = "^1.0.0"
supabase = "^2.5.0"
pypdf = "^4.2.0"
python-docx = "^1.1.0"
httpx = "^0.27.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.0"
pytest-cov = "^5.0.0"
ruff = "^0.4.0"
mypy = "^1.10.0"
pre-commit = "^3.7.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--cov=. --cov-report=term-missing"
```

---

## .env Files and python-dotenv

A `.env` file holds environment-specific secrets and configuration that should never be committed to git.

### .env file format

```env
# .env  (NEVER commit this file)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost:5432/docsense
REDIS_URL=redis://localhost:6379
DEBUG=true
ENVIRONMENT=development
```

### Loading with python-dotenv

```python
from dotenv import load_dotenv
import os

# Load .env before anything else that reads os.environ
load_dotenv()  # reads .env in current directory

# For specific files:
load_dotenv(".env.development")

# With override (env vars already set won't be overwritten by default)
load_dotenv(override=True)

api_key = os.getenv("ANTHROPIC_API_KEY")
```

### .gitignore entries

```gitignore
.env
.env.*
!.env.example
!.env.*.example
```

---

## Why NEVER Hardcode API Keys

**Git history is forever. Even deleted files.**

If you commit a secret:
1. `git rm` does not remove it from history
2. Anyone who clones the repo after you can run `git log -p` and see it
3. GitHub scans all public repos for known secret patterns and notifies providers (Anthropic, OpenAI, AWS) — your key may be auto-revoked within minutes
4. Private repos that later become public expose all historical secrets
5. Forks preserve history — even if you delete the repo

If you accidentally commit a secret:
```bash
# Immediately revoke the key in the provider dashboard
# Then remove from history (requires force push):
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/file" \
  --prune-empty --tag-name-filter cat -- --all

# Better: use git-filter-repo (faster, safer)
pip install git-filter-repo
git filter-repo --path secrets.py --invert-paths
```

The safest rule: treat every commit as if it will be public forever.

---

## pydantic-settings BaseSettings: The Standard Pattern for FastAPI AI Apps

`pydantic-settings` provides type-safe, validated configuration loaded from environment variables. It is the production standard for FastAPI.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Required — will raise ValidationError if missing
    anthropic_api_key: SecretStr
    database_url: str

    # Optional with defaults
    debug: bool = False
    environment: str = "development"
    max_tokens: int = 4096
    model_name: str = "claude-sonnet-4-6"

@lru_cache
def get_settings() -> Settings:
    return Settings()

# In FastAPI:
from fastapi import Depends

def get_api_key(settings: Settings = Depends(get_settings)):
    return settings.anthropic_api_key.get_secret_value()
```

`SecretStr` masks the value in logs and repr, preventing accidental leakage. `lru_cache` means settings are only parsed once — not on every request.

---

## Secrets in Production

Local `.env` files are for development only. In production, use proper secrets management.

### AWS Secrets Manager

```python
import boto3
import json

def get_secret(secret_name: str, region: str = "us-east-1") -> dict:
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

# Usage
secrets = get_secret("docsense/production")
api_key = secrets["ANTHROPIC_API_KEY"]
```

### ECS Environment Variables

In ECS task definitions, environment variables are injected at runtime — never baked into the Docker image:

```json
{
  "environment": [
    {"name": "ENVIRONMENT", "value": "production"}
  ],
  "secrets": [
    {
      "name": "ANTHROPIC_API_KEY",
      "valueFrom": "arn:aws:secretsmanager:us-east-1:123:secret:docsense/prod-abc123"
    }
  ]
}
```

The `secrets` field pulls from Secrets Manager at container startup. The value is never visible in the task definition or Docker image.

---

## Managing Multiple .env Files

```
.env                    # base defaults (non-secret, safe to commit selectively)
.env.development        # local dev overrides
.env.test               # used during pytest runs
.env.production         # loaded by deployment pipeline (or ignored — use Secrets Manager)
.env.example            # committed — shows all required keys with dummy values
```

### Loading the right file

```python
import os
from dotenv import load_dotenv

env = os.getenv("ENVIRONMENT", "development")
load_dotenv(f".env.{env}")  # load environment-specific first
load_dotenv(".env")          # then fall back to base
```

### .env.example (commit this)

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
REDIS_URL=redis://localhost:6379
DEBUG=true
ENVIRONMENT=development
```

---

## pre-commit Hooks to Block Secret Commits

`pre-commit` runs checks before every `git commit`, blocking the commit if any check fails.

### Setup

```bash
pip install pre-commit
```

### .pre-commit-config.yaml

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: detect-private-key       # blocks RSA/PEM private keys
      - id: check-added-large-files  # blocks files > 500kb
      - id: check-merge-conflict
      - id: trailing-whitespace

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets           # scans for API keys, passwords, tokens
        args: ['--baseline', '.secrets.baseline']

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

```bash
pre-commit install      # installs git hook
pre-commit run --all-files  # run manually on all files
```

Once installed, `git commit` will automatically scan staged files. A detected API key pattern causes the commit to abort with a clear error message.
