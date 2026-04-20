# 1.8 Testing AI Applications

## pytest Fundamentals

pytest is the standard Python testing framework. It discovers tests automatically (files named `test_*.py`, functions named `test_*`), provides rich assertion introspection, and has a massive plugin ecosystem.

### Running Tests

```bash
pytest                          # run all tests
pytest tests/unit/              # run a specific directory
pytest -v                       # verbose: show each test name
pytest -k "test_parse"          # run tests whose name contains "parse"
pytest -x                       # stop after first failure
pytest --tb=short               # shorter traceback format
pytest -s                       # don't capture stdout (show print statements)
pytest test_examples.py -v      # run specific file
```

### Fixtures

Fixtures are reusable setup/teardown functions injected into tests by name. They replace setUp/tearDown from unittest.

```python
import pytest

@pytest.fixture
def db_connection():
    """Set up a test database connection before each test."""
    conn = create_test_db()
    yield conn           # test runs here
    conn.close()         # teardown after test

def test_insert_document(db_connection):
    doc = db_connection.insert({"title": "test"})
    assert doc["id"] is not None
```

### Fixture Scopes

| Scope | When Created | When Destroyed | Use For |
|---|---|---|---|
| `function` (default) | Before each test | After each test | DB rows, temp files |
| `module` | Once per test file | End of file | Expensive setup shared across file |
| `session` | Once per test run | End of test run | DB connections, HTTP clients |
| `class` | Once per test class | End of class | Class-level shared state |

```python
@pytest.fixture(scope="session")
def supabase_client():
    """One client for the entire test suite."""
    return create_client(url=TEST_URL, key=TEST_KEY)

@pytest.fixture(scope="module")
def uploaded_document(supabase_client):
    """Upload once, use across multiple tests in this file."""
    doc_id = upload_test_pdf(supabase_client)
    yield doc_id
    delete_document(supabase_client, doc_id)  # cleanup
```

### conftest.py

`conftest.py` is a special file that pytest automatically loads. Define fixtures here to share them across multiple test files without importing.

```
tests/
├── conftest.py              # fixtures available to all tests below
├── unit/
│   ├── conftest.py          # fixtures available to unit/ tests only
│   └── test_parser.py
├── integration/
│   └── test_api.py
```

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="session")
def client():
    return TestClient(app)

@pytest.fixture
def sample_pdf_bytes():
    with open("tests/fixtures/sample.pdf", "rb") as f:
        return f.read()
```

### Parametrize

Run the same test with multiple inputs in one declaration:

```python
@pytest.mark.parametrize("input_text,expected_token_count", [
    ("Hello world", 2),
    ("" , 0),
    ("a" * 1000, 250),  # ~4 chars per token
    ("GPT-4 is great!", 5),
])
def test_token_count(input_text, expected_token_count):
    count = estimate_token_count(input_text)
    assert abs(count - expected_token_count) <= 2  # within 2 tokens
```

### Custom Marks

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (> 1s)",
    "integration: marks integration tests that hit real APIs",
    "unit: marks fast unit tests",
]
```

```python
@pytest.mark.slow
def test_process_large_pdf():
    ...  # takes 10 seconds

@pytest.mark.integration
def test_upload_to_supabase():
    ...  # hits real Supabase instance

# Run only fast tests during development:
# pytest -m "not slow and not integration"
# Run everything in CI:
# pytest -m ""
```

---

## Mocking LLM API Calls

### Why Mock?

| Reason | Explanation |
|---|---|
| Cost | GPT-4o at $5/million input tokens — a test suite with 50 tests calling the real API costs money every run |
| Speed | Real API calls take 2–30 seconds. With mocking, the same test runs in milliseconds |
| Determinism | LLMs are non-deterministic. "assert response == expected" fails randomly without mocking |
| Offline | Tests should pass with no internet connection (CI environments, planes) |
| Rate limits | Repeated test runs can hit API rate limits and throttle your account |

### Mocking the Anthropic Client

```python
from unittest.mock import patch, MagicMock
import pytest

# The function under test (in services/claude.py):
def analyze_document(text: str) -> dict:
    from anthropic import Anthropic
    client = Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Analyze: {text}"}]
    )
    return parse_response(message.content[0].text)

# The test:
def test_analyze_document_returns_parsed_result():
    mock_response = MagicMock()
    mock_response.content[0].text = '{"summary": "Test doc", "confidence": 0.9}'

    with patch("services.claude.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = mock_response
        result = analyze_document("This is a test document.")

    assert result["summary"] == "Test doc"
    assert result["confidence"] == 0.9
    # Verify the API was called with the right parameters
    MockAnthropic.return_value.messages.create.assert_called_once()
```

### Mocking OpenAI

```python
@patch("openai.OpenAI")
def test_openai_call(MockOpenAI):
    mock_choice = MagicMock()
    mock_choice.message.content = '{"result": "parsed"}'

    MockOpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[mock_choice]
    )

    result = call_openai_api("Summarize this")
    assert result == {"result": "parsed"}
```

---

## What TO Test in AI Apps

### Test These Things

| Category | What to Test | Example |
|---|---|---|
| Input validation | Reject empty/too-long/malformed inputs | `assert raises ValueError for "" input` |
| Response parsing | Parse LLM JSON output correctly | Mock LLM, assert parsed fields match |
| Error handling | Graceful failure when API is down | Mock raises `APIError`, assert 503 returned |
| Retry logic | Retries on transient failures | Mock fails twice, succeeds third — assert 3 calls |
| Token counting | Estimates stay within reasonable bounds | `assert 10 <= count("hello world") <= 15` |
| Business logic | Confidence thresholds, routing rules | `assert route_to_human(confidence=0.3) == True` |
| API schemas | Request/response Pydantic validation | `assert raises ValidationError for missing field` |

### What NOT to Test

**Do not assert on the exact wording of LLM responses.** The model is non-deterministic and gets updated. Tests like `assert "red flag" in response.lower()` will flake.

Do not test that the LLM makes good decisions — test that your code correctly handles whatever the LLM returns.

Do not integration-test against real LLM APIs in your CI pipeline. Use mocks for unit tests, and run real API tests manually or in a separate expensive CI step.

---

## pytest-asyncio for Async FastAPI Route Testing

FastAPI routes are often async. Testing them requires either `TestClient` (which handles the event loop for you) or `pytest-asyncio` for testing async functions directly.

```python
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"   # auto-detect async test functions
```

```python
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_upload_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/api/upload",
            files={"file": ("test.txt", b"content", "text/plain")}
        )
    assert response.status_code == 200
    assert "document_id" in response.json()

@pytest.mark.asyncio
async def test_async_service_function():
    result = await my_async_service("input data")
    assert result is not None
```

### Using TestClient (synchronous — simpler for most cases)

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_upload_requires_file():
    response = client.post("/api/upload")  # no file
    assert response.status_code == 422     # Unprocessable Entity
```

---

## pytest-cov: Measuring Coverage

```bash
pip install pytest-cov

# Run tests with coverage report
pytest --cov=. --cov-report=term-missing

# Generate HTML report (open htmlcov/index.html)
pytest --cov=. --cov-report=html

# Fail the build if coverage drops below 70%
pytest --cov=. --cov-fail-under=70
```

### What Coverage Numbers Mean for AI Apps

| Coverage | Interpretation |
|---|---|
| < 50% | Large untested surface area — risky to deploy |
| 50–70% | Acceptable for early-stage apps |
| 70–80% | Good for most AI apps — happy path + common errors covered |
| 80–90% | Well-tested — most edge cases covered |
| > 90% | Diminishing returns — often achieved by testing trivial code |
| 100% | Usually a waste of time — line coverage doesn't guarantee correct behavior |

**For AI apps, prioritize testing**: input validation, parsing functions, retry/error handling, and API schema correctness. The LLM service layer is mocked, so focus on the code around it.

### pyproject.toml coverage config

```toml
[tool.coverage.run]
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/migrations/*",
    "main.py",          # startup code — tested via integration tests
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

---

## GitHub Actions CI: Running Tests on Every PR

Create `.github/workflows/test.yml` to automatically run tests, linting, and type checking on every pull request.

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip packages
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Lint with ruff
        run: ruff check . --output-format=github

      - name: Format check with ruff
        run: ruff format --check .

      - name: Type check with mypy
        run: mypy . --ignore-missing-imports
        continue-on-error: true  # Don't fail CI on type errors initially

      - name: Run tests with coverage
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          DATABASE_URL: "sqlite:///test.db"
          ENVIRONMENT: test
        run: |
          pytest tests/ \
            -v \
            --tb=short \
            --cov=. \
            --cov-report=term-missing \
            --cov-fail-under=70 \
            -m "not integration"

      - name: Upload coverage report
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          fail_ci_if_error: false
```

### Key CI Patterns for AI Apps

- **Never put real API keys in CI config** — use GitHub Secrets (`Settings → Secrets → Actions`)
- **Skip integration tests in CI** by default (`-m "not integration"`) — run them nightly or manually
- **Matrix testing** across Python versions catches compatibility issues early
- **Cache pip** to speed up CI runs from ~3 minutes to ~30 seconds
- **continue-on-error: true** for mypy lets you add type checking incrementally without blocking deploys
