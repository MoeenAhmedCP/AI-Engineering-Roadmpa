# 1.2 REST APIs & HTTP

## The Mental Model

HTTP is a request-response protocol. Your AI app makes HTTP calls constantly — to OpenAI, to your own API, to Supabase. Understanding the mechanics prevents debugging in the dark.

```
Client                          Server
  │──── POST /analyze ─────────▶│
  │     Headers: {Authorization} │  ← who are you?
  │     Body: {text: "..."}      │  ← what do you want?
  │                              │
  │◀─── 200 OK ─────────────────│
  │     Body: {summary: "..."}   │
```

---

## HTTP Methods

| Method | Use Case | Has Body? | Idempotent? |
|---|---|---|---|
| GET | Retrieve data | No | Yes |
| POST | Create / trigger action | Yes | No |
| PUT | Replace a resource entirely | Yes | Yes |
| PATCH | Update specific fields | Yes | No |
| DELETE | Remove a resource | No | Yes |

**AI-specific pattern:** You'll mostly use `POST` because LLM calls are actions (trigger analysis, generate text), not data retrieval.

---

## Status Codes You Must Know

```
2xx — Success
  200 OK              — standard success
  201 Created         — resource was created (after POST)
  204 No Content      — success, no body (after DELETE)

4xx — Client error (your fault)
  400 Bad Request     — malformed request, invalid JSON
  401 Unauthorized    — missing/invalid auth
  403 Forbidden       — authenticated but not permitted
  404 Not Found       — resource doesn't exist
  422 Unprocessable   — valid JSON but fails validation (FastAPI default for bad input)
  429 Too Many Requests — rate limited ← common with LLM APIs

5xx — Server error (their fault)
  500 Internal Error  — server crashed
  502 Bad Gateway     — proxy error
  503 Service Unavail — server down / overloaded
```

---

## Anatomy of a Request

```http
POST /api/analyze HTTP/1.1
Host: api.myapp.com
Authorization: Bearer sk-ant-...
Content-Type: application/json
X-Request-ID: abc123

{
  "document_id": 42,
  "model": "claude-sonnet-4-6",
  "options": {"temperature": 0.2}
}
```

- **Headers** — metadata about the request (auth, content type, custom)
- **Body** — the payload (JSON in AI apps almost always)
- **Query params** — `?page=2&limit=10` — for filtering/pagination on GET

---

## FastAPI — The Standard for AI Apps

FastAPI is preferred over Flask because:
1. **Async native** — `async def` routes work out of the box
2. **Pydantic validation** — request/response schemas validated automatically
3. **Auto docs** — `/docs` (Swagger) generated from your code
4. **Type hints** → validation + editor support

### Basic FastAPI Structure

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI()

class AnalyzeRequest(BaseModel):
    text: str
    max_tokens: int = 1000

class AnalyzeResponse(BaseModel):
    summary: str
    tokens_used: int

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    if len(request.text) < 10:
        raise HTTPException(status_code=422, detail="Text too short")
    # call Claude/OpenAI here
    return AnalyzeResponse(summary="...", tokens_used=500)
```

### Dependency Injection

```python
from fastapi import Header

async def verify_api_key(x_api_key: str = Header()):
    if x_api_key != "secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze(request: AnalyzeRequest):
    ...
```

---

## Authentication Patterns

### API Key (simplest — used by OpenAI, Anthropic)
```python
# Header-based
Authorization: Bearer sk-ant-...

# Python
headers = {"Authorization": f"Bearer {api_key}"}
```

### JWT Bearer Token
- Stateless — server doesn't store session
- Contains: user ID, expiry, scope
- Verify with a secret key

### OAuth2 (when acting on behalf of a user)
- Get authorization code → exchange for access token → use token
- More complex, needed for Google, GitHub integrations

---

## Making HTTP Calls in Python

```python
import httpx
import asyncio

# Synchronous (fine for scripts, bad in async contexts)
response = httpx.get("https://api.example.com/data",
                     headers={"Authorization": "Bearer token"},
                     timeout=30.0)
response.raise_for_status()    # raises on 4xx/5xx
data = response.json()

# Async (use in FastAPI routes)
async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data",
                                    timeout=30.0)
        response.raise_for_status()
        return response.json()
```

---

## Rate Limiting & Retries

OpenAI and Anthropic rate limit by tokens-per-minute and requests-per-minute. You will hit these in production.

```python
import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True,
)
async def call_with_retry(client: httpx.AsyncClient, payload: dict) -> dict:
    response = await client.post("https://api.openai.com/v1/chat/completions",
                                 json=payload, timeout=60)
    if response.status_code == 429:
        retry_after = int(response.headers.get("retry-after", 5))
        await asyncio.sleep(retry_after)
        raise httpx.HTTPStatusError("Rate limited", request=response.request, response=response)
    response.raise_for_status()
    return response.json()
```

**Exponential backoff:** wait 1s, then 2s, then 4s, then 8s... — prevents hammering an already-overloaded API.

---

## JSON Serialization

```python
import json
from datetime import datetime

# Python dict → JSON string
data = {"id": 1, "text": "hello", "created": datetime.now()}
# json.dumps(data)  # fails — datetime not serializable

# Custom encoder
class DatetimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

json_str = json.dumps(data, cls=DatetimeEncoder)

# In FastAPI, Pydantic handles this automatically
from pydantic import BaseModel

class Response(BaseModel):
    id: int
    created: datetime        # automatically serialized to ISO string
```

---

## Error Handling in APIs

```python
from fastapi import Request
from fastapi.responses import JSONResponse

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": str(exc),
            "path": str(request.url)
        }
    )

# Per-route error handling
@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        result = await call_claude(request.text)
        return result
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Rate limited. Retry in {e.retry_after}s")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
```
