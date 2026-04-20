"""
1.2 REST APIs — Working FastAPI App
A complete FastAPI app demonstrating AI engineering patterns.

Install: pip install fastapi uvicorn httpx python-dotenv pydantic
Run:     uvicorn main:app --reload --port 8000
Docs:    http://localhost:8000/docs
"""

import os
import time
import hashlib
import asyncio
from datetime import datetime
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator


# ─────────────────────────────────────────
# Startup / Shutdown (lifespan)
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Initialize shared resources on startup, clean up on shutdown."""
    app.state.http_client = httpx.AsyncClient(timeout=60.0)
    app.state.request_log: list[dict] = []
    print("✓ HTTP client initialized")
    yield
    await app.state.http_client.aclose()
    print("✓ HTTP client closed")


# ─────────────────────────────────────────
# App
# ─────────────────────────────────────────

app = FastAPI(
    title="AI Engineering Learning API",
    description="Demonstrates REST API patterns for AI apps",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Schemas (Pydantic models = validation + docs)
# ─────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v):
        if v not in ("system", "user", "assistant"):
            raise ValueError(f"Role must be system/user/assistant, got: {v}")
        return v


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = False

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_reasonable(cls, v):
        if v < 1 or v > 16000:
            raise ValueError("max_tokens must be between 1 and 16000")
        return v


class ChatResponse(BaseModel):
    id: str
    content: str
    model: str
    usage: dict
    created_at: datetime


class SummarizeRequest(BaseModel):
    text: str
    style: str = "bullet_points"    # bullet_points | paragraph | one_sentence

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if len(v.strip()) < 20:
            raise ValueError("Text must be at least 20 characters")
        return v.strip()


class SummarizeResponse(BaseModel):
    original_length: int
    summary: str
    style: str


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


# ─────────────────────────────────────────
# Dependencies (reusable auth / rate limiting)
# ─────────────────────────────────────────

_start_time = time.time()
_rate_limit_store: dict[str, list[float]] = {}


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    """
    Verify API key from header.
    In production: check against database, not a hardcoded value.
    """
    valid_keys = os.getenv("VALID_API_KEYS", "test-key-123").split(",")
    if not x_api_key or x_api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


async def rate_limiter(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    """Simple in-memory rate limiter: 10 requests per minute per API key."""
    key = x_api_key or request.client.host
    now = time.time()
    window = 60.0
    limit = 10

    # Clean old entries
    _rate_limit_store.setdefault(key, [])
    _rate_limit_store[key] = [t for t in _rate_limit_store[key] if now - t < window]

    if len(_rate_limit_store[key]) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {limit} requests per minute",
            headers={"Retry-After": "60"},
        )
    _rate_limit_store[key].append(now)


def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


# ─────────────────────────────────────────
# Simulated LLM call (swap for real API)
# ─────────────────────────────────────────

async def call_llm(messages: list[dict], model: str, max_tokens: int) -> dict:
    """
    In production, replace this with a real API call:

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={"model": model, "messages": messages, "max_tokens": max_tokens},
        )
        response.raise_for_status()
        return response.json()
    """
    await asyncio.sleep(0.1)    # simulate latency
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return {
        "id": f"chatcmpl-{hashlib.md5(last_user_msg.encode()).hexdigest()[:8]}",
        "choices": [{"message": {"content": f"[Simulated response to: {last_user_msg[:50]}]"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "model": model,
    }


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — used by load balancers and monitoring."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_api_key), Depends(rate_limiter)],
    summary="Multi-turn chat completion",
)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Send a conversation history and receive the next assistant message.
    Supports both regular and streaming responses.
    """
    messages = [m.model_dump() for m in request.messages]

    if request.stream:
        # Return a streaming response
        async def generate():
            words = f"[Streaming response to: {messages[-1]['content'][:30]}...]".split()
            for word in words:
                yield f"data: {word}\n\n"
                await asyncio.sleep(0.05)
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    raw = await call_llm(messages, request.model, request.max_tokens)

    # Log in background (don't block the response)
    background_tasks.add_task(
        log_request,
        endpoint="/chat",
        model=request.model,
        tokens=raw["usage"]["total_tokens"],
    )

    return ChatResponse(
        id=raw["id"],
        content=raw["choices"][0]["message"]["content"],
        model=raw["model"],
        usage=raw["usage"],
        created_at=datetime.now(),
    )


@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Summarize text in different styles",
)
async def summarize(request: SummarizeRequest):
    """
    Summarize a document in bullet_points, paragraph, or one_sentence style.
    Demonstrates prompt template injection.
    """
    style_prompts = {
        "bullet_points": "Summarize the following text as 3-5 bullet points:\n\n",
        "paragraph": "Write a concise paragraph summarizing:\n\n",
        "one_sentence": "Summarize in exactly one sentence:\n\n",
    }

    if request.style not in style_prompts:
        raise HTTPException(status_code=422, detail=f"Unknown style: {request.style}")

    prompt = style_prompts[request.style] + request.text
    messages = [
        {"role": "system", "content": "You are a helpful summarization assistant."},
        {"role": "user", "content": prompt},
    ]

    raw = await call_llm(messages, "gpt-4o-mini", max_tokens=500)
    summary = raw["choices"][0]["message"]["content"]

    return SummarizeResponse(
        original_length=len(request.text),
        summary=summary,
        style=request.style,
    )


@app.get(
    "/models",
    dependencies=[Depends(verify_api_key)],
    summary="List available models",
)
async def list_models() -> dict:
    """Return available models with pricing info."""
    return {
        "models": [
            {"id": "gpt-4o", "context_window": 128000, "cost_per_1k_input": 0.005},
            {"id": "gpt-4o-mini", "context_window": 128000, "cost_per_1k_input": 0.00015},
            {"id": "claude-sonnet-4-6", "context_window": 200000, "cost_per_1k_input": 0.003},
        ]
    }


# ─────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions — never expose raw tracebacks."""
    print(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An unexpected error occurred"},
    )


# ─────────────────────────────────────────
# Background Task
# ─────────────────────────────────────────

async def log_request(endpoint: str, model: str, tokens: int):
    """Log usage asynchronously (don't slow down the response)."""
    print(f"[LOG] {endpoint} | model={model} | tokens={tokens}")


# ─────────────────────────────────────────
# Run (for development)
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
