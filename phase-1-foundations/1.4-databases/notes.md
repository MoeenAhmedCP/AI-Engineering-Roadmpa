# 1.4 Databases — SQL + NoSQL

## Which Database for What

| Database | Type | Use in AI Apps |
|---|---|---|
| PostgreSQL | Relational (SQL) | User data, conversation history, document metadata |
| Redis | Key-value store | Caching LLM responses, rate limiting, sessions |
| MongoDB | Document store | Semi-structured logs, flexible schemas |
| pgvector | Vector (SQL extension) | Storing embeddings in Postgres |
| Pinecone / Qdrant | Vector DB | Dedicated vector search at scale |

**Starting rule:** Use PostgreSQL for everything. Add Redis when you need caching. Add a vector DB when Postgres can't handle your embedding workload (millions of vectors with < 100ms latency).

---

## SQL Essentials

```sql
-- Basic SELECT
SELECT id, text, created_at
FROM documents
WHERE status = 'done'
ORDER BY created_at DESC
LIMIT 10;

-- JOIN — combine tables
SELECT d.id, d.filename, a.summary, a.confidence_score
FROM documents d
INNER JOIN analyses a ON d.id = a.document_id
WHERE d.user_id = 42;

-- LEFT JOIN — keep all documents even if no analysis yet
SELECT d.id, d.filename, a.summary
FROM documents d
LEFT JOIN analyses a ON d.id = a.document_id;

-- GROUP BY — aggregate
SELECT model, COUNT(*) as calls, AVG(latency_ms) as avg_latency, SUM(cost_usd) as total_cost
FROM api_logs
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY model
HAVING COUNT(*) > 100
ORDER BY total_cost DESC;

-- Upsert (insert or update)
INSERT INTO users (id, email, plan)
VALUES (1, 'user@example.com', 'free')
ON CONFLICT (id)
DO UPDATE SET plan = EXCLUDED.plan, updated_at = NOW();
```

---

## Indexes

Without an index, every query scans the entire table. With an index, it jumps straight to matching rows.

```sql
-- Always index foreign keys
CREATE INDEX idx_analyses_document_id ON analyses(document_id);

-- Index columns you filter on frequently
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_status ON documents(status);

-- Composite index — when you always filter by both columns together
CREATE INDEX idx_logs_user_date ON api_logs(user_id, created_at DESC);

-- Partial index — index only a subset of rows (efficient)
CREATE INDEX idx_documents_pending ON documents(created_at)
WHERE status = 'uploaded';

-- Check if queries are using indexes
EXPLAIN ANALYZE SELECT * FROM documents WHERE user_id = 42;
```

**Rule:** add an index if you see `Seq Scan` in `EXPLAIN ANALYZE` on a large table.

---

## ACID Properties

| Property | Meaning | Why It Matters |
|---|---|---|
| **Atomicity** | All or nothing — a transaction either fully succeeds or fully rolls back | Prevent partial state (e.g., charge user but don't update their plan) |
| **Consistency** | DB moves from one valid state to another | Constraints (NOT NULL, FOREIGN KEY) are never violated |
| **Isolation** | Concurrent transactions don't interfere | Two users updating the same row don't corrupt each other |
| **Durability** | Committed data survives crashes | Written to disk before confirming success |

```python
# Atomic transaction — both operations succeed or neither does
async with db.transaction():
    await db.execute("UPDATE wallets SET balance = balance - 10 WHERE user_id = $1", user_id)
    await db.execute("INSERT INTO transactions (user_id, amount) VALUES ($1, $2)", user_id, 10)
```

---

## SQLAlchemy ORM

```python
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
from sqlalchemy.sql import func
from datetime import datetime

DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/mydb"
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    content = Column(Text)
    status = Column(String(50), default="uploaded")
    user_id = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())

    analyses = relationship("Analysis", back_populates="document", cascade="all, delete-orphan")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    summary = Column(Text)
    confidence_score = Column(Float)
    tokens_used = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())

    document = relationship("Document", back_populates="analyses")


# Dependency for FastAPI
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# CRUD operations
from sqlalchemy import select


async def create_document(db: AsyncSession, filename: str, user_id: int) -> Document:
    doc = Document(filename=filename, user_id=user_id)
    db.add(doc)
    await db.flush()     # get the generated ID without committing
    return doc


async def get_document(db: AsyncSession, doc_id: int) -> Document | None:
    result = await db.execute(select(Document).where(Document.id == doc_id))
    return result.scalar_one_or_none()


async def get_user_documents(db: AsyncSession, user_id: int) -> list[Document]:
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
        .limit(50)
    )
    return result.scalars().all()


async def update_document_status(db: AsyncSession, doc_id: int, status: str) -> None:
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if doc:
        doc.status = status
```

---

## Redis — Caching LLM Responses

Redis is perfect for AI apps because LLM responses are expensive and often repeated (same question → same answer).

```python
import redis.asyncio as redis
import hashlib
import json

redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)


def make_cache_key(model: str, messages: list[dict]) -> str:
    """Deterministic key from model + conversation."""
    content = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return f"llm:response:{hashlib.sha256(content.encode()).hexdigest()}"


async def get_cached_response(key: str) -> dict | None:
    cached = await redis_client.get(key)
    return json.loads(cached) if cached else None


async def cache_response(key: str, response: dict, ttl_seconds: int = 3600) -> None:
    await redis_client.setex(key, ttl_seconds, json.dumps(response))


# Usage
async def cached_llm_call(model: str, messages: list[dict]) -> dict:
    key = make_cache_key(model, messages)

    # Check cache first
    cached = await get_cached_response(key)
    if cached:
        print("Cache HIT")
        return cached

    # Call the actual LLM
    print("Cache MISS — calling API")
    response = await call_openai(model, messages)

    # Store in cache
    await cache_response(key, response, ttl_seconds=3600)
    return response


# Rate limiting with Redis
async def check_rate_limit(user_id: int, limit: int = 100, window: int = 3600) -> bool:
    """Returns True if under limit, False if exceeded."""
    key = f"rate_limit:{user_id}"
    count = await redis_client.incr(key)
    if count == 1:
        await redis_client.expire(key, window)
    return count <= limit
```

---

## Designing a Chatbot History Schema

```sql
CREATE TABLE users (
    id          BIGSERIAL PRIMARY KEY,
    email       VARCHAR(255) UNIQUE NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE conversations (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       VARCHAR(255),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    token_count     INTEGER,
    model           VARCHAR(100),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at ASC);
CREATE INDEX idx_conversations_user ON conversations(user_id, updated_at DESC);
```

```python
# Fetch conversation history for LLM context
async def get_conversation_messages(db, conversation_id: int, limit: int = 20) -> list[dict]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = result.scalars().all()
    # Reverse to chronological order
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]
```

---

## When to Use What

| Scenario | Database Choice | Why |
|---|---|---|
| Store user conversation history | PostgreSQL | Relational, ACID, query with SQL |
| Cache identical LLM responses | Redis | Fast, TTL, exact match |
| Cache semantically similar queries | Qdrant + Redis | Vector similarity + fast lookup |
| Store document chunks + embeddings | pgvector or Pinecone | Vector search on embeddings |
| Store raw uploaded files | S3 / Supabase Storage | Object storage, not a DB |
| Feature flags, A/B test config | Redis | Fast reads, easy updates |
