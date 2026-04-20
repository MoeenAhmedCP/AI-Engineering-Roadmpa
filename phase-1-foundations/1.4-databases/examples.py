"""
1.4 Databases — Working Examples
Demonstrates SQLAlchemy (async), Redis caching patterns, and chatbot history.
Uses SQLite in-memory so it runs without Postgres or Redis installed.

Run: python examples.py
Install: pip install sqlalchemy aiosqlite
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Optional


# ─────────────────────────────────────────
# 1. SQLAlchemy Async ORM with SQLite
# ─────────────────────────────────────────

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Float, select, func
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)   # user | assistant | system
    content = Column(Text, nullable=False)
    token_count = Column(Integer)
    model = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")


# Setup in-memory SQLite
engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def create_conversation(db: AsyncSession, user_id: int, title: str = "New Conversation") -> Conversation:
    convo = Conversation(user_id=user_id, title=title)
    db.add(convo)
    await db.flush()
    return convo


async def add_message(db: AsyncSession, conversation_id: int, role: str, content: str, model: str = None) -> Message:
    token_count = len(content.split()) * 4 // 3  # rough estimate
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        token_count=token_count,
        model=model,
    )
    db.add(msg)
    await db.flush()
    return msg


async def get_conversation_history(db: AsyncSession, conversation_id: int, limit: int = 20) -> list[dict]:
    """Fetch last N messages formatted for LLM API."""
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]


async def get_user_conversations(db: AsyncSession, user_id: int) -> list[Conversation]:
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
    )
    return result.scalars().all()


async def demo_orm():
    print("\n=== SQLAlchemy ORM Demo ===")
    await init_db()

    async with AsyncSessionLocal() as db:
        async with db.begin():
            # Create a conversation
            convo = await create_conversation(db, user_id=42, title="AI Questions Session")
            print(f"Created conversation: id={convo.id}")

            # Add messages (simulating a chat)
            await add_message(db, convo.id, "system", "You are a helpful AI assistant.")
            await add_message(db, convo.id, "user", "What is RAG?")
            await add_message(db, convo.id, "assistant",
                "RAG stands for Retrieval-Augmented Generation. It gives LLMs access to external knowledge.",
                model="claude-sonnet-4-6")
            await add_message(db, convo.id, "user", "How does it differ from fine-tuning?")
            await add_message(db, convo.id, "assistant",
                "RAG retrieves knowledge at query time; fine-tuning bakes knowledge into model weights.",
                model="claude-sonnet-4-6")

        # Fetch conversation history (ready for LLM context)
        async with db.begin():
            history = await get_conversation_history(db, convo.id)
            print(f"\nConversation history ({len(history)} messages):")
            for msg in history:
                print(f"  [{msg['role']}]: {msg['content'][:60]}...")


# ─────────────────────────────────────────
# 2. Redis-style Caching (dict stand-in)
# ─────────────────────────────────────────

class InMemoryRedis:
    """Simulates Redis: key-value store with TTL."""

    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}  # key → (value, expires_at)

    def set(self, key: str, value: str, ex: int = None):
        expires_at = time.time() + ex if ex else float("inf")
        self._store[key] = (value, expires_at)

    def get(self, key: str) -> Optional[str]:
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def incr(self, key: str) -> int:
        current = int(self.get(key) or 0) + 1
        ttl = None
        if key in self._store:
            _, expires_at = self._store[key]
            if expires_at != float("inf"):
                ttl = int(expires_at - time.time())
        self.set(key, str(current), ex=ttl)
        return current

    def expire(self, key: str, seconds: int):
        if key in self._store:
            value, _ = self._store[key]
            self._store[key] = (value, time.time() + seconds)

    def delete(self, key: str):
        self._store.pop(key, None)


redis = InMemoryRedis()


def make_cache_key(model: str, messages: list[dict]) -> str:
    """Deterministic SHA-256 cache key from model + messages."""
    content = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return f"llm:response:{hashlib.sha256(content.encode()).hexdigest()}"


def get_cached(key: str) -> Optional[dict]:
    raw = redis.get(key)
    return json.loads(raw) if raw else None


def set_cached(key: str, response: dict, ttl: int = 3600):
    redis.set(key, json.dumps(response), ex=ttl)


def check_rate_limit(user_id: int, limit: int = 10, window: int = 60) -> bool:
    """Returns True if under limit. Increments counter."""
    key = f"rate:{user_id}:{int(time.time() // window)}"
    count = redis.incr(key)
    if count == 1:
        redis.expire(key, window)
    return count <= limit


def demo_caching():
    print("\n=== Redis Caching Demo ===")

    model = "gpt-4o-mini"
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
    ]

    key = make_cache_key(model, messages)

    # Miss → call LLM → store
    cached = get_cached(key)
    print(f"Cache MISS: {cached is None}")

    fake_response = {"content": "RAG is Retrieval-Augmented Generation.", "tokens": 50}
    set_cached(key, fake_response, ttl=3600)

    # Hit → return from cache
    cached = get_cached(key)
    print(f"Cache HIT: {cached}")

    # Rate limiting demo
    print("\nRate limiting (limit=3):")
    for i in range(5):
        allowed = check_rate_limit(user_id=99, limit=3, window=60)
        print(f"  Request {i+1}: {'ALLOWED' if allowed else 'BLOCKED'}")


# ─────────────────────────────────────────
# 3. Full Chatbot Storage Flow
# ─────────────────────────────────────────

async def chatbot_turn(
    db: AsyncSession,
    conversation_id: int,
    user_message: str,
) -> str:
    """
    Simulates one turn of a chatbot:
    1. Save user message
    2. Fetch history for context
    3. Check cache / call LLM (simulated)
    4. Save assistant reply
    5. Return reply
    """
    # Save user message
    await add_message(db, conversation_id, "user", user_message)

    # Build context
    history = await get_conversation_history(db, conversation_id, limit=10)

    # Check cache
    cache_key = make_cache_key("gpt-4o-mini", history)
    cached = get_cached(cache_key)
    if cached:
        reply = cached["content"]
        print(f"  [CACHE HIT] Returning cached response")
    else:
        # Simulate LLM call
        reply = f"[Simulated response to: '{user_message[:40]}']"
        set_cached(cache_key, {"content": reply}, ttl=300)
        print(f"  [LLM CALL] Generated new response")

    # Save assistant reply
    await add_message(db, conversation_id, "assistant", reply, model="gpt-4o-mini")
    return reply


async def demo_full_chatbot():
    print("\n=== Full Chatbot Storage Flow ===")
    await init_db()

    async with AsyncSessionLocal() as db:
        async with db.begin():
            convo = await create_conversation(db, user_id=1, title="Test Session")

        # Simulate a multi-turn conversation
        questions = [
            "What is a transformer?",
            "How does attention work?",
            "What is a transformer?",   # repeated — should hit cache
        ]

        for q in questions:
            async with db.begin():
                print(f"\n  User: {q}")
                reply = await chatbot_turn(db, convo.id, q)
                print(f"  Bot:  {reply[:60]}")

        # Show final history
        async with db.begin():
            history = await get_conversation_history(db, convo.id)
            print(f"\n  Total messages stored: {len(history)}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(demo_orm())
    demo_caching()
    asyncio.run(demo_full_chatbot())
    print("\n✓ Done")
