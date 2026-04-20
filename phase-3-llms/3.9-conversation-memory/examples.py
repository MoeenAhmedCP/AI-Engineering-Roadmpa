"""
3.9 Conversation Memory — Examples
====================================
Implements five memory strategies without any real API calls.

Run: python examples.py
"""

import re
import time
from typing import Optional

SIMULATE = True

# Token estimation: ~1.3 tokens per whitespace-delimited word
def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))

def _total_tokens(messages: list[dict]) -> int:
    return sum(_estimate_tokens(m.get("content", "")) for m in messages)


# ===========================================================================
# 1. BufferMemory — keeps the full history
# ===========================================================================

class BufferMemory:
    """Stores every message. Simple; token cost grows linearly."""

    def __init__(self):
        self._messages: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def token_count(self) -> int:
        return _total_tokens(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()


# ===========================================================================
# 2. WindowMemory — keeps the last N messages
# ===========================================================================

class WindowMemory(BufferMemory):
    """Keeps only the most recent `max_messages` messages."""

    def __init__(self, max_messages: int = 6):
        super().__init__()
        self.max_messages = max_messages

    def add_message(self, role: str, content: str) -> None:
        super().add_message(role, content)
        if len(self._messages) > self.max_messages:
            dropped = len(self._messages) - self.max_messages
            self._messages = self._messages[-self.max_messages:]
            # (In production you'd log the dropped count)


# ===========================================================================
# 3. SummaryMemory — summarises older turns, keeps recent ones
# ===========================================================================

class SummaryMemory:
    """
    Keeps a running prose summary of old messages plus the last
    `keep_recent` messages verbatim.

    compress_older() is called when history grows beyond a threshold.
    """

    def __init__(self, keep_recent: int = 4, compress_after: int = 8):
        self._messages: list[dict] = []
        self._summary: str = ""
        self.keep_recent = keep_recent
        self.compress_after = compress_after

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) >= self.compress_after:
            self.compress_older(self._mock_llm_summarise)

    def compress_older(self, llm_fn) -> None:
        """Summarise all but the last `keep_recent` messages."""
        to_compress = self._messages[:-self.keep_recent]
        recent      = self._messages[-self.keep_recent:]
        if not to_compress:
            return
        combined_text = self._summary + "\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in to_compress
        )
        self._summary = llm_fn(combined_text)
        self._messages = recent

    def _mock_llm_summarise(self, text: str) -> str:
        """Simulates an LLM summarisation call."""
        lines = [l for l in text.splitlines() if l.strip()]
        snippet = "; ".join(l[:60] for l in lines[:3])
        return f"[Summary] The conversation covered: {snippet}..."

    def get_messages(self) -> list[dict]:
        """Returns summary as a system message + recent messages."""
        result = []
        if self._summary:
            result.append({"role": "system", "content": self._summary})
        result.extend(self._messages)
        return result

    def token_count(self) -> int:
        return _total_tokens(self.get_messages())

    def add_message_raw(self, role: str, content: str) -> None:
        """Add without triggering compression — useful for testing."""
        self._messages.append({"role": role, "content": content})


# ===========================================================================
# 4. EntityMemory — extract and track named facts
# ===========================================================================

class EntityMemory:
    """
    Regex-based entity extraction. Tracks names, emails, and numbers
    mentioned across the conversation.
    """

    _EMAIL_RE   = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    _PHONE_RE   = re.compile(r"\+?[\d][\d\s\-().]{7,14}[\d]")
    # Capitalised word sequences (crude name heuristic — good enough for demo)
    _NAME_RE    = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b")

    def __init__(self):
        self._entities: dict[str, list] = {
            "names":  [],
            "emails": [],
            "phones": [],
            "numbers": [],
        }

    def extract_entities(self, text: str) -> dict:
        """Return entities found in `text`."""
        found = {
            "names":   list(set(self._NAME_RE.findall(text))),
            "emails":  list(set(self._EMAIL_RE.findall(text))),
            "phones":  list(set(self._PHONE_RE.findall(text))),
        }
        return found

    def update_entities(self, text: str) -> None:
        found = self.extract_entities(text)
        for key, values in found.items():
            for v in values:
                if v not in self._entities.get(key, []):
                    self._entities.setdefault(key, []).append(v)

    def get_context(self) -> str:
        """Returns a compact string summarising known entities."""
        parts = []
        if self._entities["names"]:
            parts.append("Names: " + ", ".join(self._entities["names"]))
        if self._entities["emails"]:
            parts.append("Emails: " + ", ".join(self._entities["emails"]))
        if self._entities["phones"]:
            parts.append("Phones: " + ", ".join(self._entities["phones"]))
        if not parts:
            return "(no entities extracted yet)"
        return "[Known entities] " + " | ".join(parts)

    @property
    def entities(self) -> dict:
        return dict(self._entities)


# ===========================================================================
# 5. ConversationStore — multi-user session isolation
# ===========================================================================

class ConversationStore:
    """
    In-memory store keyed by session_id.
    In production, replace the dict with a DB (Postgres, Redis, etc.).
    """

    def __init__(self):
        self._store: dict[str, list[dict]] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._store.setdefault(session_id, []).append(
            {"role": role, "content": content, "ts": time.time()}
        )

    def get_messages(self, session_id: str) -> list[dict]:
        return list(self._store.get(session_id, []))

    def list_sessions(self) -> list[str]:
        return list(self._store.keys())

    def session_token_count(self, session_id: str) -> int:
        return _total_tokens(self.get_messages(session_id))

    def delete_session(self, session_id: str) -> None:
        self._store.pop(session_id, None)


# ===========================================================================
# 6. simulate_conversation helper
# ===========================================================================

def simulate_conversation(memory, turns: list[dict]) -> None:
    """
    Walk through a list of {role, content} turns, add each to memory,
    and print what the model would see (get_messages()) at each user turn.
    """
    for i, turn in enumerate(turns, 1):
        role    = turn["role"]
        content = turn["content"]

        # Add to memory
        if hasattr(memory, "add_message"):
            memory.add_message(role, content)
        elif isinstance(memory, SummaryMemory):
            memory.add_message(role, content)

        # Only print the model's context view on user turns
        if role == "user":
            msgs = memory.get_messages()
            tok  = memory.token_count()
            print(f"  Turn {i:02d} ({role}): \"{content[:50]}\"")
            print(f"           → model sees {len(msgs)} messages | ~{tok} tokens")


# ===========================================================================
# Main demo
# ===========================================================================

# A 10-turn conversation for demos
DEMO_TURNS = [
    {"role": "user",      "content": "Hi, I'm Alice Nguyen from Acme Corp. alice@acmecorp.com."},
    {"role": "assistant", "content": "Hello Alice! How can I help you today?"},
    {"role": "user",      "content": "I'm building a RAG pipeline using Chroma as the vector store."},
    {"role": "assistant", "content": "Great choice. Chroma works well for local and production RAG setups."},
    {"role": "user",      "content": "I'm having trouble with chunking strategy — what size do you recommend?"},
    {"role": "assistant", "content": "For most documents, 512 tokens with 50-token overlap is a solid default."},
    {"role": "user",      "content": "Does that change for code files? I'm indexing Python source code."},
    {"role": "assistant", "content": "Yes — for code, chunk at function/class boundaries rather than fixed tokens."},
    {"role": "user",      "content": "Good tip. What embedding model pairs best with Chroma?"},
    {"role": "assistant", "content": "text-embedding-3-small is fast and cheap; nomic-embed-text is a strong open-source option."},
]


def demo_buffer():
    print("=" * 60)
    print("DEMO: BufferMemory (full history)")
    print("=" * 60)
    mem = BufferMemory()
    simulate_conversation(mem, DEMO_TURNS)
    print(f"  Final: {len(mem)} messages | ~{mem.token_count()} tokens\n")


def demo_window():
    print("=" * 60)
    print("DEMO: WindowMemory (last 4 messages)")
    print("=" * 60)
    mem = WindowMemory(max_messages=4)
    simulate_conversation(mem, DEMO_TURNS)
    print(f"  Final: {len(mem)} messages | ~{mem.token_count()} tokens\n")


def demo_summary():
    print("=" * 60)
    print("DEMO: SummaryMemory (compress after 6 messages)")
    print("=" * 60)
    mem = SummaryMemory(keep_recent=4, compress_after=6)
    simulate_conversation(mem, DEMO_TURNS)
    msgs = mem.get_messages()
    print(f"  Final: {len(msgs)} messages | ~{mem.token_count()} tokens")
    if msgs and msgs[0]["role"] == "system":
        print(f"  Summary: {msgs[0]['content']}\n")


def demo_entity():
    print("=" * 60)
    print("DEMO: EntityMemory (fact extraction)")
    print("=" * 60)
    mem = EntityMemory()
    for turn in DEMO_TURNS:
        mem.update_entities(turn["content"])
    print(f"  Entities found:")
    for key, vals in mem.entities.items():
        if vals:
            print(f"    {key}: {vals}")
    print(f"  Context string: {mem.get_context()}\n")


def demo_store():
    print("=" * 60)
    print("DEMO: ConversationStore (multi-user isolation)")
    print("=" * 60)
    store = ConversationStore()

    # Two separate users, each with their own session
    for i, turn in enumerate(DEMO_TURNS[:6]):
        store.add_message("session-alice", turn["role"], turn["content"])
    for turn in DEMO_TURNS[6:]:
        store.add_message("session-bob", turn["role"], turn["content"])

    for sid in store.list_sessions():
        msgs = store.get_messages(sid)
        toks = store.session_token_count(sid)
        print(f"  Session '{sid}': {len(msgs)} messages | ~{toks} tokens")
    print()


if __name__ == "__main__":
    demo_buffer()
    demo_window()
    demo_summary()
    demo_entity()
    demo_store()

    print("=" * 60)
    print("All conversation-memory demos complete.")
    print("=" * 60)
