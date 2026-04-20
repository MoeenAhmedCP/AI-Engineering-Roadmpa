# 3.9 Conversation Memory

## Why LLMs Are Stateless

Every call to an LLM API is completely independent. The model has no awareness of previous requests. When you call `client.messages.create()`, you are sending a self-contained payload. The model processes it, returns a response, and forgets everything immediately.

This is a deliberate architectural choice: it makes serving scalable and cheap (no per-user server-side state), but it means that "memory" — any sense of continuity across turns — is entirely your responsibility as the application developer.

The naive solution is to stuff the entire conversation history into every API call. This works for short conversations. For longer ones, you hit context limits and see costs balloon. Designing memory well is one of the most consequential decisions in an LLM application.

---

## Buffer Memory: Simple, Linear Cost

The simplest approach: maintain a list of all messages and send the full list with every request.

```python
messages = []
messages.append({"role": "user",      "content": "My name is Alice."})
messages.append({"role": "assistant", "content": "Nice to meet you, Alice!"})
messages.append({"role": "user",      "content": "What's my name?"})
# Send all three messages → model answers "Alice"
```

**Pros:** Zero information loss. Model has full context. Easy to implement.

**Cons:** Token count grows with every turn. A 100-turn conversation can cost 10x more than a 10-turn one. For GPT-4 class models at $10/1M tokens, 100 turns × 1000 tokens/turn = $1 per conversation — unacceptable at scale.

**When to use:** Short, bounded conversations (customer support, Q&A sessions with natural endpoints). Fine if you control the conversation length.

---

## Window Memory: Fixed Cost, Fixed Horizon

Keep only the last N messages. When the buffer fills, drop the oldest messages (or oldest pairs of user+assistant turns).

```python
MAX_MESSAGES = 10
messages.append(new_message)
if len(messages) > MAX_MESSAGES:
    messages = messages[-MAX_MESSAGES:]
```

**Pros:** Bounded, predictable token cost. Easy to implement.

**Cons:** The model loses early context. If the user mentioned their name in turn 1 and you're now on turn 15, that's gone. Can cause the model to appear "forgetful."

**When to use:** Transactional interactions where only recent context matters — customer support for a specific order, step-by-step task completion.

---

## Summary Memory: Best of Both Worlds

Periodically compress old conversation history into a running summary, then keep only the summary plus the most recent N messages.

```
Summary so far: "User is Alice, a senior engineer at Acme Corp. She is building a RAG pipeline and wants to use Chroma as the vector store. We covered embedding strategies in the last session."

[Last 4 messages: current technical discussion]
```

The summary is regenerated every K turns by asking the model: "Summarise this conversation so far in 3-5 sentences, preserving all important facts and decisions."

**Pros:** Retains semantic content of old turns at low token cost. Summary grows slowly (logarithmically) as conversation grows.

**Cons:** Summarisation itself costs tokens and adds latency. Information can be lost if the summary omits details the model deemed unimportant. The model's summary may not match what you'd want preserved.

**When to use:** Long research or planning sessions, personal assistants, ongoing projects spanning multiple sessions.

---

## Entity Memory: Fact Tracking

Instead of storing all turns or a prose summary, extract and maintain a structured fact store — a dict of entities and their attributes.

```python
entities = {
    "user_name": "Alice",
    "company": "Acme Corp",
    "project": "RAG pipeline",
    "preferred_db": "Chroma",
}
```

Prepend this context to every request as a structured block. Update it after each turn by running a small extraction step.

**Pros:** Very token-efficient (only the facts, not the conversation). Easy to inspect, audit, and edit. Can be persisted in a structured database.

**Cons:** Only captures explicit facts. Misses tone, reasoning, narrative arc. The extraction step can miss or misattribute facts.

**When to use:** Customer profiles, preference tracking, CRM-style assistants where facts about the user matter more than the conversation flow.

---

## Vector Memory: Semantic Retrieval

Embed every conversation turn as a vector. When building context for a new request, retrieve the K most semantically similar past turns.

```python
# On each new user message:
query_embedding = embed(new_message)
relevant_turns = vector_db.search(query_embedding, k=5)
context = format_turns(relevant_turns) + [new_message]
```

**Pros:** Can recall relevant information from arbitrarily long conversation histories. Naturally surfaces the most relevant past context, not just the most recent.

**Cons:** Requires a vector store. Embedding step adds latency. Retrieval can miss relevant turns if the query is phrased differently. Complex to debug ("why did the model remember X but not Y?").

**When to use:** Very long-running sessions (many hours, many topics), research assistants, personal AI that spans weeks of interaction.

---

## Database Persistence: Multi-Session Memory

For memory that survives beyond a single session, store conversations in a database (PostgreSQL, SQLite, or a document store).

Schema:

```sql
CREATE TABLE conversations (
    id          UUID PRIMARY KEY,
    session_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL,          -- 'user' | 'assistant' | 'system'
    content     TEXT NOT NULL,
    token_count INT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON conversations(session_id, created_at);
CREATE INDEX ON conversations(user_id, created_at);
```

On each request:
1. Load messages for this `session_id` (with window/summary applied).
2. Send to the LLM.
3. Append both the user message and assistant response to the DB.

**Pros:** Memory survives server restarts, user can return days later. Enables analytics over conversation data.

**Cons:** DB read on every request adds latency (~5-20ms for indexed queries). Need to manage storage growth.

---

## Multi-User Isolation

Always scope memory to `user_id`. Never let one user's conversation bleed into another's.

```python
class ConversationStore:
    def __init__(self):
        self._store: dict[str, list] = {}  # {session_id: [messages]}

    def get(self, session_id: str) -> list:
        return self._store.get(session_id, [])

    def add(self, session_id: str, role: str, content: str):
        self._store.setdefault(session_id, []).append(
            {"role": role, "content": content}
        )
```

For production, use `user_id + session_id` as the compound key to allow multiple conversations per user.

---

## Token Budget Management

Before sending messages to the API, count tokens and trim if needed. A rough heuristic: `n_tokens ≈ len(text.split()) * 1.3`. For exact counts, use the tokenizer library for your model.

Trim strategy:
1. Remove oldest messages first (preserve system prompt and most recent turns).
2. If a single message exceeds the budget, truncate its content.
3. Log when trimming occurs — frequent trimming signals that window size or summarisation frequency needs adjustment.

---

## When to Use Which Memory Type

| Use case | Recommended memory |
|---|---|
| Short customer support chat | Buffer or Window |
| Long research session (1 hour) | Summary + Window |
| Personal assistant (ongoing weeks) | Vector + DB persistence |
| Preference/profile tracking | Entity |
| Multi-turn task completion | Window |
| High-volume, cost-sensitive chat | Window (small N) |
| Audit-required conversations | DB persistence (all turns) |

The right choice is rarely one type — production systems often layer them: entity memory for facts, window for recent context, summary for medium-term, DB for persistence.
