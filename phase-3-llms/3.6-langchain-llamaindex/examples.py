"""
3.6 LangChain & LlamaIndex — Examples

SIMULATE=True (default): uses stub classes, no installs required.
SIMULATE=False: uses real LangChain/LlamaIndex if installed.

Run: python examples.py
"""

import os
import json
from typing import Any, Callable, Iterator

SIMULATE = os.getenv("SIMULATE", "true").lower() != "false"

# ---------------------------------------------------------------------------
# Try to import real libraries; fall back to stubs gracefully
# ---------------------------------------------------------------------------

LANGCHAIN_AVAILABLE = False
LLAMAINDEX_AVAILABLE = False

if not SIMULATE:
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from langchain_community.vectorstores import FAISS
        from langchain.memory import ConversationBufferMemory
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        print("[INFO] langchain not installed; using stubs. pip install langchain langchain-community")

    try:
        from llama_index.core import VectorStoreIndex, Document as LlamaDocument
        LLAMAINDEX_AVAILABLE = True
    except ImportError:
        print("[INFO] llama_index not installed; using stubs. pip install llama-index")


# ---------------------------------------------------------------------------
# Shared Document type (mirrors LangChain's Document)
# ---------------------------------------------------------------------------

class Document:
    """Simple document with content and metadata."""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        source = self.metadata.get("source", "?")
        return f"Document(source={source!r}, content={self.page_content[:50]!r}...)"


# ---------------------------------------------------------------------------
# Stub LLM — deterministic mock, no network calls
# ---------------------------------------------------------------------------

class StubLLM:
    """
    Mimics LangChain's ChatModel interface.
    invoke(prompt) returns a deterministic mock string.
    """

    def invoke(self, prompt: Any) -> str:
        # Extract text from various input types
        if isinstance(prompt, str):
            text = prompt
        elif isinstance(prompt, list):
            # Simulate chat messages list
            text = " ".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in prompt
            )
        else:
            text = str(prompt)

        key = text[:40].strip()
        return f"[StubLLM] Answer for: '{key}...'"

    def stream(self, prompt: Any) -> Iterator[str]:
        words = self.invoke(prompt).split()
        for word in words:
            yield word + " "

    def __or__(self, other):
        """Support pipe syntax: llm | parser"""
        return _PipedRunnable(self, other)


# ---------------------------------------------------------------------------
# Stub vector store
# ---------------------------------------------------------------------------

class StubVectorStore:
    """
    In-memory vector store stub with naive keyword matching for similarity.
    """

    def __init__(self, documents: list[Document] = None):
        self._docs = documents or []

    def add_documents(self, docs: list[Document]):
        self._docs.extend(docs)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return docs whose content shares the most words with the query."""
        query_words = set(query.lower().split())

        def score(doc: Document) -> int:
            doc_words = set(doc.page_content.lower().split())
            return len(query_words & doc_words)

        ranked = sorted(self._docs, key=score, reverse=True)
        return ranked[:k]

    def as_retriever(self, search_kwargs: dict = None) -> "StubRetriever":
        k = (search_kwargs or {}).get("k", 4)
        return StubRetriever(self, k)


class StubRetriever:
    def __init__(self, store: StubVectorStore, k: int = 4):
        self._store = store
        self._k = k

    def invoke(self, query: str) -> list[Document]:
        return self._store.similarity_search(query, k=self._k)


# ---------------------------------------------------------------------------
# Stub output parser
# ---------------------------------------------------------------------------

class StubStrOutputParser:
    """Passes through string output unchanged."""

    def invoke(self, text: Any) -> str:
        return str(text)

    def __or__(self, other):
        return _PipedRunnable(self, other)


# ---------------------------------------------------------------------------
# Pipe composition helper
# ---------------------------------------------------------------------------

class _PipedRunnable:
    """Allows (a | b).invoke(x) by chaining .invoke calls."""

    def __init__(self, left, right):
        self._left = left
        self._right = right

    def invoke(self, inputs: Any) -> Any:
        intermediate = self._left.invoke(inputs)
        return self._right.invoke(intermediate)

    def stream(self, inputs: Any) -> Iterator[Any]:
        # Stream from left, collect, then stream from right
        intermediate = self._left.invoke(inputs)
        if hasattr(self._right, "stream"):
            yield from self._right.stream(intermediate)
        else:
            yield self._right.invoke(intermediate)

    def __or__(self, other):
        return _PipedRunnable(self, other)


# ---------------------------------------------------------------------------
# Stub prompt template
# ---------------------------------------------------------------------------

class StubPromptTemplate:
    def __init__(self, template: str, input_variables: list[str] = None):
        self.template = template
        self.input_variables = input_variables or []

    def invoke(self, variables: dict) -> str:
        result = self.template
        for k, v in variables.items():
            # Support both {key} and {{key}} style
            if isinstance(v, list):
                v = "\n".join(
                    doc.page_content if hasattr(doc, "page_content") else str(doc)
                    for doc in v
                )
            result = result.replace("{" + k + "}", str(v))
        return result

    def __or__(self, other):
        return _PipedRunnable(self, other)

    @classmethod
    def from_template(cls, template: str) -> "StubPromptTemplate":
        return cls(template)


# ---------------------------------------------------------------------------
# 1. Build a RAG chain
# ---------------------------------------------------------------------------

def build_rag_chain(documents: list[str], simulate: bool = True) -> Callable[[str], str]:
    """
    Build a RAG chain from a list of raw text strings.

    Returns:
        A callable that takes a question string and returns an answer string.
    """
    doc_objects = [
        Document(page_content=text, metadata={"source": f"doc_{i}"})
        for i, text in enumerate(documents)
    ]

    if not simulate and LANGCHAIN_AVAILABLE:
        # Real LangChain implementation
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        store = FAISS.from_documents(doc_objects, embeddings)
        retriever = store.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template(
            "Answer the question using only the context below.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        llm = ChatAnthropic(model="claude-sonnet-4-6")
        parser = StrOutputParser()

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | parser
        )
        return lambda q: chain.invoke(q)

    else:
        # Stub implementation
        store = StubVectorStore(doc_objects)
        retriever = store.as_retriever(search_kwargs={"k": 3})
        prompt = StubPromptTemplate.from_template(
            "Answer the question using only the context below.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        llm = StubLLM()
        parser = StubStrOutputParser()

        def answer(question: str) -> str:
            retrieved = retriever.invoke(question)
            ctx = "\n".join(d.page_content for d in retrieved)
            formatted = prompt.invoke({"context": ctx, "question": question})
            raw = llm.invoke(formatted)
            return parser.invoke(raw)

        return answer


def demo_rag_chain():
    print("\n" + "=" * 60)
    print("DEMO 1: RAG Chain")
    print("=" * 60)

    docs = [
        "Python was created by Guido van Rossum and first appeared in 1991. It emphasizes code readability.",
        "FastAPI is a modern web framework for building APIs with Python. It uses type hints and async support.",
        "Docker containers are lightweight, portable execution environments that include all dependencies.",
        "PostgreSQL is a reliable open-source relational database with support for JSON and vector search.",
        "RAG combines retrieval with generation to ground LLM responses in real documents.",
    ]

    chain = build_rag_chain(docs, simulate=SIMULATE)

    questions = [
        "Who created Python?",
        "What does FastAPI use for validation?",
    ]

    for q in questions:
        answer = chain(q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")


# ---------------------------------------------------------------------------
# 2. LCEL-style chain demo
# ---------------------------------------------------------------------------

def lcel_demo(simulate: bool = SIMULATE):
    """
    Demonstrates LCEL pipe-syntax chain composition:
      prompt | llm | output_parser
    """
    print("\n" + "=" * 60)
    print("DEMO 2: LCEL Chain Composition (prompt | llm | parser)")
    print("=" * 60)

    if not simulate and LANGCHAIN_AVAILABLE:
        from langchain_anthropic import ChatAnthropic
        prompt = ChatPromptTemplate.from_template("Explain {topic} in exactly one sentence.")
        llm = ChatAnthropic(model="claude-sonnet-4-6")
        parser = StrOutputParser()
        chain = prompt | llm | parser
    else:
        prompt = StubPromptTemplate.from_template("Explain {topic} in exactly one sentence.")
        llm = StubLLM()
        parser = StubStrOutputParser()
        chain = prompt | llm | parser

    topics = ["vector embeddings", "prompt caching", "tokenization"]
    for topic in topics:
        result = chain.invoke({"topic": topic})
        print(f"  [{topic}]: {result}")

    print("\nStreaming output for 'RAG systems':")
    print("  ", end="")
    for chunk in chain.stream({"topic": "RAG systems"}):
        print(chunk, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 3. Conversation chain with memory
# ---------------------------------------------------------------------------

def conversation_chain_demo(simulate: bool = SIMULATE):
    """
    Multi-turn conversation with memory.
    Shows 3 exchanges where later answers reference earlier context.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Conversation Chain with Memory")
    print("=" * 60)

    conversation_history: list[dict] = []
    llm = StubLLM()

    def chat(user_message: str) -> str:
        conversation_history.append({"role": "user", "content": user_message})

        if not simulate and LANGCHAIN_AVAILABLE:
            pass  # Real implementation would use ConversationBufferMemory

        # For stub: include history context in prompt
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in conversation_history[:-1]
        )
        prompt_text = (
            f"Previous conversation:\n{history_str}\n\n"
            f"User: {user_message}\nAssistant:"
        ) if history_str else f"User: {user_message}\nAssistant:"

        response = llm.invoke(prompt_text)
        conversation_history.append({"role": "assistant", "content": response})
        return response

    exchanges = [
        "Hi! I'm learning about machine learning.",
        "What should I learn first — supervised or unsupervised learning?",
        "Can you give me a concrete example of supervised learning?",
    ]

    for turn, user_msg in enumerate(exchanges, start=1):
        print(f"\nTurn {turn}:")
        print(f"  User:      {user_msg}")
        response = chat(user_msg)
        print(f"  Assistant: {response}")

    print(f"\n[Memory] Stored {len(conversation_history)} messages in history.")


# ---------------------------------------------------------------------------
# 4. Framework comparison
# ---------------------------------------------------------------------------

def compare_frameworks():
    print("\n" + "=" * 60)
    print("DEMO 4: Framework Comparison Table")
    print("=" * 60)

    rows = [
        ("Feature",             "LangChain",          "LlamaIndex",         "From Scratch"),
        ("-" * 22,              "-" * 20,             "-" * 20,             "-" * 18),
        ("Primary focus",       "General chains",     "Document Q&A",       "Whatever you need"),
        ("RAG setup speed",     "Medium",             "Fast",               "Manual"),
        ("Debugging",           "verbose/LangSmith",  "source_nodes",       "print()"),
        ("Agent support",       "Excellent",          "Good",               "Complex"),
        ("Doc loaders",         "Many types",         "Excellent",          "Write your own"),
        ("Version stability",   "Changes often",      "Changes often",      "You own it"),
        ("Learning curve",      "Medium",             "Low-Medium",         "Low (for basics)"),
        ("Best for",            "Complex agents",     "Document Q&A",       "Simple/core paths"),
    ]

    col_widths = [22, 20, 20, 18]
    for row in rows:
        line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(f"  {line}")

    print("\nKey takeaway:")
    print("  Use frameworks deliberately. Start without them, add only when the")
    print("  time savings outweigh the debugging cost and version instability.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("3.6 LangChain & LlamaIndex — Examples")
    print(f"Mode:      {'SIMULATE (stubs, no installs needed)' if SIMULATE else 'REAL'}")
    print(f"LangChain: {'available' if LANGCHAIN_AVAILABLE else 'not installed (using stubs)'}")
    print(f"LlamaIndex:{'available' if LLAMAINDEX_AVAILABLE else 'not installed (using stubs)'}")
    print("=" * 60)

    demo_rag_chain()
    lcel_demo()
    conversation_chain_demo()
    compare_frameworks()

    print("\n" + "=" * 60)
    print("All demos complete.")
    if SIMULATE:
        print("To use real frameworks:")
        print("  pip install langchain langchain-anthropic langchain-community langchain-openai faiss-cpu")
        print("  pip install llama-index")
        print("  SIMULATE=false ANTHROPIC_API_KEY=sk-ant-... python examples.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
