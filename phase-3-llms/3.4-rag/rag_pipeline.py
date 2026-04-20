"""
3.4 RAG Pipeline — Complete Implementation

Set SIMULATE=True (default) to run without API keys.
Set SIMULATE=False and provide ANTHROPIC_API_KEY to use real Claude.

Run: python rag_pipeline.py
"""

import os
import math
import json
import hashlib
from typing import Optional

SIMULATE = os.getenv("SIMULATE", "true").lower() != "false"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Document:
    """A chunk of text with metadata."""

    def __init__(self, content: str, metadata: Optional[dict] = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        source = self.metadata.get("source", "unknown")
        chunk_id = self.metadata.get("chunk_id", "?")
        preview = self.content[:60].replace("\n", " ")
        return f"Document(source={source!r}, chunk={chunk_id}, content={preview!r}...)"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source: str = "unknown",
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[Document]:
    """
    Split text into overlapping fixed-size chunks.

    Args:
        text:       Full document text.
        source:     Source identifier (filename, URL, etc.)
        chunk_size: Maximum characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Returns:
        List of Document objects.
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to end on a sentence boundary to avoid splitting mid-sentence
        if end < len(text):
            for boundary in (". ", ".\n", "! ", "? ", "\n\n", "\n"):
                boundary_pos = text.rfind(boundary, start, end)
                if boundary_pos != -1 and boundary_pos > start + overlap:
                    end = boundary_pos + len(boundary)
                    break

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append(Document(
                content=chunk_content,
                metadata={
                    "source": source,
                    "chunk_id": chunk_id,
                    "start_char": start,
                    "end_char": end,
                },
            ))
            chunk_id += 1

        # Advance with overlap
        start = end - overlap
        if start >= len(text) - overlap:
            break

    return chunks


# ---------------------------------------------------------------------------
# Vector store (in-memory cosine similarity)
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Simple in-memory vector store using cosine similarity.

    SIMULATE=True  → deterministic hash-based 64-dim fake embeddings (no API).
    SIMULATE=False → calls OpenAI-compatible embeddings endpoint via Anthropic SDK
                     (or you can swap in any embeddings API).
    """

    DIMS = 64  # dimensionality used in simulation mode

    def __init__(self):
        self._documents: list[Document] = []
        self._vectors: list[list[float]] = []

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _hash_embed(self, text: str) -> list[float]:
        """
        Deterministic fake embedding: hash chunks of the text into floats.
        Similar words don't cluster meaningfully — this is only for demo purposes.
        """
        vector = []
        # Use multiple hash seeds to fill DIMS dimensions
        for i in range(self.DIMS):
            seed = f"{i}:{text}"
            digest = hashlib.md5(seed.encode()).hexdigest()
            # Convert first 8 hex chars to a float in [-1, 1]
            raw = int(digest[:8], 16)
            vector.append((raw / 0x7FFFFFFF) - 1.0)
        return self._normalize(vector)

    def _real_embed(self, text: str) -> list[float]:
        """Call a real embeddings API. Requires anthropic or openai package."""
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            # Anthropic does not yet expose an embeddings endpoint;
            # fall back to openai-compatible if available.
            raise NotImplementedError("Anthropic embeddings not yet in SDK; use OpenAI.")
        except (ImportError, NotImplementedError):
            pass

        try:
            import openai  # type: ignore
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except ImportError:
            print("[WARNING] openai package not installed; falling back to hash embedding.")
            return self._hash_embed(text)

    def embed(self, text: str) -> list[float]:
        if SIMULATE:
            return self._hash_embed(text)
        return self._real_embed(text)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0:
            return vector
        return [x / magnitude for x in vector]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            # Pad shorter vector with zeros
            length = max(len(a), len(b))
            a = a + [0.0] * (length - len(a))
            b = b + [0.0] * (length - len(b))
        dot = sum(x * y for x, y in zip(a, b))
        return dot  # already normalized vectors → dot product = cosine sim

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[Document]) -> None:
        """Embed and store a list of documents."""
        for doc in docs:
            vec = self.embed(doc.content)
            self._documents.append(doc)
            self._vectors.append(vec)
        print(f"[VectorStore] Indexed {len(docs)} documents. Total: {len(self._documents)}.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """
        Return the k most similar documents to the query.

        Returns:
            List of (Document, similarity_score) tuples, sorted by score descending.
        """
        if not self._documents:
            return []

        query_vec = self.embed(query)
        scores = [
            (doc, self._cosine_similarity(query_vec, vec))
            for doc, vec in zip(self._documents, self._vectors)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_rag_prompt(question: str, context_docs: list[Document]) -> str:
    """
    Build the full augmented prompt to send to the LLM.

    Injects retrieved context chunks with source citations and instructs
    the model to answer only from the provided context.
    """
    context_blocks = []
    for i, doc in enumerate(context_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", i)
        context_blocks.append(
            f"[Source {i}: {source}, chunk {chunk_id}]\n{doc.content}"
        )

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the context does not contain sufficient information to answer, respond with:
"I don't have enough information in the provided documents to answer this question."

Cite your sources by referencing [Source N] inline.

--- CONTEXT ---
{context_str}
--- END CONTEXT ---

Question: {question}

Answer:"""
    return prompt


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _mock_llm_response(prompt: str) -> str:
    """
    Simulate an LLM response by extracting key phrases from the context.
    Not intelligent — purely for demo without API keys.
    """
    # Extract context section
    if "--- CONTEXT ---" in prompt and "--- END CONTEXT ---" in prompt:
        start = prompt.index("--- CONTEXT ---") + len("--- CONTEXT ---")
        end = prompt.index("--- END CONTEXT ---")
        context = prompt[start:end].strip()
        # Return first 200 chars of context as a "summary"
        summary = context[:300].split("\n")
        # Filter out source lines
        content_lines = [l for l in summary if not l.startswith("[Source")]
        answer_text = " ".join(content_lines).strip()[:200]
        return f"Based on [Source 1]: {answer_text}... (simulated answer)"
    return "I don't have enough information in the provided documents to answer this question."


def _real_llm_response(prompt: str) -> str:
    """Call Claude via the Anthropic SDK."""
    try:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except ImportError:
        print("[ERROR] anthropic package not installed. Run: pip install anthropic")
        return _mock_llm_response(prompt)
    except KeyError:
        print("[ERROR] ANTHROPIC_API_KEY not set in environment.")
        return _mock_llm_response(prompt)


def answer_question(question: str, store: VectorStore, k: int = 4) -> str:
    """
    Full RAG query pipeline:
      1. Retrieve top-k relevant chunks
      2. Build augmented prompt
      3. Call LLM (real or mock)
      4. Return answer
    """
    results = store.search(question, k=k)

    if not results:
        return "No documents in the knowledge base."

    context_docs = [doc for doc, score in results]

    # Show retrieval info
    print(f"\n  [Retrieval] Top {len(results)} chunks for: {question!r}")
    for doc, score in results:
        source = doc.metadata.get("source", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        print(f"    score={score:.4f}  source={source}  chunk={chunk_id}")

    prompt = build_rag_prompt(question, context_docs)

    if SIMULATE:
        return _mock_llm_response(prompt)
    else:
        return _real_llm_response(prompt)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_DOCUMENTS = [
    {
        "source": "python_overview.txt",
        "content": (
            "Python is a high-level, interpreted programming language created by Guido van Rossum "
            "and first released in 1991. Python's design philosophy emphasizes code readability. "
            "Python supports multiple programming paradigms, including structured, object-oriented, "
            "and functional programming. It uses dynamic typing and garbage collection. "
            "Python is widely used in web development, data science, artificial intelligence, "
            "scientific computing, and automation scripting."
        ),
    },
    {
        "source": "fastapi_guide.txt",
        "content": (
            "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on "
            "standard Python type hints. It was created by Sebastian Ramirez and released in 2018. "
            "FastAPI is built on top of Starlette for the web parts and Pydantic for the data parts. "
            "Key features include automatic OpenAPI documentation generation, dependency injection, "
            "async support with asyncio, and extremely high performance comparable to NodeJS and Go. "
            "FastAPI automatically validates request data using Pydantic models."
        ),
    },
    {
        "source": "docker_basics.txt",
        "content": (
            "Docker is a platform for developing, shipping, and running applications in containers. "
            "A container is a lightweight, standalone, executable software package that includes "
            "everything needed to run an application: code, runtime, system tools, libraries, and settings. "
            "Docker containers are isolated from each other and the host system. "
            "The Docker image is a read-only template used to create containers. "
            "Images are built from a Dockerfile, which contains a sequence of instructions. "
            "Docker Hub is the default public registry for Docker images."
        ),
    },
    {
        "source": "postgres_notes.txt",
        "content": (
            "PostgreSQL is a powerful, open source object-relational database system with over 35 years "
            "of active development. It is known for reliability, feature robustness, and performance. "
            "PostgreSQL supports advanced data types including JSON, JSONB, arrays, hstore, and geometric types. "
            "JSONB stores JSON data in a decomposed binary format, which is slower to input but significantly "
            "faster to process. PostgreSQL also supports full-text search, table inheritance, "
            "window functions, and common table expressions (CTEs). "
            "The pgvector extension adds vector similarity search capabilities to PostgreSQL."
        ),
    },
    {
        "source": "rag_concepts.txt",
        "content": (
            "Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model. "
            "The retrieval system finds relevant documents from a knowledge base using semantic search. "
            "Semantic search converts text to vector embeddings and finds similar vectors. "
            "The retrieved documents are injected into the LLM prompt as context. "
            "This allows the LLM to answer questions about private or recent data not in its training set. "
            "RAG reduces hallucination because the model has actual source text to reference. "
            "Common vector stores used in RAG systems include FAISS, Chroma, Pinecone, and pgvector."
        ),
    },
]

DEMO_QUESTIONS = [
    "Who created FastAPI and what are its key features?",
    "What is the difference between JSONB and JSON in PostgreSQL?",
    "How does RAG help reduce hallucination in LLMs?",
]


def demo():
    print("=" * 65)
    print("3.4 RAG Pipeline Demo")
    print(f"Mode: {'SIMULATE (no API keys needed)' if SIMULATE else 'REAL (calling Claude)'}")
    print("=" * 65)

    # ----- Build knowledge base -----
    print("\n[1/3] Building knowledge base...")
    store = VectorStore()

    all_chunks: list[Document] = []
    for doc_data in DEMO_DOCUMENTS:
        chunks = chunk_text(
            doc_data["content"],
            source=doc_data["source"],
            chunk_size=300,
            overlap=50,
        )
        all_chunks.extend(chunks)
        print(f"  {doc_data['source']}: {len(chunks)} chunk(s)")

    store.add_documents(all_chunks)

    # ----- Ask questions -----
    print("\n[2/3] Answering questions...\n")
    for i, question in enumerate(DEMO_QUESTIONS, start=1):
        print(f"Q{i}: {question}")
        answer = answer_question(question, store, k=3)
        print(f"\nA{i}: {answer}")
        print("-" * 65)

    # ----- Show prompt structure -----
    print("\n[3/3] Example prompt structure:")
    sample_results = store.search("What is Python?", k=2)
    sample_docs = [doc for doc, _ in sample_results]
    sample_prompt = build_rag_prompt("What is Python?", sample_docs)
    print(sample_prompt[:600] + "\n... [truncated]")

    print("\n[Done] RAG pipeline demo complete.")
    print("To use real Claude: SIMULATE=false ANTHROPIC_API_KEY=sk-ant-... python rag_pipeline.py")


if __name__ == "__main__":
    demo()
