# 3.6 LangChain and LlamaIndex

## LangChain Core Concepts

LangChain is a framework for building LLM-powered applications by composing reusable components. Its core model is a chain: a sequence of steps where the output of one step becomes the input of the next.

### The Chain Model

In early LangChain, chains were Python classes (`LLMChain`, `RetrievalQA`, `ConversationalRetrievalChain`). These worked but were opaque and hard to customize. LangChain now recommends LCEL — the LangChain Expression Language.

### LCEL — LangChain Expression Language

LCEL uses the pipe operator `|` to compose runnables, borrowing from Unix shell pipes. Every component in LangChain — prompts, models, parsers, retrievers — implements the `Runnable` interface with three methods: `invoke`, `stream`, and `batch`.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
    | ChatAnthropic(model="claude-sonnet-4-6")
    | StrOutputParser()
)

# Three ways to run:
result = chain.invoke({"topic": "transformers"})        # returns string
for chunk in chain.stream({"topic": "transformers"}):   # streaming
    print(chunk, end="")
results = chain.batch([{"topic": "RAG"}, {"topic": "agents"}])  # parallel
```

**RunnablePassthrough:** Passes the input through unchanged. Used to merge the original question back into the chain after retrieval.

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## Document Loaders

LangChain provides loaders that convert raw files into `Document` objects (content + metadata).

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

loader = TextLoader("report.txt")
docs = loader.load()  # returns list[Document]

pdf_loader = PyPDFLoader("contract.pdf")
pages = pdf_loader.load_and_split()

web_loader = WebBaseLoader("https://example.com/docs")
web_docs = web_loader.load()
```

Each `Document` has `.page_content` (the text) and `.metadata` (source, page number, etc.).

---

## Text Splitters

After loading, documents are split into chunks for embedding.

**RecursiveCharacterTextSplitter** — the default choice. Tries separators in order: `\n\n`, `\n`, `. `, space, falling back to characters. Respects natural text boundaries as much as possible within the size budget.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
```

**CharacterTextSplitter** — splits only on a single separator (default `\n\n`). Simpler but less robust.

---

## Vector Stores in LangChain

LangChain wraps all major vector stores with a uniform interface.

```python
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store = FAISS.from_documents(chunks, embeddings)

# Convert to retriever
retriever = store.as_retriever(search_kwargs={"k": 4})
docs = retriever.invoke("What is RAG?")
```

The `.as_retriever()` interface is what plugs into LCEL chains. You can swap FAISS for Chroma or Pinecone without changing the chain code.

---

## Memory

Memory lets a chain remember previous turns in a conversation.

**ConversationBufferMemory** — stores every message verbatim. Simple, but grows without bound. Suitable for short conversations.

**ConversationSummaryMemory** — uses an LLM to periodically summarize older history. Keeps token usage bounded at the cost of some detail loss.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "Hi"}, {"output": "Hello!"})
history = memory.load_memory_variables({})
```

In LCEL, memory is managed manually — you maintain the messages list and pass it into the chain on each turn.

---

## LlamaIndex Core Concepts

LlamaIndex (formerly GPT Index) is optimized specifically for indexing and querying over documents. Where LangChain is general-purpose, LlamaIndex is document-first.

### Key Abstractions

**SimpleDirectoryReader** — loads all files from a directory automatically, handling PDF, DOCX, TXT, Markdown, and others.

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader("./docs/").load_data()
index = VectorStoreIndex.from_documents(documents)
```

**VectorStoreIndex** — handles chunking, embedding, and storage in one step. Opinionated defaults that work well out of the box.

**QueryEngine** — the high-level interface for asking questions.

```python
engine = index.as_query_engine()
response = engine.query("What are the contract termination terms?")
print(response)           # answer
print(response.source_nodes)  # retrieved chunks with scores
```

**response_mode** — controls how retrieved nodes are combined:
- `compact` (default): fits as much context as possible into one LLM call.
- `tree_summarize`: hierarchical summarization for very long context.
- `no_text`: returns only retrieved nodes without LLM generation.

---

## LangChain vs LlamaIndex vs From Scratch

| | LangChain | LlamaIndex | From Scratch |
|---|---|---|---|
| Learning curve | Medium | Low–Medium | Low (for simple cases) |
| RAG pipeline | Explicit, flexible | Opinionated, fast | Full control |
| Debugging | Verbose mode, LangSmith | Response metadata | Print statements |
| Multi-step agents | Excellent | Good | Complex |
| Document loading | Many loaders | Excellent | Manual |
| Production use | Common | Common | Common for core paths |
| Version stability | Changes often | Changes often | Stable |
| Best for | Complex agents, custom chains | Document Q&A | Simple, fast, understandable |

**Recommendation:** Use LlamaIndex for document-centric Q&A applications where you want speed of iteration. Use LangChain when you need chains with complex branching, agents, or custom memory. Use from-scratch (as in 3.4) when you want to understand exactly what is happening or avoid framework overhead.

---

## When Frameworks Hurt

Frameworks add real costs:

1. **Magic:** When something breaks, the error originates 5 levels deep in library code you do not own. Stack traces are unreadable.
2. **Hard to debug:** The `verbose=True` flag helps, but you are still debugging someone else's abstraction.
3. **Version hell:** LangChain has broken backward compatibility multiple times. Upgrading `langchain` from 0.1 to 0.2 to 0.3 often requires rewriting chains.
4. **Unnecessary overhead:** If your RAG pipeline is 50 lines of Python, adding LangChain to express the same logic in 30 lines is a net loss once you factor in the dependency, the learning curve, and future upgrade risk.
5. **Leaky abstractions:** When the default behavior does not match your use case (custom metadata filtering, hybrid search, reranking), you fight the framework rather than build freely.

Use frameworks deliberately. Start without them, add them when they provide a concrete time saving that outweighs the cost.

---

## Debugging

**LangChain verbose mode:**
```python
llm = ChatAnthropic(model="claude-sonnet-4-6", verbose=True)
chain = prompt | llm | parser  # verbose propagates
```

**Callbacks:**
```python
from langchain.callbacks import StdOutCallbackHandler
chain.invoke({"question": "..."}, config={"callbacks": [StdOutCallbackHandler()]})
```

**LangSmith** — Anthropic's (LangChain's) cloud tracing platform. Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to see every LLM call, token count, latency, and intermediate output in a UI. Extremely useful for debugging multi-step chains and agents.

**LlamaIndex debugging:**
```python
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
```

Or inspect `response.source_nodes` to see exactly what was retrieved and scored.
