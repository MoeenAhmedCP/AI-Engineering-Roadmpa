# AI Engineering Roadmap
## From Beginner to Interview-Ready

> **How to use this file with Claude Code:**  
> Open this file in your IDE. As you study each topic, ask Claude Code to explain concepts, generate practice exercises, build mini-projects, and quiz you. Tick off checkboxes as you complete topics. Every section has suggested prompts you can run directly in Claude Code.

---

## What is an AI Engineer?

An AI engineer **builds products powered by AI models**. They don't train models from scratch (that's an ML researcher) — they know enough to fine-tune, prompt, evaluate, and deploy them reliably at scale.

| Role | Main Focus | Key Skills |
|---|---|---|
| **AI Engineer** | Building products with AI models | APIs, prompting, RAG, deployment, evaluation |
| **ML Engineer** | Training and optimizing models | PyTorch, training loops, distributed training |
| **Data Scientist** | Analysis and model selection | Statistics, notebooks, experimentation |

**Timeline:** ~22 weeks at 2–3 hours/day. Adjust to your pace.

---

## The 5-Phase Journey

```
Phase 1 (Weeks 1–3)   → Software Foundations
Phase 2 (Weeks 4–7)   → Core AI & ML Concepts  
Phase 3 (Weeks 8–12)  → LLMs & Modern AI APIs       ← Most job postings live here
Phase 4 (Weeks 13–17) → Production AI Systems
Phase 5 (Weeks 18–22) → Advanced Topics
```

---

## Phase 1 — Software Foundations
**Weeks 1–3 · Prerequisites every AI engineer must own**

> Don't skip this even if you know some Python. AI engineers who skip foundations end up unable to deploy or debug anything they build.

### 1.1 Python Proficiency

- [ ] Lists, dicts, sets, tuples — and when to use each
- [ ] Classes, inheritance, dunder methods (`__init__`, `__repr__`, `__len__`)
- [ ] Decorators (`@property`, `@staticmethod`, custom decorators)
- [ ] List/dict comprehensions, generator expressions
- [ ] `async`/`await` — essential for non-blocking LLM calls
- [ ] Type hints (`str`, `Optional[str]`, `list[dict]`, `TypedDict`)
- [ ] Context managers (`with` statement, `__enter__`/`__exit__`)
- [ ] Error handling (`try/except/finally`, custom exceptions)
- [ ] NumPy basics — array operations, broadcasting, slicing
- [ ] Pandas basics — DataFrames, groupby, filtering, reading CSV/JSON

**What to build:** A data processing script that reads a CSV, cleans it, filters rows, and outputs a summary.

**Claude Code prompts to try:**
```
"Explain Python decorators to me and show me 3 practical examples I'd use in AI engineering"
"Give me 10 Python exercises focused on async/await, from beginner to intermediate"
"What's the difference between list and generator comprehensions? When should I use each?"
"Review my Python code and suggest improvements for readability and performance"
```

**Resources:**
- Python official docs — https://docs.python.org
- "Fluent Python" by Luciano Ramalho (book)
- Real Python — https://realpython.com

---

### 1.2 REST APIs & HTTP

- [ ] HTTP methods: GET, POST, PUT, PATCH, DELETE — and when to use each
- [ ] Status codes: 200, 201, 400, 401, 403, 404, 422, 429, 500
- [ ] Request/response anatomy: headers, body, query params
- [ ] Authentication: API keys, Bearer tokens, OAuth basics
- [ ] Build a simple API with **FastAPI** (preferred for AI) or Flask
- [ ] Call external APIs with `httpx` or `requests`
- [ ] Handle rate limits, retries, and timeouts gracefully
- [ ] Understand JSON serialization/deserialization

**What to build:** A FastAPI endpoint that accepts a text body, calls the OpenAI API, and returns the response. Add proper error handling and rate limiting.

**Claude Code prompts to try:**
```
"Build me a FastAPI app with 3 endpoints: health check, text summarization using OpenAI, and conversation chat. Include proper error handling."
"Explain HTTP status codes with examples specific to AI API development"
"Show me how to implement retry logic with exponential backoff for API calls in Python"
```

**Resources:**
- FastAPI docs — https://fastapi.tiangolo.com
- HTTP in depth — https://developer.mozilla.org/en-US/docs/Web/HTTP

---

### 1.3 Git & Version Control

- [ ] Basic workflow: `init`, `add`, `commit`, `push`, `pull`
- [ ] Branching: `branch`, `checkout`, `merge`, `rebase`
- [ ] Pull requests and code review workflow
- [ ] Resolving merge conflicts
- [ ] `.gitignore` — never commit `.env`, model weights, `__pycache__`
- [ ] Writing meaningful commit messages (conventional commits)
- [ ] Git tags and releases

**Claude Code prompts to try:**
```
"What should always be in a .gitignore for an AI engineering Python project?"
"Explain git rebase vs merge with examples. When should I use each?"
"Show me a good git branching strategy for an AI project with multiple engineers"
```

---

### 1.4 Databases — SQL + NoSQL Basics

- [ ] SQL: `SELECT`, `WHERE`, `JOIN` (INNER, LEFT, RIGHT), `GROUP BY`, `HAVING`, `ORDER BY`
- [ ] Indexes — why they matter, how to choose columns to index
- [ ] Transactions and ACID properties
- [ ] PostgreSQL via psycopg2 or SQLAlchemy ORM
- [ ] SQLAlchemy — defining models, sessions, queries
- [ ] MongoDB basics — documents, collections, queries (optional but useful)
- [ ] Redis — key-value store, TTL, use cases (caching, sessions, queues)
- [ ] When to use SQL vs NoSQL vs a vector database

**What to build:** Store conversation history from a chatbot in PostgreSQL using SQLAlchemy. Add a Redis cache layer for frequently asked questions.

**Claude Code prompts to try:**
```
"Design a PostgreSQL schema for storing multi-turn LLM conversation history with user metadata"
"Show me how to use SQLAlchemy ORM to create, read, update, and delete records in Python"
"When would I use Redis instead of PostgreSQL in an AI application? Give concrete examples"
"Write SQL queries to analyze chatbot usage: messages per day, average session length, most common topics"
```

---

### 1.5 Docker & Containers

- [ ] What containers solve (reproducibility, "works on my machine" problem)
- [ ] Writing a `Dockerfile` for a Python AI app
- [ ] Build, run, tag, and push images (`docker build`, `docker run`, `docker push`)
- [ ] `docker-compose` for multi-service setups (app + Redis + Postgres)
- [ ] Environment variables in Docker (`ENV`, `--env-file`)
- [ ] Multi-stage builds for smaller images
- [ ] Common gotchas: file permissions, networking between containers

**What to build:** Dockerize your FastAPI AI app from 1.2. Add a `docker-compose.yml` that runs the app + Redis + PostgreSQL together.

**Claude Code prompts to try:**
```
"Write a Dockerfile for a Python FastAPI app that calls OpenAI. Optimize for small image size."
"Write a docker-compose.yml for: FastAPI app, PostgreSQL, Redis. Include environment variables and health checks."
"What are the most common Docker mistakes Python developers make?"
```

---

### 1.6 Environment Management & Secrets

- [ ] Python virtual environments: `venv`, `conda`, `poetry`
- [ ] `requirements.txt` vs `pyproject.toml`
- [ ] `.env` files and `python-dotenv` — loading secrets from environment
- [ ] Never hardcode API keys — understand why and how to enforce it
- [ ] Secrets in production: AWS Secrets Manager, HashiCorp Vault basics
- [ ] `pre-commit` hooks to catch secrets before committing

**Claude Code prompts to try:**
```
"Show me the proper way to manage API keys and secrets in a Python AI project, from development to production"
"Set up a pre-commit hook that prevents committing .env files or hardcoded secrets"
```

---

### 1.7 Cloud Basics

- [ ] Core concepts: regions, availability zones, VPCs
- [ ] EC2 — launch a server, SSH in, deploy an app
- [ ] S3 — store and retrieve files programmatically with boto3
- [ ] IAM — roles, policies, principle of least privilege
- [ ] Lambda basics — serverless functions for AI tasks
- [ ] ECS or App Runner — deploy Docker containers
- [ ] Cost awareness — understand what costs money and how to estimate bills

**Claude Code prompts to try:**
```
"Walk me through deploying a FastAPI AI app on AWS using ECS Fargate, step by step"
"Show me how to use boto3 to upload files to S3, generate presigned URLs, and download files"
"What AWS services does an AI engineer use most? Explain each one's role."
```

---

### 1.8 Testing & Code Quality

- [ ] `pytest` — writing unit tests, fixtures, parametrize
- [ ] Mocking — `unittest.mock`, mocking OpenAI API calls in tests
- [ ] Integration tests vs unit tests — when to use each
- [ ] Test coverage with `pytest-cov`
- [ ] Linting with `ruff` or `flake8`
- [ ] Type checking with `mypy`
- [ ] CI with GitHub Actions — run tests on every PR

**What to build:** Write tests for your FastAPI AI app. Mock the OpenAI API calls so tests run without hitting the real API.

**Claude Code prompts to try:**
```
"Write pytest tests for a FastAPI app that calls OpenAI. Show me how to mock the API calls."
"Set up a GitHub Actions workflow that runs tests, linting, and type checking on every PR"
"What should I test in an AI application? What's hard to test and how do engineers handle it?"
```

---

## Phase 2 — Core AI & ML Concepts
**Weeks 4–7 · The "why" behind everything**

> You don't need to implement backpropagation by hand. But if you don't understand how models learn, you'll never know why your model is hallucinating, why it drifts over time, or how to debug failures. This is what separates good AI engineers from great ones in interviews.

### 2.1 How Machine Learning Works

- [ ] Supervised vs unsupervised vs reinforcement learning
- [ ] The training loop: data → model → prediction → loss → gradient → update
- [ ] Loss functions: MSE for regression, cross-entropy for classification
- [ ] Gradient descent: batch, stochastic (SGD), mini-batch
- [ ] Learning rate — too high (diverges), too low (slow), learning rate schedules
- [ ] Overfitting vs underfitting — the bias-variance tradeoff
- [ ] Train/validation/test splits — why you need all three
- [ ] Regularization: L1 (Lasso), L2 (Ridge), dropout
- [ ] Hyperparameters vs parameters — what's the difference

**Claude Code prompts to try:**
```
"Explain gradient descent to me visually using a simple example. How does a model actually learn?"
"What is overfitting? Show me a Python example that demonstrates it and how regularization fixes it"
"Explain the bias-variance tradeoff with a concrete example. How does it affect model selection?"
"Quiz me on machine learning fundamentals with 10 questions at increasing difficulty"
```

---

### 2.2 Neural Networks Fundamentals

- [ ] Neurons: inputs × weights + bias → activation function → output
- [ ] Activation functions: ReLU, sigmoid, softmax, tanh — and when to use each
- [ ] Layers: input, hidden, output
- [ ] Forward pass: compute output from input
- [ ] Backpropagation: compute gradients using chain rule
- [ ] Batch normalization — what it does and why it helps
- [ ] Dropout — random neurons turned off during training
- [ ] Universal approximation theorem — what it means intuitively

**Claude Code prompts to try:**
```
"Build a simple neural network from scratch in pure Python (no libraries) that learns XOR. Walk me through every step."
"Explain backpropagation intuitively. How does the gradient actually flow backwards?"
"What is the vanishing gradient problem? Why does it occur and how do modern architectures fix it?"
```

---

### 2.3 Key ML Algorithms

- [ ] **Linear regression** — predict continuous values, understand coefficients
- [ ] **Logistic regression** — binary classification, sigmoid output as probability
- [ ] **Decision trees** — splitting on features, information gain, Gini impurity
- [ ] **Random forests** — ensemble of trees, feature importance
- [ ] **Gradient boosting** (XGBoost, LightGBM) — sequential tree ensemble
- [ ] **k-Means clustering** — unsupervised grouping
- [ ] **k-NN** — classification by nearest neighbours
- [ ] **SVM** — maximum margin classifier (conceptual understanding)
- [ ] Feature engineering — encoding categoricals, scaling, creating new features

**Claude Code prompts to try:**
```
"Show me a full scikit-learn example: train a random forest, evaluate it, tune hyperparameters, interpret feature importance"
"When would I use gradient boosting vs a neural network? Walk through the decision."
"Build a k-means clustering example from scratch. When is clustering useful in AI products?"
```

---

### 2.4 NLP Basics

- [ ] Why text needs to become numbers for models
- [ ] Tokenization: character, word, subword (BPE)
- [ ] Vocabulary and out-of-vocabulary (OOV) words
- [ ] Word embeddings: word2vec, GloVe — how similar words cluster in space
- [ ] Cosine similarity — measuring distance between vectors
- [ ] TF-IDF — term frequency-inverse document frequency for search
- [ ] N-grams — sequences of tokens
- [ ] Stop words, stemming, lemmatization

**Claude Code prompts to try:**
```
"Explain how word2vec trains word embeddings. What does 'king - man + woman = queen' actually mean mathematically?"
"Show me how to compute TF-IDF from scratch and when it's better than embeddings for search"
"What is tokenization in LLMs? How does GPT's BPE tokenizer work? Show me examples of surprising tokenizations."
```

---

### 2.5 Model Evaluation Metrics

- [ ] **Accuracy** — and why it's misleading on imbalanced data
- [ ] **Precision** — of what I predicted positive, how many were correct
- [ ] **Recall** — of actual positives, how many did I find
- [ ] **F1 score** — harmonic mean of precision and recall
- [ ] **AUC-ROC** — model ranking ability across thresholds
- [ ] **Confusion matrix** — TP, FP, TN, FN
- [ ] **BLEU score** — for machine translation evaluation
- [ ] **ROUGE score** — for text summarization evaluation
- [ ] **Perplexity** — how surprised a language model is by text

**Claude Code prompts to try:**
```
"I have 99% negative examples, 1% positive. Why is accuracy useless here? What metrics should I use instead?"
"Walk me through how to choose evaluation metrics for: a spam classifier, a document summarizer, a chatbot"
"Show me how to compute a confusion matrix and all derived metrics in Python with a real example"
```

---

### 2.6 PyTorch Basics

- [ ] Tensors — create, reshape, slice, operate on
- [ ] Moving tensors to GPU (`tensor.to('cuda')`)
- [ ] Autograd — automatic differentiation
- [ ] `nn.Module` — defining a model
- [ ] Optimizers: `Adam`, `SGD`, `AdamW`
- [ ] Training loop pattern: forward → loss → backward → step
- [ ] Loading a pretrained model from Hugging Face
- [ ] Running inference on a pretrained model

**What to build:** Load a pretrained BERT model from Hugging Face. Run text classification inference on 10 sentences. Understand what the output logits mean.

**Claude Code prompts to try:**
```
"Walk me through the full PyTorch training loop from scratch: define model, loss, optimizer, train for 10 epochs, evaluate"
"Show me how to load a pretrained Hugging Face model and run inference in PyTorch. What is the pipeline API?"
"Explain autograd in PyTorch. How does it track gradients? Show me a simple example."
```

---

### 2.7 Data Pipelines & Preprocessing

- [ ] ETL: Extract, Transform, Load
- [ ] Data cleaning: handling missing values, duplicates, outliers
- [ ] Feature scaling: normalization (0–1), standardization (z-score)
- [ ] Encoding: one-hot, label encoding, ordinal
- [ ] Chunking text documents for LLMs
- [ ] Data versioning with DVC
- [ ] Building reproducible pipelines with `make` or Prefect/Airflow basics

**Claude Code prompts to try:**
```
"Build a data pipeline in Python that: loads a folder of PDFs, extracts text, cleans it, and chunks it for RAG"
"What are the best chunking strategies for documents going into a vector database? Show tradeoffs."
"Show me how to build a reproducible ML pipeline using DVC for data versioning"
```

---

### 2.8 Experiment Tracking

- [ ] Why you need experiment tracking (you will run hundreds of experiments)
- [ ] **MLflow** — log parameters, metrics, artifacts, compare runs
- [ ] **Weights & Biases (W&B)** — richer UI, sweep hyperparameter tuning
- [ ] **LangSmith** — specifically for LLM experiments
- [ ] Reproducibility: seeds, environment pinning, config files
- [ ] A/B testing model versions

**Claude Code prompts to try:**
```
"Set up MLflow experiment tracking in a Python project. Show me how to log parameters, metrics, and artifacts"
"How do I track prompt experiments? I want to compare 5 different system prompts and measure which performs best"
```

---

## Phase 3 — LLMs & Modern AI APIs
**Weeks 8–12 · The heart of AI engineering today**

> 80% of AI engineering job postings revolve around LLMs. Master this phase and you're already employable. Every section here has a buildable project attached — build them all.

### 3.1 How Transformers Work

- [ ] **Tokens** — LLMs don't see words, they see tokens (~0.75 words each)
- [ ] **Tokenization** — BPE (Byte Pair Encoding), SentencePiece
- [ ] **Embeddings** — every token becomes a vector
- [ ] **Self-attention** — each token attends to every other token
  - Query, Key, Value matrices
  - Attention score = softmax(QK^T / √d_k)
  - Multi-head attention — multiple attention patterns simultaneously
- [ ] **Positional encoding** — how the model knows word order
- [ ] **Feed-forward layers** — applied to each position independently
- [ ] **Layer normalization** — stabilizes training
- [ ] **Residual connections** — gradients flow more easily
- [ ] **Context window** — maximum tokens the model processes at once
- [ ] **Causal (decoder-only) vs encoder-decoder** architectures
- [ ] **Self-supervised pretraining** — predict next token on the entire internet
- [ ] **RLHF** — Reinforcement Learning from Human Feedback (why Claude/ChatGPT are helpful)

**Claude Code prompts to try:**
```
"Explain self-attention to me from scratch. What problem does it solve that RNNs couldn't?"
"Build a minimal transformer in PyTorch from scratch — just attention + feed-forward + embedding. Walk me through every line."
"What is the context window? What happens when I exceed it? How do different models handle very long documents?"
"Why did transformers replace LSTMs and RNNs? What fundamental problem do they solve better?"
```

**Resource:** Andrej Karpathy's "Let's build GPT" — https://youtu.be/kCc8FmEb1nY (watch this, it's essential)

---

### 3.2 Embeddings & Vector Search

- [ ] What an embedding is — a dense vector representing meaning
- [ ] Why similar meanings are geometrically close (vector space)
- [ ] **Cosine similarity** vs Euclidean distance for comparing embeddings
- [ ] OpenAI embedding models: `text-embedding-3-small`, `text-embedding-3-large`
- [ ] Open-source embeddings: `all-MiniLM-L6-v2`, `bge-large`, `nomic-embed`
- [ ] **Vector databases:**
  - `pgvector` — PostgreSQL extension (best for existing Postgres users)
  - `Pinecone` — managed, serverless (easiest to start)
  - `Qdrant` — open-source, self-hostable, fast
  - `FAISS` — local, in-memory (great for development)
  - `Chroma` — local, developer-friendly
- [ ] **ANN search** — approximate nearest neighbours (why it's fast at scale)
- [ ] **Metadata filtering** — filter by document type, date, user before vector search
- [ ] Embedding different modalities: text, images, code

**What to build:** Embed 100 Wikipedia articles, store in FAISS, build a semantic search interface.

**Claude Code prompts to try:**
```
"Show me how to embed text with OpenAI's API, store vectors in FAISS, and search for similar documents"
"Compare pgvector vs Pinecone vs Qdrant. When would I choose each for a production RAG system?"
"What is ANN (approximate nearest neighbours) search? How does HNSW indexing work and why is it fast?"
"Build a semantic search engine for a folder of text files using sentence-transformers and FAISS"
```

---

### 3.3 Prompt Engineering

- [ ] **System prompt** — sets the model's persona, rules, and output format
- [ ] **User prompt** — the actual question or task
- [ ] **Few-shot examples** — show the model 2–5 examples of good input/output
- [ ] **Chain-of-thought (CoT)** — "Think step by step before answering"
- [ ] **Zero-shot CoT** — just appending "Let's think step by step"
- [ ] **Output formatting** — ask for JSON, markdown, bullet points
- [ ] **Prompt templates** — parameterized prompts with variables
- [ ] **Negative instructions** — what NOT to do (use sparingly)
- [ ] **Temperature** — 0 for deterministic, 1+ for creative
- [ ] **Top-p (nucleus sampling)** — controls randomness differently
- [ ] **Systematic prompt testing** — test prompts against a set of examples, measure quality
- [ ] **Prompt versioning** — treat prompts like code, version control them

**What to build:** Build a prompt testing harness: given 5 prompt variants and 20 test inputs, run all combinations and score outputs with an LLM judge.

**Claude Code prompts to try:**
```
"Show me how to write a system prompt for a customer support bot that: stays on-topic, never makes up facts, and always formats responses as bullet points"
"What is chain-of-thought prompting? Show me before/after examples where it dramatically improves accuracy"
"Build a prompt testing framework in Python that compares multiple prompt variants across a test set and ranks them by LLM-judged quality"
"I'm getting inconsistent JSON output from GPT-4. How do I make it reliably output valid JSON every time?"
```

---

### 3.4 RAG — Retrieval-Augmented Generation

> **The most important AI engineering pattern.** Understand this deeply.

**The problem RAG solves:** LLMs only know what they were trained on (cutoff date, no private data). RAG gives them access to your data at query time.

**The basic RAG flow:**
1. **Ingest:** Load documents → chunk → embed → store in vector DB
2. **Retrieve:** Embed user query → find similar chunks in vector DB
3. **Augment:** Inject retrieved chunks into the prompt as context
4. **Generate:** LLM answers using the injected context

**What to learn:**
- [ ] Chunking strategies: fixed-size, recursive, semantic, parent-child
- [ ] Chunk size and overlap tradeoffs (too small = loses context, too large = irrelevant)
- [ ] Retrieval: `similarity_search`, top-k selection
- [ ] Prompt construction with retrieved context
- [ ] Citation/source attribution in responses
- [ ] Handling the case where no relevant context is found ("I don't know")
- [ ] Evaluating RAG: context relevancy, answer faithfulness, answer completeness

**What to build:**
1. **Week 10:** PDF Q&A — upload PDFs, chunk, embed, answer questions from content
2. **Week 11:** Multi-document RAG with metadata filtering (only search docs from the last 30 days)

**Claude Code prompts to try:**
```
"Build a complete RAG pipeline from scratch: load a PDF, chunk it, embed with OpenAI, store in FAISS, and answer questions"
"What chunking strategy should I use for: legal contracts, customer support emails, technical documentation? Explain the tradeoffs."
"How do I evaluate if my RAG system is actually working? What metrics should I track?"
"My RAG system gives wrong answers — walk me through a debugging process to find out why"
"Build a RAG system with metadata filtering: only retrieve chunks from documents tagged with a specific category"
```

---

### 3.5 OpenAI & Anthropic APIs

- [ ] **Chat completions** — `messages` array with `system`, `user`, `assistant` roles
- [ ] **Streaming** — receive tokens as they generate (`stream=True`)
- [ ] **Function calling / tool use** — model decides to call your function
- [ ] **Structured outputs** — force JSON conforming to a schema
- [ ] **Vision** — pass images in the messages array
- [ ] **Token counting** — `tiktoken` for OpenAI, `anthropic` SDK for Claude
- [ ] **Cost estimation** — input tokens × price + output tokens × price
- [ ] **Model selection tradeoffs:**
  - `gpt-4o-mini` / `claude-haiku` — fast, cheap, good for simple tasks
  - `gpt-4o` / `claude-sonnet` — balanced, most tasks
  - `o1` / `claude-opus` — expensive, complex reasoning
- [ ] **Batch API** — 50% cheaper, async, for non-real-time workloads
- [ ] **Prompt caching** — Anthropic and OpenAI cache repeated system prompts

**Claude Code prompts to try:**
```
"Show me the full OpenAI chat completion API: system prompt, user message, streaming response, error handling"
"Build a function-calling example where the LLM decides whether to: search the web, query a database, or answer directly"
"How do I estimate the cost of my AI application before going to production? Show me a cost calculator."
"What is prompt caching? Show me how to use it with Anthropic's API and calculate the savings."
"Build a multi-modal example: user uploads an image, GPT-4o describes it, then answers questions about it"
```

---

### 3.6 LangChain & LlamaIndex

- [ ] **LangChain core concepts:**
  - Chains — sequence of operations (prompt → LLM → parser)
  - LCEL — LangChain Expression Language (`|` pipe syntax)
  - Document loaders — PDF, web, CSV, database
  - Text splitters — chunking strategies
  - Retrievers — fetch from vector stores
  - Memory — conversation history management
  - Agents — LLM + tools (covered in Phase 5)
- [ ] **LlamaIndex concepts:**
  - Index types — VectorStoreIndex, SummaryIndex
  - Query engines
  - Node parsers and transformations
- [ ] When to use a framework vs build from scratch
- [ ] Debugging LangChain with `verbose=True` and callbacks

**Claude Code prompts to try:**
```
"Build a RAG chain in LangChain using LCEL syntax. Include a document loader, text splitter, vector store, and retrieval QA chain"
"What are the pros and cons of LangChain vs LlamaIndex vs building from scratch? When would I choose each?"
"Show me how to add conversation memory to a LangChain chain so it remembers previous messages"
"Debug this LangChain code for me: [paste your code]. Why isn't it retrieving the right chunks?"
```

---

### 3.7 Open-Source Models

- [ ] **Why open-source?** — data privacy, no per-token cost, customization
- [ ] **Key models to know:**
  - Llama 3 (Meta) — most popular open-source family
  - Mistral 7B / Mixtral — efficient, excellent for size
  - Qwen 2.5 (Alibaba) — strong multilingual
  - Phi-3 (Microsoft) — small but capable
  - Gemma (Google)
- [ ] **Running locally with Ollama** — `ollama run llama3`
- [ ] **Hugging Face Hub** — `from_pretrained()` to load any model
- [ ] **Hugging Face `transformers` pipeline API** — simple inference
- [ ] **Reading model cards** — parameters, training data, benchmarks, limitations
- [ ] **GGUF format** — quantized models for CPU inference
- [ ] **Open-source vs closed:** when to use each

**What to build:** Run Llama 3 8B locally with Ollama, build the same chatbot as 3.5 but using the local model instead of OpenAI. Compare output quality and latency.

**Claude Code prompts to try:**
```
"Show me how to run Llama 3 locally with Ollama and build a Python client that talks to it using the OpenAI-compatible API"
"How do I load a model from Hugging Face and run inference? Show me the pipeline API and the low-level approach."
"Compare Llama 3 8B vs Mistral 7B vs Phi-3 mini — what are the practical differences in quality, speed, and cost?"
"When should I use open-source models instead of GPT-4 or Claude? Walk me through the decision."
```

---

### 3.8 Structured Output & JSON Mode

- [ ] Why unstructured LLM output breaks pipelines
- [ ] **JSON mode** — forces valid JSON but not schema-conformant
- [ ] **Structured outputs** (OpenAI) — schema-conformant guarantee
- [ ] **Pydantic** — define data models with validation
- [ ] **Instructor** library — Pydantic + LLM = typed, validated output
- [ ] Handling validation errors and retrying
- [ ] Nested schemas and optional fields
- [ ] Extracting structured data from unstructured text at scale

**What to build:** Extract structured data (name, email, company, action items) from 50 unstructured meeting transcript emails using Pydantic + Instructor.

**Claude Code prompts to try:**
```
"Show me how to extract structured data from text using Pydantic and the Instructor library. I want to extract: product name, price, sentiment, and key features from product reviews."
"What's the difference between JSON mode and structured outputs in OpenAI's API? When does each fail?"
"Build a pipeline that reads customer emails, extracts structured ticket information (priority, category, sentiment, action required), and stores it in a database"
```

---

### 3.9 Conversation Memory Patterns

- [ ] Why LLMs are stateless (no memory between API calls)
- [ ] **Buffer memory** — pass full conversation history (simple, gets expensive)
- [ ] **Window memory** — keep last N messages (fixed cost)
- [ ] **Summary memory** — summarize older messages, keep recent ones
- [ ] **Entity memory** — extract and track key facts (names, preferences)
- [ ] **Vector memory** — embed past messages, retrieve relevant ones
- [ ] Managing context window limits
- [ ] Storing conversation history in a database (not just in-memory)
- [ ] Multi-user conversation isolation

**What to build:** Build a chatbot with summary memory: after every 10 messages, summarize the older ones and maintain a running summary that's prepended to the context.

**Claude Code prompts to try:**
```
"Build a chatbot with conversation memory stored in PostgreSQL — each user has their own conversation history, retrievable across sessions"
"When does a conversation context window fill up? How do I handle this gracefully without losing important context?"
"Implement summary memory in Python: after 10 messages, use an LLM to summarize the conversation so far and store only the summary + last 5 messages"
```

---

## Phase 4 — Production AI Systems
**Weeks 13–17 · Making AI reliable at scale**

> This is where most tutorials stop and where real jobs begin. Getting a demo to work takes an afternoon. Getting it to be fast, reliable, cheap, and safe for thousands of users is a different engineering discipline.

### 4.1 LLM Evaluation & Testing

- [ ] **The eval problem** — LLM output is non-deterministic, hard to assert on
- [ ] **Golden test sets** — curated input/expected output pairs
- [ ] **LLM-as-judge** — use a stronger model to score outputs (1–5 scale, pass/fail)
- [ ] **Metrics for RAG:**
  - Context relevancy — is the retrieved context relevant to the question?
  - Answer faithfulness — does the answer stay within the retrieved context?
  - Answer completeness — does it answer the whole question?
- [ ] **RAGAS framework** — automated RAG evaluation
- [ ] **Human evaluation** — periodic sampling reviewed by humans
- [ ] **Regression testing** — eval suite runs on every code change
- [ ] **Red-teaming** — trying to make your system fail
- [ ] **Tools:** LangSmith, Braintrust, PromptFoo, Athina AI

**Claude Code prompts to try:**
```
"Build an LLM evaluation framework: given 20 question/answer pairs, use GPT-4 to judge the quality of answers on a 1-5 scale. Output a report."
"How do I set up regression testing for an LLM application? I want every GitHub PR to run an eval suite."
"Show me how to use RAGAS to evaluate a RAG pipeline for context relevancy and answer faithfulness"
"What's the right way to red-team an AI customer support bot? What failure modes should I test?"
```

---

### 4.2 Observability & Tracing

- [ ] Why standard logging isn't enough for AI systems
- [ ] **LangSmith** — traces every LLM call, shows prompts/responses, latency, tokens
- [ ] **Langfuse** — open-source alternative, self-hostable
- [ ] **Helicone** — logging proxy (works with any OpenAI-compatible API)
- [ ] What to trace: prompt sent, response received, latency, token counts, cost, errors
- [ ] Structured logging with JSON (easier to query)
- [ ] Dashboards: latency percentiles, error rates, cost trends
- [ ] Alerting: spike in errors, cost anomaly, latency degradation
- [ ] Tracing in distributed systems (trace IDs across services)

**Claude Code prompts to try:**
```
"Integrate LangSmith tracing into my LangChain RAG pipeline. What do I configure and what can I see in the dashboard?"
"Design a logging strategy for an AI API: what should I log with every request? How do I avoid logging PII?"
"Set up structured JSON logging for a FastAPI AI app. Include: request ID, user ID (hashed), model used, token counts, latency, error details."
```

---

### 4.3 Cost Optimisation

> LLM costs can spiral from $100/month to $10,000/month as you scale. Plan for this.

- [ ] **Prompt caching** — Anthropic: 90% reduction on cached prefix; OpenAI: similar
- [ ] **Model tiering** — route simple queries to cheap models, hard ones to expensive
- [ ] **Query classification** — ML classifier decides which model to use
- [ ] **Semantic caching** — cache responses for similar queries (GPTCache, Momento)
- [ ] **Batch API** — 50% cheaper, use for non-real-time workloads
- [ ] **Output length control** — set `max_tokens` appropriately, instruct brevity
- [ ] **Prompt optimization** — shorter prompts cost less; remove unnecessary few-shot examples
- [ ] **Token counting before sending** — reject or truncate inputs that are too long
- [ ] **Cost monitoring** — alert when daily cost exceeds threshold

**Claude Code prompts to try:**
```
"Build a request router that classifies queries as 'simple' or 'complex' and sends them to GPT-4o-mini or GPT-4o respectively. Show me how to measure the cost savings."
"Implement semantic caching with Redis: if a new query is >95% similar to a cached one, return the cached response instead of calling the LLM"
"How do I use Anthropic's prompt caching feature? Show me the API call and calculate the savings for a system with a 2000-token system prompt"
```

---

### 4.4 Latency & Streaming

- [ ] **Why LLMs are slow** — autoregressive generation, one token at a time
- [ ] **TTFT** — Time to First Token (most important UX metric)
- [ ] **Streaming** — return tokens as they generate, not all at once
- [ ] Implementing streaming in FastAPI with Server-Sent Events (SSE)
- [ ] Streaming in the browser (EventSource API)
- [ ] **Async everything** — never block the event loop on LLM calls
- [ ] Concurrency with `asyncio.gather` for parallel LLM calls
- [ ] **Speculative decoding** — draft model generates candidates, main model verifies (advanced)
- [ ] When to use background jobs instead of synchronous response
- [ ] Queue-based architecture for high-volume AI workloads (Celery + Redis)

**What to build:** FastAPI endpoint with streaming LLM response + a simple HTML page that renders the streaming output in real time.

**Claude Code prompts to try:**
```
"Build a FastAPI endpoint that streams an LLM response using Server-Sent Events. Include a simple HTML frontend that renders the stream."
"How do I make 5 LLM API calls concurrently in Python? Show me asyncio.gather and how to handle errors."
"Design a queue-based AI pipeline: user submits a long document, gets a job ID immediately, the analysis runs in background, user polls for result"
```

---

### 4.5 Guardrails & Safety

- [ ] **Input validation** — detect and handle unsafe/off-topic inputs before the LLM
- [ ] **Prompt injection** — detect override attempts ("ignore previous instructions")
- [ ] **Output validation** — verify output format, length, and content before serving
- [ ] **Content moderation** — OpenAI moderation API, Azure Content Safety
- [ ] **PII detection** — detect names, emails, SSNs, credit cards before logging or storage
- [ ] **Guardrails AI** — framework for input/output validation rules
- [ ] **NeMo Guardrails** — NVIDIA framework for dialogue safety
- [ ] **Jailbreak detection** — patterns that try to bypass safety
- [ ] **Hallucination mitigation** — grounding, retrieval, citations
- [ ] Rate limiting per user — prevent abuse

**Claude Code prompts to try:**
```
"Build an input validation layer for a customer support bot: reject off-topic queries, detect prompt injection, strip PII before logging"
"Show me how to use the OpenAI moderation API to screen user inputs and explain the category flags"
"What are the most common prompt injection patterns? Show me examples and how to detect/defend against each"
"Implement a simple hallucination detector: given a RAG response and the source documents, use an LLM to check if the answer is grounded in the sources"
```

---

### 4.6 Deployment & Serving

**For closed-model APIs (OpenAI, Anthropic):**
- [ ] FastAPI production setup: Gunicorn + Uvicorn workers
- [ ] Containerize: Dockerfile, docker-compose, push to ECR
- [ ] Deploy: AWS ECS Fargate, AWS App Runner, Railway, Render
- [ ] Environment variables, secrets management in deployment
- [ ] Auto-scaling based on request queue depth

**For open-source models:**
- [ ] **vLLM** — high-throughput inference server (best for production)
- [ ] **Text Generation Inference (TGI)** — Hugging Face's inference server
- [ ] GPU selection: A10G (24GB) for 7–13B models, H100 for 70B+
- [ ] **Quantization** — INT8/INT4 to reduce memory (slight quality tradeoff)
- [ ] **Model parallelism** — split models across multiple GPUs

**Claude Code prompts to try:**
```
"Show me the full deployment of a FastAPI AI app to AWS ECS Fargate with: ECR, task definition, service, ALB, environment variables from Secrets Manager"
"Set up vLLM to serve Llama 3 8B with OpenAI-compatible API. How do I optimize for throughput vs latency?"
"What's the cheapest way to deploy an open-source 7B model that can handle 100 requests/hour? Walk through the options."
```

---

### 4.7 Caching for AI Applications

- [ ] **Exact caching** — identical queries get cached LLM responses (Redis)
- [ ] **Semantic caching** — similar queries get cached responses (embedding similarity)
- [ ] Cache key design for AI: hash(model + system_prompt + user_message)
- [ ] TTL strategy — how long should AI responses be cached?
- [ ] **GPTCache** — semantic caching library
- [ ] **Momento** — managed semantic cache service
- [ ] Cache invalidation — when data changes, cached answers may be wrong
- [ ] Cache warming — pre-populate cache with common queries before launch

**Claude Code prompts to try:**
```
"Implement a two-layer cache for an AI API: exact match in Redis (fast), semantic match using embeddings (slower but catches near-duplicates)"
"How do I design cache keys for an LLM API that has different models, system prompts, and users?"
"What are the failure modes of semantic caching? When does it return wrong cached results?"
```

---

### 4.8 Data Privacy & Compliance

- [ ] **GDPR principles:** data minimisation, right to erasure, purpose limitation
- [ ] PII in AI systems: what it is, how to detect, how to handle
- [ ] **Data residency** — some regulations require data to stay in a region
- [ ] **Azure OpenAI** — OpenAI models in your Azure subscription (data doesn't leave)
- [ ] **On-premises open-source models** — when cloud is not allowed
- [ ] **Audit logging** — log every AI interaction for compliance
- [ ] **Right to erasure** — how to delete a user's data including AI interaction logs
- [ ] **Terms of service** — what can and cannot be sent to each AI API
- [ ] **AI Act** (EU) — high-risk AI system requirements

**Claude Code prompts to try:**
```
"Design a GDPR-compliant logging system for an AI chatbot: what do I log, how do I anonymize, how do I handle deletion requests?"
"What PII should I scrub before sending user text to an external LLM API? Show me a Python PII scrubber."
"When should I use Azure OpenAI instead of regular OpenAI? What compliance requirements drive this decision?"
```

---

## Phase 5 — Advanced Topics
**Weeks 18–22 · What separates senior AI engineers**

### 5.1 Fine-Tuning LLMs

**When to fine-tune (vs prompt or RAG):**
- You need a very specific output style or format that prompting can't achieve
- You want to reduce prompt length significantly (compress few-shot examples into weights)
- You need domain-specific behaviour that changes model reasoning, not just knowledge

**When NOT to fine-tune:**
- Your knowledge changes frequently (use RAG instead)
- Prompting already achieves good results (fine-tuning adds complexity)
- You don't have high-quality labeled training data (>1000 examples)

**What to learn:**
- [ ] Full fine-tuning vs parameter-efficient fine-tuning (PEFT)
- [ ] **LoRA** (Low-Rank Adaptation) — train small adapter matrices, not the full model
- [ ] **QLoRA** — LoRA + 4-bit quantization (run on consumer GPUs)
- [ ] **Instruction tuning** — train on instruction/response pairs
- [ ] **RLHF** — Reinforcement Learning from Human Feedback (conceptual)
- [ ] **DPO** — Direct Preference Optimization (simpler alternative to RLHF)
- [ ] Dataset preparation for fine-tuning
- [ ] Hyperparameters: learning rate, epochs, batch size, rank (for LoRA)
- [ ] Evaluating fine-tuned models
- [ ] **Hugging Face PEFT** library — the standard tool
- [ ] **Unsloth** — faster QLoRA training
- [ ] **OpenAI fine-tuning API** — fine-tune gpt-4o-mini on your data

**Claude Code prompts to try:**
```
"Walk me through fine-tuning Llama 3 8B with QLoRA using Hugging Face PEFT. What data do I need, what hyperparameters matter?"
"When should I fine-tune vs use RAG vs prompt engineer? Give me a decision framework with examples."
"How do I prepare a fine-tuning dataset? What format does Hugging Face expect? Show me data preparation code."
"Use OpenAI's fine-tuning API to fine-tune gpt-4o-mini on customer support examples. Walk me through the full process."
```

---

### 5.2 AI Agents & Tool Use

**What an agent is:** An LLM that can use tools (web search, code execution, API calls, file operations) to accomplish multi-step tasks autonomously.

**The ReAct pattern:**
```
Thought: I need to find the current weather in Lahore
Action: search_web("current weather Lahore Pakistan")
Observation: The weather is 32°C, sunny
Thought: I have the answer
Action: final_answer("It's 32°C and sunny in Lahore")
```

**What to learn:**
- [ ] Tool definition: name, description, parameters (what the model sees)
- [ ] Tool execution: your code runs the tool, returns results to the model
- [ ] ReAct prompting pattern
- [ ] **LangGraph** — state machine for complex agent workflows
- [ ] **AutoGen** — multi-agent conversations
- [ ] Agent memory: short-term (conversation), long-term (vector store)
- [ ] **Human-in-the-loop** — pause agent, get human approval, continue
- [ ] Agent evaluation — hard because multi-step, non-deterministic
- [ ] Cost and safety guardrails — agents can take expensive or dangerous actions
- [ ] Common failure modes: infinite loops, wrong tool selection, hallucinated tool calls

**What to build:** An agent that: searches the web, reads URLs, executes Python code, and writes a research report. Add a budget cap (stop after $0.50 of API spend).

**Claude Code prompts to try:**
```
"Build an AI agent with 3 tools: web search, Python code execution, and file writing. Use LangChain's tool calling framework."
"Show me how to build a LangGraph workflow for a multi-step research agent with human approval checkpoints"
"What are the safety constraints I need to put on an AI agent? How do I prevent it from taking destructive actions?"
"My agent is looping — it keeps calling the same tool with the same parameters. How do I debug and fix this?"
```

---

### 5.3 Multimodal AI

- [ ] **Vision-language models** — GPT-4o, Claude, Gemini can see images
- [ ] Passing images to the API: base64 encoding or URL
- [ ] Use cases: document extraction, image description, visual Q&A, chart reading
- [ ] **OCR via LLMs** — extract text from images of documents
- [ ] **Document parsing** — invoices, contracts, forms with vision models
- [ ] **Audio:** Whisper (speech-to-text), TTS API (text-to-speech)
- [ ] **Video:** frame sampling + vision model (no native video support in most APIs yet)
- [ ] Open-source vision models: LLaVA, Qwen-VL, Pixtral
- [ ] **CLIP** — image embeddings for visual search
- [ ] Multimodal RAG — retrieve based on image content

**Claude Code prompts to try:**
```
"Build a document extraction pipeline: user uploads a PDF invoice, vision model extracts vendor, amount, date, line items into structured JSON"
"Show me how to transcribe audio with Whisper API and then summarize the transcript with Claude"
"Build an image search engine: embed a folder of images with CLIP, store vectors, search by text description"
```

---

### 5.4 Advanced RAG Techniques

**Why naive RAG fails:**
- Short chunks lose context; long chunks bring irrelevant info
- Pure semantic search misses exact keyword matches
- Top-k retrieval doesn't account for quality, just similarity

**Advanced techniques to learn:**
- [ ] **Hybrid search** — BM25 (keyword) + vector (semantic) combined
- [ ] **Reranking** — use a cross-encoder to reorder top-k results by relevance
- [ ] **HyDE** (Hypothetical Document Embeddings) — generate a hypothetical answer, embed it to find similar docs
- [ ] **Parent-child chunking** — embed small chunks, retrieve large parent chunks for context
- [ ] **RAPTOR** — recursive abstraction and summarization for hierarchical retrieval
- [ ] **Corrective RAG** — evaluate retrieved docs, re-query if poor quality
- [ ] **Self-RAG** — model decides when to retrieve, evaluates its own output
- [ ] **Multi-query retrieval** — generate 3 query variants, take union of results
- [ ] **Contextual retrieval** (Anthropic) — prepend chunk context before embedding

**Claude Code prompts to try:**
```
"Implement hybrid search combining BM25 keyword search with vector semantic search using Reciprocal Rank Fusion (RRF) to combine scores"
"Build a reranking step into my RAG pipeline using a cross-encoder from Hugging Face. Show me before/after retrieval quality."
"What is HyDE and when does it improve RAG? Show me an implementation."
"Implement multi-query RAG: for each user question, generate 3 different phrasings, retrieve for all, deduplicate, then generate a response"
```

---

### 5.5 AI System Design

> This is the core skill tested in senior AI engineering interviews. Practice designing systems end-to-end.

**Framework for any AI system design question:**

1. **Clarify requirements** — scale, latency SLA, accuracy requirements, cost budget
2. **Data layer** — what data, ingestion pipeline, storage
3. **ML/AI layer** — model choice, prompting, retrieval
4. **API layer** — endpoints, auth, rate limiting
5. **Evaluation** — metrics, offline eval, online monitoring
6. **Scale** — caching, queuing, horizontal scaling
7. **Cost** — estimate and optimize

**Practice these design questions:**
- [ ] Design a customer support AI for 1M users (latency < 2s, cost < $0.01/query)
- [ ] Design a document Q&A system for a law firm (100k documents, strict data privacy)
- [ ] Design a code review assistant that integrates with GitHub
- [ ] Design a content moderation system for a social platform (1M posts/day)
- [ ] Design an AI-powered search for an e-commerce site (10M products)

**Claude Code prompts to try:**
```
"Walk me through the system design for a customer support chatbot: 1M users, 10k documents, < 2 second latency, $500/month budget. Draw the full architecture."
"Design a RAG-based document Q&A for a law firm. They have 100k case documents, can't send data to OpenAI, need 99.9% uptime."
"I'm being asked to design an AI code review tool in an interview. Walk me through what I should cover and in what order."
```

---

### 5.6 Model Quantization & Optimization

- [ ] **Why quantization?** — reduce model size (float32 → int8 → int4) for faster inference
- [ ] **INT8 quantization** — small quality loss, 2x memory reduction
- [ ] **INT4 (QLoRA/GGUF)** — larger quality loss, 4x memory reduction, runs on consumer GPU/CPU
- [ ] **GGUF format** — quantized format for llama.cpp, runs on CPU
- [ ] **llama.cpp** — run LLMs on CPU (laptop deployment)
- [ ] **Flash Attention 2** — faster attention computation (less memory, same output)
- [ ] **Continuous batching** — vLLM's technique for high-throughput serving
- [ ] **KV cache** — caching key/value matrices to speed up generation
- [ ] **Speculative decoding** — draft + verify for faster generation
- [ ] **ONNX** — portable model format for cross-platform inference

**Claude Code prompts to try:**
```
"Explain model quantization. What quality do I lose going from float32 to int4? When is the tradeoff worth it?"
"Show me how to run a quantized Llama model with llama.cpp locally on CPU. What GGUF file should I download?"
"What is Flash Attention and how does it speed up transformers? Does it change the model output?"
```

---

### 5.7 MLOps for AI Systems

- [ ] **CI/CD for ML** — eval suite runs on every PR, blocks merge if quality drops
- [ ] **Model registry** — track which model version is in production (MLflow, W&B)
- [ ] **Shadow mode** — run new model in parallel without serving results, compare offline
- [ ] **A/B testing** — route x% of traffic to new model, measure real user impact
- [ ] **Canary deployment** — deploy to 5% of users first, watch metrics
- [ ] **Prompt versioning** — treat prompts like code (git, tags, rollback)
- [ ] **Data versioning** — DVC for datasets, versioned vector databases
- [ ] **Feature stores** — centralized feature computation and serving
- [ ] **Feedback loops** — use production data to improve models

**Claude Code prompts to try:**
```
"Design a CI/CD pipeline for an LLM application: what checks run on PR, what blocks merge, how do I auto-deploy?"
"How do I run an A/B test between two LLM prompts in production? Show me the architecture and analysis."
"Set up prompt versioning: I want to version my prompts like code, track which version is in production, and roll back if quality drops"
```

---

## Interview Preparation

### How AI Engineering Interviews Are Structured

Most AI engineering interviews have:
1. **Coding round** — Python, data structures, build a small AI component
2. **ML/AI concepts** — explain transformers, RAG, fine-tuning, evaluation
3. **System design** — design an AI system end-to-end
4. **Take-home project** — build a RAG system, chatbot, or evaluation pipeline

---

### Must-Know Interview Questions

#### Conceptual

**Q: How does attention work in a transformer?**
> Each token creates Q (query), K (key), V (value) vectors. Attention score = softmax(QK^T / √d_k). The output is a weighted sum of Value vectors. Multi-head attention runs multiple attention patterns simultaneously — some may attend to syntax, others semantics. Self-attention means each token can directly attend to every other, solving the vanishing gradient problem of RNNs.

**Q: When would you use RAG vs fine-tuning?**
> Use RAG when knowledge changes frequently, you need source citations, or you have private data. Use fine-tuning when you need a specific output style/format, want to compress many few-shot examples into weights, or need domain-specific reasoning (not just knowledge). Start with RAG — fine-tuning adds significant complexity.

**Q: How do you evaluate an LLM application?**
> Multi-layer: (1) offline eval — golden test set scored by LLM-as-judge before every deploy, (2) human evaluation — periodic sampling, (3) online metrics — thumbs up/down, regeneration rate, session length, (4) monitoring — latency, cost, error rates in LangSmith/Langfuse. Evaluation is continuous, not a one-time step.

**Q: What is prompt injection and how do you defend against it?**
> User includes text to override system prompt ("ignore previous instructions"). Defence: input validation (detect and reject patterns), privilege separation (model can't access what it doesn't need), output validation (check response conforms to expected format), and limiting blast radius through architecture.

**Q: Explain the difference between tokens and words.**
> Tokens are BPE subword units, ~0.75 English words each. "unbelievable" = 3 tokens. Non-English and code tokenize less efficiently (2–4x more tokens per word). Matters for: cost (billed per token), context window (in tokens), and performance.

#### System Design

**Q: Design a document Q&A system for 100k documents.**
> Ingestion: load → chunk (500 tokens, 50 overlap, recursive splitting) → embed (text-embedding-3-small) → store in pgvector with metadata. Query: embed question → hybrid search (BM25 + vector) → rerank top-20 → inject top-5 into prompt → Claude Sonnet with source citations → stream response. Evaluation: RAGAS for context relevancy and faithfulness. Scale: incremental indexing, semantic cache for repeated queries, cost monitoring.

**Q: How do you reduce LLM costs by 70%?**
> (1) Prompt caching — up to 90% on cached prefixes. (2) Model tiering — classify queries, use Haiku/mini for 70% of simple ones. (3) Semantic caching — identical/similar queries hit cache. (4) Batch API — 50% cheaper for non-real-time. (5) Output length control + prompt trimming.

#### Coding Challenges

**Typical take-home: Build a RAG pipeline**
```python
# The flow to implement:
# 1. Load documents (PyPDFLoader or BeautifulSoup)
# 2. Chunk (RecursiveCharacterTextSplitter, chunk_size=500, overlap=50)
# 3. Embed (OpenAIEmbeddings or sentence-transformers)
# 4. Store (FAISS locally, Pinecone/pgvector in production)
# 5. Retrieve (similarity_search, k=4)
# 6. Generate (ChatOpenAI with retrieved context injected)
# 7. Evaluate (RAGAS or LLM-as-judge)
# Key things to show: error handling, async, metadata filtering, eval
```

**Ask Claude Code:**
```
"Give me a RAG coding challenge and then evaluate my implementation. Be critical about: chunking strategy, retrieval quality, error handling, and evaluation."
"Quiz me on AI engineering interview questions. Ask me one at a time and grade my answers."
"Give me a system design interview: 'Design a real-time AI coding assistant'. Ask follow-up questions like an interviewer would."
```

---

## Resources

### Courses (in order)

| Resource | Cost | What you get |
|---|---|---|
| fast.ai — Practical Deep Learning | Free | Best intro to ML/DL from a practical angle |
| Andrej Karpathy — Neural Networks: Zero to Hero | Free (YouTube) | Build GPT from scratch. Essential. |
| DeepLearning.AI Short Courses | Free | RAG, LangChain, prompt engineering (1–3 hrs each) |
| Hugging Face NLP Course | Free | Transformers, fine-tuning, HF ecosystem |
| Full Stack LLM Bootcamp | Free | Production LLM apps from scratch |

### Key Papers to Read

- Attention Is All You Need (2017) — the original transformer paper
- Language Models are Few-Shot Learners (GPT-3 paper)
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- LoRA: Low-Rank Adaptation of Large Language Models
- RLHF: Learning to summarize from human feedback

### Blogs to Follow

- **Lilian Weng** — lilianweng.github.io (OpenAI Head of Safety, deep technical essays)
- **Simon Willison** — simonwillison.net (practical, opinionated LLM engineering)
- **Eugene Yan** — eugeneyan.com (applied ML in production)
- **The Batch** — deeplearning.ai/the-batch (weekly newsletter)
- **Papers With Code** — paperswithcode.com (SOTA benchmarks + code)

### Tools to Know

| Category | Tool | Why |
|---|---|---|
| LLM APIs | OpenAI, Anthropic, Google | The main providers |
| Open-source models | Ollama, Hugging Face | Local and self-hosted |
| Orchestration | LangChain, LlamaIndex | Build LLM pipelines |
| Vector DBs | pgvector, Pinecone, Qdrant | Store embeddings |
| Evaluation | LangSmith, RAGAS, Braintrust | Measure quality |
| Fine-tuning | Hugging Face PEFT, Unsloth | Adapt models |
| Serving | vLLM, TGI | High-throughput inference |
| Observability | Langfuse, Helicone | Trace LLM calls |
| Experiment tracking | W&B, MLflow | Log experiments |

---

## Projects Checklist

Build these in order — each one unlocks the next:

- [ ] **Week 2** — FastAPI REST API with authentication and tests
- [ ] **Week 6** — ML model (sklearn) trained, evaluated, and served via API
- [ ] **Week 8** — CLI chatbot with OpenAI API + streaming + conversation history
- [ ] **Week 10** — PDF Q&A RAG system (FAISS + LangChain + OpenAI)
- [ ] **Week 12** — Structured data extraction pipeline (Pydantic + Instructor)
- [ ] **Week 13** — Eval framework: LLM-as-judge scoring 20 test cases
- [ ] **Week 14** — FastAPI AI microservice: streaming, rate limiting, Docker
- [ ] **Week 16** — Production RAG: pgvector + hybrid search + reranking + LangSmith tracing
- [ ] **Week 18** — Fine-tune a small model (QLoRA on Llama) on custom data
- [ ] **Week 20** — AI agent with 3+ tools, budget cap, human-in-the-loop
- [ ] **Week 22** — Capstone: full AI product with eval, monitoring, cost optimization, docs

---

## Progress Tracker

### Phase 1 — Software Foundations
- [ ] 1.1 Python proficiency
- [ ] 1.2 REST APIs & HTTP
- [ ] 1.3 Git & version control
- [ ] 1.4 Databases
- [ ] 1.5 Docker & containers
- [ ] 1.6 Environment management
- [ ] 1.7 Cloud basics
- [ ] 1.8 Testing & code quality

### Phase 2 — Core AI & ML
- [ ] 2.1 How machine learning works
- [ ] 2.2 Neural networks fundamentals
- [ ] 2.3 Key ML algorithms
- [ ] 2.4 NLP basics
- [ ] 2.5 Model evaluation metrics
- [ ] 2.6 PyTorch basics
- [ ] 2.7 Data pipelines
- [ ] 2.8 Experiment tracking

### Phase 3 — LLMs & AI APIs
- [ ] 3.1 How transformers work
- [ ] 3.2 Embeddings & vector search
- [ ] 3.3 Prompt engineering
- [ ] 3.4 RAG
- [ ] 3.5 OpenAI & Anthropic APIs
- [ ] 3.6 LangChain & LlamaIndex
- [ ] 3.7 Open-source models
- [ ] 3.8 Structured output
- [ ] 3.9 Conversation memory

### Phase 4 — Production AI
- [ ] 4.1 LLM evaluation & testing
- [ ] 4.2 Observability & tracing
- [ ] 4.3 Cost optimisation
- [ ] 4.4 Latency & streaming
- [ ] 4.5 Guardrails & safety
- [ ] 4.6 Deployment & serving
- [ ] 4.7 Caching for AI apps
- [ ] 4.8 Data privacy & compliance

### Phase 5 — Advanced
- [ ] 5.1 Fine-tuning LLMs
- [ ] 5.2 AI agents & tool use
- [ ] 5.3 Multimodal AI
- [ ] 5.4 Advanced RAG techniques
- [ ] 5.5 AI system design
- [ ] 5.6 Model quantization & optimization
- [ ] 5.7 MLOps for AI systems

---

*Generated for use with Claude Code. Ask Claude Code anything — it's your AI pair programmer for this entire journey.*