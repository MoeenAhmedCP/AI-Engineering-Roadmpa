# 3.2 Embeddings & Vector Search — Notes

Embeddings are how meaning becomes math. They are the foundation of semantic search, RAG, recommendation systems, duplicate detection, and clustering. Understanding them properly — especially the difference between cosine and Euclidean distance — prevents a category of common bugs in production AI systems.

---

## What an Embedding Is

An embedding is a dense vector of floating-point numbers that encodes the meaning of a piece of text. "Dense" means every dimension carries information (contrast with sparse vectors like TF-IDF where most values are zero).

The key geometric property: **similar meanings end up close together in the vector space.** After training, "dog" and "puppy" will have vectors that point in nearly the same direction. "Paris" and "France" will be geometrically close. "Python" (programming) and "Python" (snake) will be in very different regions of the space, and the model learns the correct region from context.

This is not hand-crafted. It emerges from training on massive text by predicting words from their contexts (or by contrastive learning, where similar pairs are pulled together and dissimilar pairs pushed apart).

**Dimensionality:** text-embedding-3-small produces 1536-dimensional vectors. Each is a list of 1536 floats. You can truncate these to lower dimensions (OpenAI supports this) with some quality loss. More dimensions = more expressive but more expensive to store and search.

---

## Why Cosine Similarity, Not Euclidean Distance

This is one of the most common mistakes when first working with embeddings.

**Euclidean distance** measures the straight-line distance between two points in space. This is great for many geometric problems, but it conflates two things for text embeddings:
1. The angle (direction) between vectors — which encodes semantic meaning
2. The magnitude (length) of vectors — which carries much less meaning

Two embeddings for "happy" and "joyful" might have different magnitudes depending on how the model was trained, but they should point in nearly the same direction. Euclidean distance would penalize magnitude differences that are meaningless. Cosine similarity only looks at the angle between vectors, ignoring magnitude entirely.

**Cosine similarity formula:** `cos(θ) = (A · B) / (|A| × |B|)`

Result is in [-1, 1]:
- 1.0 = identical direction (semantically identical)
- 0.0 = perpendicular (semantically unrelated)
- -1.0 = opposite directions (semantically opposite)

**The practical consequence:** always normalize your embeddings (divide by their L2 norm) before storing. With normalized embeddings, cosine similarity equals dot product, and dot product is much faster to compute — no square roots needed. FAISS and most vector DBs can exploit this.

---

## OpenAI Embedding Models

**text-embedding-3-small:**
- 1536 dimensions
- Cost: $0.02 per 1M tokens
- Best for: high-volume use cases, search, classification
- Supports Matryoshka representation (truncate to fewer dims)

**text-embedding-3-large:**
- 3072 dimensions
- Cost: $0.13 per 1M tokens
- Best for: highest quality retrieval, when accuracy > cost
- ~6.5x more expensive than small, meaningful quality bump

**Legacy text-embedding-ada-002:**
- 1536 dims, $0.10/1M tokens — worse than 3-small at higher cost. Do not use for new projects.

---

## Open-Source Embedding Models

| Model | Dims | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Very fast | Good | Default for local dev, low-latency search |
| all-mpnet-base-v2 | 768 | Fast | Better | Better quality, still fast |
| bge-large-en-v1.5 | 1024 | Medium | Very good | High-quality retrieval, competitive with OpenAI |
| nomic-embed-text | 768 | Fast | Very good | Long documents (8192 token context) |
| e5-mistral-7b-instruct | 4096 | Slow | Excellent | Best open-source, needs GPU |

For most local projects: start with `all-MiniLM-L6-v2`. If quality matters more than speed: `bge-large-en-v1.5`.

---

## Vector Database Comparison

| DB | Type | Language | ANN Index | When to Use |
|----|------|----------|-----------|-------------|
| FAISS | Library | Python/C++ | HNSW, IVF | Millions of vectors, full control, no server |
| Chroma | Library/Server | Python | HNSW | Local dev, prototyping, easy to start |
| pgvector | PostgreSQL extension | SQL | HNSW, IVF | You already use Postgres, want SQL+vectors |
| Pinecone | Managed cloud | API | Proprietary | Production, no infra management, serverless |
| Qdrant | Server/Cloud | Rust | HNSW | Production, good filtering, open-source |
| Weaviate | Server/Cloud | Go | HNSW | Rich metadata, GraphQL queries |
| Milvus | Server | Go | Multiple | Billions of vectors, enterprise scale |

**Decision guide:**
- Prototype or local dev → Chroma (one-liner setup)
- Already on Postgres → pgvector (zero new infrastructure)
- Production, team, no infra ops → Pinecone or Qdrant Cloud
- Billions of vectors or special ANN requirements → Milvus or FAISS
- Need rich filtering + open-source + self-host → Qdrant

---

## Approximate Nearest Neighbor (ANN) Search

Exact nearest neighbor search requires computing the distance from your query to every single stored vector — O(n) complexity. With 10 million vectors, this is too slow for real-time search.

**HNSW (Hierarchical Navigable Small World)** is the dominant algorithm:
- Builds a multi-layer graph where each layer is a "skip list" of connections
- Search navigates from coarse (top layer) to fine (bottom layer) greedily following nearest neighbors
- Complexity: O(log n) for search
- Trade-off: uses more memory, index build time is O(n log n), and results are approximate (usually 99%+ recall)
- Parameters: `M` (connections per node, higher=better recall, more memory), `ef_construction` (build quality), `ef_search` (search quality)

**IVF (Inverted File Index):**
- Clusters vectors into centroids (like k-means)
- At search time, search only the nearest `nprobe` clusters
- Good for very large collections where HNSW memory is prohibitive

---

## Metadata Filtering

Pure vector similarity is often not enough. You almost always need to filter by metadata:
- Only search documents from the last 30 days
- Only search documents belonging to this user
- Only search documents with category = "legal"

Two approaches:
1. **Pre-filter:** filter by metadata first, then vector search within the filtered set. Simple but can reduce recall if the filtered set is too small.
2. **Post-filter:** vector search to get top-K, then filter by metadata. Can fail if all top-K results are filtered out.
3. **Hybrid (best):** most production vector DBs (Qdrant, Pinecone, Weaviate) support native filtered ANN — they integrate metadata filter into the index traversal itself.

Always index metadata fields you plan to filter on. Unindexed metadata filters force a linear scan.

---

## Embedding Other Modalities

Embeddings generalize beyond text:

**Images:** CLIP (OpenAI) embeds images and text into the same space. This enables cross-modal search: search images with text queries. ViT models embed images directly.

**Audio:** Wav2Vec 2.0, Whisper embeddings. Music retrieval systems use audio embeddings for "find songs similar to this one."

**Code:** CodeBERT, StarCoder embeddings. Used for code search, duplicate detection in large codebases, and semantic code completion.

**Multimodal:** GPT-4V, Gemini 1.5 Pro can embed images+text together. Useful for product search (image of shoe → find similar shoes), document understanding (PDF with charts).

---

## Production Considerations

**Embedding cost at scale:**
- 1 million product descriptions × 100 tokens each = 100M tokens
- At $0.02/1M tokens (text-embedding-3-small) = $2 for initial embedding
- Re-embedding when content changes adds ongoing cost

**Storage:**
- 1536-dimensional float32 vector = 6,144 bytes ≈ 6KB per embedding
- 1 million embeddings = ~6GB of raw vector data
- Use float16 or int8 quantization to reduce storage by 2-4x (small quality loss)

**Staleness:**
- Embeddings reflect model knowledge at time of embedding
- If you update the embedding model (e.g., switch from ada-002 to text-embedding-3-small), you must re-embed everything — they live in incompatible vector spaces

**Chunking before embedding:**
- Embedding models have token limits (512 tokens for most open-source, 8192 for OpenAI)
- Long documents must be chunked first (see Section 3.4 for deep coverage)
- What you chunk determines what you can retrieve
