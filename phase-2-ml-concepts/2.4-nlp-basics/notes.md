# 2.4 NLP Basics

## Why Text Needs Numbers

Neural networks and ML models operate on numbers — they multiply matrices, compute dot products, and apply calculus. Text is discrete, symbolic, and variable-length. Before a model can process language, every word (or subword or character) must become a vector of numbers. The choice of how you convert text to numbers profoundly affects what patterns the model can discover.

---

## Tokenization

Tokenization is the process of splitting text into the discrete units (tokens) that a model will process. Different granularities capture different trade-offs.

### Character-Level Tokenization
Split text into individual characters: "hello" → ['h', 'e', 'l', 'l', 'o'].
- Tiny vocabulary (~256 characters for ASCII)
- Can handle any word, including misspellings and novel words
- Very long sequences — every word is many tokens
- Harder to learn word-level meaning from character sequences

### Word-Level Tokenization
Split on whitespace and punctuation: "I love NLP." → ['I', 'love', 'NLP', '.'].
- Intuitive, short sequences
- Large vocabulary (100k+ for real text)
- **Out-of-vocabulary (OOV) problem:** words not seen during training are unknown. "ChatGPT" would be unknown to a model trained in 2019.
- Cannot handle morphology: "run", "runs", "running", "ran" are all separate tokens

### Subword Tokenization (BPE — Byte Pair Encoding)
The standard for modern LLMs. Vocabulary contains frequent full words AND common subwords.

**Intuition for BPE:**
1. Start with a character-level vocabulary
2. Count all character pair frequencies in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat for many thousands of merge operations
5. Result: common words become single tokens, rare words split into subword pieces

Example merges on "aaabdaaabac":
- Most frequent pair: "aa" → merge into "X": "XabdXabac"
- Most frequent pair: "Xa" → merge into "Y": "YbdYbac"
- Continue...

In practice, GPT uses BPE with ~50,000 vocabulary tokens. Common English words like "the", "and", "neural" are single tokens. Rare words like "photosynthesis" might split into ["photo", "synth", "esis"].

### Vocabulary and OOV
The vocabulary is the complete set of tokens a model knows. With BPE, OOV is essentially eliminated — any text can be encoded by falling back to individual characters, which are always in the vocabulary. The tradeoff: unusual words use more tokens (and more context window).

---

## Word Embeddings

### Word2Vec — Intuition
Word2Vec learns dense vector representations (embeddings) of words from large text corpora by predicting context. Two training objectives:
- **CBOW:** predict the target word from surrounding context words
- **Skip-gram:** predict surrounding context words from the target word

Training the network is not the goal — the learned weight matrix becomes the embedding lookup table. Words that appear in similar contexts get similar vectors.

**The famous example:** king - man + woman ≈ queen

This works because the vector arithmetic captures semantic relationships geometrically. The vector (king - man) encodes the concept of "royalty without maleness." Adding "woman" shifts to the female royal.

The embedding space encodes many analogies: Paris - France + Germany ≈ Berlin, doctor - man + woman ≈ nurse (unfortunately also captures biases in training data).

### GloVe (Global Vectors for Word Representation)
Instead of local context windows like Word2Vec, GloVe directly factorizes a word co-occurrence matrix built from the entire corpus. The result: embeddings where the ratio of co-occurrence probabilities is encoded in the vector differences.

Both Word2Vec and GloVe produce static embeddings — each word has one fixed vector regardless of context. "Bank" has the same embedding in "river bank" and "bank account." Contextual embeddings (BERT, GPT) solve this.

---

## Cosine Similarity

Cosine similarity measures the angle between two vectors, not their magnitude:

```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

- Range: -1 (opposite direction) to 1 (same direction)
- 0 means orthogonal (unrelated)

**Why it works for embeddings:** Embedding magnitude encodes word frequency, not meaning. A rare word might have a smaller magnitude than a common word even if semantically similar. Cosine similarity ignores magnitude and focuses on direction — the direction encodes meaning in embedding space.

Cosine similarity is the standard metric for semantic search, embedding-based retrieval, clustering of embeddings, and nearest-neighbor lookups.

---

## TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) scores words by how important they are to a specific document relative to a collection of documents.

```
TF(term, doc)  = (count of term in doc) / (total words in doc)
IDF(term)      = log(total_docs / (1 + docs_containing_term))
TF-IDF(t, d)   = TF(t, d) * IDF(t)
```

**Intuition:**
- High TF: this term appears a lot in this document → likely important to this doc
- High IDF: this term appears in few documents → it's specific and discriminating
- Low IDF (near 0): "the", "and", "is" appear in almost every document → not distinctive

**When TF-IDF beats embeddings:**
- Keyword search: the query contains specific technical terms or product names
- Short documents: little context for embeddings to work with
- No GPU: TF-IDF is extremely fast to compute
- Exact terminology matters: legal or medical text where precise terms are critical

**Limitation:** No semantic understanding. "automobile" and "car" are completely unrelated tokens in TF-IDF, even though they mean the same thing.

---

## N-grams

N-grams are contiguous sequences of n tokens. They capture local word order — information that bag-of-words and TF-IDF miss.

- **Unigram:** single word — "machine", "learning"
- **Bigram:** two consecutive words — "machine learning", "deep neural"
- **Trigram:** three consecutive words — "deep neural network"

Uses:
- Language modeling: predict next word given previous n-1 words
- Text classification features: bigrams capture phrases that have different meaning than individual words ("not good" vs "good")
- BLEU score computation: measures n-gram overlap between translation and reference
- Spell checking: character n-grams for fuzzy matching

---

## Preprocessing: Stop Words, Stemming, Lemmatization

### Stop Words
Common words with little semantic content: "the", "a", "is", "in", "it". Removing them:
- Reduces vocabulary size and noise in TF-IDF
- Speeds up processing
- Can hurt performance for exact phrase matching or when stop words carry meaning ("to be or not to be")

### Stemming
Crudely chops word endings using rules: "running" → "run", "studies" → "studi". Fast but can produce non-words.

### Lemmatization
Converts words to their dictionary base form using vocabulary and morphology: "running" → "run", "better" → "good". Requires a vocabulary. Slower but more accurate.

Both aim to reduce vocabulary size by grouping inflections of the same word. Less critical for LLMs which handle morphology naturally through subword tokenization.

---

## Surprising Tokenizations in LLMs

Numbers and dates behave unexpectedly with BPE tokenizers:
- "1234567" might tokenize as ["123", "45", "67"] — digits are grouped arbitrarily
- This is one reason LLMs struggle with arithmetic: numbers don't have consistent tokenizations
- "2024-01-15" might be ["2024", "-", "01", "-", "15"] — 5 tokens for a date

Unicode text:
- Emoji: a single emoji like 🤔 might be 3-4 tokens (it's multiple bytes in UTF-8)
- Non-Latin scripts: Chinese, Arabic, Thai often use more tokens per character than English
- This affects cost (more tokens = more expensive API calls) and context limits

Token count ≠ word count. The ratio varies by language, content type, and model. Counting tokens before sending to an API is a good practice.
