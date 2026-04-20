# 2.7 Data Pipelines

Data pipelines transform raw, messy data into the clean, structured format that models actually learn from. In AI engineering, the quality of your pipeline often matters more than the sophistication of your model.

---

## ETL: Extract, Transform, Load

**Extract** — Pull data from its source: files on disk, databases, APIs, S3 buckets. The key concern here is correctness: did you read every record? Did you handle encoding issues, corrupt files, or partial downloads?

**Transform** — Clean, normalize, and reshape the data. This is where most of the engineering effort lives: handling missing values, encoding categoricals, normalizing numerics, tokenizing text, deduplicating records, and enforcing schema constraints.

**Load** — Write the result to its destination: a feature store, a vector database, a training dataset on disk, or directly into a model's dataloader. In AI pipelines the "load" step often means batching data for efficient training.

---

## Loading Patterns

| Pattern | When to Use |
|---------|-------------|
| **In-memory** | Dataset fits in RAM (<~1 GB). Load all at once, shuffle freely. Simple but doesn't scale. |
| **Streaming / generators** | Large datasets (logs, web crawls). Read one record or batch at a time; memory stays constant. |
| **Memory-mapped** | Binary files (numpy `.npy`, Arrow/Parquet). OS maps the file into virtual memory — random access is fast without loading everything. |

```python
# Streaming with a generator (never loads more than one line)
def stream_file(path, batch_size=32):
    batch = []
    with open(path) as f:
        for line in f:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
```

---

## Preprocessing Numeric Data

**Min-max normalization** — scales values to [0, 1]. Sensitive to outliers.
```
x_norm = (x - x_min) / (x_max - x_min)
```

**Z-score standardization** — centers to mean 0, std 1. More robust to outliers, preferred for neural networks.
```
x_std = (x - mean) / std
```

**Encoding categoricals:**
- One-hot encoding: creates a binary column per category. Good for low-cardinality features; explodes in size for high-cardinality.
- Label encoding: maps category → integer. Only appropriate when ordinal relationship exists (small < medium < large).

**Handling missing values:**
- Numeric: fill with mean, median, or a learned value; add a boolean "was_missing" indicator column.
- Categorical: fill with mode or a dedicated "unknown" token.
- Time series: forward-fill or interpolate.

---

## Text Preprocessing

Standard NLP preprocessing pipeline:

1. **Lowercase** — reduce vocabulary size; "Cat" and "cat" become the same token.
2. **Remove punctuation / special characters** — depends on task; punctuation matters for sentiment.
3. **Tokenize** — split into words (whitespace), subwords (BPE/WordPiece), or characters.
4. **Remove stop words** — high-frequency words ("the", "is") that carry little signal. Skip this step for LLMs, which handle them naturally.
5. **Lemmatize / stem** — reduce words to base form ("running" → "run"). Useful for classical ML; unnecessary for embedding-based models.

For LLMs specifically, use the model's own tokenizer (e.g., `AutoTokenizer`) rather than custom preprocessing — the model was trained on specific tokenized inputs and assumes that exact format.

---

## Chunking Documents for LLMs

LLMs have a finite context window. Long documents must be split into chunks before embedding or prompting.

| Strategy | Description | Tradeoff |
|----------|-------------|----------|
| **Fixed-size** | Split every N tokens or characters | Simple; may break mid-sentence |
| **Recursive** | Split on `\n\n`, `\n`, `.`, ` ` in order until chunk is small enough | Preserves structure better |
| **Semantic** | Group sentences by embedding similarity | Coherent chunks; expensive to compute |

**Overlap** — include the last K tokens of each chunk at the start of the next. Prevents losing context at chunk boundaries. Typical values: 10–20% of chunk size.

```
Chunk 1: [tokens 0–200]
Chunk 2: [tokens 180–380]   ← 20-token overlap
Chunk 3: [tokens 360–560]
```

---

## Batching

A **batch** is a group of samples processed together in one forward pass. Batching enables:
- **GPU parallelism** — matrix operations over a batch are much faster than sequential single-sample operations.
- **Stable gradient estimates** — averaging gradients over a batch reduces noise compared to single-sample (SGD).

**Batch size tradeoffs:**

| Batch Size | Memory | Training Speed | Gradient Quality |
|------------|--------|----------------|------------------|
| Too small (1–4) | Low | Slow (many steps) | Noisy, unstable |
| Good (32–256) | Moderate | Fast | Stable |
| Too large (>2048) | High (OOM risk) | Fast per step | Sharp minima, may generalize worse |

Start with 32 or 64 and adjust based on GPU memory.

---

## Reproducibility

A pipeline that produces different results each run cannot be debugged or compared. Best practices:

- **Random seeds**: set `random.seed(42)`, `numpy.random.seed(42)`, and `torch.manual_seed(42)` at the start.
- **Stratified splits**: when splitting train/val/test, preserve class proportions per split.
- **DVC (Data Version Control)**: tracks dataset versions alongside code in git. Running `dvc repro` regenerates outputs from a locked set of inputs + pipeline steps.
- **Environment pinning**: `pip freeze > requirements.txt` or use `poetry.lock` / `conda-lock`.

---

## Class Imbalance

When one class vastly outnumbers another (e.g., 99% negative, 1% positive), naive training collapses to always predicting the majority class.

**Solutions:**

| Technique | How | When |
|-----------|-----|------|
| **Class weights** | Scale loss by `1/class_frequency`; PyTorch `CrossEntropyLoss(weight=...)` | Easiest; built in |
| **Oversampling** | Duplicate minority class samples (random) or synthesize new ones (SMOTE: interpolate between neighbors) | Small datasets |
| **Undersampling** | Randomly remove majority class samples | Large datasets; accepts information loss |
| **Threshold tuning** | Adjust decision threshold away from 0.5 at inference time | When model outputs probabilities |

SMOTE (Synthetic Minority Oversampling Technique) generates new minority samples by interpolating between existing ones in feature space — not just exact copies.

---

## Pipeline Design Principles

**Functional transformations** — each step should be a pure function: same input, same output, no side effects. This makes steps independently testable and composable.

**sklearn `Pipeline`** — chains preprocessing steps and a final estimator. `pipeline.fit(X_train)` fits all steps; `pipeline.predict(X_test)` applies all transforms then predicts. Prevents data leakage: fit parameters (e.g., scaler mean) are computed only on training data, then applied to val/test.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression()),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)   # scaler uses train statistics — no leakage
```

In AI engineering, the same principle applies: fit your tokenizer vocabulary, scaler statistics, and encoders on training data only, then apply those fitted objects to validation and test sets.
