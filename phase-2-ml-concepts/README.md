# Phase 2: ML Concepts

Core machine learning foundations for AI engineers. Each section pairs conceptual notes with runnable Python examples.

---

## Overview Table

| # | Topic | Key Concept | Mini-Project |
|---|-------|-------------|--------------|
| 2.1 | How ML Works | Training loop, loss functions, gradient descent | Linear regression from scratch with numpy |
| 2.2 | Neural Networks | Backprop, activations, vanishing gradients | XOR solver + activation function table |
| 2.3 | ML Algorithms | Trees, boosting, clustering | RandomForest vs GradientBoosting comparison |
| 2.4 | NLP Basics | Tokenization, TF-IDF, embeddings, BPE | TF-IDF search engine from scratch |
| 2.5 | Evaluation Metrics | Precision/recall, BLEU, ROUGE, perplexity | Full eval report + threshold tuning |
| 2.6 | PyTorch Basics | Tensors, autograd, nn.Module, HuggingFace | 3-layer MLP + sentiment pipeline |
| 2.7 | Data Pipelines | ETL, chunking strategies, feature scaling | 4 chunking strategies implemented |
| 2.8 | Experiment Tracking | MLflow, W&B, LangSmith, reproducibility | PromptExperiment tracker class |

---

## Progress Checklist

### 2.1 How ML Works
- [ ] Read notes.md (supervised/unsupervised/RL, training loop, loss functions)
- [ ] Run examples.py (linear regression from scratch)
- [ ] Understand 3 learning rate behaviors
- [ ] Understand overfitting demo (degree-8 polynomial)

### 2.2 Neural Networks
- [ ] Read notes.md (neuron math, activations, backprop, vanishing gradients)
- [ ] Run examples.py (XOR neural network from scratch)
- [ ] Study activation function comparison table
- [ ] Understand vanishing gradient demo (10-layer sigmoid)

### 2.3 ML Algorithms
- [ ] Read notes.md (linear regression through k-NN, feature engineering)
- [ ] Run examples.py (RandomForest vs GradientBoosting, KMeans)
- [ ] Complete exercises.py (k-NN from scratch, elbow method, one-hot encoding)

### 2.4 NLP Basics
- [ ] Read notes.md (tokenization, Word2Vec, TF-IDF, BPE)
- [ ] Run examples.py (TF-IDF search engine, n-grams, BPE demo)
- [ ] Understand why subword tokenization matters for LLMs

### 2.5 Evaluation Metrics
- [ ] Read notes.md (all classification metrics, BLEU, ROUGE, perplexity)
- [ ] Run examples.py (confusion matrix, imbalanced data trap, threshold sweep)
- [ ] Complete exercises.py (compute_metrics, f_beta_score, LLM-as-judge prompt)

### 2.6 PyTorch Basics
- [ ] Read notes.md (tensors, autograd, nn.Module, HuggingFace)
- [ ] Run examples.py (MLP training loop, sentiment analysis pipeline)
- [ ] Understand model.train() vs model.eval() difference

### 2.7 Data Pipelines
- [ ] Read notes.md (ETL, chunking strategies, feature scaling, encoding)
- [ ] Run examples.py (messy data cleaning, 4 chunking strategies)
- [ ] Understand when to use fixed-size vs recursive vs semantic chunking

### 2.8 Experiment Tracking
- [ ] Read notes.md (MLflow, W&B, LangSmith, reproducibility)
- [ ] Run examples.py (PromptExperiment class, ExperimentTracker)
- [ ] Understand what to log: params, metrics, artifacts, git hash
