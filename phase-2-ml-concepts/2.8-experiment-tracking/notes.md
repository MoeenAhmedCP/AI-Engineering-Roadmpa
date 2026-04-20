# 2.8 Experiment Tracking

Machine learning development is inherently iterative. You run dozens of experiments before finding what works. Without systematic tracking, you will eventually face the question: "which hyperparameters produced that result I saw three weeks ago?" Experiment tracking is the practice of recording everything needed to reproduce, compare, and understand each run.

---

## Why Experiment Tracking Matters

The **reproducibility crisis** in ML is real: researchers frequently cannot reproduce their own results when conditions change — different random seeds, a different library version, or an undocumented preprocessing step. In production, failing to reproduce a model means you cannot safely audit, retrain, or debug it.

Practically, experiment tracking solves everyday engineering problems:
- "Which model should I deploy — the one from Tuesday or Thursday?"
- "Did increasing dropout actually help?"
- "What was the batch size when we got 94% accuracy?"

Without a tracking system, these questions can only be answered by rereading notes, digging through shell history, or running everything again.

---

## What to Log

At minimum, log everything that could change the outcome:

**Hyperparameters:** learning rate, batch size, optimizer, architecture choices (number of layers, hidden dim), dropout rate, weight decay, loss function.

**Metrics per epoch:** training loss, validation loss, validation accuracy/F1/BLEU — whatever your task requires. Logging per epoch (not just final) lets you catch overfitting early.

**Final metrics:** the definitive numbers for the run.

**Model artifacts:** saved weights (`state_dict`), ONNX exports, vocabulary files. Link artifacts to the run that produced them.

**Environment metadata:** git commit hash, Python version, library versions (`pip freeze`), dataset path + hash (so you know which version of data was used).

**What NOT to log:** raw PII or sensitive data, excessively large intermediate files (use artifact pointers instead), noisy debug `print()` output that floods the log.

---

## MLflow

MLflow is an open-source platform for managing the ML lifecycle. It has four main components:

- **Tracking** — logs parameters, metrics, and artifacts to a local or remote backend.
- **Projects** — packages ML code in a reusable format.
- **Models** — standardizes model packaging for serving.
- **Model Registry** — versioned model store with staging/production lifecycle.

### Core API

```python
import mlflow

# Start a run (context manager auto-ends it)
with mlflow.start_run(run_name="experiment-1"):
    mlflow.log_param("lr", 0.001)
    mlflow.log_param("batch_size", 64)

    for epoch in range(num_epochs):
        loss = train_one_epoch(...)
        mlflow.log_metric("train_loss", loss, step=epoch)

    mlflow.log_artifact("model.pt")
    mlflow.log_metric("val_f1", 0.87)
```

Run `mlflow ui` in the directory containing `mlruns/` to open the browser dashboard. It shows all runs in a table, lets you sort/filter, and plots metrics over time.

---

## Weights & Biases (W&B)

W&B provides a richer cloud-based tracking UI with tighter integration for common frameworks.

Key differentiators from MLflow:
- **Sweeps** — automated hyperparameter search (grid, random, Bayesian). Define a search space in YAML; W&B agents run experiments in parallel.
- **System monitoring** — CPU, GPU, memory usage logged automatically.
- **Richer plots** — built-in histogram logging, image logging, confusion matrices.
- **Team collaboration** — shared project dashboards, report builder.

```python
import wandb

wandb.init(project="my-project", config={"lr": 0.001, "batch_size": 64})

for epoch in range(num_epochs):
    loss = train_one_epoch(...)
    wandb.log({"train_loss": loss, "epoch": epoch})

wandb.finish()
```

Use MLflow when you need a fully local/private setup with no external service. Use W&B when you want richer tooling, team dashboards, or hyperparameter sweeps.

---

## LangSmith (for LLMs)

LangSmith is Anthropic/LangChain's experiment tracking tool designed specifically for LLM applications.

What it adds beyond MLflow/W&B:
- **Prompt/response logging** — records the full input prompt, model response, latency, and token counts.
- **Prompt version comparison** — run the same test suite against two prompt variants and compare outputs side-by-side.
- **Chain tracing** — visualises multi-step LLM chains (retrieval → prompt → LLM → parsing).
- **Human evaluation** — UI for annotators to rate model outputs.

For classical ML experiments (training loss curves), use MLflow or W&B. For LLM prompt experiments, use LangSmith.

---

## Reproducibility Best Practices

1. **Fix random seeds** at the top of every training script:
   ```python
   import random, numpy as np, torch
   SEED = 42
   random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
   ```

2. **Use config files** (YAML/JSON/TOML), not hardcoded constants. Pass the config path as a CLI argument and log the entire config at the start of the run.

3. **Pin environments** — commit `requirements.txt` or use `poetry.lock`. Log the output of `pip freeze` as a run artifact.

4. **Containerize training** — Docker ensures the full environment (OS, CUDA version, system libraries) is reproducible, not just Python packages.

5. **Log the git hash** — log `git rev-parse HEAD` as a run parameter. This tells you exactly which code produced the result.

---

## A/B Testing Model Versions

When you have a new model version and want to verify it is actually better in production:

**Shadow mode:** Route all production traffic to Model A (current). Simultaneously send the same requests to Model B (new) but do not serve its responses to users. Log both sets of outputs. Compare offline. Zero risk to users.

**Canary deployment:** Route a small fraction of traffic (e.g., 5%) to Model B. Monitor error rates, latency, and business metrics. If stable, gradually ramp up. Roll back instantly if metrics degrade.

Track both experiments in your experiment system with the same metrics so you can make a data-driven decision to promote the new model.
