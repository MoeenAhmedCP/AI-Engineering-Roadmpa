"""
2.8 Experiment Tracking — Examples
No external dependencies required at top level.
Optional: pip install mlflow
Run: python examples.py
"""

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. InMemoryTracker — drop-in experiment logger
# ---------------------------------------------------------------------------

class InMemoryTracker:
    """
    Lightweight, in-memory experiment tracker that mirrors the MLflow API.
    Use when mlflow is not installed or for unit tests.
    """

    def __init__(self):
        self._runs: list = []
        self._current: dict = {}
        self._active: bool = False

    # ---- context manager ----

    def start_run(self, run_name: str = ""):
        """Begin recording a new run. Returns self for use as context manager."""
        if self._active:
            raise RuntimeError("A run is already active. Call end_run() first.")
        self._current = {
            "run_id":   hashlib.md5(f"{run_name}{time.time()}".encode()).hexdigest()[:8],
            "run_name": run_name or f"run_{len(self._runs)}",
            "params":   {},
            "metrics":  defaultdict(list),   # metric_name -> [(step, value), ...]
            "artifacts":{},
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "end_time": None,
            "status": "RUNNING",
        }
        self._active = True
        return self

    def end_run(self, status: str = "FINISHED"):
        if not self._active:
            return
        self._current["end_time"] = datetime.now().isoformat(timespec="seconds")
        self._current["status"] = status
        # Freeze defaultdict into plain dict
        self._current["metrics"] = dict(self._current["metrics"])
        self._runs.append(self._current)
        self._current = {}
        self._active = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status)
        return False   # do not suppress exceptions

    # ---- logging methods ----

    def _assert_active(self):
        if not self._active:
            raise RuntimeError("No active run. Call start_run() first.")

    def log_param(self, key: str, value):
        self._assert_active()
        self._current["params"][key] = value

    def log_metric(self, key: str, value: float, step: int = None):
        self._assert_active()
        step = step if step is not None else len(self._current["metrics"][key])
        self._current["metrics"][key].append((step, value))

    def log_artifact(self, name: str, content: str):
        """Store arbitrary string content as a named artifact."""
        self._assert_active()
        self._current["artifacts"][name] = content

    # ---- retrieval ----

    def get_summary(self) -> dict:
        """Return summary of the most recently completed run."""
        if not self._runs:
            return {}
        run = self._runs[-1]
        # Summarise metrics: take the last value per metric
        metric_summary = {
            k: round(v[-1][1], 6) for k, v in run["metrics"].items()
        }
        return {
            "run_id":   run["run_id"],
            "run_name": run["run_name"],
            "params":   run["params"],
            "metrics":  metric_summary,
            "status":   run["status"],
        }

    @staticmethod
    def compare_runs(summaries: list):
        """Print a comparison table of run summaries."""
        if not summaries:
            print("  No runs to compare.")
            return

        # Collect all metric keys
        all_metrics = sorted({k for s in summaries for k in s.get("metrics", {})})
        all_params  = sorted({k for s in summaries for k in s.get("params", {})})

        col_w = 18

        # Header
        header = f"{'Run':12s} {'Status':10s}"
        for p in all_params:
            header += f"  {p[:col_w]:>{col_w}}"
        for m in all_metrics:
            header += f"  {m[:col_w]:>{col_w}}"
        print("  " + header)
        print("  " + "-" * len(header))

        for s in summaries:
            row = f"{s['run_name'][:12]:12s} {s['status'][:10]:10s}"
            for p in all_params:
                val = str(s["params"].get(p, "-"))[:col_w]
                row += f"  {val:>{col_w}}"
            for m in all_metrics:
                val = str(s["metrics"].get(m, "-"))[:col_w]
                row += f"  {val:>{col_w}}"
            print("  " + row)


# ---------------------------------------------------------------------------
# 2. Try real mlflow, fall back to InMemoryTracker
# ---------------------------------------------------------------------------

try:
    import mlflow
    _TRACKER_BACKEND = "mlflow"
except ImportError:
    mlflow = None
    _TRACKER_BACKEND = "in_memory"

_global_tracker = InMemoryTracker()


# ---------------------------------------------------------------------------
# 3. generate_run_id
# ---------------------------------------------------------------------------

def generate_run_id(config: dict) -> str:
    """
    Deterministic run ID: SHA-256 hash of sorted config key-value pairs.
    Same config always produces the same ID — useful for caching.
    """
    serialised = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(serialised).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 4. track_experiment
# ---------------------------------------------------------------------------

def track_experiment(config: dict, simulate_training: bool = True) -> dict:
    """
    Log an experiment run.
    If mlflow is available and a tracking URI is set, uses real mlflow.
    Otherwise uses the in-process InMemoryTracker.

    Returns a summary dict of the completed run.
    """
    run_name = (
        f"lr={config.get('lr','?')}_bs={config.get('batch_size','?')}"
    )
    run_id = generate_run_id(config)

    if _TRACKER_BACKEND == "mlflow":
        return _track_with_mlflow(config, run_name, simulate_training)
    else:
        return _track_with_inmemory(config, run_name, simulate_training)


def _track_with_inmemory(config: dict, run_name: str, simulate: bool) -> dict:
    global _global_tracker
    with _global_tracker.start_run(run_name=run_name):
        for k, v in config.items():
            _global_tracker.log_param(k, v)

        if simulate:
            rng = random.Random(hash(run_name) % (2**31))
            epochs = config.get("epochs", 10)
            lr = float(config.get("lr", 0.01))
            loss = 1.0

            for epoch in range(epochs):
                # Simulated loss decay with noise
                loss *= (1 - lr * (0.5 + 0.5 * rng.random()))
                loss = max(0.01, loss + rng.gauss(0, 0.005))
                val_loss = loss * (1 + rng.gauss(0, 0.03))
                _global_tracker.log_metric("train_loss", round(loss, 5), step=epoch)
                _global_tracker.log_metric("val_loss",   round(val_loss, 5), step=epoch)

            val_acc = min(0.99, 0.5 + (1 - loss) * 0.45 + rng.gauss(0, 0.01))
            _global_tracker.log_metric("val_accuracy", round(val_acc, 4))
            _global_tracker.log_artifact(
                "config.json", json.dumps(config, indent=2)
            )

    return _global_tracker.get_summary()


def _track_with_mlflow(config: dict, run_name: str, simulate: bool) -> dict:
    """Real MLflow tracking (only called if mlflow is importable)."""
    import mlflow

    with mlflow.start_run(run_name=run_name) as run:
        for k, v in config.items():
            mlflow.log_param(k, v)

        if simulate:
            rng = random.Random(hash(run_name) % (2**31))
            epochs = config.get("epochs", 10)
            lr = float(config.get("lr", 0.01))
            loss = 1.0

            for epoch in range(epochs):
                loss *= (1 - lr * (0.5 + 0.5 * rng.random()))
                loss = max(0.01, loss + rng.gauss(0, 0.005))
                val_loss = loss * (1 + rng.gauss(0, 0.03))
                mlflow.log_metric("train_loss", round(loss, 5), step=epoch)
                mlflow.log_metric("val_loss",   round(val_loss, 5), step=epoch)

            val_acc = min(0.99, 0.5 + (1 - loss) * 0.45 + rng.gauss(0, 0.01))
            mlflow.log_metric("val_accuracy", round(val_acc, 4))

        run_id = run.info.run_id

    # Build a summary compatible with compare_runs
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)
    return {
        "run_id":   run_id[:8],
        "run_name": run_name,
        "params":   dict(run_data.data.params),
        "metrics":  {k: round(v, 6) for k, v in run_data.data.metrics.items()},
        "status":   run_data.info.status,
    }


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sep = "=" * 60
    print(sep)
    print(f"2.8 Experiment Tracking — Demo")
    print(f"Backend: {_TRACKER_BACKEND}")
    print(sep)

    # Define three configurations to compare
    configs = [
        {"lr": 0.1,   "batch_size": 32,  "epochs": 10, "optimizer": "adam",  "dropout": 0.1},
        {"lr": 0.01,  "batch_size": 64,  "epochs": 10, "optimizer": "adam",  "dropout": 0.2},
        {"lr": 0.001, "batch_size": 128, "epochs": 10, "optimizer": "adamw", "dropout": 0.3},
    ]

    print("\nRunning 3 simulated experiments...\n")
    summaries = []
    for cfg in configs:
        summary = track_experiment(cfg, simulate_training=True)
        summaries.append(summary)
        print(f"  Completed: {summary['run_name'][:40]:40s}  "
              f"val_accuracy={summary['metrics'].get('val_accuracy', 'N/A')}")

    # Compare in table
    print(f"\n{sep}")
    print("Run Comparison Table")
    print(sep)
    InMemoryTracker.compare_runs(summaries)

    # Best run by val_accuracy
    best = max(summaries, key=lambda s: s["metrics"].get("val_accuracy", 0))
    print(f"\n  Best run  : {best['run_name']}")
    print(f"  val_acc   : {best['metrics'].get('val_accuracy', 'N/A')}")
    print(f"  val_loss  : {best['metrics'].get('val_loss', 'N/A')}")
    print(f"  Params    : {best['params']}")

    # Demonstrate generate_run_id determinism
    print(f"\n{sep}")
    print("generate_run_id demo (deterministic)")
    print(sep)
    cfg_a = {"lr": 0.01, "batch_size": 64}
    cfg_b = {"batch_size": 64, "lr": 0.01}  # same params, different order
    cfg_c = {"lr": 0.02, "batch_size": 64}  # different value
    print(f"  config A run_id : {generate_run_id(cfg_a)}")
    print(f"  config B run_id : {generate_run_id(cfg_b)}  (same as A — order-independent)")
    print(f"  config C run_id : {generate_run_id(cfg_c)}  (different)")
    print(f"  A==B: {generate_run_id(cfg_a) == generate_run_id(cfg_b)}")
    print(f"  A==C: {generate_run_id(cfg_a) == generate_run_id(cfg_c)}")

    print(f"\n{sep}")
    print("Done.")
    print(sep)
