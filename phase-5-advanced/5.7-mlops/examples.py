"""
5.7 MLOps for AI Systems — Examples
Demonstrates: model registry, A/B testing, prompt versioning, canary deployment, feedback loops.
Run: python examples.py
"""

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Registry
# ─────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    """Track model versions and manage production promotion."""

    def __init__(self):
        self._models: dict[str, list[dict]] = {}   # name → list of versions
        self._production: dict[str, str] = {}       # name → version

    def register(self, name: str, version: str, metadata: dict) -> dict:
        if name not in self._models:
            self._models[name] = []
        entry = {
            "name": name,
            "version": version,
            "metadata": metadata,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "stage": "registered",
        }
        self._models[name].append(entry)
        return entry

    def promote(self, name: str, version: str) -> None:
        """Promote a version to production. Demotes previous production version."""
        for entry in self._models.get(name, []):
            if entry["version"] == self._production.get(name):
                entry["stage"] = "archived"
            if entry["version"] == version:
                entry["stage"] = "production"
        self._production[name] = version

    def rollback(self, name: str) -> Optional[str]:
        """Rollback to the previous non-archived version."""
        versions = self._models.get(name, [])
        archived = [v for v in versions if v["stage"] == "archived"]
        if not archived:
            print(f"  No archived version to rollback to for {name}")
            return None
        prev = archived[-1]
        self.promote(name, prev["version"])
        print(f"  Rolled back {name} to {prev['version']}")
        return prev["version"]

    def get_production(self, name: str) -> Optional[dict]:
        prod_ver = self._production.get(name)
        if not prod_ver:
            return None
        for entry in self._models.get(name, []):
            if entry["version"] == prod_ver:
                return entry
        return None

    def list_versions(self, name: str) -> None:
        print(f"\n  Model: {name}")
        print(f"  {'Version':<12} {'Stage':<14} {'Eval Score':<14} Registered")
        print(f"  {'-'*60}")
        for v in self._models.get(name, []):
            score = v["metadata"].get("eval_score", "N/A")
            print(f"  {v['version']:<12} {v['stage']:<14} {str(score):<14} {v['registered_at'][:19]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. A/B Test Router
# ─────────────────────────────────────────────────────────────────────────────

class ABTestRouter:
    """Route requests to model A or B. Consistent per request_id. Track outcomes."""

    def __init__(self, model_a: str, model_b: str, traffic_split: float = 0.1):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split   # fraction going to model_b
        self._outcomes: dict[str, dict] = {}

    def route(self, request_id: str) -> str:
        """Consistent hash: same request_id always goes to same model."""
        h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        fraction = (h % 10000) / 10000.0
        return self.model_b if fraction < self.traffic_split else self.model_a

    def record_outcome(self, request_id: str, success: bool, rating: int = 0) -> None:
        model = self.route(request_id)
        self._outcomes[request_id] = {"model": model, "success": success, "rating": rating}

    def get_stats(self) -> dict:
        stats: dict[str, dict] = {self.model_a: {"total": 0, "success": 0, "ratings": []},
                                   self.model_b: {"total": 0, "success": 0, "ratings": []}}
        for o in self._outcomes.values():
            m = o["model"]
            stats[m]["total"] += 1
            if o["success"]:
                stats[m]["success"] += 1
            if o.get("rating"):
                stats[m]["ratings"].append(o["rating"])

        result = {}
        for model, s in stats.items():
            n = s["total"]
            result[model] = {
                "requests": n,
                "success_rate": round(s["success"] / n, 3) if n > 0 else 0,
                "avg_rating": round(sum(s["ratings"]) / len(s["ratings"]), 2) if s["ratings"] else None,
            }
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prompt Version Control
# ─────────────────────────────────────────────────────────────────────────────

class PromptVersionControl:
    """Store, version, and manage prompt lifecycle."""

    def __init__(self):
        self._prompts: dict[str, list[dict]] = {}
        self._production: dict[str, int] = {}

    def save(self, name: str, content: str, metadata: dict = None) -> int:
        if name not in self._prompts:
            self._prompts[name] = []
        version = len(self._prompts[name]) + 1
        self._prompts[name].append({
            "version": version,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_production": False,
        })
        return version

    def get(self, name: str, version: int = None) -> Optional[str]:
        versions = self._prompts.get(name, [])
        if not versions:
            return None
        if version is None:
            prod_ver = self._production.get(name)
            if prod_ver:
                for v in versions:
                    if v["version"] == prod_ver:
                        return v["content"]
            return versions[-1]["content"]
        for v in versions:
            if v["version"] == version:
                return v["content"]
        return None

    def promote_to_production(self, name: str, version: int) -> None:
        for v in self._prompts.get(name, []):
            v["is_production"] = (v["version"] == version)
        self._production[name] = version

    def diff(self, name: str, v1: int, v2: int) -> str:
        c1 = self.get(name, v1) or ""
        c2 = self.get(name, v2) or ""
        lines1 = c1.splitlines()
        lines2 = c2.splitlines()
        out = [f"diff v{v1} → v{v2}"]
        for line in lines1:
            if line not in lines2:
                out.append(f"  - {line}")
        for line in lines2:
            if line not in lines1:
                out.append(f"  + {line}")
        return "\n".join(out) if len(out) > 1 else "  (no changes)"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Canary Deployment
# ─────────────────────────────────────────────────────────────────────────────

class CanaryDeployment:
    """Gradually shift traffic from old to new version."""

    def __init__(self, old_version: str, new_version: str, start_pct: int = 5):
        self.old_version = old_version
        self.new_version = new_version
        self.canary_pct = start_pct
        self._rolled_back = False

    def get_version(self, request_id: str) -> str:
        if self._rolled_back:
            return self.old_version
        h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        fraction = (h % 100)
        return self.new_version if fraction < self.canary_pct else self.old_version

    def increment_canary(self, pct: int) -> None:
        self.canary_pct = min(100, pct)
        print(f"  Canary → {self.canary_pct}% on {self.new_version}")

    def rollback(self) -> None:
        self._rolled_back = True
        self.canary_pct = 0
        print(f"  ROLLBACK: all traffic → {self.old_version}")

    def status(self) -> str:
        if self._rolled_back:
            return f"ROLLED BACK: 100% {self.old_version}"
        if self.canary_pct >= 100:
            return f"COMPLETE: 100% {self.new_version}"
        return f"{self.old_version}: {100 - self.canary_pct}% | {self.new_version}: {self.canary_pct}%"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feedback Collector
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackCollector:
    """Collect and analyze production feedback signals."""

    def __init__(self):
        self._records: list[dict] = []

    def record(self, session_id: str, rating: int, regenerated: bool = False,
               copied: bool = False, escalated: bool = False, response: str = "") -> None:
        self._records.append({
            "session_id": session_id,
            "rating": rating,        # 1-5 or 0 if no rating
            "regenerated": regenerated,
            "copied": copied,
            "escalated": escalated,
            "response_preview": response[:50],
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    def quality_metrics(self) -> dict:
        n = len(self._records)
        if n == 0:
            return {}
        rated = [r for r in self._records if r["rating"] > 0]
        avg_rating = sum(r["rating"] for r in rated) / len(rated) if rated else 0
        regen_rate = sum(1 for r in self._records if r["regenerated"]) / n
        copy_rate = sum(1 for r in self._records if r["copied"]) / n
        escalation_rate = sum(1 for r in self._records if r["escalated"]) / n
        return {
            "total_sessions": n,
            "avg_rating": round(avg_rating, 2),
            "thumbs_down_rate": round(sum(1 for r in rated if r["rating"] <= 2) / len(rated), 3) if rated else 0,
            "regeneration_rate": round(regen_rate, 3),
            "copy_rate": round(copy_rate, 3),
            "escalation_rate": round(escalation_rate, 3),
        }

    def export_for_finetuning(self, min_rating: int = 4) -> list[dict]:
        """Export high-quality responses as fine-tuning candidates."""
        return [
            {"response": r["response_preview"], "rating": r["rating"]}
            for r in self._records
            if r["rating"] >= min_rating and not r["regenerated"]
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    print("=" * 60)
    print("MLOps for AI Systems — Demo")
    print("=" * 60)

    # ── Model Registry ──
    print("\n── Model Registry ──")
    registry = ModelRegistry()
    registry.register("rag-assistant", "v1.0.0", {"eval_score": 3.8, "notes": "baseline"})
    registry.register("rag-assistant", "v1.1.0", {"eval_score": 4.1, "notes": "improved chunking"})
    registry.register("rag-assistant", "v1.2.0", {"eval_score": 4.3, "notes": "hybrid search"})
    registry.promote("rag-assistant", "v1.0.0")  # first production
    registry.promote("rag-assistant", "v1.2.0")  # latest promotion
    registry.list_versions("rag-assistant")
    print(f"  Production: {registry.get_production('rag-assistant')['version']}")
    registry.rollback("rag-assistant")
    registry.list_versions("rag-assistant")

    # ── A/B Test ──
    print("\n── A/B Test Router (10% canary) ──")
    ab = ABTestRouter("sonnet-v1-prompt", "sonnet-v2-prompt", traffic_split=0.1)
    for i in range(200):
        req_id = f"req_{i:04d}"
        model = ab.route(req_id)
        # Simulate v2 being slightly better
        success = random.random() < (0.88 if model == "sonnet-v2-prompt" else 0.82)
        rating = random.randint(3, 5) if model == "sonnet-v2-prompt" else random.randint(2, 5)
        ab.record_outcome(req_id, success, rating)
    stats = ab.get_stats()
    print(f"  {'Model':<25} {'Requests':>10} {'Success':>10} {'Avg Rating':>12}")
    print(f"  {'-'*60}")
    for model, s in stats.items():
        print(f"  {model:<25} {s['requests']:>10} {s['success_rate']:>10.1%} {str(s['avg_rating']):>12}")

    # ── Prompt Versioning ──
    print("\n── Prompt Version Control ──")
    pvc = PromptVersionControl()
    v1 = pvc.save("support", "You are a helpful customer support agent. Answer questions.", {"eval": 3.9})
    v2 = pvc.save("support", "You are a precise, empathetic support agent. Be concise and cite sources.", {"eval": 4.2})
    pvc.promote_to_production("support", v1)
    print(f"  v1 in production: {pvc.get('support')[:60]}...")
    pvc.promote_to_production("support", v2)
    print(f"  v2 in production: {pvc.get('support')[:60]}...")
    print(pvc.diff("support", v1, v2))

    # ── Canary Deployment ──
    print("\n── Canary Deployment ──")
    canary = CanaryDeployment("llama-3-8b-v1", "llama-3-8b-v2", start_pct=5)
    reqs = [f"r{i}" for i in range(100)]
    v_counts: dict[str, int] = {}
    for r in reqs:
        v = canary.get_version(r)
        v_counts[v] = v_counts.get(v, 0) + 1
    print(f"  Initial: {canary.status()} → actual split: {v_counts}")
    canary.increment_canary(25)
    canary.increment_canary(100)
    print(f"  Final:   {canary.status()}")

    # ── Feedback Loop ──
    print("\n── Feedback Collection ──")
    fc = FeedbackCollector()
    for i in range(50):
        fc.record(
            session_id=f"s{i}",
            rating=random.choices([1, 2, 3, 4, 5], weights=[2, 3, 10, 20, 15])[0],
            regenerated=random.random() < 0.12,
            copied=random.random() < 0.35,
            escalated=random.random() < 0.05,
            response=f"The answer to query {i} is...",
        )
    metrics = fc.quality_metrics()
    print(f"  Total sessions:     {metrics['total_sessions']}")
    print(f"  Avg rating:         {metrics['avg_rating']}/5")
    print(f"  Thumbs-down rate:   {metrics['thumbs_down_rate']:.1%}")
    print(f"  Regeneration rate:  {metrics['regeneration_rate']:.1%}")
    print(f"  Copy rate:          {metrics['copy_rate']:.1%}")
    print(f"  Escalation rate:    {metrics['escalation_rate']:.1%}")
    finetuning_data = fc.export_for_finetuning(min_rating=4)
    print(f"  Fine-tuning candidates (rating ≥ 4): {len(finetuning_data)}")
