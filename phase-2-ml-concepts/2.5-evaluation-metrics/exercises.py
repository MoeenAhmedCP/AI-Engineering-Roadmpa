"""
2.5 Evaluation Metrics — Exercises
Attempt each before reading solutions.
Run: python exercises.py
"""

import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# EXERCISE 1 — Macro-average precision, recall, and F1 for multi-class
# y_true and y_pred are lists of class labels (strings or ints).
# Return dict: {"precision": float, "recall": float, "f1": float}
# ---------------------------------------------------------------------------

def macro_prf1(y_true: list, y_pred: list) -> dict:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 2 — Find optimal threshold from ROC data
# roc_points: list of (fpr, tpr, threshold) tuples (from sklearn or hand-built)
# Maximise F1 by trying each threshold. Return the best threshold value.
# Hint: precision = TP/(TP+FP), recall = TPR — but here use fpr/tpr + base rate.
# Simpler: iterate thresholds, recompute TP/FP/FN from y_true, y_score, pick best F1.
# ---------------------------------------------------------------------------

def optimal_threshold(y_true: list, y_score: list) -> float:
    """Return threshold in y_score that maximises F1 on binary labels."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 3 — BLEU-2 score (bigram)
# reference: list of reference tokens, hypothesis: list of hypothesis tokens
# BLEU-2 = BP * exp(0.5 * log(p1) + 0.5 * log(p2))
# BP = min(1, exp(1 - len(ref)/len(hyp)))
# p1, p2 = clipped unigram and bigram precision
# ---------------------------------------------------------------------------

def bleu2(reference: list, hypothesis: list) -> float:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 4 — Mean Average Precision at K (MAP@K)
# predictions: list of lists (ranked results per query)
# actuals:     list of sets (relevant items per query)
# Return MAP@K as float.
# ---------------------------------------------------------------------------

def map_at_k(predictions: list, actuals: list, k: int) -> float:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 5 — Accuracy vs F1 on imbalanced dataset
# Create a synthetic dataset: 990 class-0, 10 class-1.
# Model A predicts all 0. Model B predicts correctly (85% class-1 recall, no FP).
# Print accuracy and F1 for both. Show why accuracy is misleading here.
# No skeleton needed — implement directly.
# ---------------------------------------------------------------------------

def imbalanced_demo():
    raise NotImplementedError


# ---------------------------------------------------------------------------
# EXERCISE 6 — Expected Calibration Error (ECE) with 10 equal-width bins
# probs:   list of predicted probabilities (floats 0–1)
# actuals: list of binary labels (0 or 1)
# ECE = sum_b (|B_b| / N) * |avg_confidence(b) - avg_accuracy(b)|
# ---------------------------------------------------------------------------

def expected_calibration_error(probs: list, actuals: list, n_bins: int = 10) -> float:
    raise NotImplementedError


# ===========================================================================
# SOLUTIONS
# ===========================================================================

def _sol_macro_prf1(y_true: list, y_pred: list) -> dict:
    classes = set(y_true) | set(y_pred)
    precisions, recalls, f1s = [], [], []

    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    n = len(classes)
    return {
        "precision": round(sum(precisions) / n, 4),
        "recall":    round(sum(recalls) / n, 4),
        "f1":        round(sum(f1s) / n, 4),
    }


def _sol_optimal_threshold(y_true: list, y_score: list) -> float:
    """Sweep every unique score as candidate threshold; maximise F1."""
    best_f1, best_thresh = -1.0, 0.5

    for thresh in sorted(set(y_score)):
        y_pred = [1 if s >= thresh else 0 for s in y_score]
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return round(best_thresh, 4)


def _ngram_counts(tokens: list, n: int) -> dict:
    counts = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i : i + n])] += 1
    return counts


def _sol_bleu2(reference: list, hypothesis: list) -> float:
    if not hypothesis:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(reference) / len(hypothesis)))

    # Clipped precision for n=1 and n=2
    def clipped_precision(n):
        ref_counts = _ngram_counts(reference, n)
        hyp_counts = _ngram_counts(hypothesis, n)
        clipped = sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
        total = sum(hyp_counts.values())
        return clipped / total if total > 0 else 0.0

    p1 = clipped_precision(1)
    p2 = clipped_precision(2)

    if p1 == 0 or p2 == 0:
        return 0.0

    log_bleu = 0.5 * math.log(p1) + 0.5 * math.log(p2)
    return round(bp * math.exp(log_bleu), 4)


def _sol_map_at_k(predictions: list, actuals: list, k: int) -> float:
    def average_precision_at_k(pred, relevant, k):
        hits = 0
        score = 0.0
        for i, item in enumerate(pred[:k]):
            if item in relevant:
                hits += 1
                score += hits / (i + 1)
        denom = min(len(relevant), k)
        return score / denom if denom > 0 else 0.0

    if not predictions:
        return 0.0

    ap_scores = [
        average_precision_at_k(pred, set(rel), k)
        for pred, rel in zip(predictions, actuals)
    ]
    return round(sum(ap_scores) / len(ap_scores), 4)


def _sol_imbalanced_demo():
    # Dataset: 990 class-0, 10 class-1
    y_true = [0] * 990 + [1] * 10
    n = len(y_true)

    # Model A: always predicts 0
    pred_a = [0] * n

    # Model B: correctly identifies 8/10 class-1 (85% recall) with 0 false positives
    pred_b = [0] * 990 + [1] * 8 + [0] * 2  # 8 TP, 0 FP, 2 FN

    def compute_metrics(y_t, y_p, name):
        acc = sum(1 for a, b in zip(y_t, y_p) if a == b) / len(y_t)
        tp = sum(1 for a, b in zip(y_t, y_p) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_t, y_p) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_t, y_p) if a == 1 and b == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {name}: accuracy={acc:.3f}  precision={prec:.3f}  "
              f"recall={rec:.3f}  F1={f1:.3f}")

    print("  Dataset: 990 class-0, 10 class-1 (1% minority)")
    compute_metrics(y_true, pred_a, "Model A (all-zero)")
    compute_metrics(y_true, pred_b, "Model B (real detection)")
    print("  => Model A has 99% accuracy but F1=0.0 — it learns nothing.")
    print("  => Model B has lower accuracy but meaningful F1.")


def _sol_expected_calibration_error(probs: list, actuals: list, n_bins: int = 10) -> float:
    bins = [[] for _ in range(n_bins)]
    bin_width = 1.0 / n_bins

    for p, a in zip(probs, actuals):
        idx = min(int(p / bin_width), n_bins - 1)
        bins[idx].append((p, a))

    ece = 0.0
    N = len(probs)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(p for p, _ in b) / len(b)
        avg_acc  = sum(a for _, a in b) / len(b)
        ece += (len(b) / N) * abs(avg_conf - avg_acc)

    return round(ece, 6)


# ---------------------------------------------------------------------------

def solutions():
    sep = "=" * 55
    print(sep)
    print("SOLUTIONS — 2.5 Evaluation Metrics Exercises")
    print(sep)

    # --- Exercise 1 ---
    print("\n[Exercise 1] macro_prf1 (multi-class)")
    y_true = ["cat", "dog", "cat", "bird", "dog", "cat", "bird", "dog"]
    y_pred = ["cat", "cat", "cat", "bird", "dog",  "dog", "bird", "cat"]
    result = _sol_macro_prf1(y_true, y_pred)
    print(f"  y_true: {y_true}")
    print(f"  y_pred: {y_pred}")
    print(f"  Macro P={result['precision']}  R={result['recall']}  F1={result['f1']}")

    # --- Exercise 2 ---
    print("\n[Exercise 2] optimal_threshold")
    import random
    random.seed(42)
    y_true_bin = [1]*30 + [0]*70
    # Positive scores ~ 0.7, negative ~ 0.3 (noisy)
    y_score = [0.5 + random.gauss(0.25, 0.15) for _ in range(30)] + \
              [0.5 + random.gauss(-0.2, 0.15) for _ in range(70)]
    y_score = [max(0.0, min(1.0, s)) for s in y_score]
    thresh = _sol_optimal_threshold(y_true_bin, y_score)
    print(f"  Best threshold = {thresh}")

    # --- Exercise 3 ---
    print("\n[Exercise 3] BLEU-2 score")
    ref  = "the cat sat on the mat".split()
    hyp1 = "the cat sat on the mat".split()   # perfect
    hyp2 = "a dog lay on the floor".split()   # poor
    hyp3 = "the cat sat on a mat".split()     # near-perfect
    print(f"  Ref : {ref}")
    print(f"  Hyp1 (perfect)   : BLEU-2 = {_sol_bleu2(ref, hyp1)}")
    print(f"  Hyp2 (poor)      : BLEU-2 = {_sol_bleu2(ref, hyp2)}")
    print(f"  Hyp3 (near-perf) : BLEU-2 = {_sol_bleu2(ref, hyp3)}")

    # --- Exercise 4 ---
    print("\n[Exercise 4] MAP@K")
    preds   = [["a","b","c","d"], ["b","c","a","d"], ["c","a","b","d"]]
    actuals = [{"a","c"},          {"a","b"},          {"a","b","c"}]
    for k in [2, 3, 4]:
        score = _sol_map_at_k(preds, actuals, k)
        print(f"  MAP@{k} = {score}")

    # --- Exercise 5 ---
    print("\n[Exercise 5] imbalanced dataset demo")
    _sol_imbalanced_demo()

    # --- Exercise 6 ---
    print("\n[Exercise 6] Expected Calibration Error (ECE)")
    import random
    random.seed(7)
    # Well-calibrated: prob ~ actual rate
    probs_good   = [random.uniform(0, 1) for _ in range(200)]
    actuals_good = [1 if random.random() < p else 0 for p in probs_good]
    # Overconfident: always predicts near 0.9, but true rate is 50%
    probs_bad    = [random.uniform(0.75, 0.99) for _ in range(200)]
    actuals_bad  = [random.randint(0, 1) for _ in range(200)]

    ece_good = _sol_expected_calibration_error(probs_good, actuals_good)
    ece_bad  = _sol_expected_calibration_error(probs_bad, actuals_bad)
    print(f"  Well-calibrated ECE : {ece_good:.4f}  (expect: low)")
    print(f"  Overconfident ECE   : {ece_bad:.4f}  (expect: high)")

    print(f"\n{sep}")
    print("All solutions complete.")
    print(sep)


if __name__ == "__main__":
    print("Attempt the exercises above, then call solutions() to compare.\n")
    solutions()
