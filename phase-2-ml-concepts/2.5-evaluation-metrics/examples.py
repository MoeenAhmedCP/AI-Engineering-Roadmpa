"""
2.5 Evaluation Metrics — Examples
Stdlib only.

Covers:
- Confusion matrix from scratch
- Precision, recall, F1, accuracy
- Imbalanced data problem demonstration
- BLEU score from scratch (1-gram and 2-gram)
- ROUGE-1 from scratch
- eval_report() function
- Threshold tuning: precision/recall tradeoff
"""

import math
from collections import Counter


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# PART 1: Confusion Matrix from Scratch
# ---------------------------------------------------------------------------

def confusion_matrix(y_pred, y_true, positive_class=1):
    """
    Compute TP, FP, TN, FN from prediction and ground truth lists.
    Returns dict with all four values.
    """
    tp = fp = tn = fn = 0
    for pred, true in zip(y_pred, y_true):
        if pred == positive_class and true == positive_class:
            tp += 1
        elif pred == positive_class and true != positive_class:
            fp += 1
        elif pred != positive_class and true == positive_class:
            fn += 1
        else:
            tn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def print_confusion_matrix(cm):
    """Print a formatted confusion matrix."""
    tp, fp = cm["TP"], cm["FP"]
    fn, tn = cm["FN"], cm["TN"]
    print(f"  {'':25} {'Pred: POS':>12} {'Pred: NEG':>12}")
    print(f"  {'Actual: POS':<25} {tp:>12} {fn:>12}")
    print(f"  {'Actual: NEG':<25} {fp:>12} {tn:>12}")


def derive_metrics(cm):
    """Derive precision, recall, F1, accuracy from confusion matrix."""
    tp, fp = cm["TP"], cm["FP"]
    fn, tn = cm["FN"], cm["TN"]
    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
    }


def demo_confusion_matrix():
    # Concrete example: email spam detection (20 emails)
    y_true = [1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1]
    y_pred = [1,1,1,0,0, 1,0,0,0,0, 0,0,0,0,0, 1,1,0,0,1]
    # Meaning: 10 spam (1), 10 legit (0)
    # TP=6 (correctly caught spam), FP=1 (legit marked spam)
    # FN=4 (spam that got through), TN=9 (correctly cleared legit)

    cm = confusion_matrix(y_pred, y_true)
    metrics = derive_metrics(cm)

    print("  Spam detection example (20 emails, 10 spam):")
    print()
    print_confusion_matrix(cm)
    print()
    print(f"  TP={cm['TP']}, FP={cm['FP']}, TN={cm['TN']}, FN={cm['FN']}")
    print()
    print(f"  Precision: {metrics['precision']:.4f}  (of emails we flagged, how many were spam?)")
    print(f"  Recall:    {metrics['recall']:.4f}  (of all spam, how many did we catch?)")
    print(f"  F1 Score:  {metrics['f1']:.4f}  (harmonic mean of precision and recall)")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}  (overall fraction correct)")


# ---------------------------------------------------------------------------
# PART 2: Imbalanced Data Problem
# ---------------------------------------------------------------------------

def demo_imbalanced_data():
    """
    990 negatives, 10 positives.
    Naive classifier (always predict negative) gets 99% accuracy, 0 recall.
    """
    n_neg, n_pos = 990, 10
    y_true = [0] * n_neg + [1] * n_pos

    # Naive classifier: always predict negative
    y_pred_naive = [0] * (n_neg + n_pos)

    # Smart classifier: catches 6 of 10 positives, 5 false alarms
    y_pred_smart = [0] * n_neg + [0] * 4 + [1] * 6   # misses 4, catches 6
    # Add 5 false positives among the negatives
    y_pred_smart[:5] = [1] * 5

    print("  Dataset: 990 negatives, 10 positives (99% class imbalance)")
    print()

    for name, y_pred in [("Naive (always predict 0)", y_pred_naive),
                          ("Smart (catches 6/10 positives)", y_pred_smart)]:
        cm = confusion_matrix(y_pred, y_true)
        m = derive_metrics(cm)
        print(f"  [{name}]")
        print(f"    Accuracy:  {m['accuracy']:.4f}  <-- looks great for naive!")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall:    {m['recall']:.4f}  <-- 0.0 for naive = detects NOTHING")
        print(f"    F1:        {m['f1']:.4f}  <-- honest summary")
        print()

    print("  Key lesson: 99% accuracy on an imbalanced dataset is meaningless.")
    print("  Always use F1, precision, and recall on imbalanced data.")


# ---------------------------------------------------------------------------
# PART 3: BLEU Score from Scratch
# ---------------------------------------------------------------------------

def ngram_counts(tokens, n):
    """Count all n-grams in a token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_ngram_precision(hypothesis, reference, n):
    """Compute clipped n-gram precision."""
    hyp_ngrams = ngram_counts(hypothesis, n)
    ref_ngrams = ngram_counts(reference, n)

    # Clipped count: can't claim more than reference has
    clipped = sum(min(count, ref_ngrams.get(gram, 0))
                  for gram, count in hyp_ngrams.items())
    total_hyp = max(sum(hyp_ngrams.values()), 1)
    return clipped / total_hyp


def bleu_score(hypothesis, reference, max_n=2):
    """Compute BLEU score for n=1..max_n with brevity penalty."""
    hyp_tokens = hypothesis.split()
    ref_tokens = reference.split()

    # Brevity penalty
    bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
    bp = min(1.0, bp)

    # Log average of n-gram precisions
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p_n = bleu_ngram_precision(hyp_tokens, ref_tokens, n)
        if p_n == 0:
            return 0.0
        log_sum += (1 / max_n) * math.log(p_n)

    return bp * math.exp(log_sum)


def demo_bleu():
    reference  = "the cat sat on the mat"
    hypotheses = [
        ("Perfect match", "the cat sat on the mat"),
        ("Good translation", "a cat was sitting on the mat"),
        ("Poor translation", "the dog ran in the park"),
        ("Too short", "cat mat"),
    ]

    print(f"  Reference: '{reference}'")
    print()
    print(f"  {'Hypothesis':<30} {'BLEU-1':>8} {'BLEU-2':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for name, hyp in hypotheses:
        b1 = bleu_score(hyp, reference, max_n=1)
        b2 = bleu_score(hyp, reference, max_n=2)
        print(f"  {name:<30} {b1:>8.4f} {b2:>8.4f}")


# ---------------------------------------------------------------------------
# PART 4: ROUGE-1 from Scratch
# ---------------------------------------------------------------------------

def rouge_1(hypothesis, reference):
    """
    ROUGE-1 recall: fraction of reference unigrams that appear in hypothesis.
    """
    hyp_tokens = set(hypothesis.lower().split())
    ref_tokens  = reference.lower().split()

    if not ref_tokens:
        return 0.0

    overlap = sum(1 for t in ref_tokens if t in hyp_tokens)
    return overlap / len(ref_tokens)


def demo_rouge():
    reference  = "the cat sat on the mat near the window"
    hypotheses = [
        ("Full summary", "the cat sat on the mat near the window"),
        ("Partial summary", "the cat sat on the mat"),
        ("Different words", "a feline rested on a rug by the glass"),
        ("Empty", ""),
    ]

    print(f"  Reference: '{reference}'")
    print()
    print(f"  {'Hypothesis':<35} {'ROUGE-1':>8}")
    print(f"  {'-'*35} {'-'*8}")
    for name, hyp in hypotheses:
        r1 = rouge_1(hyp, reference)
        print(f"  {name:<35} {r1:>8.4f}")


# ---------------------------------------------------------------------------
# PART 5: eval_report()
# ---------------------------------------------------------------------------

def eval_report(y_predictions, y_ground_truth, label="Evaluation Report"):
    """
    Compute and print a full metrics table.
    Works with binary 0/1 labels.
    """
    cm = confusion_matrix(y_predictions, y_ground_truth)
    m  = derive_metrics(cm)

    print(f"  {label}")
    print(f"  {'='*40}")
    print_confusion_matrix(cm)
    print(f"  {'-'*40}")
    print(f"  {'Metric':<15} {'Value':>10}")
    print(f"  {'-'*15} {'-'*10}")
    for metric_name, value in m.items():
        print(f"  {metric_name:<15} {value:>10.4f}")
    print(f"  {'TP':<15} {cm['TP']:>10}")
    print(f"  {'FP':<15} {cm['FP']:>10}")
    print(f"  {'TN':<15} {cm['TN']:>10}")
    print(f"  {'FN':<15} {cm['FN']:>10}")


# ---------------------------------------------------------------------------
# PART 6: Threshold Tuning
# ---------------------------------------------------------------------------

def demo_threshold_tuning():
    """
    For a list of (probability, label) pairs, sweep threshold 0.1..0.9.
    Show precision/recall tradeoff.
    """
    # Simulated model probabilities and true labels
    data = [
        (0.95, 1), (0.88, 1), (0.82, 0), (0.76, 1), (0.71, 1),
        (0.65, 0), (0.60, 1), (0.55, 0), (0.48, 0), (0.42, 1),
        (0.38, 0), (0.30, 0), (0.22, 1), (0.15, 0), (0.08, 0),
    ]
    probs  = [d[0] for d in data]
    labels = [d[1] for d in data]

    thresholds = [round(t * 0.1, 1) for t in range(1, 10)]

    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Positives':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for thresh in thresholds:
        y_pred = [1 if p >= thresh else 0 for p in probs]
        cm = confusion_matrix(y_pred, labels)
        m  = derive_metrics(cm)
        n_predicted_pos = sum(y_pred)
        print(f"  {thresh:>10.1f} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {n_predicted_pos:>10}")

    print()
    print("  Interpretation:")
    print("  - Low threshold (0.1): predict positive aggressively → high recall, low precision")
    print("  - High threshold (0.9): only very confident predictions → high precision, low recall")
    print("  - Choose threshold based on the cost of FP vs FN in your application")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_section("PART 1: Confusion Matrix from Scratch")
    demo_confusion_matrix()

    print_section("PART 2: Imbalanced Data Problem")
    demo_imbalanced_data()

    print_section("PART 3: BLEU Score (translation evaluation)")
    demo_bleu()

    print_section("PART 4: ROUGE-1 (summarization evaluation)")
    demo_rouge()

    print_section("PART 5: eval_report() — Full Metrics Table")
    y_pred  = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_true  = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    eval_report(y_pred, y_true, label="Sample Classification Task")

    print_section("PART 6: Threshold Tuning — Precision/Recall Tradeoff")
    demo_threshold_tuning()

    print_section("SUMMARY")
    print("  1. Accuracy is misleading on imbalanced data — always check precision+recall")
    print("  2. Precision = when we say yes, are we right? (spam filter priority)")
    print("  3. Recall = did we find all the positives? (cancer detection priority)")
    print("  4. F1 = harmonic mean, penalizes extreme imbalance between P and R")
    print("  5. BLEU measures n-gram precision (translation), ROUGE measures recall (summaries)")
    print("  6. Threshold tuning lets you trade precision for recall based on use case")
