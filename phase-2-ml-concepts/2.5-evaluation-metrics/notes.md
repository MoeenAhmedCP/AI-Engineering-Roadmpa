# 2.5 Evaluation Metrics

## Why Metrics Matter

Choosing the wrong metric is one of the most common mistakes in ML. A model can score 99% accuracy and be completely useless. Every metric measures something specific — you need to pick the one that measures what actually matters for your application.

---

## The Confusion Matrix

The foundation of classification evaluation. For binary classification (positive/negative):

```
                    Predicted Positive   Predicted Negative
Actual Positive:         TP                    FN
Actual Negative:         FP                    TN
```

**TP (True Positive):** Correctly predicted positive. Spam email correctly flagged as spam.
**FP (False Positive):** Predicted positive, actually negative. Legitimate email flagged as spam. (Type I error)
**FN (False Negative):** Predicted negative, actually positive. Spam email allowed through. (Type II error)
**TN (True Negative):** Correctly predicted negative. Legitimate email correctly passed.

Concrete example — cancer detection (100 patients, 10 have cancer):
- Model predicts cancer for 12 patients
- 8 actual cancer patients correctly identified (TP=8)
- 4 healthy patients incorrectly flagged (FP=4)
- 2 cancer patients missed (FN=2)
- 86 healthy patients correctly cleared (TN=86)

---

## Accuracy

```
Accuracy = (TP + TN) / (TP + FP + TN + FN)
```

**The imbalanced data trap:** If 990 of 1000 examples are negative and 10 are positive, a model that always predicts "negative" achieves 99% accuracy while detecting zero positive cases. This is worthless for the actual task (detecting rare positives).

Rule: If classes are imbalanced (< 80/20 split), don't use accuracy as your primary metric.

---

## Precision

```
Precision = TP / (TP + FP)
```

"Of everything the model predicted as positive, what fraction was actually positive?"

High precision means few false alarms. When you say "yes," you're usually right.

**When precision matters:** Spam filter — you'd rather miss some spam than put legitimate emails in spam. You want: when we say "spam," we're almost certainly right. A false positive (real email marked spam) is more costly than a false negative (spam getting through).

---

## Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

"Of all the actual positives, what fraction did the model find?"

High recall means you catch most of the positives — few are missed.

**When recall matters:** Cancer detection, fraud detection — you'd rather have false alarms than miss real cases. Missing a cancer patient (FN) is catastrophic. A false alarm (FP) just means more tests. You want: find as many actual positives as possible.

---

## F1 Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

The harmonic mean of precision and recall. Useful when you want balance between the two, and especially when data is imbalanced.

The harmonic mean penalizes extreme cases more than the arithmetic mean. If precision=1.0 and recall=0.01, arithmetic mean=0.505 (misleadingly high), harmonic mean=0.02 (more honest).

**F-beta score** generalizes F1: `F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)`. Beta > 1 weights recall higher. Beta < 1 weights precision higher.

---

## AUC-ROC

**ROC curve:** Plot True Positive Rate (recall) vs False Positive Rate at every possible classification threshold (0 to 1).

```
True Positive Rate  = TP / (TP + FN)   = Recall
False Positive Rate = FP / (FP + TN)   = 1 - Specificity
```

**AUC (Area Under the Curve):** The area under the ROC curve, ranging from 0 to 1.
- AUC = 1.0: perfect classifier
- AUC = 0.5: random classifier (diagonal line on the plot)
- AUC = 0.0: perfectly wrong

**Why AUC is useful:** It's threshold-independent. It measures ranking ability — "how well does the model rank actual positives above actual negatives?" A model with AUC=0.85 correctly ranks a random positive above a random negative 85% of the time. Robust to class imbalance.

---

## BLEU Score (for Machine Translation)

BLEU (Bilingual Evaluation Understudy) measures n-gram precision between a generated text and reference translations.

```
BLEU = BP * exp( sum( w_n * log(p_n) ) )
```

Where:
- `p_n` = n-gram precision (for n=1,2,3,4): fraction of n-grams in hypothesis that appear in reference
- `w_n` = weight (uniform: 0.25 for n=1..4)
- `BP` = brevity penalty: penalizes outputs shorter than reference (to prevent "gaming" with short outputs)

**Limitation:** BLEU is a poor proxy for translation quality. It doesn't measure fluency, meaning preservation, or semantic similarity. Two semantically identical sentences with different word choices get low BLEU. Still widely used because it's cheap and correlates weakly with human judgment at scale.

---

## ROUGE (for Summarization)

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between generated summary and reference summary.

**ROUGE-1:** Unigram overlap. Measures recall of individual words.
```
ROUGE-1 Recall = (# unigrams in both generated and reference) / (# unigrams in reference)
```

**ROUGE-L:** Longest Common Subsequence (LCS) based. Measures sentence-level structure similarity. Better than ROUGE-1 for capturing fluency.

Unlike BLEU (precision-focused), ROUGE is recall-focused: does the generated summary contain the important information from the reference? Use ROUGE-1/ROUGE-2/ROUGE-L together for a complete picture.

---

## Perplexity

Perplexity measures how "surprised" a language model is by a test text — how well it predicts the text.

```
Perplexity = exp( -(1/N) * sum( log p(w_i | context) ) )
```

Equivalently, perplexity is the exponentiated average negative log-likelihood per token.

**Interpretation:**
- Perplexity 10: the model is as uncertain as if it were choosing uniformly among 10 equally likely tokens at each step
- Perplexity 50: as uncertain as choosing among 50 options — worse
- Perplexity 1: perfect prediction every time

Lower perplexity = model assigns higher probabilities to the actual text = better language model. GPT-4 achieves perplexity of ~3-5 on standard benchmarks; a random model would be tens of thousands.

---

## LLM-as-Judge: The Modern Approach

For open-ended generation tasks (summarization, question answering, reasoning), traditional metrics fail — they can't capture semantic quality, coherence, or factual accuracy.

**LLM-as-judge:** Use a capable LLM (e.g., Claude or GPT-4) to evaluate the output of another model on structured criteria.

Advantages:
- Can evaluate dimensions no automated metric captures: relevance, coherence, factual accuracy, helpfulness, tone
- Flexible criteria: score on a rubric you define
- Much cheaper than human annotation at scale

Common patterns:
- **Pointwise scoring:** "Rate this response 1-5 on helpfulness"
- **Pairwise comparison:** "Which response A or B better answers the question?"
- **Reference-based:** "Does this response correctly answer the question given this context?"

Key considerations:
- Position bias: LLMs tend to prefer the first response shown
- Verbose bias: LLMs often prefer longer responses even when shorter is better
- Self-preference bias: a model may prefer its own style
- Mitigation: swap order of responses, use multiple judges, calibrate with human labels
