# 2.3 ML Algorithms

## Linear Regression

**Problem:** Predict a continuous numeric output from input features.

**How it works:** Learns a weighted sum of input features: `y = w1*x1 + w2*x2 + ... + wn*xn + b`. Each coefficient w_i represents how much that feature influences the output. You find these coefficients by minimizing MSE loss (closed-form via the Normal Equation, or iteratively via gradient descent).

**When to use:** Fast baseline for any regression task. Works well when the relationship between features and target is approximately linear. Quick to train and interpret.

**Key hyperparameters:** Regularization strength (L1=Lasso produces sparse solutions, L2=Ridge produces small weights).

**Feature importance insight:** The magnitude of each coefficient (after scaling features) tells you how much each feature contributes. Positive coefficient = higher feature → higher prediction. Zero coefficient (L1) = feature not useful.

---

## Logistic Regression

**Problem:** Binary or multi-class classification (despite the name, it's a classifier).

**How it works:** Applies a linear combination of features, then passes the result through a sigmoid function to produce a probability. Decision boundary: predict class 1 if p > 0.5.

```
p(y=1) = sigmoid(w · x + b) = 1 / (1 + e^(-(w·x+b)))
```

The log-odds (logit) is linear: `log(p/(1-p)) = w·x + b`. A coefficient of 2.0 for feature x means one unit increase in x multiplies the odds of class 1 by e^2 ≈ 7.4x.

**Decision boundary:** The set of points where p = 0.5, i.e., where `w·x + b = 0`. This is always a hyperplane (linear boundary). Logistic regression cannot learn curved decision boundaries without feature engineering.

**When to use:** Strong linear baseline for classification. Fast, interpretable, well-calibrated probabilities. Often beats complex models when data is limited or features are well-engineered.

**Key hyperparameters:** C (inverse regularization strength), penalty (L1/L2).

---

## Decision Trees

**Problem:** Classification or regression with interpretable, rule-based predictions.

**How it works:** Recursively splits the data by choosing the feature and threshold that best separates classes. At each node, it picks the split that maximizes **information gain** (most purity improvement).

**Gini impurity:** Measures how often a randomly chosen element would be incorrectly classified.
```
Gini = 1 - sum(p_i^2)
```
Gini = 0 means perfectly pure (all one class). Gini = 0.5 means maximally impure (equal classes).

**Information gain** = parent impurity - weighted sum of child impurities. Choose the split that maximizes this.

**Complexity control:** Tree depth directly controls overfitting. Shallow tree → underfit (high bias). Deep tree → overfit (memorizes training data). `max_depth` is the most important hyperparameter.

**When to use:** Need interpretability (can visualize and explain every decision). When you have categorical features without encoding. Works without feature scaling.

**Key hyperparameters:** `max_depth`, `min_samples_split`, `min_samples_leaf`.

---

## Random Forests

**Problem:** Improve decision trees' accuracy and reduce overfitting.

**How it works (Bagging — Bootstrap Aggregating):**
1. Sample N training points with replacement (bootstrap sample) — each tree sees ~63% of data
2. At each split, consider only a random subset of features (typically sqrt(n_features) for classification)
3. Build a fully grown tree on the bootstrap sample
4. Repeat for many trees (e.g., 100–1000)
5. Prediction = majority vote (classification) or mean (regression) across all trees

**Why it works:** Each tree overfits differently (different bootstrap sample, different random features). Averaging cancels out individual errors. The diversity between trees is key — correlated trees give less benefit.

**Feature importance:** Average decrease in Gini impurity from splits on each feature, across all trees. More reliable than single decision tree feature importance.

**OOB (Out-of-Bag) error:** The ~37% of data each tree never saw can be used for validation without a separate val set. Free internal cross-validation.

**When to use:** Strong baseline for tabular data. Robust to outliers and irrelevant features. Little hyperparameter tuning needed. Works well out of the box.

**Key hyperparameters:** `n_estimators` (more=better, diminishing returns), `max_features`, `max_depth`.

---

## Gradient Boosting (XGBoost / LightGBM)

**Problem:** Achieve state-of-the-art results on tabular data competitions and production systems.

**How it works (Sequential, not parallel like Random Forest):**
1. Start with a simple prediction (e.g., the mean of y)
2. Compute the residuals (errors) of the current model
3. Train a new shallow tree to predict these residuals
4. Add this tree to the ensemble with a small weight (learning rate)
5. Repeat: each new tree corrects the previous ensemble's mistakes

```
F_m(x) = F_{m-1}(x) + lr * tree_m(x)
```

**Why it often beats Random Forest:** Boosting focuses each new tree on the hardest examples (largest errors). More efficient use of model capacity than averaging.

**XGBoost vs LightGBM:** Both are gradient boosting implementations. LightGBM is faster on large datasets (histogram-based splitting, leaf-wise tree growth). XGBoost has more regularization options.

**When to use:** Top-performing model for structured/tabular data. Industry standard for Kaggle competitions. When you need the last percentage point of accuracy and can afford more tuning.

**Key hyperparameters:** `n_estimators`, `learning_rate` (lr), `max_depth`, `subsample`, `colsample_bytree`.

---

## k-Means Clustering

**Problem:** Unsupervised grouping — find k natural clusters in data.

**How it works:**
1. Initialize k centroids (randomly, or with k-means++ for better initialization)
2. Assign each point to its nearest centroid (by Euclidean distance)
3. Recompute centroids as the mean of all assigned points
4. Repeat steps 2-3 until centroids stop moving (convergence)

**Inertia:** Sum of squared distances from each point to its assigned centroid. Lower = tighter clusters. Always decreases as k increases — not useful alone for choosing k.

**Elbow method for choosing k:** Plot inertia vs k. Find the "elbow" where adding another cluster gives diminishing returns in inertia reduction. The elbow = good k.

**When to use:** Customer segmentation, document clustering, grouping similar queries, anomaly detection (points far from all centroids). Assumes clusters are convex and roughly equal size.

**Key hyperparameters:** `n_clusters` (k), `n_init` (number of random starts — always run multiple times).

---

## k-Nearest Neighbors (k-NN)

**Problem:** Classification or regression using similarity to training examples.

**How it works (Lazy learning — no training phase):**
- At prediction time, find the k training examples closest to the query point
- Classification: return the majority class among k neighbors
- Regression: return the mean target value among k neighbors

**Curse of dimensionality:** In high dimensions, all points become equidistant — the concept of "nearest neighbor" breaks down. k-NN degrades badly with more than ~20 features. Dimensionality reduction (PCA) or embedding-based similarity (approximate nearest neighbor) is used instead.

**When to use:** Small datasets, low-dimensional data, when you need no training time (but can afford slow inference). Good for recommendation systems when using embedding similarity.

**Key hyperparameters:** `k` (small k = more complex boundary, high variance; large k = smoother boundary, high bias), distance metric.

---

## Feature Engineering

Raw data rarely comes in a form ready for ML models. Feature engineering transforms raw columns into representations models can learn from effectively.

### One-Hot Encoding
Convert a categorical variable with n distinct values into n binary columns. `color = red` → `[1, 0, 0]` (for red/green/blue).
- Use for **low-cardinality** categoricals (< ~50 unique values)
- Creates sparse, interpretable features
- Tree models handle one-hot fine; neural networks prefer learned embeddings

### Ordinal Encoding
Map categories to integers that preserve order. `small→1, medium→2, large→3`.
- Only appropriate when the order is meaningful
- Arbitrary numbers for nominal categories mislead models

### Feature Scaling

**StandardScaler (z-score):** Subtract mean, divide by std. Result has mean=0, std=1.
```
x_scaled = (x - mean) / std
```
Use for: neural networks, SVMs, logistic regression, any gradient-based model. Gradient descent assumes features have similar scale — otherwise it zigzags.

**MinMaxScaler:** Scale to [0, 1].
```
x_scaled = (x - min) / (max - min)
```
Use for: when you need bounded values (e.g., pixel values in image models). Sensitive to outliers (one extreme value squashes everything else).

**Tree models (decision trees, random forests, gradient boosting) don't need scaling** — they split on thresholds, not distances or magnitudes.

### Target Encoding
Replace a categorical value with the mean target value for that category (computed on the training set). Powerful for high-cardinality categoricals but risks leaking label information — always compute on training set only, use cross-validation to prevent leakage.
