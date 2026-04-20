# 2.1 How ML Works

## Types of Machine Learning

### Supervised Learning
The model learns a mapping from inputs to outputs using labeled training data. Every training example has a correct answer. The model adjusts its internal parameters to produce outputs that match those correct answers.

AI engineering examples:
- Document classification (input: text, output: category label)
- Sentiment analysis (input: review, output: positive/negative/neutral)
- Named entity recognition (input: sentence, output: tagged entities)
- Regression on tabular data (input: house features, output: price)

### Unsupervised Learning
No labels. The model finds structure in data on its own — patterns, clusters, compressed representations.

AI engineering examples:
- Clustering customer queries to discover intent categories before you have labels
- Topic modeling on a document corpus
- Anomaly detection in API logs (no labeled anomalies needed)
- Dimensionality reduction (PCA, t-SNE) to visualize high-dimensional embeddings

### Reinforcement Learning (RL)
An agent takes actions in an environment, receives a reward signal, and learns to maximize cumulative reward. No fixed dataset — the agent generates its own experience.

AI engineering examples:
- RLHF (Reinforcement Learning from Human Feedback) — how GPT and Claude were trained after supervised pretraining. Human preferences become the reward signal.
- RL for tool use: agent learns which tool to call to maximize task completion rate
- Dialogue policy optimization: learn which response strategy leads to better user ratings

---

## The Training Loop — Step by Step

Training is the iterative process of adjusting model parameters to reduce error. One full pass over the dataset is called an **epoch**. Within each epoch, you process data in **batches**.

### Step 1: Data
Prepare a batch of (input, label) pairs. Inputs are represented as numbers (vectors, matrices, tensors). Labels are target outputs.

### Step 2: Forward Pass
Feed the input through the model, layer by layer, to produce a **prediction** (also called the output or logit). Each layer applies a linear transformation followed by a nonlinear activation function.

```
prediction = model(input)
```

### Step 3: Compute Loss
Measure how wrong the prediction is compared to the true label. The loss function returns a single scalar — lower is better. This number is the signal that drives all learning.

```
loss = loss_function(prediction, true_label)
```

### Step 4: Backward Pass (Backpropagation)
Compute the gradient of the loss with respect to every parameter in the model. Gradients tell you: "if I increase this parameter slightly, does the loss go up or down — and by how much?" PyTorch and other frameworks do this automatically via automatic differentiation.

```
loss.backward()   # computes gradients for all parameters
```

### Step 5: Optimizer Update
Adjust every parameter in the direction that reduces the loss, scaled by the learning rate.

```
optimizer.step()   # w = w - lr * gradient
optimizer.zero_grad()  # clear gradients before next batch
```

Repeat for thousands of batches across many epochs.

---

## Loss Functions

### Mean Squared Error (MSE) — Regression
Measures the average squared difference between predictions and targets.

```
MSE = (1/n) * sum((y_pred - y_true)^2)
```

Squaring penalizes large errors more than small ones. Sensitive to outliers. Use when your output is a continuous real number.

### Binary Cross-Entropy — Classification
Measures how well a predicted probability matches a binary label (0 or 1).

```
BCE = -(1/n) * sum(y*log(p) + (1-y)*log(1-p))
```

Where p is the predicted probability of class 1. When y=1, the loss is -log(p) — high if p is near 0, zero if p=1. Forces the model to be confidently correct, not just directionally correct.

For multi-class classification, use Categorical Cross-Entropy (also called Softmax Loss).

---

## Gradient Descent Variants

### Batch Gradient Descent
Compute gradients using the **entire** dataset before updating parameters. Most accurate gradient estimate, but slow and requires all data in memory. Impractical for large datasets.

### Stochastic Gradient Descent (SGD)
Update parameters after **each single example**. Very fast, lots of updates, but gradients are noisy — they jump around. The noise can help escape local minima, but makes convergence erratic.

### Mini-batch Gradient Descent
**The standard in practice.** Update after each batch of N examples (typically 32–512). Balances the accuracy of batch GD with the speed of SGD. GPU parallelism makes batching efficient. Adam/AdamW use mini-batch GD internally.

**Tradeoffs:**
- Larger batch → smoother gradients, but less regularization effect, more memory
- Smaller batch → noisier gradients, often better generalization, less memory

---

## Learning Rate Effects

The learning rate (lr) controls how large each parameter update step is.

**Too high (e.g., lr=1.0):** Parameters overshoot the minimum. Loss oscillates or diverges (goes to infinity). You'll see NaN losses.

**Too low (e.g., lr=0.00001):** Training is slow. Loss decreases but takes forever to converge. May get stuck in plateaus.

**Just right (e.g., lr=0.001 for Adam):** Loss decreases smoothly and converges in a reasonable number of steps.

### Learning Rate Schedules
Rather than a fixed lr, schedules adjust it during training:
- **Step decay**: reduce by factor every N epochs (e.g., lr × 0.1 every 30 epochs)
- **Cosine annealing**: smoothly decay from lr_max to lr_min following a cosine curve
- **Warmup + decay**: start very low, ramp up for first N steps, then decay. Standard for Transformers (used in BERT, GPT training).
- **Cyclical LR**: oscillate between low and high — helps escape saddle points

---

## Overfitting vs Underfitting

**Underfitting (High Bias):** The model is too simple to capture the pattern in the data. Training loss is high. Think of fitting a straight line to data that curves — the model has the wrong assumptions baked in.

**Overfitting (High Variance):** The model memorizes the training data, including its noise. Training loss is very low, but validation loss is high. It learned specific patterns that don't generalize.

### Bias-Variance Tradeoff
- High bias models are consistent but wrong (a broken clock analogy: it's always off by the same amount)
- High variance models are right on average but unpredictable (they perform differently on different samples)
- Goal: find the sweet spot that minimizes both

You diagnose this by plotting training loss and validation loss against epochs. If val loss stops decreasing and starts increasing while train loss continues down — that's overfitting.

---

## Train / Validation / Test Splits

**Training set (~70-80%):** The data the model actually learns from. Parameters are updated using only this data.

**Validation set (~10-15%):** Used during development to tune hyperparameters and make decisions like "should I use 2 layers or 4?" The model never trains on this, but you use its loss to guide choices. This means the val set influences training indirectly — it guides your hyperparameter decisions.

**Test set (~10-15%):** Locked away until the very end. Used exactly once to report final performance. If you tune on the test set, you're leaking information and your reported numbers are optimistic.

Why all three: if you use the validation set for final reporting, your estimate is biased (you chose the model that happened to do best on val). The test set gives an honest estimate of real-world performance.

---

## Regularization

Regularization adds constraints or noise to prevent overfitting.

**L2 Regularization (Ridge / Weight Decay):** Adds a penalty proportional to the sum of squared weights to the loss. This pushes all weights toward zero but doesn't zero them out. Result: small, spread-out weights. Standard in neural networks as weight decay in Adam/AdamW.

**L1 Regularization (Lasso):** Adds a penalty proportional to the sum of absolute values of weights. This produces **sparse** solutions — many weights become exactly zero. Useful when you suspect most features are irrelevant.

**Dropout:** During training, randomly zero out a fraction of neurons on each forward pass. Forces the network to not rely on any single neuron — builds redundancy. **Always turn off during inference** (`model.eval()` in PyTorch handles this).

---

## Parameters vs Hyperparameters

**Parameters:** Values learned during training. Weights and biases in a neural network. You don't set them manually — the optimizer updates them automatically based on gradients.

**Hyperparameters:** Choices you make before training that control the training process or model structure. Examples:
- Learning rate
- Batch size
- Number of layers, hidden size
- Dropout rate
- Regularization strength (lambda)
- Number of epochs
- Optimizer choice (Adam vs SGD)

Hyperparameter tuning is one of the main jobs of an ML engineer. Grid search, random search, and Bayesian optimization are common strategies.
