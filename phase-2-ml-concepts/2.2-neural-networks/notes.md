# 2.2 Neural Networks

## The Neuron — Math

A single artificial neuron computes:

```
output = activation( sum(x_i * w_i) + b )
       = activation( X · W + b )
```

Where:
- `x_i` are the input values (features, or outputs of the previous layer)
- `w_i` are the learnable weights — one per input
- `b` is the learnable bias — shifts the activation threshold
- `activation` is a nonlinear function applied to the dot product result

Without the activation function, stacking multiple layers would be equivalent to a single linear transformation — no matter how many layers you add. Nonlinearity is what gives neural networks their expressiveness.

A layer is just a collection of neurons operating in parallel. For a layer with 512 neurons receiving 256 inputs: you have a weight matrix of shape (256, 512) and a bias vector of shape (512,).

---

## Activation Functions

### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
```
- Outputs x if x > 0, otherwise outputs 0
- Default for hidden layers in feedforward networks and CNNs
- Computationally fast, does not suffer from vanishing gradients for positive inputs
- **Dying ReLU problem:** If a neuron's input is always negative (e.g., due to a bad weight initialization or high learning rate), the gradient is always zero — the neuron never updates and "dies." Leaky ReLU (max(0.01x, x)) addresses this.

### Sigmoid
```
sigmoid(x) = 1 / (1 + e^(-x))
```
- Maps any input to (0, 1) — useful for binary classification outputs (probability interpretation)
- **Vanishing gradient problem:** When |x| is large, the sigmoid saturates — its gradient approaches zero. During backpropagation, this near-zero gradient gets multiplied across layers, shrinking exponentially. In deep networks, gradients in early layers become so tiny that those layers barely update. This made deep networks nearly untrainable before ReLU became standard.
- Still used in the output layer for binary classification and in LSTM gates.

### Softmax
```
softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)
```
- Converts a vector of raw scores (logits) into a probability distribution that sums to 1
- Used exclusively in the output layer for multi-class classification
- The class with the highest logit gets the highest probability
- Numerically stable implementation subtracts the max logit before exponentiation

### Tanh
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- Maps input to (-1, 1) — zero-centered, unlike sigmoid
- Slightly better than sigmoid because zero-centered outputs help gradient flow
- Still suffers from vanishing gradients at extremes
- Used in RNNs and LSTM cell state updates

### GELU (Gaussian Error Linear Unit)
```
GELU(x) ≈ x * sigmoid(1.702 * x)
```
- Smooth, differentiable approximation to ReLU — nonzero gradient everywhere
- Outperforms ReLU empirically on many Transformer tasks
- **Default activation in modern Transformers**: GPT, BERT, Claude all use GELU
- Intuition: unlike ReLU which hard-gates at 0, GELU softly gates — small negative inputs are "mostly" gated out, not completely zeroed

**When to use which:**
- Hidden layers in standard networks: ReLU (or Leaky ReLU)
- Transformers and large language models: GELU
- Binary classification output: Sigmoid
- Multi-class classification output: Softmax
- Never use sigmoid/tanh for deep hidden layers — vanishing gradients will kill training

---

## Forward Pass — Layer by Layer

For a 3-layer network:

```
Layer 1: h1 = ReLU(X @ W1 + b1)     # shape: (batch, hidden1)
Layer 2: h2 = ReLU(h1 @ W2 + b2)    # shape: (batch, hidden2)
Output:  y  = sigmoid(h2 @ W3 + b3)  # shape: (batch, 1)
```

Each layer's output becomes the next layer's input. The final layer's activation depends on the task (sigmoid for binary, softmax for multi-class, none/linear for regression).

---

## Backpropagation — Intuition

Backpropagation is the algorithm for computing gradients of the loss with respect to every parameter in the network. It uses the **chain rule** from calculus.

**Chain rule:** If loss L depends on output y, y depends on h, and h depends on weight w, then:
```
dL/dw = dL/dy * dy/dh * dh/dw
```

Backprop works **backwards** — it starts at the loss and propagates gradient information layer by layer back toward the input.

Step by step:
1. Compute dL/d(output) — how does loss change with the final output?
2. Multiply by d(output)/d(layer2) — chain rule through the output layer
3. Multiply by d(layer2)/d(layer1) — chain rule through layer 2
4. Continue until you reach every weight and bias

Modern frameworks (PyTorch, JAX) do this automatically by building a computational graph during the forward pass and traversing it in reverse during `.backward()`.

---

## Vanishing and Exploding Gradients

### Vanishing Gradients
As gradients flow backward through many layers, they can shrink toward zero. Each layer's gradient is multiplied by the activation function's derivative. If that derivative is consistently less than 1 (as with sigmoid/tanh at saturation), the product of these derivatives across 20 layers can be astronomically small. Early layers learn extremely slowly or not at all.

**Fixes:**
- Use ReLU/GELU activations (derivative is 1 for positive inputs — no shrinking)
- **Residual connections (skip connections):** Add the input directly to the output of a block: `output = block(x) + x`. This creates a gradient highway — gradients can flow directly backward without being multiplied through activation functions.
- **Layer Normalization:** Normalize activations within each layer to prevent them from saturating, which keeps gradients healthy.

### Exploding Gradients
The opposite — gradients grow exponentially. Can cause NaN parameters. Fix with **gradient clipping**: if the gradient norm exceeds a threshold, scale it down.

---

## Batch Normalization

Normalizes the activations of each layer across the batch dimension to have zero mean and unit variance. Benefits:
- Faster training (can use higher learning rates)
- Acts as a mild regularizer
- Reduces sensitivity to weight initialization
- Uses running statistics during inference (not per-batch), so behavior is consistent

Layer Normalization (used in Transformers) normalizes across the feature dimension instead of the batch dimension — works better for variable-length sequences.

---

## Dropout

During training, randomly set a fraction (dropout rate, typically 0.1–0.5) of neuron outputs to zero on each forward pass. Different neurons are dropped each batch.

**Why it works:** Forces the network to learn redundant representations. No single neuron can be relied upon, so the network distributes information across many neurons. Acts as training an ensemble of sub-networks.

**Critical: disable during inference.** `model.eval()` in PyTorch turns off dropout automatically. During inference, all neurons are active, and their outputs are scaled by (1 - dropout_rate) to account for the difference.

---

## Universal Approximation Theorem

A neural network with at least one hidden layer with a sufficient number of neurons and a nonlinear activation function can approximate **any continuous function** to arbitrary precision.

Practical intuition: each neuron can learn a "bump" or "step" — a simple local pattern. By combining enough of these bumps, you can approximate any shape. More neurons = more bumps = more complex functions you can represent.

The theorem guarantees expressiveness exists, not that gradient descent will find the right parameters. That's why architecture design, initialization, and optimization all still matter enormously.
