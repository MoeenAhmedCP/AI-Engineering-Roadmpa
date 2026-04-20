"""
2.2 Neural Networks — Examples
Runnable with just numpy.

Covers:
- 2-layer neural network from scratch (XOR problem)
- Activation function comparison table
- Vanishing gradient demonstration (10-layer sigmoid network)
"""

import numpy as np

# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    # Numerically stable sigmoid
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_fn(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


# ---------------------------------------------------------------------------
# PART 1: Activation Function Table
# ---------------------------------------------------------------------------

def print_activation_table():
    """Print activation values at x = -2, -1, 0, 1, 2 for ReLU/sigmoid/tanh."""
    xs = [-2, -1, 0, 1, 2]

    header = f"  {'x':>5} | {'ReLU':>10} | {'sigmoid':>10} | {'tanh':>10}"
    separator = "  " + "-" * (len(header) - 2)
    print(header)
    print(separator)

    for x in xs:
        x_arr = np.array([x], dtype=float)
        r = relu(x_arr)[0]
        s = sigmoid(x_arr)[0]
        t = tanh_fn(x_arr)[0]
        print(f"  {x:>5} | {r:>10.4f} | {s:>10.4f} | {t:>10.4f}")

    print()
    print("  Notes:")
    print("  - ReLU: 0 for negative inputs, identity for positive")
    print("  - Sigmoid: compresses all reals to (0, 1)")
    print("  - Tanh: compresses all reals to (-1, 1), zero-centered")


# ---------------------------------------------------------------------------
# PART 2: 2-Layer Neural Network — XOR Problem
# ---------------------------------------------------------------------------
# XOR cannot be solved by a linear model (it's not linearly separable).
# A hidden layer with nonlinear activation learns the required representation.
#
# Inputs:  [0,0], [0,1], [1,0], [1,1]
# Outputs:    0,    1,    1,    0

def binary_cross_entropy(y_hat, y_true):
    eps = 1e-12
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))


class TwoLayerNet:
    """
    Architecture: Input(2) -> Hidden(4, sigmoid) -> Output(1, sigmoid)
    """

    def __init__(self, seed=42):
        np.random.seed(seed)
        # Weights initialized with Xavier initialization
        self.W1 = np.random.randn(2, 4) * np.sqrt(2 / 2)   # (input, hidden)
        self.b1 = np.zeros(4)
        self.W2 = np.random.randn(4, 1) * np.sqrt(2 / 4)   # (hidden, output)
        self.b2 = np.zeros(1)

    def forward(self, X):
        # Layer 1
        self.z1 = X @ self.W1 + self.b1        # (n, 4)
        self.a1 = sigmoid(self.z1)              # (n, 4)
        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2  # (n, 1)
        self.a2 = sigmoid(self.z2)              # (n, 1)
        return self.a2

    def backward(self, X, y_true, lr=0.1):
        n = X.shape[0]

        # --- Output layer gradients ---
        # dL/dz2 = (a2 - y) for binary cross-entropy + sigmoid (convenient form)
        dL_dz2 = (self.a2 - y_true) / n          # (n, 1)

        dL_dW2 = self.a1.T @ dL_dz2              # (4, 1)
        dL_db2 = np.sum(dL_dz2, axis=0)          # (1,)

        # --- Hidden layer gradients (chain rule through sigmoid) ---
        dL_da1 = dL_dz2 @ self.W2.T              # (n, 4)
        dL_dz1 = dL_da1 * sigmoid_grad(self.z1)  # (n, 4)

        dL_dW1 = X.T @ dL_dz1                    # (2, 4)
        dL_db1 = np.sum(dL_dz1, axis=0)          # (4,)

        # --- Update weights ---
        self.W2 -= lr * dL_dW2
        self.b2 -= lr * dL_db2
        self.W1 -= lr * dL_dW1
        self.b1 -= lr * dL_db1


def train_xor(n_steps=10000, lr=0.5):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    net = TwoLayerNet(seed=0)
    print(f"  {'Step':>6} | {'Loss':>10} | Predictions")
    print(f"  {'-'*6} | {'-'*10} | {'-'*30}")

    for step in range(n_steps + 1):
        y_hat = net.forward(X)
        loss = binary_cross_entropy(y_hat, y)

        if step % 2000 == 0:
            preds = y_hat.flatten().round(3)
            print(f"  {step:>6} | {loss:>10.6f} | {preds}")

        if step < n_steps:
            net.backward(X, y, lr=lr)

    # Final predictions
    y_hat_final = net.forward(X)
    print()
    print("  Final predictions (should be [~0, ~1, ~1, ~0]):")
    for i, (xi, yi_true, yi_hat) in enumerate(zip(X, y, y_hat_final)):
        correct = "CORRECT" if abs(yi_hat[0] - yi_true[0]) < 0.5 else "WRONG"
        print(f"    Input {xi.astype(int)} -> True: {yi_true[0]:.0f}, Predicted: {yi_hat[0]:.4f} [{correct}]")


# ---------------------------------------------------------------------------
# PART 3: Vanishing Gradient Demonstration
# ---------------------------------------------------------------------------

def demonstrate_vanishing_gradient(n_layers=10):
    """
    Show how gradient magnitude shrinks through a deep sigmoid network.
    Each layer's gradient is multiplied by sigmoid'(z), which is at most 0.25.
    Across 10 layers: 0.25^10 ≈ 0.000001
    """
    np.random.seed(42)
    x = np.array([0.5])  # A single input

    # Small random weights (realistic for a network)
    weights = [np.random.randn(1, 1) * 0.5 for _ in range(n_layers)]
    biases  = [np.zeros(1) for _ in range(n_layers)]

    # --- Forward pass: store pre-activations ---
    zs = []
    a = x
    for W, b in zip(weights, biases):
        z = a @ W + b
        zs.append(z.flatten()[0])
        a = sigmoid(z)

    # --- Backward pass: track gradient magnitude at each layer ---
    # Start with gradient of 1.0 from the loss
    grad = np.array([1.0])
    grad_magnitudes = []

    for layer_idx in reversed(range(n_layers)):
        z = np.array([zs[layer_idx]])
        # Chain rule: multiply by sigmoid'(z) and weight
        sg = sigmoid_grad(z)
        grad = grad * sg * abs(weights[layer_idx].flatten()[0])
        grad_magnitudes.insert(0, abs(grad[0]))

    print(f"  Gradient magnitude at each layer of a {n_layers}-layer sigmoid network:")
    print(f"  (flowing backward from output layer to input layer)")
    print()
    print(f"  {'Layer':<8} | {'Gradient Magnitude':<20} | {'Visual'}")
    print(f"  {'-'*8} | {'-'*20} | {'-'*30}")

    max_mag = max(grad_magnitudes) if max(grad_magnitudes) > 0 else 1.0
    for i, mag in enumerate(grad_magnitudes):
        layer_name = f"Layer {n_layers - i}" if i == 0 else f"Layer {n_layers - i}"
        # Scale bar
        bar_len = int((mag / max_mag) * 25)
        bar = "#" * bar_len
        print(f"  {layer_name:<8} | {mag:<20.2e} | {bar}")

    print()
    print(f"  Layer {n_layers} (output): gradient = {grad_magnitudes[0]:.6f}")
    print(f"  Layer 1 (input):  gradient = {grad_magnitudes[-1]:.2e}")
    ratio = grad_magnitudes[0] / (grad_magnitudes[-1] + 1e-30)
    print(f"  Ratio (output/input): {ratio:.0f}x larger at output layer")
    print()
    print("  This is the vanishing gradient problem.")
    print("  Early layers barely update because their gradient is nearly zero.")
    print("  Fix: use ReLU/GELU activations + residual connections (skip connections).")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_section("PART 1: Activation Functions at x = -2, -1, 0, 1, 2")
    print_activation_table()

    print_section("PART 2: XOR Neural Network (10,000 training steps)")
    print("  XOR is not linearly separable — requires hidden layer + nonlinearity")
    print()
    train_xor(n_steps=10000, lr=0.5)

    print_section("PART 3: Vanishing Gradient (10-layer sigmoid network)")
    demonstrate_vanishing_gradient(n_layers=10)

    print_section("SUMMARY")
    print("  1. Neurons: dot product of inputs and weights, then activation function")
    print("  2. Activation functions add nonlinearity — without them, deep=shallow")
    print("  3. ReLU/GELU preferred for hidden layers; sigmoid for binary output")
    print("  4. Backprop = chain rule applied backward through the network")
    print("  5. Vanishing gradients: sigmoid gradients < 0.25, shrink exponentially")
    print("  6. Fix: ReLU + residual connections + layer normalization")
