"""
2.1 How ML Works — Examples
Runnable with just numpy.

Covers:
- Linear regression from scratch (forward pass, MSE loss, gradients, weight update)
- Three learning rates and their effects
- Overfitting demonstration with degree-8 polynomial
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
# PART 1: Linear Regression from Scratch
# ---------------------------------------------------------------------------

def linear_regression_from_scratch(lr=0.01, n_steps=1000, seed=42):
    """
    Learn y = 2x + 3 from noisy data.
    Implements: init → forward → loss → gradients → update
    """
    np.random.seed(seed)

    # --- Generate synthetic data: y = 2x + 3 + noise ---
    n_samples = 100
    X = np.random.randn(n_samples)           # input features
    noise = np.random.randn(n_samples) * 0.5
    y_true = 2.0 * X + 3.0 + noise          # true relationship: slope=2, bias=3

    # --- Random initialization ---
    w = np.random.randn()   # weight (slope)
    b = np.random.randn()   # bias (intercept)

    losses = []

    for step in range(n_steps):
        # FORWARD PASS: y_hat = w*x + b
        y_hat = w * X + b

        # LOSS: Mean Squared Error
        errors = y_hat - y_true
        loss = np.mean(errors ** 2)
        losses.append(loss)

        # BACKWARD PASS: compute gradients
        # dL/dw = (2/n) * sum(errors * X)  [chain rule: dL/dy_hat * dy_hat/dw]
        # dL/db = (2/n) * sum(errors)      [chain rule: dL/dy_hat * dy_hat/db]
        dL_dw = (2 / n_samples) * np.sum(errors * X)
        dL_db = (2 / n_samples) * np.sum(errors)

        # OPTIMIZER UPDATE: gradient descent step
        w = w - lr * dL_dw
        b = b - lr * dL_db

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1:4d} | Loss: {loss:.6f} | w={w:.4f}, b={b:.4f}")

    print(f"\n  True parameters:    w=2.0000, b=3.0000")
    print(f"  Learned parameters: w={w:.4f}, b={b:.4f}")
    return losses, w, b


# ---------------------------------------------------------------------------
# PART 2: Learning Rate Effects
# ---------------------------------------------------------------------------

def demonstrate_learning_rates():
    """
    Show what happens with lr too small, just right, and too large.
    """
    np.random.seed(42)
    n_samples = 50
    X = np.random.randn(n_samples)
    y_true = 2.0 * X + 3.0 + np.random.randn(n_samples) * 0.3

    learning_rates = [0.0001, 0.01, 0.5]
    descriptions = ["too small (slow convergence)", "just right", "too large (diverges)"]

    for lr, desc in zip(learning_rates, descriptions):
        w, b = 0.0, 0.0
        final_losses = []
        diverged = False

        for step in range(200):
            y_hat = w * X + b
            errors = y_hat - y_true
            loss = np.mean(errors ** 2)

            if np.isnan(loss) or np.isinf(loss) or loss > 1e8:
                diverged = True
                break

            dL_dw = (2 / n_samples) * np.sum(errors * X)
            dL_db = (2 / n_samples) * np.sum(errors)
            w = w - lr * dL_dw
            b = b - lr * dL_db
            final_losses.append(loss)

        if diverged:
            status = "DIVERGED (NaN/Inf loss)"
            final_loss_str = "N/A"
        else:
            final_loss_str = f"{final_losses[-1]:.6f}"
            status = "OK"

        print(f"  lr={lr:<8} | {desc:<30} | Final loss: {final_loss_str:<12} | {status}")


# ---------------------------------------------------------------------------
# PART 3: Overfitting Demonstration
# ---------------------------------------------------------------------------

def polynomial_features(X, degree):
    """Create polynomial feature matrix [1, x, x^2, ..., x^degree]."""
    return np.column_stack([X ** d for d in range(degree + 1)])


def fit_polynomial(X_train, y_train, degree):
    """Fit polynomial using least squares (closed form solution)."""
    Phi = polynomial_features(X_train, degree)
    # Normal equation: w = (Phi^T Phi)^-1 Phi^T y
    # Add small regularization for numerical stability
    reg = 1e-10 * np.eye(Phi.shape[1])
    w = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ y_train)
    return w


def predict_polynomial(X, w, degree):
    Phi = polynomial_features(X, degree)
    return Phi @ w


def demonstrate_overfitting():
    """
    Fit degree-1 (underfit), degree-3 (just right), and degree-8 (overfit)
    polynomials to 10 data points. Watch train vs val loss diverge for degree-8.
    """
    np.random.seed(7)

    # True function: y = sin(x) + noise
    # Only 10 training points — easy to overfit
    n_train = 10
    X_train = np.linspace(-2, 2, n_train)
    y_train = np.sin(X_train) + np.random.randn(n_train) * 0.2

    # Validation: 50 points on the same range (the true pattern)
    X_val = np.linspace(-2, 2, 50)
    y_val = np.sin(X_val)  # no noise — measure true generalization

    print(f"  {'Degree':<8} {'Train MSE':<15} {'Val MSE':<15} {'Verdict'}")
    print(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*20}")

    for degree in [1, 3, 5, 8]:
        w = fit_polynomial(X_train, y_train, degree)

        y_hat_train = predict_polynomial(X_train, w, degree)
        y_hat_val   = predict_polynomial(X_val, w, degree)

        train_mse = np.mean((y_hat_train - y_train) ** 2)
        val_mse   = np.mean((y_hat_val - y_val) ** 2)

        # Heuristic: overfitting when val >> train
        gap = val_mse / (train_mse + 1e-10)
        if degree == 1:
            verdict = "underfit"
        elif gap < 5:
            verdict = "good fit"
        elif gap < 20:
            verdict = "mild overfit"
        else:
            verdict = "SEVERE OVERFIT"

        print(f"  {degree:<8} {train_mse:<15.6f} {val_mse:<15.6f} {verdict}")

    print()
    print("  Interpretation:")
    print("  - Degree 1 (line): can't capture sine curve → underfit, high train AND val loss")
    print("  - Degree 3-5:      captures the curve → low train AND val loss")
    print("  - Degree 8:        memorizes 10 training points exactly (train MSE ≈ 0)")
    print("                     but wildly wrong on validation → OVERFIT")
    print("  This is the bias-variance tradeoff in action.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_section("PART 1: Linear Regression from Scratch (lr=0.01, 1000 steps)")
    print("  Training on y = 2x + 3 + noise...")
    print()
    losses, final_w, final_b = linear_regression_from_scratch(lr=0.01, n_steps=1000)
    print(f"\n  Initial loss (step 100): {losses[99]:.6f}")
    print(f"  Final loss  (step 1000): {losses[-1]:.6f}")

    print_section("PART 2: Learning Rate Effects (200 steps each)")
    demonstrate_learning_rates()
    print()
    print("  Key insight:")
    print("  - lr=0.0001: loss barely moves, training is painfully slow")
    print("  - lr=0.01:   smooth convergence, reaches near-zero loss")
    print("  - lr=0.5:    overshoots the minimum repeatedly, diverges")

    print_section("PART 3: Overfitting (degree-8 polynomial on 10 points)")
    demonstrate_overfitting()

    print_section("SUMMARY")
    print("  1. Training loop: forward → loss → backward → update (repeat)")
    print("  2. Gradients tell us: 'nudge weight THIS direction to reduce loss'")
    print("  3. Learning rate is the most important hyperparameter to tune")
    print("  4. Overfitting: model memorizes training data, fails on new data")
    print("  5. Always monitor BOTH train and validation loss during training")
