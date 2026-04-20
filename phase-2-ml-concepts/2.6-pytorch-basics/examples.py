"""
2.6 PyTorch Basics — Examples
Requires: pip install torch
Optional: pip install transformers
Run: python examples.py
"""

import io
import math

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[INFO] torch not installed. Run: pip install torch")
    print("[INFO] Running with stub demonstrations only.\n")


# ---------------------------------------------------------------------------
# 1. Tensor Operations
# ---------------------------------------------------------------------------

def tensor_operations():
    if not TORCH_AVAILABLE:
        print("[tensor_operations] torch not available — skipping.")
        return

    print("=" * 50)
    print("1. Tensor Operations")
    print("=" * 50)

    # Creation
    zeros = torch.zeros(2, 3)
    ones  = torch.ones(2, 3)
    rand  = torch.randn(2, 3)
    print(f"zeros:\n{zeros}")
    print(f"ones:\n{ones}")
    print(f"randn:\n{rand}")

    # Shape, dtype, device
    t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"\nt.shape  = {t.shape}")
    print(f"t.dtype  = {t.dtype}")
    print(f"t.device = {t.device}")
    print(f"t.ndim   = {t.ndim}")

    # Arithmetic & broadcasting
    a = torch.tensor([[1.0], [2.0], [3.0]])  # shape (3,1)
    b = torch.tensor([[10.0, 20.0]])           # shape (1,2)
    print(f"\nBroadcast a+b (3x1 + 1x2 -> 3x2):\n{a + b}")

    # Matrix multiply
    A = torch.randn(4, 5)
    B = torch.randn(5, 3)
    C = A @ B
    print(f"\nA @ B shape: {A.shape} @ {B.shape} = {C.shape}")

    # Autograd
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1   # (x+1)^2
    y.backward()
    print(f"\nAutograd: y = (x+1)^2 at x=3")
    print(f"  dy/dx = {x.grad.item():.1f}  (expected: 2*(3+1) = 8.0)")

    # no_grad context
    with torch.no_grad():
        z = x ** 2
    print(f"  z.requires_grad inside no_grad: {z.requires_grad}")


# ---------------------------------------------------------------------------
# 2. Simple Neural Network
# ---------------------------------------------------------------------------

class SimpleNN(nn.Module):
    """Two fully-connected layers with ReLU activation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def describe_model(model: "nn.Module"):
    """Print parameter summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    for name, param in model.named_parameters():
        print(f"    {name:30s} {tuple(param.shape)}")


# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------

def train_loop(model: "nn.Module", X: "torch.Tensor", y: "torch.Tensor",
               epochs: int = 20):
    if not TORCH_AVAILABLE:
        print("[train_loop] torch not available — skipping.")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn   = nn.MSELoss()

    print("\n  Training (MSELoss, Adam, lr=0.01):")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()          # 1. zero gradients
        preds = model(X)               # 2. forward
        loss  = loss_fn(preds, y)      # 3. loss
        loss.backward()                # 4. backward
        optimizer.step()               # 5. update

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}  loss={loss.item():.6f}")

    return model


# ---------------------------------------------------------------------------
# 4. Save / Load via state_dict
# ---------------------------------------------------------------------------

def save_load_demo(model: "nn.Module"):
    if not TORCH_AVAILABLE:
        print("[save_load_demo] torch not available — skipping.")
        return

    print("\n  Saving model state_dict to in-memory buffer...")
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)

    # Reconstruct
    loaded_model = SimpleNN(in_dim=8, hidden_dim=16, out_dim=1)
    loaded_model.load_state_dict(torch.load(buf, weights_only=True))
    loaded_model.eval()

    # Verify: check first param matches
    orig_w  = list(model.parameters())[0].data
    loaded_w = list(loaded_model.parameters())[0].data
    match = torch.allclose(orig_w, loaded_w)
    print(f"  Weights match after reload: {match}")


# ---------------------------------------------------------------------------
# 5. Hugging Face Pipeline Demo
# ---------------------------------------------------------------------------

def hf_pipeline_demo():
    print("\n" + "=" * 50)
    print("5. Hugging Face pipeline() Demo")
    print("=" * 50)

    try:
        from transformers import pipeline
    except ImportError:
        print("  [INFO] transformers not installed. Run: pip install transformers")
        print("  Showing what the output looks like:")
        mock = [
            {"label": "POSITIVE", "score": 0.9998},
            {"label": "NEGATIVE", "score": 0.9975},
            {"label": "POSITIVE", "score": 0.6431},
        ]
        sentences = [
            "PyTorch makes deep learning approachable and fun.",
            "Debugging NaN losses at 2am is not my favorite activity.",
            "The model sort of works, I guess.",
        ]
        for sent, res in zip(sentences, mock):
            print(f"  [{res['label']:8s} {res['score']:.4f}] {sent}")
        return

    print("  Loading distilbert sentiment pipeline (downloads on first run)...")
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    sentences = [
        "PyTorch makes deep learning approachable and fun.",
        "Debugging NaN losses at 2am is not my favorite activity.",
        "The model sort of works, I guess.",
    ]

    results = classifier(sentences)
    for sent, res in zip(sentences, results):
        print(f"  [{res['label']:8s} {res['score']:.4f}] {sent}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("2.6 PyTorch Basics — Live Demos")
    print("=" * 50)

    # --- Demo 1: Tensor operations ---
    tensor_operations()

    if TORCH_AVAILABLE:
        # --- Demo 2: Model architecture ---
        print("\n" + "=" * 50)
        print("2. SimpleNN Architecture")
        print("=" * 50)
        model = SimpleNN(in_dim=8, hidden_dim=16, out_dim=1)
        describe_model(model)

        # --- Demo 3: Training loop ---
        print("\n" + "=" * 50)
        print("3. Training Loop on Synthetic Data")
        print("=" * 50)
        torch.manual_seed(42)
        # Synthetic: y = sum(x) + noise
        X_synth = torch.randn(64, 8)
        y_synth  = X_synth.sum(dim=1, keepdim=True) + 0.1 * torch.randn(64, 1)
        trained_model = train_loop(model, X_synth, y_synth, epochs=20)

        # --- Demo 4: Save/Load ---
        print("\n" + "=" * 50)
        print("4. Save / Load State Dict")
        print("=" * 50)
        save_load_demo(trained_model)

    # --- Demo 5: HF pipeline ---
    hf_pipeline_demo()

    print("\nDone.")
