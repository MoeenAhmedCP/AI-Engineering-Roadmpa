# 2.6 PyTorch Basics

PyTorch is the dominant framework for deep learning research and AI engineering. Even when you use higher-level tools like Hugging Face, you will need to read PyTorch code, write custom training loops, and debug tensor shapes. This section covers the essentials.

---

## Tensors

A tensor is the fundamental data container — essentially an n-dimensional array with extra capabilities: it knows its device (CPU vs GPU) and can optionally record a computation graph for automatic differentiation.

### Creation

```python
import torch

torch.zeros(3, 4)          # 3x4 of zeros (float32 by default)
torch.ones(2, 3)           # 2x3 of ones
torch.randn(5, 5)          # 5x5 standard normal
torch.arange(10)           # [0,1,...,9]
torch.tensor([1.0, 2.0])   # from Python list

# From pretrained weights (e.g., loading a numpy array)
import numpy as np
arr = np.array([[1.0, 2.0], [3.0, 4.0]])
t = torch.from_numpy(arr)  # shares memory — careful with in-place ops
```

### Key Attributes

```python
t = torch.randn(8, 16)
t.shape       # torch.Size([8, 16])
t.dtype       # torch.float32
t.device      # device(type='cpu')
t.ndim        # 2
t.numel()     # 128  — total number of elements
```

### Operations

```python
a = torch.randn(3, 4)
b = torch.randn(4, 5)

a + 1.0           # broadcast scalar
a * 2.0           # elementwise
a @ b             # matrix multiply — equivalent to torch.matmul(a, b)
a.T               # transpose (2D only shorthand)
a.sum()           # scalar sum
a.mean(dim=0)     # mean along axis 0 → shape (4,)
a.unsqueeze(0)    # add dim at position 0 → shape (1,3,4)
a.squeeze()       # remove all size-1 dims
a.reshape(6, 2)   # reinterpret shape; same data
a.view(6, 2)      # like reshape but requires contiguous memory
```

**Broadcasting rules:** Dimensions are aligned from the right. A dimension of size 1 expands to match the other tensor. `shape (3,1) + shape (1,4)` gives `shape (3,4)`.

---

## Autograd

Autograd is PyTorch's automatic differentiation engine. Every tensor operation is recorded on a computation graph; calling `.backward()` traverses it in reverse to populate `.grad` attributes.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1   # y = (x+1)^2
y.backward()              # compute dy/dx
print(x.grad)             # tensor(8.) — dy/dx at x=3 is 2*(x+1) = 8

# Gradients accumulate — always zero them before a new step
x.grad.zero_()
```

**`torch.no_grad()` context:** Disables gradient tracking. Use it during inference and evaluation to save memory and computation.

```python
with torch.no_grad():
    output = model(inputs)   # no graph is built; faster + less memory
```

---

## nn.Module

`nn.Module` is the base class for all neural network layers and models. Subclass it to define your architecture.

```python
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = MyNet(10, 64, 3)
model.parameters()         # iterator of all learnable tensors
model.named_parameters()   # iterator of (name, tensor) pairs
model.eval()               # put in eval mode (disables Dropout, BatchNorm train behavior)
model.train()              # put back in training mode
```

---

## Training Loop Pattern

Every PyTorch training loop follows the same structure:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()          # 1. clear gradients from last step
        logits = model(X_batch)        # 2. forward pass
        loss = loss_fn(logits, y_batch) # 3. compute loss
        loss.backward()                # 4. backpropagate
        optimizer.step()               # 5. update weights
    # evaluation outside the loop with model.eval() + torch.no_grad()
```

The order matters: zero gradients BEFORE the forward pass; step AFTER backward.

---

## Optimizers

| Optimizer | When to Use |
|-----------|-------------|
| **SGD** | Simple baselines, CV fine-tuning (with momentum). Predictable but slow to converge. |
| **Adam** | Default choice for most tasks. Adaptive learning rates per parameter. |
| **AdamW** | Adam + decoupled weight decay. Preferred for LLM fine-tuning and transformers. |

```python
torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))
torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01)
```

Use a learning rate scheduler to reduce `lr` over training:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scheduler.step()   # call once per epoch
```

---

## Loss Functions

| Loss | Use Case |
|------|----------|
| `nn.MSELoss()` | Regression |
| `nn.CrossEntropyLoss()` | Multi-class classification (takes raw logits, applies softmax internally) |
| `nn.BCEWithLogitsLoss()` | Binary classification (numerically stable; takes logits, not probabilities) |
| `nn.NLLLoss()` | When you apply `log_softmax` yourself |

**Important:** `CrossEntropyLoss` expects integer class indices as targets, not one-hot. `BCEWithLogitsLoss` expects float targets (0.0 or 1.0).

---

## GPU Acceleration

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On Apple Silicon:
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = model.to(device)
X = X.to(device)
y = y.to(device)
```

All tensors involved in a computation must be on the same device. A common bug is forgetting to move input data to the GPU. Move data in the training loop batch-by-batch, not all at once.

---

## Saving and Loading

```python
# Save (weights only — preferred)
torch.save(model.state_dict(), "model.pt")

# Load
model = MyNet(10, 64, 3)              # must reconstruct architecture first
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# Save entire model (ties you to the exact class definition at load time)
torch.save(model, "model_full.pt")
```

Always use `state_dict()` for production — it decouples weights from code.

---

## Hugging Face Integration

Hugging Face's `transformers` library wraps PyTorch models in a convenient API.

```python
from transformers import pipeline, AutoModel, AutoTokenizer

# High-level: pipeline handles tokenization, forward pass, decoding
pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
pipe("This movie was great!")

# Mid-level: direct model access, still PyTorch under the hood
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # shape: (1, seq_len, 768)
```

**When to use what:**
- `pipeline` — for quick inference, demos, and standard tasks.
- `AutoModel` + raw PyTorch — when you need custom heads, intermediate activations, or efficient batching.
- Full custom PyTorch — when architecture isn't in HF Hub, or you need maximum control for research.

---

## AI Engineering Context

In production AI engineering, you usually spend more time on **inference** than training:

- **Just inference:** Use HF `pipeline` or `model.forward()` under `torch.no_grad()`. Export via `torch.onnx.export` or `torch.jit.script` for deployment.
- **Fine-tuning a pretrained model:** Load HF weights, attach a custom head, train with AdamW and a small LR (1e-5 to 3e-4). Freeze lower layers early.
- **Training from scratch:** Only when you have a truly novel architecture or massive proprietary data. Rare in practice.
- **Evaluation:** Always `model.eval()` + `torch.no_grad()`. Forgetting either is a silent bug that leaks memory or changes BatchNorm/Dropout behavior.

Understanding PyTorch internals lets you debug shape errors, write custom loss functions, and integrate pre-trained components without being boxed in by high-level APIs.
