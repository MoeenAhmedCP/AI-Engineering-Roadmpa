"""
5.6 Model Quantization & Optimization — Examples
Demonstrates: int8/int4 quantization, memory estimation, GGUF naming guide.
Run: python examples.py  (no external deps required)
"""

import struct
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. Simple INT8 Quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_int8(weights: list[float]) -> tuple[list[int], float, float]:
    """
    Quantize float32 weights to int8 using symmetric quantization.
    Returns (quantized_weights, scale, zero_point).
    """
    w_min, w_max = min(weights), max(weights)
    abs_max = max(abs(w_min), abs(w_max))
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    zero_point = 0.0  # symmetric: zero maps to zero

    quantized = [max(-128, min(127, round(w / scale))) for w in weights]
    return quantized, scale, zero_point


def dequantize_int8(quantized: list[int], scale: float, zero_point: float = 0.0) -> list[float]:
    return [(q - zero_point) * scale for q in quantized]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Simple INT4 Quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_int4(weights: list[float]) -> tuple[list[int], float]:
    """
    Quantize to 4-bit signed integers [-8, 7].
    Returns (quantized_weights, scale).
    """
    abs_max = max(abs(w) for w in weights) if weights else 1.0
    scale = abs_max / 7.0 if abs_max > 0 else 1.0
    quantized = [max(-8, min(7, round(w / scale))) for w in weights]
    return quantized, scale


def dequantize_int4(quantized: list[int], scale: float) -> list[float]:
    return [q * scale for q in quantized]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Measure Quantization Error
# ─────────────────────────────────────────────────────────────────────────────

def measure_error(original: list[float], reconstructed: list[float]) -> dict:
    n = len(original)
    mse = sum((o - r) ** 2 for o, r in zip(original, reconstructed)) / n
    max_err = max(abs(o - r) for o, r in zip(original, reconstructed))
    mean_abs = sum(abs(o - r) for o, r in zip(original, reconstructed)) / n
    return {
        "mse": round(mse, 8),
        "max_error": round(max_err, 6),
        "mean_abs_error": round(mean_abs, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def compare_quantization_levels(weights: list[float]) -> None:
    """Print a table comparing float32, int8, int4 memory and reconstruction error."""

    # INT8
    q8, s8, z8 = quantize_int8(weights)
    dq8 = dequantize_int8(q8, s8, z8)
    err8 = measure_error(weights, dq8)

    # INT4
    q4, s4 = quantize_int4(weights)
    dq4 = dequantize_int4(q4, s4)
    err4 = measure_error(weights, dq4)

    n = len(weights)
    f32_bytes = n * 4
    i8_bytes = n * 1
    i4_bytes = n * 0.5

    print("\n── Quantization Comparison ──")
    print(f"  Weights: {n} values")
    print(f"\n  {'Format':<10} {'Bytes':>10} {'Reduction':>12} {'Max Error':>12} {'MSE':>14}")
    print(f"  {'-'*62}")
    print(f"  {'float32':<10} {f32_bytes:>10} {'baseline':>12} {'0.000000':>12} {'0.00000000':>14}")
    print(f"  {'int8':<10} {i8_bytes:>10} {f'{f32_bytes/i8_bytes:.1f}×':>12} "
          f"{err8['max_error']:>12.6f} {err8['mse']:>14.8f}")
    print(f"  {'int4':<10} {i4_bytes:>10} {f'{f32_bytes/i4_bytes:.1f}×':>12} "
          f"{err4['max_error']:>12.6f} {err4['mse']:>14.8f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. GGUF naming guide
# ─────────────────────────────────────────────────────────────────────────────

def gguf_naming_guide() -> None:
    print("\n── GGUF Quantization Naming Guide ──")
    formats = [
        ("Q2_K",   2,  "Extreme compression, noticeable quality loss"),
        ("Q3_K_M", 3,  "Very small, acceptable for simple tasks"),
        ("Q4_0",   4,  "Original 4-bit, decent quality"),
        ("Q4_K_S", 4,  "K-quant small, better than Q4_0"),
        ("Q4_K_M", 4,  "K-quant medium — RECOMMENDED default"),
        ("Q5_K_M", 5,  "Higher quality, ~20% more memory than Q4_K_M"),
        ("Q5_K_S", 5,  "K-quant small 5-bit"),
        ("Q6_K",   6,  "Near-lossless on most benchmarks"),
        ("Q8_0",   8,  "Essentially lossless, 2× Q4_K_M memory"),
        ("F16",    16, "Half precision, full quality"),
    ]
    print(f"\n  {'Format':<12} {'Bits':>5} {'Description'}")
    print(f"  {'-'*65}")
    for name, bits, desc in formats:
        marker = " ← recommended" if name == "Q4_K_M" else ""
        print(f"  {name:<12} {bits:>5}  {desc}{marker}")

    print("\n  Naming logic:")
    print("  Q = quantized | 4 = bits | K = k-quant algorithm | M/S = Medium/Small variant")
    print("  K-quants are smarter about which weights get more bits (non-uniform).")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Model memory estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_model_memory(params_billions: float) -> None:
    """Estimate GPU/RAM required for different quantization levels."""
    params = params_billions * 1e9

    dtypes = [
        ("float32", 4, "Training"),
        ("bfloat16", 2, "Inference, fine-tuning"),
        ("int8",    1, "Quantized inference"),
        ("int4",  0.5, "QLoRA, GGUF Q8_0 approx"),
        ("int4 K", 0.45, "GGUF Q4_K_M approx"),
        ("int2",  0.25, "Extreme (quality loss)"),
    ]

    print(f"\n── Memory Estimate for {params_billions}B parameter model ──")
    print(f"\n  {'Format':<14} {'GB RAM':>8}  {'Use case'}")
    print(f"  {'-'*55}")
    for dtype, bytes_per, use in dtypes:
        gb = params * bytes_per / 1e9
        print(f"  {dtype:<14} {gb:>8.1f}  {use}")

    print(f"\n  Common GPU VRAM: RTX 4090=24GB, A10G=24GB, A100=40/80GB, H100=80GB")


# ─────────────────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    print("=" * 60)
    print("Model Quantization & Optimization Examples")
    print("=" * 60)

    # Generate sample weights that look like a neural network layer
    weights = [random.gauss(0, 0.02) for _ in range(256)]

    # Show quantization comparison
    compare_quantization_levels(weights)

    # Show a few individual values
    print("\n── Sample Weight Reconstruction ──")
    sample = weights[:5]
    q8, s8, z8 = quantize_int8(sample)
    dq8 = dequantize_int8(q8, s8, z8)
    q4, s4 = quantize_int4(sample)
    dq4 = dequantize_int4(q4, s4)

    print(f"\n  {'Original':<16} {'INT8 reconstructed':<22} {'INT4 reconstructed'}")
    print(f"  {'-'*65}")
    for orig, r8, r4 in zip(sample, dq8, dq4):
        print(f"  {orig:>14.6f}   {r8:>18.6f}   {r4:>18.6f}")

    # GGUF guide
    gguf_naming_guide()

    # Memory estimates for common model sizes
    for size in [7.0, 13.0, 34.0, 70.0]:
        estimate_model_memory(size)

    print("\n── Key Takeaways ──")
    print("  • INT8: 4× memory reduction, < 3% quality loss — good default for serving")
    print("  • INT4/Q4_K_M: 8× memory reduction, 5-10% quality loss — fits consumer GPUs")
    print("  • Flash Attention 2: same quality, 2-4× faster — always enable if GPU supports it")
    print("  • KV cache: crucial for long contexts — budget it separately from model weights")
