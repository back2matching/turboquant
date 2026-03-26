# Architecture

> How TurboQuant works, from math to code.

## Overview

TurboQuant compresses LLM KV cache entries from 16 bits to 3-4 bits per element. The pipeline has four layers:

```
core.py          Algorithm: rotation + scalar quantization
    |
cache.py         HuggingFace DynamicCache with residual window
    |
cuda_accel.py    Optional CUDA acceleration (2.4x speedup)
    |
server.py        OpenAI-compatible HTTP server
```

Each layer builds on the one above. You can use any layer independently.

## Core Algorithm (core.py, 344 lines)

Implements Algorithms 1 and 2 from the TurboQuant paper (Zandieh et al., arXiv:2504.19874, ICLR 2026).

### TurboQuantMSE (Algorithm 1)

MSE-optimal vector quantization. Four steps:

**1. Random rotation (one-time setup)**
Generate a random orthogonal matrix Pi via QR decomposition of a Gaussian matrix. This rotation is computed once per (dim, bits, seed) combination and reused for all vectors.

```
gaussian = torch.randn(dim, dim)
Q, R = torch.linalg.qr(gaussian)
Pi = Q * sign(diag(R))     # Ensure det = +1
```

**Why it works:** Multiplying a unit vector by a random orthogonal matrix makes each coordinate follow a Beta((d-1)/2, (d-1)/2) distribution on [-1, 1]. In high dimensions (d >= 64), coordinates become nearly independent. This means scalar quantization per coordinate is near-optimal for the full vector.

**2. Optimal codebook (one-time setup)**
Compute 2^b centroids that minimize MSE over the Beta distribution. For each bit width:
- Divide the Beta CDF into 2^b equal-probability intervals (using `betaincinv` from scipy)
- Place each centroid at the conditional expectation E[X | X in interval]
- Special case for 1-bit: centroids at +/- sqrt(2/(pi*d))

**3. Quantize (per vector)**
```
norms = ||x||
x_unit = x / ||x||
y = x_unit @ Pi^T           # Rotate
indices = argmin |y - c|     # Nearest centroid per coordinate
```

Output: uint8 indices (shape ..., dim) + float32 norms (shape ..., 1).

**4. Dequantize (per vector)**
```
y_hat = codebook[indices]    # Look up centroids
x_hat = y_hat @ Pi           # Inverse rotation
x_hat = x_hat * norms        # Rescale
```

### TurboQuantIP (Algorithm 2)

Extends MSE quantization to preserve inner products (important for attention scores). Two-stage approach:

**Stage 1:** Run TurboQuantMSE at (bits-1) to get MSE-optimal compression.

**Stage 2:** Compute residual = x_unit - x_mse, then apply Quantized Johnson-Lindenstrauss (QJL):
- Project residual through random Gaussian matrix S
- Store only the signs (1 bit per dimension)
- Dequantize via: gamma * sqrt(pi/2) / dim * (signs @ S)

Total storage: (bits-1) for MSE indices + 1 bit for QJL signs = bits total per dimension.

The QJL correction makes inner products unbiased (E[approx_ip] = true_ip), which matters for attention.

### Memory Computation

`compute_memory_bytes()` calculates exact storage for quantized vectors:
- MSE mode: index_bytes + norm_bytes
- IP mode: index_bytes + qjl_bytes + norm_bytes + residual_norm_bytes
- Reports compression ratio vs FP16 baseline

### Theoretical Bounds

MSE distortion for b-bit quantization: sqrt(3) * pi/2 * (1/4^b)

| Bits | Theoretical Bound | Compression vs FP16 |
|------|-------------------|---------------------|
| 1 | 0.680 | 12.8x |
| 2 | 0.170 | 7.1x |
| 3 | 0.043 | 4.9x |
| 4 | 0.011 | 3.8x |

## Cache Integration (cache.py, 166 lines)

### TurboQuantLayer

Subclasses HuggingFace's `DynamicLayer`. Implements a residual window pattern (from KIVI):

- **Residual window** (default 128 tokens): kept in full FP16. No quantization applied.
- **Overflow**: when the residual exceeds `residual_len`, the oldest tokens get quantized via TurboQuantMSE and moved to the quantized buffer.
- **Full view**: attention sees quantized (old) + residual (recent) concatenated along the sequence dimension.

This means the model always has exact FP16 values for the most recent 128 tokens, which are the ones attention focuses on most.

### TurboQuantCache

Subclasses HuggingFace's `DynamicCache`. Drop-in replacement: just pass it as `past_key_values`.

- Creates `TurboQuantLayer` instances as needed per transformer layer
- Shares quantizer instances across layers via `_get_quantizer()` registry (keyed by head_dim, bits, device)
- Full API compatibility with transformers 4.40.0+

### Data Flow

```
model.forward(past_key_values=cache)
  -> cache.update(key_states, value_states, layer_idx)
    -> TurboQuantLayer.update(key_states, value_states)
      -> Append to residual window
      -> If residual > 128 tokens:
           Quantize overflow (flatten -> quantize -> dequantize -> reshape)
           Move dequantized to quantized buffer
           Trim residual
      -> Return: cat(quantized, residual)
```

Note: the quantized buffer stores dequantized (lossy FP16) values, not raw indices. This avoids a dequantization step on every attention pass but uses more memory than storing indices directly. A future optimization could store compressed indices and dequantize on-the-fly.

## CUDA Acceleration (cuda_accel.py + cuda/, 76 + 194 + 19 lines)

### Wrapper (cuda_accel.py)

Provides `cuda_quantize()` and `cuda_dequantize()` functions that:
- Try to import the `cuda_turboquant` extension module
- If available and input is on CUDA, use the fused kernels
- Otherwise, fall back to pure PyTorch (identical math)

The wrapper exposes `is_cuda_available()` to check at runtime.

### CUDA Kernel (turboquant_kernel.cu)

Two fused kernels, both operating in FP32:

**`turboquant_quantize_kernel`** -- One block per vector:
1. Warp-reduce to compute vector norm
2. Block-reduce via shared memory
3. For each coordinate: compute rotated value (mat-vec), find nearest centroid
4. Output: uint8 indices + float32 norm

**`turboquant_dequantize_kernel`** -- One block per vector:
1. Look up centroid values from indices
2. Inverse rotation (mat-vec with un-transposed Pi)
3. Scale by norm

Both kernels use BLOCK_SIZE=256 threads and FP32 throughout (avoids CUDA 12.1 bf16 header issues).

### Build

```bash
cd cuda/
python setup.py build_ext --inplace
```

Requires: CUDA toolkit, PyTorch with CUDA support. Compiles with `-O3 -std=c++17 --use_fast_math`.

## Inference Server (server.py, 263 lines)

OpenAI-compatible HTTP server using Python's `http.server` (no framework dependencies).

### Endpoints

| Method | Path | What |
|--------|------|------|
| POST | `/v1/chat/completions` | Chat completion with TurboQuant KV cache |
| GET | `/v1/models` | List loaded model |
| GET | `/health` | Health check + GPU stats |
| OPTIONS | `*` | CORS preflight |

### Generation Pipeline

```
load_model()              -- Load HuggingFace model + tokenizer (supports INT8/INT4 weight quantization)
    |
generate_response()       -- Prefill + autoregressive generation
    |                        Uses TurboQuantCache as past_key_values
    |
TurboQuantHandler         -- HTTP handler, JSON in/out
    |
HTTPServer                -- Serve on 0.0.0.0:{port}
```

The server is lazy-loaded: model loads on first request (or at startup if called via `main()`). Supports:
- Temperature sampling (greedy argmax when temp != 1.0)
- Max token limits
- Tools parameter (passed through but not executed)
- Custom model selection via `--model`
- Weight quantization via `--quantize int8|int4` (bitsandbytes)

Response includes a `turboquant` field with KV bits, generation time, tokens/sec, and VRAM usage.

### CLI Entry Point

Registered as `turboquant-server` in pyproject.toml:

```bash
turboquant-server --model Qwen/Qwen2.5-3B-Instruct --bits 4 --port 8000
```

## Dependencies

| Package | Version | Used for |
|---------|---------|----------|
| torch | >= 2.2.0 | Tensor ops, QR decomposition, CUDA |
| transformers | >= 4.40.0 | DynamicCache base class, model loading |
| scipy | >= 1.10.0 | Beta distribution inverse CDF (codebook computation) |
| numpy | >= 1.24.0 | Numerical integration for conditional expectations |

Optional: `bitsandbytes` (for INT8/INT4 weight quantization in server).

## Design Decisions

**Why store dequantized FP16 instead of indices?** Simpler implementation. Avoids dequantization in the attention hot path. Trades memory savings for speed. A production implementation would store indices and dequantize on-the-fly.

**Why residual window = 128?** Matches KIVI's default. Recent tokens dominate attention weights, so keeping them in full precision has the biggest quality impact. Configurable per-layer.

**Why QR decomposition for the rotation?** Ensures exact orthogonality (unlike Gram-Schmidt which accumulates numerical error). The paper specifies this approach.

**Why scipy for codebook computation?** The optimal codebook requires the inverse regularized incomplete beta function. NumPy doesn't have it; scipy does (`betaincinv`). This is a one-time cost at initialization.

**Why no streaming in the server?** Kept it simple for v0.1.0. The `stream` parameter is accepted but ignored. Server-Sent Events would be the natural addition.
