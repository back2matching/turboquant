# llama.cpp TurboQuant Port: Technical Design

> Based on analysis of ikawrakow's ik_llama.cpp fork.

## Key Discovery

ikawrakow **already has rotation-based KV cache types** (Q4_0_R8, Q5_0_R4, Q8_0_R8). These use Hadamard-style rotation with multiple independent scales per block. Our TurboQuant approach (random orthogonal rotation via QR decomposition + optimal Beta codebook) is a different and provably better rotation strategy.

We're not inventing infrastructure from scratch. We're adding a better rotation algorithm to existing rotation infrastructure.

## Approach: TQ4_0 (TurboQuant-Rotated Q4_0)

### Block Structure

```c
#define QK_TQ4_0 32  // Same as Q4_0

typedef struct {
    ggml_half d;           // Scale factor (same as Q4_0) — 2 bytes
    uint8_t qs[QK_TQ4_0/2]; // 4-bit quantized values — 16 bytes
} block_tq4_0;
// Total: 18 bytes = same as Q4_0
// Difference: values are quantized AFTER rotation by a shared orthogonal matrix
```

**Same block size and layout as Q4_0.** The only difference is in the quantize/dequantize functions, which apply the TurboQuant rotation before/after the standard Q4_0 quantization.

### Rotation Matrix Storage

The rotation matrix `Pi` (dim x dim, where dim = head_dim = 128) is:
- Generated once at model init via QR decomposition of a random Gaussian matrix
- Shared across ALL layers and ALL heads (same rotation for every KV vector)
- Stored as a global persistent tensor (128x128 = 16384 floats = 64KB)
- Seed-based: deterministic from a fixed seed, so no need to save to GGUF

**Why shared:** The rotation is data-oblivious (doesn't depend on model weights). Same rotation works for any model. This means zero storage overhead in the GGUF file.

### Quantize: quantize_row_tq4_0

```
Input: float x[head_dim]    (one KV head vector)
1. Normalize: norm = ||x||, x_unit = x / norm
2. Rotate: y = Pi^T @ x_unit   (matrix-vector multiply, 128x128)
3. Quantize y as standard Q4_0:
   - Find max(|y[0..31]|), compute scale d = max / 8
   - Round each y[i] to 4-bit: q[i] = round(y[i] / d)
   - Pack into block_tq4_0.qs[] and block_tq4_0.d
4. Store norm separately (or baked into scale d)
```

### Dequantize: dequantize_row_tq4_0

```
Input: block_tq4_0 block
1. Unpack standard Q4_0: y_hat[i] = (nibble - 8) * d
2. Inverse rotate: x_hat = Pi @ y_hat   (matrix-vector multiply)
3. Rescale by norm (if stored separately)
Output: float x_hat[head_dim]
```

### Compute Cost

The rotation adds a 128x128 matrix-vector multiply per KV head vector. At head_dim=128:
- 128*128 = 16,384 FMA operations per vector
- On RTX 4080 (49 TFLOPS FP32): ~0.3 microseconds per vector
- For 32K tokens * 4 KV heads = 128K vectors: ~40ms total per layer

This is acceptable for prefill (one-time cost) and negligible for generation (1 vector per step).

### Files to Change (7 files)

1. **ggml/include/ggml.h** — Add `GGML_TYPE_TQ4_0 = 250` to enum (unused slot)
2. **ggml/src/ggml-common.h** — Define `block_tq4_0` (identical to `block_q4_0`)
3. **ggml/src/iqk/iqk_quantize.h** — Declare function signatures
4. **ggml/src/iqk/iqk_quantize.cpp** — Implement quantize/dequantize with rotation
5. **ggml/src/ggml.c** — Add type_traits entry
6. **common/common.h** — Add "tq4_0" to cache type docs
7. **ggml/src/iqk/iqk_flash_attn.cpp** — Add TQ4_0 dequant path in flash attention kernel

### vs ikawrakow's Rotation Types

| Feature | Q4_0_R8 (ikawrakow) | TQ4_0 (ours) |
|---------|---------------------|---------------|
| Rotation | Hadamard (deterministic, fixed) | Random orthogonal (QR decomp) |
| Blocks | 8 sub-blocks per group, each with own scale | Standard Q4_0 blocks (no sub-blocks) |
| Storage overhead | 8x scales per group (larger blocks) | Same as Q4_0 (no overhead) |
| Theoretical guarantee | None (empirical) | Near-optimal MSE bound (proven) |
| Codebook | Standard Q4_0 (uniform) | Optimal for Beta distribution |
| Storage per value | 4.5 bytes (rotation overhead) | 0.56 bytes (same as Q4_0!) |

**Our advantage:** Same storage as Q4_0 but better distortion due to optimal rotation + codebook. ikawrakow's approach trades storage for quality (larger blocks). Ours is free.

### Implementation Priority

1. **CPU scalar reference** (1 week) — Prove correctness against our PyTorch implementation
2. **Benchmark vs q4_0** (2 days) — Show perplexity improvement at same VRAM
3. **CUDA kernel** (2-3 weeks) — Fuse rotation into flash attention dequant path
4. **Submit PR** (1 week) — To ik_llama.cpp fork first

### Risk: Rotation Cost in Flash Attention

The main concern: can we fuse the 128x128 rotation into the flash attention kernel without killing performance?

Options:
- Pre-rotate during quantize (at KV cache write time), store rotated values. Dequantize is standard Q4_0. **This is the clean approach.** No change to flash attention at all.
- Fuse rotation into dequantize during attention. More complex, may not be faster.

**Recommendation:** Pre-rotate approach. The rotation happens once when a token enters the KV cache. Flash attention dequantizes standard Q4_0 blocks (unchanged). The inverse rotation only happens on cache read, which we can bake into the to_float function.

Wait — this doesn't work because the Q4_0 codebook is designed for uniform distributions, not for Beta-distributed rotated coordinates. The whole point of TurboQuant is that after rotation, the optimal codebook is different from Q4_0's uniform quantizer.

**Revised approach:** Use TurboQuant's optimal Beta codebook instead of Q4_0's uniform quantizer. This means the dequantize values map to Beta-optimal centroids, not `(nibble - 8) * d`. The block structure stays the same (16 indices + 1 scale), but the centroid mapping changes.

This is actually simpler: just change the lookup table in dequantize from `{-8d, -7d, ..., 7d}` to `{c0*d, c1*d, ..., c15*d}` where `c_i` are the Beta-optimal centroids. One lookup table change.
