# KV Cache Compression Landscape

> Every method that exists, what's in production, what's research-only.

## Production Engines (Shipping Today)

| Engine | KV Cache Types | Minimum Bits | Notes |
|--------|---------------|-------------|-------|
| **llama.cpp** | f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 | 4-bit | Flash attention required |
| **Ollama** | q8_0, q4_0 (subset of llama.cpp) | 4-bit | `OLLAMA_KV_CACHE_TYPE` env var |
| **vLLM** | FP8 E4M3, FP8 E5M2 | 8-bit | No sub-8-bit support |
| **TensorRT-LLM** | FP8, INT8, NVFP4 (Blackwell) | 4-bit (Blackwell only) | Best throughput improvement |

**Gap:** No production engine supports sub-4-bit KV cache quantization. Everything below q4_0 is research-only.

## Research Methods (Not In Production)

| Method | Venue | Bits | Key Insight |
|--------|-------|------|-------------|
| **TurboQuant** | ICLR 2026 | 2.5-3.5 | Random rotation + optimal scalar quant + QJL residual |
| **KIVI** | ICML 2024 | 2 | Asymmetric: per-channel keys, per-token values |
| **KVQuant** | NeurIPS 2024 | 2-3 | Pre-RoPE quantization, outlier isolation |
| **PALU** | ICLR 2025 | Low-rank | Decompose KV projections, 91% reduction |
| **H2O** | NeurIPS 2023 | N/A (eviction) | Keep top-k tokens by attention score |
| **StreamingLLM** | 2023 | N/A (eviction) | Attention sinks + sliding window |

## KV Cache Memory Formula

```
Per-token KV = 2 (K+V) x Layers x KV_Heads x Head_Dim x Bytes
```

| Model | Standard KV/token | With Hybrid (DeltaNet) |
|-------|-------------------|----------------------|
| Llama-3.1-8B | 144 KB | 144 KB (no hybrid) |
| Qwen3.5-9B | 144 KB | **~36 KB** (75% DeltaNet) |
| Qwen3.5-27B | ~256 KB | **~64 KB** (75% DeltaNet) |

Qwen3.5's hybrid architecture already gives 4x KV reduction. TurboQuant on top of this gives diminishing returns.

## Integration Points

Adding new KV quant to **llama.cpp** requires:
- ggml.h (enum + type traits)
- ggml-quants.c (quantize/dequantize)
- CUDA FlashAttention kernels (HARDEST — can't reuse existing dequant code)
- 8-15 files total, weeks of work

**Ollama** inherits from llama.cpp automatically.

See [SYNTHESIS.md](SYNTHESIS.md) for the full integration strategy.
