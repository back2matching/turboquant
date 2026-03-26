# Roadmap

> What shipped, what's possible next. Status: published on PyPI, not actively developed.

## Shipped (v0.1.0)

✅ TurboQuantMSE -- MSE-optimal vector quantization (Algorithm 1)
✅ TurboQuantIP -- Inner-product optimal quantization (Algorithm 2)
✅ TurboQuantCache -- Drop-in DynamicCache replacement for HuggingFace
✅ Residual window (128 tokens FP16, older tokens quantized)
✅ CUDA kernel -- Fused rotation+quantize (2.4x speedup)
✅ OpenAI-compatible inference server (`turboquant-server` CLI)
✅ 13 tests (core algorithms + cache integration)
✅ Benchmark harness (FP16 vs TQ 3-bit vs TQ 4-bit, `--context` sweep, per-model JSON output)
✅ **RTX 4080 benchmark data** -- 45 data points, 4 models (7B/3B/1.6B/0.5B), contexts up to 8K
✅ Published on PyPI as `turboquant 0.1.0`
⚠️ llama.cpp PR submitted ([#20995](https://github.com/ggml-org/llama.cpp/pull/20995)) — closed, premature. Multiple competing implementations in progress.

## Possible Next Steps

These are ideas, not planned work. The package is stable at 0.1.0.

### Performance

⬜ **Store compressed indices instead of dequantized FP16** -- Currently the quantized buffer stores lossy FP16 values after a quantize-dequantize roundtrip. Storing raw uint8 indices + float32 norms and dequantizing on-the-fly during attention would cut memory further. This is the main optimization left.

⬜ **Shared memory tiling in CUDA kernel** -- Current kernel does naive mat-vec per thread. Tiling the rotation matrix in shared memory would reduce global memory reads from O(D^2) to O(D^2 / tile_size).

⬜ **Bitpacking for sub-8-bit indices** -- 3-bit indices are stored as uint8 (wasting 5 bits). Packing 8 3-bit values into 3 bytes would reduce index storage by 62%.

⬜ **Streaming generation in server** -- Accept `stream: true` and return Server-Sent Events.

### Quality

⬜ **Per-layer bit allocation** -- Some layers are more sensitive than others. Allocating more bits to sensitive layers and fewer to others could improve quality at the same total budget.

⬜ **Adaptive residual window** -- Grow/shrink the residual window based on attention entropy. High-entropy layers benefit more from FP16 precision.

### Integration

⬜ **vLLM PagedAttention support** -- vLLM's paged KV cache uses a different memory layout. Supporting it would bring TurboQuant to production serving.

⬜ **GGML quantization type** -- The llama.cpp PR adds TQ4_0 as a cache type. Getting it merged and adding TQ3_0 would cover the GGML ecosystem.

⬜ **Triton kernel** -- A Triton implementation would be more portable than raw CUDA and easier to maintain.

## Related Work

| Project | Relationship |
|---------|-------------|
| [turboquant-vectors](https://github.com/back2matching/turboquant-vectors) | Applies TurboQuant to embeddings (not KV cache). Separate repo, separate use case. |
| [kvcache-bench](https://github.com/back2matching/kvcache-bench) | Benchmarking tool that tests TurboQuant alongside other KV compression methods. |
| [KIVI](https://arxiv.org/abs/2402.02750) | Prior work on KV cache quantization. TurboQuant's residual window pattern comes from KIVI. |
| [QJL](https://arxiv.org/abs/2406.03482) | Quantized JL transform used in TurboQuantIP's Stage 2. |

## Paper Reference

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research)
ICLR 2026 | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
