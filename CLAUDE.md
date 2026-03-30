# CLAUDE.md — turboquant

> Operating instructions for Claude Code on this repo.

## What Is This?

KV cache compression for HuggingFace Transformers. First open-source implementation of Google's TurboQuant algorithm (ICLR 2026, arXiv:2504.19874). Compresses KV cache entries from 16 bits to 3-4 bits per element using random rotation + optimal scalar quantization.

**Status:** v0.2.0 — compressed index storage shipped. Published on PyPI.

## Current State

| Metric | Value |
|--------|-------|
| Version | 0.2.0 (PyPI) |
| Tests | 15 (8 core + 7 cache) |
| Python source | 926 lines (5 files) |
| CUDA source | 194 lines (1 kernel) |
| Benchmark data | 45 points (4 models, RTX 4080) |
| Total lines | ~1,590 code + ~1,400 docs |
| Dependencies | torch, numpy, scipy, transformers |
| License | Apache 2.0 |
| Python | >= 3.10 |

## Architecture

```
turboquant/core.py         TurboQuantMSE + TurboQuantIP (the algorithms)
turboquant/cache.py         TurboQuantCache (HuggingFace DynamicCache drop-in)
turboquant/cuda_accel.py    Optional CUDA wrapper with PyTorch fallback
turboquant/server.py        OpenAI-compatible inference server
cuda/turboquant_kernel.cu   Fused rotation+quantize CUDA kernels
```

**Algorithm pipeline:** Generate rotation matrix (QR) -> Compute codebook (Beta distribution) -> Quantize: rotate + nearest centroid -> Dequantize: centroid lookup + inverse rotation.

**Cache pipeline:** New tokens go to FP16 residual window (128 tokens). When residual overflows, oldest tokens get quantized and moved to quantized buffer. Attention sees: cat(quantized_old, fp16_recent).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details.

## Key Files

| File | Lines | What |
|------|-------|------|
| `turboquant/core.py` | 344 | TurboQuantMSE (Algorithm 1), TurboQuantIP (Algorithm 2), compute_memory_bytes() |
| `turboquant/cache.py` | 224 | TurboQuantLayer (compressed index storage), TurboQuantCache, memory_usage_bytes(), shared quantizer registry |
| `turboquant/cuda_accel.py` | 76 | cuda_quantize(), cuda_dequantize(), is_cuda_available() |
| `turboquant/server.py` | 263 | load_model(), generate_response(), TurboQuantHandler, main() |
| `cuda/turboquant_kernel.cu` | 194 | Fused quantize + dequantize CUDA kernels |
| `tests/test_core.py` | 92 | 8 tests: MSE bounds, norm preservation, compression ratio, IP unbiasedness, edge cases |
| `tests/test_cache.py` | 93 | 7 tests: basic update, incremental gen, multi-layer, residual quality, bit widths, compressed storage, memory tracking |
| `benchmarks/benchmark_kv.py` | 270 | FP16 vs TQ 3-bit vs TQ 4-bit on real models. Flags: --model, --quick, --context |
| `examples/basic_usage.py` | 40 | Drop-in usage with any HuggingFace model |

## Commands

```bash
pip install -e ".[dev]"              # Install with dev deps
python -m pytest tests/ -v           # Run all 13 tests
python -m turboquant.core            # Sanity test (core algorithms)
python -m turboquant.cache           # Sanity test (cache integration)
turboquant-server --model Qwen/Qwen2.5-3B-Instruct --bits 4 --port 8000

# CUDA extension (optional)
cd cuda/ && python setup.py build_ext --inplace

# Benchmark
python benchmarks/benchmark_kv.py --model Qwen/Qwen2.5-0.5B-Instruct --quick
python benchmarks/benchmark_kv.py --model Qwen/Qwen2.5-3B-Instruct --context "512,1024,2048,4096"

# Build + publish
python -m build
twine upload dist/*
```

## Key Algorithms

**TurboQuantMSE (Algorithm 1):** Rotate unit vector by random orthogonal Pi. Each coordinate now follows Beta((d-1)/2, (d-1)/2). Quantize each coordinate to nearest centroid in an optimal codebook derived from that distribution. Dequantize: centroid lookup + inverse rotation.

**TurboQuantIP (Algorithm 2):** Run MSE at (bits-1). Compute residual. Apply QJL (Quantized Johnson-Lindenstrauss): project residual through random Gaussian S, store signs. Total: (bits-1) MSE + 1 QJL = bits per dimension. Makes inner products unbiased.

**Theoretical MSE bound:** sqrt(3) * pi/2 * (1/4^b) for b-bit quantization.

## Key Classes

| Class | Base | File | Purpose |
|-------|------|------|---------|
| `TurboQuantMSE` | object | core.py | MSE-optimal quantization |
| `TurboQuantIP` | TurboQuantMSE | core.py | Inner-product optimal quantization |
| `TurboQuantLayer` | DynamicLayer | cache.py | Per-layer KV cache with residual window |
| `TurboQuantCache` | DynamicCache | cache.py | Multi-layer cache (drop-in for HF models) |
| `TurboQuantHandler` | BaseHTTPRequestHandler | server.py | HTTP handler (OpenAI-compatible API) |

## Gotchas

- **~~Dequantized storage~~ FIXED in v0.2.0**: Cache now stores uint8 indices + float32 norms. Dequantizes on-the-fly in `update()`. Real compression achieved.
- **scipy dependency for codebook**: `betaincinv` (inverse regularized incomplete beta) is only in scipy, not numpy. This makes the package heavier than you'd expect for "just quantization."
- **Quantizer registry is global**: `_quantizers` in `cache.py` caches quantizer instances by (head_dim, bits, device). This is a module-level dict. If you create caches with different seeds, they'll share quantizers (always seed=42).
- **TurboQuantIP.dequantize() signature differs**: Takes 4 args (mse_indices, norms, qjl_signs, residual_norms) vs TurboQuantMSE's 2 args (indices, norms). Not a clean override.
- **Server uses http.server, not async**: The inference server is synchronous. One request at a time. Fine for testing, not for production.
- **CUDA kernel is FP32 only**: Operates in FP32 to avoid CUDA 12.1 bf16 header compatibility issues. The PyTorch fallback also uses FP32 for the rotation math.
- **1-bit codebook is special-cased**: Uses analytical formula (+/- sqrt(2/(pi*d))) instead of the general Beta quantile approach.
- **stream parameter is ignored**: Server accepts `stream: true` but always returns the full response.

## Related Projects

| Project | What | Repo |
|---------|------|------|
| **turboquant-vectors** | Embedding privacy + compression (ACTIVE) | github.com/back2matching/turboquant-vectors |
| **kvcache-bench** | KV cache benchmarking tool | github.com/back2matching/kvcache-bench |
| **FlockRun** | Parent project, agent runtime | github.com/back2matching/flockrun |
| **llama.cpp PR** | TQ4_0 cache type in GGML | github.com/ggml-org/llama.cpp/pull/20995 |

## PyPI

- Account: back2matching
- Package: turboquant 0.2.0
- Wheel: pure Python (CUDA extension not included, must be built locally)

## Paper

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Zandieh, Daliri, Hadian, Mirrokni (Google Research)
ICLR 2026 | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## Documentation

Full docs in `docs/`:
- [docs/README.md](docs/README.md) -- Doc index
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) -- How the code works
- [docs/ROADMAP.md](docs/ROADMAP.md) -- What shipped, what's possible next
- [docs/reference/API.md](docs/reference/API.md) -- Full API reference
- [docs/reference/CODEBASE-MAP.md](docs/reference/CODEBASE-MAP.md) -- Every file, line counts
- [docs/guides/WORKFLOW.md](docs/guides/WORKFLOW.md) -- ExecPlan workflow (Read->Plan->Execute->Test->Docs->Commit)

Internal docs (local only, not tracked in git):
- `docs/research/` -- Background analysis, competitive landscape, paper deep-dives
- `docs/marketing/` -- Blog post, Reddit drafts, strategy, portfolio blog plan
- `docs/plans/` -- Active and archived project plans
