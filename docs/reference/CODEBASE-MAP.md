# Codebase Map

> Every file in the repo, what it does, and line counts.

## Summary

| Category | Files | Lines |
|----------|-------|-------|
| Python source | 4 | 849 |
| CUDA | 1 | 194 |
| Tests | 2 (+1 __init__) | 155 |
| Benchmarks | 1 | 254 |
| Examples | 1 | 40 |
| Config/build | 4 | 155 |
| Docs | 2 (README + CLAUDE) | 214 |
| **Total** | **16** | **1,861** |

## Package: turboquant/

| File | Lines | What it does |
|------|-------|-------------|
| `__init__.py` | 19 | Exports TurboQuantMSE, TurboQuantIP, TurboQuantCache. Defines `__version__ = "0.1.0"`. |
| `core.py` | 344 | Core algorithms. `TurboQuantMSE` (Algorithm 1): random rotation via QR + optimal Beta-distribution codebook + scalar quantization. `TurboQuantIP` (Algorithm 2): MSE at (bits-1) + QJL residual correction. Also `compute_memory_bytes()`. Includes inline sanity test in `__main__`. |
| `cache.py` | 166 | HuggingFace integration. `TurboQuantLayer` subclasses `DynamicLayer` with residual window (128 tokens FP16, overflow quantized). `TurboQuantCache` subclasses `DynamicCache`, creates TurboQuantLayer per transformer layer. Shared quantizer registry `_get_quantizer()`. |
| `cuda_accel.py` | 76 | Optional CUDA wrapper. `cuda_quantize()` and `cuda_dequantize()` with automatic fallback to PyTorch. `is_cuda_available()` check. |
| `server.py` | 263 | OpenAI-compatible inference server. `load_model()` with INT8/INT4 weight quantization. `generate_response()` with TurboQuantCache. `TurboQuantHandler` HTTP handler (3 endpoints). CLI entry point via `main()`. |

## CUDA: cuda/

| File | Lines | What it does |
|------|-------|-------------|
| `turboquant_kernel.cu` | 194 | Two fused CUDA kernels. `turboquant_quantize_kernel`: warp-reduce norm, mat-vec rotation, nearest-centroid quantization. `turboquant_dequantize_kernel`: centroid lookup, inverse rotation, norm scaling. Python bindings via pybind11. All FP32. |
| `setup.py` | 19 | Build script for the CUDA extension. Uses `torch.utils.cpp_extension.CUDAExtension`. |

## Tests: tests/

| File | Lines | Tests | What it covers |
|------|-------|-------|---------------|
| `__init__.py` | 0 | -- | Package marker |
| `test_core.py` | 92 | 8 | `TestTurboQuantMSE`: 1-bit and 2-bit within theoretical bounds, roundtrip norm preservation, compression ratio. `TestTurboQuantIP`: unbiased inner product. `TestEdgeCases`: single vector, zero vector, different dimensions. |
| `test_cache.py` | 63 | 5 | `TestTurboQuantCache`: basic update, incremental generation (prefill + 10 tokens), multi-layer (8 layers), residual window quality (FP16 exactness), different bit widths. |

**Total: 13 tests.**

## Benchmarks: benchmarks/

| File | Lines | What it does |
|------|-------|-------------|
| `benchmark_kv.py` | 254 | Full model benchmarking. Loads a HuggingFace model, runs FP16 baseline vs TurboQuant 3-bit and 4-bit on short and long prompts. Measures peak VRAM, KV cache estimate, tokens/sec, prefill time. Saves results to JSON. CLI: `--model`, `--quick`. |

## Examples: examples/

| File | Lines | What it does |
|------|-------|-------------|
| `basic_usage.py` | 40 | Three-step drop-in usage: load model, create `TurboQuantCache(bits=4)`, generate text. Token-by-token autoregressive loop. |

## Config & Build

| File | Lines | What it does |
|------|-------|-------------|
| `pyproject.toml` | 46 | Package metadata. Name: turboquant. Version: 0.1.0. Python >= 3.10. Dependencies: torch, transformers, scipy, numpy. Optional: pytest, uvicorn/fastapi. Entry point: `turboquant-server`. |
| `LICENSE` | 16 | Apache 2.0 |
| `.gitignore` | 10 | Standard Python ignores (__pycache__, dist, .egg-info, etc.) |

## Distribution: dist/

| File | Size | What it is |
|------|------|-----------|
| `turboquant-0.1.0.tar.gz` | 17.5 KB | Source distribution |
| `turboquant-0.1.0-py3-none-any.whl` | 15.0 KB | Wheel (pure Python, no CUDA) |

## Root Docs

| File | Lines | What it does |
|------|-------|-------------|
| `README.md` | 165 | PyPI/GitHub README. Installation, quick start, benchmarks, algorithm explanation, comparison with alternatives. |
| `CLAUDE.md` | -- | Claude Code operating instructions (updated separately). |
