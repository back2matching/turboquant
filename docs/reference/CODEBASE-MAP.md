# Codebase Map

> Every file in the repo, what it does, and line counts. Updated 2026-03-26.

## Summary

| Category | Files | Lines |
|----------|-------|-------|
| Python source | 5 | 926 |
| CUDA | 2 (kernel + setup) | 213 |
| Tests | 2 (+1 __init__) | 155 |
| Benchmarks | 1 (+4 JSON results) | 270 |
| Examples | 1 | 40 |
| Config/build | 3 | 72 |
| Docs (core) | 8 | 1,170 |
| Docs (research) | 8 | 2,081 |
| Docs (plans) | 1 archive | 166 |
| **Total** | **~35** | **~5,035** |

## Package: turboquant/

| File | Lines | What it does |
|------|-------|-------------|
| `__init__.py` | 19 | Exports TurboQuantMSE, TurboQuantIP, TurboQuantCache. Defines `__version__ = "0.2.0"`. |
| `core.py` | 344 | Core algorithms. `TurboQuantMSE` (Algorithm 1): random rotation via QR + optimal Beta-distribution codebook + scalar quantization. `TurboQuantIP` (Algorithm 2): MSE at (bits-1) + QJL residual correction. Also `compute_memory_bytes()`. Includes inline sanity test in `__main__`. |
| `cache.py` | 224 | HuggingFace integration. `TurboQuantLayer` stores compressed uint8 indices + float32 norms (v0.2.0), dequantizes on-the-fly. Residual window (128 tokens FP16). `TurboQuantCache` subclasses `DynamicCache`. `memory_usage_bytes()` reports compression stats. |
| `cuda_accel.py` | 76 | Optional CUDA wrapper. `cuda_quantize()` and `cuda_dequantize()` with automatic fallback to PyTorch. `is_cuda_available()` check. |
| `server.py` | 263 | OpenAI-compatible inference server. `load_model()` with INT8/INT4 weight quantization. `generate_response()` with TurboQuantCache. `TurboQuantHandler` HTTP handler (3 endpoints). CLI entry point via `main()`. |

## CUDA: cuda/

| File | Lines | What it does |
|------|-------|-------------|
| `turboquant_kernel.cu` | 194 | Two fused CUDA kernels. `turboquant_quantize_kernel`: warp-reduce norm, mat-vec rotation, nearest-centroid quantization. `turboquant_dequantize_kernel`: centroid lookup, inverse rotation, norm scaling. Python bindings via pybind11. All FP32. |
| `setup.py` | 19 | Build script for the CUDA extension. Uses `torch.utils.cpp_extension.CUDAExtension`. Optional — PyTorch fallback if not built. |

## Tests: tests/

| File | Lines | Tests | What it covers |
|------|-------|-------|---------------|
| `__init__.py` | 0 | -- | Package marker |
| `test_core.py` | 92 | 8 | `TestTurboQuantMSE`: 1-bit and 2-bit within theoretical bounds, roundtrip norm preservation, compression ratio. `TestTurboQuantIP`: unbiased inner product. `TestEdgeCases`: single vector, zero vector, different dimensions. |
| `test_cache.py` | 93 | 7 | `TestTurboQuantCache`: basic update, incremental generation, multi-layer, residual window quality, different bit widths, compressed index storage verification, memory usage tracking. |

**Total: 15 tests.**

## Benchmarks: benchmarks/

| File | Lines | What it does |
|------|-------|-------------|
| `benchmark_kv.py` | 270 | Full model benchmarking. Loads a HuggingFace model, runs FP16 baseline vs TurboQuant 3-bit and 4-bit across configurable context lengths. Measures peak VRAM, KV cache estimate, tokens/sec, prefill time. Saves per-model + combined JSON. CLI: `--model`, `--quick`, `--context "512,1024,2048"`. |
| `benchmark_results.json` | -- | Combined results across all models (45 data points). |
| `results_qwen2.5-*.json` | -- | Per-model results (7B: 6 pts, 3B: 12 pts, 0.5B: 15 pts). |
| `results_stablelm-*.json` | -- | StableLM-2-1.6B cross-architecture results (12 pts). |

## Examples: examples/

| File | Lines | What it does |
|------|-------|-------------|
| `basic_usage.py` | 40 | Three-step drop-in usage: load model, create `TurboQuantCache(bits=4)`, generate text. Token-by-token autoregressive loop. |

## Config & Build

| File | Lines | What it does |
|------|-------|-------------|
| `pyproject.toml` | 46 | Package metadata. Name: turboquant. Version: 0.2.0. Python >= 3.10. Dependencies: torch, transformers, scipy, numpy. Optional: pytest, uvicorn/fastapi. Entry point: `turboquant-server`. |
| `LICENSE` | 16 | Apache 2.0 |
| `.gitignore` | 10 | Standard Python ignores (__pycache__, dist, .egg-info, etc.) |

## Distribution: dist/

| File | Size | What it is |
|------|------|-----------|
| `turboquant-0.2.0.tar.gz` | -- | Source distribution |
| `turboquant-0.2.0-py3-none-any.whl` | -- | Wheel (pure Python, no CUDA) |

## Root Docs

| File | Lines | What it does |
|------|-------|-------------|
| `README.md` | 217 | PyPI/GitHub README. Installation, quick start, benchmarks (4 models, RTX 4080 data), algorithm explanation, comparison with alternatives. |
| `CLAUDE.md` | 133 | Claude Code operating instructions. Current state, architecture, commands, gotchas. |

## docs/

| File | Lines | What it does |
|------|-------|-------------|
| `docs/README.md` | 44 | Doc index and project summary. |
| `docs/ARCHITECTURE.md` | 233 | Full architecture walkthrough: core algorithms, cache pipeline, CUDA kernels, server. |
| `docs/ROADMAP.md` | 59 | What shipped in 0.1.0, what's possible next. |
| `docs/reference/API.md` | 356 | Complete API reference for all public classes and functions. |
| `docs/guides/WORKFLOW.md` | 51 | ExecPlan workflow: Read -> Plan -> Execute -> Test -> Docs -> Commit. |

## docs/research/

| File | Lines | What it does |
|------|-------|-------------|
| `SYNTHESIS.md` | 191 | Strategic decision document (2026-03-25). Cross-repo analysis and phase planning. |
| `TURBOQUANT-DEEP-DIVE.md` | 47 | Paper claims vs empirical findings. Comparison table. |
| `KV-CACHE-LANDSCAPE.md` | 51 | Production engines and their KV cache compression support. |
| `LLAMA-CPP-FEASIBILITY.md` | 540 | Detailed llama.cpp integration feasibility analysis. |
| `LLAMACPP-PORT-DESIGN.md` | 114 | Technical design for TQ4_0 GGML type. |
| `INFERENCE-SERVER-RESEARCH.md` | 425 | Server architecture options (stdlib vs FastAPI vs vLLM). |
| `FLOCKRUN-GPU-ANALYSIS.md` | 55 | Multi-agent GPU VRAM scenarios. |
| `TURBOQUANT-COMBINATIONS.md` | 658 | Feature interaction analysis across TurboQuant variants. |

## docs/plans/archive/

| File | Lines | What it does |
|------|-------|-------------|
| `PLAN-turboquant-kv-compression.md` | 166 | Archived: llama.cpp integration plan. PR #20995 submitted then closed. |
