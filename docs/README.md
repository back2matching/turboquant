# TurboQuant Documentation

> KV cache compression for HuggingFace Transformers. First open-source implementation of Google's TurboQuant (ICLR 2026).

## Docs

| Document | What's in it |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the code works: core algorithm, cache integration, CUDA acceleration, server |
| [ROADMAP.md](ROADMAP.md) | What shipped, what's possible next |
| [reference/API.md](reference/API.md) | Full API reference for all classes and functions |
| [reference/CODEBASE-MAP.md](reference/CODEBASE-MAP.md) | Every file in the repo, what it does, line counts |
| [guides/WORKFLOW.md](guides/WORKFLOW.md) | ExecPlan workflow (Read -> Plan -> Execute -> Test -> Docs -> Commit) |

## Research

Background analysis documents (written pre-implementation, updated with real data where applicable):

| Document | What's in it |
|----------|-------------|
| [research/SYNTHESIS.md](research/SYNTHESIS.md) | Strategic decision document — cross-repo analysis and phase planning |
| [research/TURBOQUANT-DEEP-DIVE.md](research/TURBOQUANT-DEEP-DIVE.md) | Paper claims vs empirical RTX 4080 findings |
| [research/KV-CACHE-LANDSCAPE.md](research/KV-CACHE-LANDSCAPE.md) | Production engines and their KV cache compression support |
| [research/LLAMA-CPP-FEASIBILITY.md](research/LLAMA-CPP-FEASIBILITY.md) | llama.cpp integration feasibility analysis |
| [research/LLAMACPP-PORT-DESIGN.md](research/LLAMACPP-PORT-DESIGN.md) | Technical design for TQ4_0 GGML type |
| [research/INFERENCE-SERVER-RESEARCH.md](research/INFERENCE-SERVER-RESEARCH.md) | Server architecture options (stdlib vs FastAPI vs vLLM) |
| [research/TURBOQUANT-COMBINATIONS.md](research/TURBOQUANT-COMBINATIONS.md) | Feature interaction analysis across TurboQuant variants |
| [research/FLOCKRUN-GPU-ANALYSIS.md](research/FLOCKRUN-GPU-ANALYSIS.md) | Multi-agent GPU VRAM scenarios |

## Quick Links

- **Paper:** [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **PyPI:** [turboquant 0.1.0](https://pypi.org/project/turboquant/)
- **License:** Apache 2.0
- **llama.cpp PR:** [ggml-org/llama.cpp#20995](https://github.com/ggml-org/llama.cpp/pull/20995) (closed)

## At a Glance

```
turboquant/
  core.py       -- TurboQuantMSE + TurboQuantIP algorithms (344 lines)
  cache.py      -- HuggingFace DynamicCache integration (166 lines)
  cuda_accel.py -- Optional CUDA wrapper with fallback (76 lines)
  server.py     -- OpenAI-compatible inference server (263 lines)
cuda/
  turboquant_kernel.cu -- Fused rotation+quantize CUDA kernels (194 lines)
tests/          -- 13 tests (core + cache)
benchmarks/     -- Real model benchmarking (45 data points, 4 models, RTX 4080)
examples/       -- Drop-in usage example
```

Total: ~870 lines of core Python + 270 lines benchmarks + 155 lines tests + 194 lines CUDA.

## Related Projects

| Project | What | Status |
|---------|------|--------|
| [turboquant-vectors](https://github.com/back2matching/turboquant-vectors) | Embedding privacy + compression | Published 0.1.0b1 |
| [kvcache-bench](https://github.com/back2matching/kvcache-bench) | KV cache benchmarking tool | Published |
| [llama.cpp TQ4_0](https://github.com/ggml-org/llama.cpp/pull/20995) | TurboQuant in C/GGML | PR closed (premature) |
