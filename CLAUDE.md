# CLAUDE.md — turboquant

> Operating instructions for Claude Code on this repo.

## What Is This?

KV cache compression for HuggingFace Transformers using TurboQuant (ICLR 2026). Provides a `TurboQuantMSE` class that subclasses HuggingFace's `DynamicCache` for drop-in KV cache compression during inference.

**Status:** Published 0.1.0 on PyPI. Thin package — the more complete implementation lives in FlockRun's `tests/kv-compression/turboquant.py` with CUDA kernels.

## Current State

| Metric | Value |
|--------|-------|
| Version | 0.1.0 (PyPI) |
| Tests | 13 |
| Dependencies | torch, numpy, scipy, transformers |

## What This Repo IS

- PyPI name reservation for `turboquant`
- Minimal KV cache compression implementation
- Reference for the TurboQuant algorithm (rotation + scalar quantization)

## What This Repo IS NOT

- Not the main development repo (that's FlockRun for KV cache work)
- Not actively maintained beyond keeping the PyPI package working
- Not where new TurboQuant features should go

## Related Projects

| Project | What | Repo |
|---------|------|------|
| **turboquant-vectors** | Embedding privacy + compression (ACTIVE) | github.com/back2matching/turboquant-vectors |
| **kvcache-bench** | KV cache benchmarking tool | github.com/back2matching/kvcache-bench |
| **FlockRun** | Parent project, agent runtime | github.com/back2matching/flockrun |

## Commands

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## PyPI

- Account: back2matching
- Package: turboquant 0.1.0
