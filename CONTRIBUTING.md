# Contributing to turboquant

Thanks for considering a contribution. turboquant is a small project maintained primarily by one person; the sections below set expectations so your time isn't wasted.

## Response SLA

**First response within 7 calendar days** for issues and PRs. If you do not hear back in that window, please leave a comment — things do fall through the cracks.

PRs with requested changes that sit unaddressed for 30 days will be labeled `stale`. After 14 more days, `stale` PRs are closed with a specific reason (out of scope / superseded / insufficient test coverage / inactive contributor). Closed PRs can always be reopened if the contributor returns.

## How PRs get merged

Every PR gets an area label on first triage:

- `area:core` — `turboquant/core.py` (algorithm)
- `area:cache` — `turboquant/cache.py` (HuggingFace integration)
- `area:cuda` — `cuda/*.cu`, `turboquant/cuda_accel.py`
- `area:server` — `turboquant/server.py`
- `area:eval` — `turboquant/eval.py`, `benchmarks/`
- `area:docs` — README, ARCHITECTURE, reference/

**Algorithmic changes** (`area:core`, `area:cache`) require measured evidence before merge: before/after on ≥ 1 real model + ≥ 1 metric (PPL via `turboquant-eval`, LongBench in the future, KV VRAM, tok/s). Micro-benchmarks on random tensors don't count.

**Kernel changes** (`area:cuda`) must preserve PyTorch-fallback parity (bit-identical or documented-tolerance-equivalent) and must build on CUDA 12.0 / 12.1 / 12.4.

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Running benchmarks

```bash
# KV VRAM + tok/s sweep
python benchmarks/benchmark_kv.py --model Qwen/Qwen2.5-0.5B-Instruct --quick

# Perplexity (new in v0.3.1)
turboquant-eval --model Qwen/Qwen2.5-0.5B-Instruct --bits 4 --dataset wikitext-2
```

## CODEOWNERS

A `.github/CODEOWNERS` file auto-assigns the primary maintainer as reviewer for everything. Contributors who maintain a subsystem for 3+ merged PRs are invited to add themselves.

## Filing a good issue

Include: `turboquant --version`, `python --version`, `torch --version`, GPU model, the exact command you ran, the full traceback. For performance issues, a reproduction script + the benchmark command you used to measure.

## License

By contributing, you agree your contributions will be licensed under Apache 2.0.
