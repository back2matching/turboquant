# TurboQuant Marketing Strategy

## What It Is

TurboQuant compresses the KV cache in LLM inference from 16 bits to 3-4 bits. `pip install turboquant`, add 3 lines of code, your model uses less VRAM. Works with any HuggingFace transformer model.

## Who It's For

**Primary:** Developers running LLMs on consumer GPUs (RTX 3060-4090, 8-24 GB VRAM) who hit VRAM limits at long contexts.

**Secondary:** ML researchers exploring KV cache compression methods who need a reference implementation of the TurboQuant paper.

**Not for:** Production serving teams (they use vLLM FP8 or Ollama q4_0 with optimized kernels).

## Competitive Position

| Us | Them |
|----|------|
| Only pip-installable sub-8-bit KV compression for HuggingFace | Ollama/vLLM have built-in cache quantization but are locked to their ecosystems |
| First independent 7B consumer GPU benchmarks | Google's paper tested on H100. tonbistudio tested 3B on RTX 3060. |
| v0.2.0: real compressed index storage | Most community implementations still store dequantized values |
| Works with any HF model, zero config | Alternatives require specific frameworks or custom builds |

## Proof Points

- **45 benchmark data points** across 4 models on RTX 4080
- **1,016 MB saved** at 4K context on Qwen2.5-3B (v0.2.0)
- **74% faster** than FP16 at 4K context under memory pressure
- **2 GB saved** at 8K context on Qwen2.5-0.5B
- **15 tests** passing, 926 lines of Python
- Published on PyPI, Apache 2.0 licensed

## Key Messages

1. **"First independent TurboQuant benchmarks at 7B on consumer GPU"** — Nobody else has this data. Google tested on H100, we tested on RTX 4080.

2. **"Under memory pressure, compression makes you faster, not slower"** — At 4K context, FP16 drops to 3.5 tok/s while TQ runs at 6.1 tok/s. The compression frees VRAM for actual computation.

3. **"Three lines of code"** — `from turboquant import TurboQuantCache; cache = TurboQuantCache(bits=4); model(..., past_key_values=cache)`. That's it.

## Target Channels

| Channel | Angle | Timing |
|---------|-------|--------|
| r/LocalLLaMA | Consumer GPU benchmarks, VRAM savings | First — this is the core audience |
| r/MachineLearning | Paper validation, academic rigor | Second — cross-post |
| r/programming | Engineering quality, pip-installable | Third |
| X/Twitter | Thread with benchmark tables | Same day as Reddit |
| HN | "Show HN" with benchmark data | After Reddit momentum |

## Timing

Google's official TurboQuant code is expected Q2 2026. Once that drops, community implementations become secondary. Window is **weeks, not months**. Ship the blog post and Reddit posts ASAP.

## Honest Weaknesses

- Speed is 30-40% slower at short contexts (rotation math overhead)
- Only 1.9x compression currently (uint8 indices, not sub-byte packed). Bitpacking will get to 3-4x.
- Doesn't work with all model architectures (Phi-3.5 cache API incompatible)
- No perplexity benchmarks yet (quality measured by output coherence, not metrics)
- Single developer, no community yet
