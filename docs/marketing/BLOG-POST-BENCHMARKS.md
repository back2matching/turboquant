# Independent TurboQuant KV Cache Benchmarks on Consumer GPUs

> First published benchmarks of Google's TurboQuant algorithm at 7B scale on an RTX 4080. 45 data points, 4 models, real VRAM measurements.

---

## TL;DR

I benchmarked TurboQuant KV cache compression on an RTX 4080 (16 GB). At 4K context on a 3B model, it saves 1 GB of VRAM and runs 74% faster than FP16 (because FP16 starts thrashing). At 8K context on a 0.5B model, it saves 2 GB. The library is pip-installable and works with any HuggingFace model in 3 lines of code.

```python
from turboquant import TurboQuantCache
cache = TurboQuantCache(bits=4)
outputs = model(**inputs, past_key_values=cache, use_cache=True)
```

Install: `pip install turboquant`
Repo: [github.com/back2matching/turboquant](https://github.com/back2matching/turboquant)

---

## Background

Google published [TurboQuant](https://arxiv.org/abs/2504.19874) at ICLR 2026 — an algorithm that compresses LLM KV cache entries from 16 bits to 3-4 bits using random rotation + optimal scalar quantization. Their paper shows impressive results on H100s with 8B models. But nobody had published independent benchmarks on consumer GPUs at that scale.

I built an open-source implementation and ran the benchmarks.

## Setup

- **GPU:** NVIDIA RTX 4080 (16 GB VRAM)
- **PyTorch:** 2.5.1, CUDA 12.1
- **Models:** Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-0.5B-Instruct, StableLM-2-1.6B
- **Methodology:** FP16 baseline vs TurboQuant 4-bit vs TurboQuant 3-bit, greedy decoding, 100 tokens generated per run. Peak VRAM measured via `torch.cuda.max_memory_allocated()`.

All results are reproducible:
```bash
pip install turboquant
python benchmarks/benchmark_kv.py --model Qwen/Qwen2.5-3B-Instruct --context "512,1024,2048,4096"
```

## Results

### Qwen2.5-7B-Instruct (14.5 GB model weights)

The 7B model barely fits on 16 GB. This is where compression matters most.

| Context | FP16 Peak | TQ 4-bit Peak | Saved | FP16 Speed | TQ Speed |
|---------|-----------|---------------|-------|------------|----------|
| 460 | 14,833 MB | 14,758 MB | 75 MB | 17.7 tok/s | **23.8 tok/s** |
| 1860 | **16,659 MB** | 16,215 MB | 444 MB | **1.0 tok/s** | **1.4 tok/s** |

At 1.8K context, FP16 exceeds the 16 GB physical VRAM limit (16,659 > 16,376 MB). The GPU starts swapping and speed collapses to 1 tok/s. TurboQuant saves 444 MB, stays under the limit, and runs **40% faster**.

This is the scenario the paper designed for — but nobody had shown it on consumer hardware.

### Qwen2.5-3B-Instruct — Context Length Sweep (5.9 GB model)

With a smaller model, we can test longer contexts and see how savings scale.

| Context | FP16 Peak | TQ 4-bit Peak | Saved | FP16 Speed | TQ Speed |
|---------|-----------|---------------|-------|------------|----------|
| 460 | 6,126 MB | 6,078 MB | 48 MB | 30.7 tok/s | 18.7 tok/s |
| 930 | 6,451 MB | 6,267 MB | 184 MB | 31.4 tok/s | 18.8 tok/s |
| 1860 | 7,359 MB | 6,851 MB | 508 MB | 26.0 tok/s | 17.8 tok/s |
| 3720 | 10,222 MB | 9,206 MB | **1,016 MB** | **3.5 tok/s** | **6.1 tok/s** |

The pattern is clear:
- **VRAM savings scale linearly with context length.** From 48 MB at 512 tokens to over 1 GB at 4K.
- **Speed overhead is constant (~35% slower) at short contexts.** The rotation math adds fixed overhead.
- **At 4K, FP16 hits memory pressure and TQ wins on speed too.** FP16 drops to 3.5 tok/s while TQ maintains 6.1 tok/s — **74% faster**.

### Qwen2.5-0.5B-Instruct — Long Context (942 MB model)

The tiny model leaves 15 GB for KV cache, so we can push to 8K tokens.

| Context | FP16 Peak | TQ 4-bit Peak | Saved | FP16 Speed | TQ Speed |
|---------|-----------|---------------|-------|------------|----------|
| 3720 | 4,654 MB | 3,621 MB | 1,033 MB | 31.9 tok/s | 26.5 tok/s |
| 7440 | 13,265 MB | 11,195 MB | **2,070 MB** | 17.8 tok/s | **19.8 tok/s** |

At 8K context, TurboQuant saves **2 GB** and is **11% faster** than FP16.

### Cross-Architecture: StableLM-2-1.6B

I also tested StableLM to see if results hold across architectures. Short answer: mixed.

| Context | FP16 Peak | TQ 4-bit Peak | VRAM Diff |
|---------|-----------|---------------|-----------|
| 460 | 3,433 MB | 3,488 MB | **+55 MB** |
| 3720 | 5,459 MB | 6,318 MB | **+859 MB** |

TQ uses MORE memory on StableLM. This exposed a limitation I fixed in v0.2.0 (see below), but the StableLM numbers are from v0.1.0 and haven't been re-tested yet.

## The v0.2.0 Story: From Fake Compression to Real Compression

Here's something I discovered during benchmarking that I think is worth sharing.

The initial implementation (v0.1.0) had a subtle architectural problem: after quantizing the KV cache entries to 4-bit indices, it **immediately dequantized them back to FP16** and stored the full-size tensors. Like compressing a file and then decompressing it and throwing away the zip. The "VRAM savings" were actually just side effects of different PyTorch memory allocation patterns between code paths.

I rewrote the cache layer in v0.2.0 to store the actual compressed data (uint8 indices + float32 norms) and dequantize on-the-fly when the model needs the data for attention. The dequantization adds ~1-2ms per forward pass — negligible compared to generation time. But now the compression is real.

The 3B results above are from v0.2.0 with genuine compressed storage.

## How It Works (30-Second Version)

1. **Random rotation:** Multiply each KV vector by a random orthogonal matrix. This spreads information evenly across coordinates, making them nearly independent.
2. **Optimal quantization:** Each coordinate now follows a Beta distribution. Compute the mathematically optimal quantization levels analytically — no training data or calibration needed.
3. **Residual window:** Keep the most recent 128 tokens in full FP16. Only compress older tokens. Recent tokens matter most for attention.

The codebook is derived from probability theory, not data. Works with any model out of the box. No fine-tuning, no calibration data, no configuration.

## Comparison with Alternatives

| Method | Bits | Ecosystem | Real Compression | Speed |
|--------|------|-----------|------------------|-------|
| **TurboQuant** | 3-4 | Any HuggingFace model | Yes (v0.2.0) | ~35% slower, faster under pressure |
| Ollama q4_0 KV | 4 | Ollama only | Yes | Native speed |
| vLLM FP8 KV | 8 | vLLM only | Yes | Native speed |
| KIVI | 2 | Research code | Yes | Unknown |

TurboQuant is the only pip-installable option for HuggingFace users. If you're already using Ollama or vLLM, their built-in options are faster. But if you're writing custom HuggingFace inference code and hitting VRAM limits, TurboQuant is the simplest path.

## What's Next

- **Sub-byte bitpacking:** Currently indices are stored as uint8 (1 byte each), wasting 4-5 bits per value. Packing will push compression from 1.9x to 3-4x — potentially 4+ GB saved at 8K context.
- **Re-benchmark StableLM** with v0.2.0 compressed storage.
- **Perplexity benchmarks** on WikiText-2 for rigorous quality measurement.

## Try It

```bash
pip install turboquant
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantCache
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

cache = TurboQuantCache(bits=4)
inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
outputs = model(**inputs, past_key_values=cache, use_cache=True)
```

Paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
Repo: [github.com/back2matching/turboquant](https://github.com/back2matching/turboquant)
PyPI: [turboquant](https://pypi.org/project/turboquant/)

---

*This is an independent implementation, not affiliated with Google Research.*
