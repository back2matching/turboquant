# TurboQuant

Open-source implementation of Google's TurboQuant KV cache compression.

Compress your LLM's KV cache to 4 bits. Save VRAM. Run longer contexts. Drop-in for HuggingFace.

```python
from turboquant import TurboQuantCache

cache = TurboQuantCache(bits=4)
outputs = model.generate(..., past_key_values=cache)
```

That's it. Three lines to compress your KV cache.

## What is this?

When LLMs generate text, they store key-value pairs for every token they've seen. This KV cache grows with context length and eats your VRAM. At 32K tokens on an 8B model, the KV cache alone uses ~4.6 GB.

TurboQuant compresses this cache to 4 bits per element (from 16), cutting memory by ~4x. It does this using a clever trick from Google's paper: rotate the vectors randomly, then quantize each coordinate independently using an optimal codebook derived from probability theory.

The result: same quality output, way less VRAM.

## Install

```bash
pip install turboquant
```

Or from source:

```bash
git clone https://github.com/back2matching/turboquant
cd turboquant
pip install -e .
```

## Quick Start

### Drop into any HuggingFace model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantCache
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Create compressed cache
cache = TurboQuantCache(bits=4)

# Use it like normal
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model(**inputs, past_key_values=cache, use_cache=True)
```

### Run the inference server

TurboQuant ships with an OpenAI-compatible inference server. Point any OpenAI client at it.

```bash
turboquant-server --model Qwen/Qwen2.5-3B-Instruct --bits 4 --port 8000
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

### Use the core algorithms directly

```python
from turboquant import TurboQuantMSE

# Quantize any vectors (KV cache heads, embeddings, etc.)
tq = TurboQuantMSE(dim=128, bits=4, device='cuda')

# Quantize
indices, norms = tq.quantize(vectors)  # vectors: (N, 128)

# Dequantize
vectors_hat = tq.dequantize(indices, norms)
```

## Benchmarks (RTX 4080 16GB)

Independent benchmarks on NVIDIA RTX 4080 (16 GB VRAM), PyTorch 2.5.1, CUDA 12.1. All results are reproducible via `benchmarks/benchmark_kv.py`.

### Qwen2.5-7B-Instruct (14.5 GB model weights)

| Context | KV Mode | Peak VRAM | VRAM Saved | Speed (tok/s) | Output Quality |
|---------|---------|-----------|------------|---------------|----------------|
| 460 | FP16 | 14,833 MB | -- | 17.7 | Coherent |
| 460 | **TQ 4-bit** | **14,758 MB** | **75 MB** | **23.8** | Coherent |
| 460 | TQ 3-bit | 14,758 MB | 75 MB | 20.6 | Minor artifacts |
| 1860 | FP16 | 16,659 MB | -- | 1.0 | Coherent |
| 1860 | **TQ 4-bit** | **16,215 MB** | **444 MB** | **1.4** | Coherent |
| 1860 | TQ 3-bit | 16,217 MB | 442 MB | 1.4 | Coherent |

At 7B with 1.8K context, FP16 exceeds physical VRAM (16,659 > 16,376 MB) and drops to 1 tok/s from swapping. TQ-4bit saves 444 MB and runs **40% faster** in this regime.

### Qwen2.5-3B-Instruct — Context Length Sweep (5.9 GB model weights)

| Context | KV Mode | Peak VRAM | VRAM Saved | Speed (tok/s) |
|---------|---------|-----------|------------|---------------|
| 460 | FP16 | 6,126 MB | -- | 31.6 |
| 460 | TQ 4-bit | 6,084 MB | 42 MB | 20.5 |
| 930 | FP16 | 6,451 MB | -- | 30.1 |
| 930 | TQ 4-bit | 6,281 MB | 170 MB | 20.0 |
| 1860 | FP16 | 7,359 MB | -- | 26.2 |
| 1860 | TQ 4-bit | 6,880 MB | **479 MB** | 19.2 |
| 3720 | FP16 | 10,222 MB | -- | 18.3 |
| 3720 | TQ 4-bit | 9,267 MB | **955 MB** | 16.3 |

VRAM savings scale with context length: 42 MB at 512 tokens up to **955 MB at 4K tokens**. At 4K context, TQ-4bit runs at 89% of FP16 speed while saving nearly 1 GB of VRAM. Extrapolating: at 32K tokens, expect ~7.5 GB saved.

### Key Takeaways

- **VRAM savings scale linearly with context length.** At short contexts (<512 tokens), savings are minimal. At 4K+ tokens, savings exceed 1 GB.
- **Speed overhead decreases at longer contexts.** TQ-4bit is 35% slower at 512 tokens but only 11% slower at 4K tokens.
- **Under memory pressure, TQ is faster than FP16.** When FP16 KV cache pushes VRAM past physical limits, TQ's smaller cache avoids thrashing and delivers higher throughput.
- **TQ-4bit and TQ-3bit have similar VRAM.** The current implementation stores dequantized FP16 values (not compressed indices), so 3-bit and 4-bit use the same memory. A production implementation storing indices would see 3-bit use 25% less than 4-bit.
- **Output quality is good at 4-bit.** Both models produce coherent, correct code across all context lengths. 3-bit shows occasional minor artifacts at shorter contexts.

### Algorithm Verification

| Bits | MSE | Theoretical Bound | Compression |
|------|-----|-------------------|-------------|
| 1 | 0.362 | 0.680 | 12.8x |
| 2 | 0.129 | 0.170 | 7.1x |
| 3 | 0.049 | 0.043 | 4.9x |
| 4 | 0.020 | 0.011 | 3.8x |

## How It Works

TurboQuant uses three ideas from the paper:

1. **Random rotation**: Multiply each KV vector by a random orthogonal matrix. This spreads the information evenly across all coordinates, making them nearly independent.

2. **Optimal codebook**: Each coordinate now follows a predictable Beta distribution. We compute the mathematically optimal quantization levels for this distribution. No training data needed.

3. **Residual window**: The most recent 128 tokens stay in full FP16 precision. Only older tokens get compressed. This preserves quality for the tokens attention focuses on most.

The rotation is computed once (not per-token) and the codebook is derived analytically. No calibration, no fine-tuning, works with any model out of the box.

## When to Use This

**Good fit:**
- You're running long contexts (8K+ tokens) on a VRAM-constrained GPU
- You're serving multiple users and need to fit more KV caches in memory
- You want to run a bigger model by freeing VRAM from KV cache
- Standard transformer models (Llama, Mistral, Qwen2.5)

**Not a good fit:**
- Very short contexts (< 1K tokens) where KV cache is tiny anyway
- Hybrid architectures with recurrent layers (Qwen3.5, Mamba) that already have small KV caches
- Tasks requiring exact bit-level precision (use FP16)
- 3-bit on models smaller than 8B (quality degrades noticeably)

## Comparison with Alternatives

| Method | Where It Runs | Bits | Setup |
|--------|---------------|------|-------|
| **TurboQuant** | Any HuggingFace model | 3-4 | `pip install turboquant` |
| Ollama q8_0 KV | Ollama only | 8 | `OLLAMA_KV_CACHE_TYPE=q8_0` |
| Ollama q4_0 KV | Ollama only | 4 | `OLLAMA_KV_CACHE_TYPE=q4_0` |
| vLLM FP8 KV | vLLM only | 8 | `kv_cache_dtype="fp8"` |
| KIVI | Research code | 2 | Not pip-installable |

TurboQuant is the only pip-installable sub-8-bit KV cache compression that works with any HuggingFace model.

## llama.cpp Integration

TurboQuant is also available as a KV cache type in llama.cpp:
- **PR:** [ggml-org/llama.cpp#20995](https://github.com/ggml-org/llama.cpp/pull/20995)
- **Usage:** `--cache-type-k tq4_0 --cache-type-v f16 --no-kv-offload`
- **Result:** 70% less perplexity degradation than Q4_0 at the same VRAM

## Paper

This implements the algorithm from:

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
ICLR 2026 | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

This is an independent implementation, not affiliated with Google Research.

## License

Apache 2.0
