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

### Qwen2.5-3B-Instruct

| KV Mode | Peak VRAM | VRAM Saved | Speed | Output Quality |
|---------|-----------|------------|-------|---------------|
| FP16 (baseline) | 6,922 MB | -- | 28 tok/s | Perfect |
| **TurboQuant 4-bit** | **6,448 MB** | **474 MB** | 17 tok/s | Good |
| TurboQuant 3-bit | 6,448 MB | 474 MB | 20 tok/s | Degraded on small models |

VRAM savings scale linearly with context length. At 32K tokens on an 8B model, expect ~3 GB saved.

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
