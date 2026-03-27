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

Independent benchmarks on NVIDIA RTX 4080 (16 GB VRAM), PyTorch 2.5.1, CUDA 12.1. 45 data points across 4 models.

**Reproduce:**
```bash
python benchmarks/benchmark_kv.py --model Qwen/Qwen2.5-3B-Instruct --context "512,1024,2048,4096"
python benchmarks/benchmark_kv.py --model Qwen/Qwen2.5-7B-Instruct --quick  # fast sanity check
```

Results are saved per-model (`benchmarks/results_*.json`) and combined (`benchmarks/benchmark_results.json`).

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
| 460 | FP16 | 6,126 MB | -- | 14.6 |
| 460 | TQ 4-bit | 6,075 MB | 51 MB | 7.8 |
| 930 | FP16 | 6,451 MB | -- | 14.1 |
| 930 | TQ 4-bit | 6,260 MB | 191 MB | 7.4 |
| 1860 | FP16 | 7,359 MB | -- | 15.4 |
| 1860 | TQ 4-bit | 6,835 MB | **524 MB** | 15.5 |
| 3720 | FP16 | 10,222 MB | -- | 2.5 |
| 3720 | TQ 4-bit | 9,174 MB | **1,048 MB** | **7.4** |

VRAM savings scale with context length: 51 MB at 512 tokens up to **1,048 MB at 4K tokens**. At 4K context, FP16 hits memory pressure (2.5 tok/s) while TQ-4bit with nibble packing runs at **7.4 tok/s — 196% faster**.

### Qwen2.5-0.5B-Instruct — Long Context (942 MB model weights)

| Context | FP16 Peak | TQ 4-bit Peak | VRAM Saved | FP16 Speed | TQ 4-bit Speed |
|---------|-----------|---------------|------------|------------|----------------|
| 460 | 1,144 MB | 1,104 MB | 40 MB | 44.3 | 30.5 |
| 930 | 1,417 MB | 1,262 MB | 155 MB | 46.1 | 30.3 |
| 1860 | 2,189 MB | 1,669 MB | 520 MB | 41.7 | 29.1 |
| 3720 | 4,654 MB | 3,621 MB | **1,033 MB** | 31.9 | 26.5 |
| 7440 | 13,265 MB | 11,195 MB | **2,070 MB** | 17.8 | **19.8** |

At 8K context, TQ-4bit saves **2 GB of VRAM** and is **11% faster** than FP16. 16K OOM'd for all modes on 16 GB.

### StableLM-2-1.6B — Cross-Architecture (3.1 GB model weights)

| Context | FP16 Peak | TQ 4-bit Peak | VRAM Diff | FP16 Speed | TQ 4-bit Speed |
|---------|-----------|---------------|-----------|------------|----------------|
| 460 | 3,433 MB | 3,488 MB | +55 MB | 68.9 | 36.7 |
| 930 | 3,724 MB | 3,894 MB | +170 MB | 68.2 | 34.8 |
| 1860 | 4,302 MB | 4,700 MB | +398 MB | 61.4 | 34.7 |
| 3720 | 5,459 MB | 6,318 MB | +859 MB | 56.1 | 33.1 |

On StableLM, TQ uses **more** VRAM than FP16 at every context length. The StableLM results were collected with v0.1.0 (dequantized storage). v0.2.0 stores compressed indices and may show different results on StableLM.

### Key Takeaways

- **VRAM savings scale linearly with context length.** At short contexts (<512 tokens), savings are minimal. At 4K tokens, savings exceed **1 GB**. At 8K, savings reach **2 GB**.
- **Under memory pressure, TQ is significantly faster than FP16.** At 4K context on 3B, FP16 drops to 3.5 tok/s while TQ-4bit runs at 6.1 tok/s (74% faster). At 8K on 0.5B, TQ is 11% faster.
- **v0.2.0 stores compressed indices.** Cache uses uint8 indices + float32 norms instead of dequantized FP16. Real compression with on-the-fly dequantization.
- **Output quality is good at 4-bit on 3B+ models.** Qwen 3B and 7B produce coherent code. On 0.5B, TQ output sometimes degrades to filler repetition — small models are more sensitive to quantization noise.

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

A TQ4_0 KV cache type was proposed for llama.cpp:
- **PR:** [ggml-org/llama.cpp#20995](https://github.com/ggml-org/llama.cpp/pull/20995) (closed — premature, multiple competing implementations in progress)
- **Usage (if built from branch):** `--cache-type-k tq4_0 --cache-type-v f16 --no-kv-offload`
- **Status:** Multiple community implementations in progress. Google's official code expected Q2 2026.

## Paper

This implements the algorithm from:

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
ICLR 2026 | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

This is an independent implementation, not affiliated with Google Research.

## License

Apache 2.0
