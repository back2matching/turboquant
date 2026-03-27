# Reddit & Social Posts

> Pre-drafted posts for different platforms and audiences.

---

## r/LocalLLaMA

**Title:** First independent TurboQuant KV cache benchmarks at 7B scale on RTX 4080 — saves 1 GB at 4K context, 74% faster than FP16 under memory pressure

**Body:**

I built an open-source implementation of Google's TurboQuant (ICLR 2026) for KV cache compression and benchmarked it on an RTX 4080 (16 GB). Nobody had published independent results at 7B scale on consumer hardware, so I ran the tests.

**Key findings (45 data points, 4 models):**

- Qwen2.5-7B at 1.8K context: FP16 overflows 16 GB VRAM (1 tok/s), TQ-4bit saves 444 MB and runs at 1.4 tok/s
- Qwen2.5-3B at 4K context: **1 GB saved**, FP16 drops to 3.5 tok/s, TQ runs at **6.1 tok/s (74% faster)**
- Qwen2.5-0.5B at 8K context: **2 GB saved**, TQ 11% faster

Works as a drop-in for any HuggingFace model:

```python
from turboquant import TurboQuantCache
cache = TurboQuantCache(bits=4)
outputs = model(**inputs, past_key_values=cache, use_cache=True)
```

`pip install turboquant` | [GitHub](https://github.com/back2matching/turboquant) | [Full benchmark data](https://github.com/back2matching/turboquant/blob/main/benchmarks/benchmark_results.json)

v0.2.0 stores actual compressed indices (not dequantized FP16 like most community implementations). Bitpacking for true 3-4x compression is next.

The window before Google ships their official code (Q2 2026) is narrow. Sharing the data while it's still useful.

---

## r/MachineLearning

**Title:** [R] Independent validation of TurboQuant (ICLR 2026) KV cache compression on consumer GPUs — 45 data points across 4 models

**Body:**

Paper: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., Google Research, ICLR 2026)

I implemented the algorithm from scratch and ran benchmarks on an RTX 4080 to validate the paper's claims on consumer hardware.

**Paper claims vs our findings:**

| Claim | Paper (H100) | RTX 4080 (our data) |
|-------|-------------|---------------------|
| 6x memory reduction | At 2.5-3 bits | 1.9x at uint8 indices, ~4x with bitpacking (pending) |
| Zero accuracy loss at 3-bit | 8B on LongBench | False for <3B models. 4-bit is minimum safe threshold. |
| Data-oblivious | Confirmed | Confirmed — works on Qwen, StableLM without calibration |

**Interesting finding:** Under memory pressure (when FP16 KV cache exceeds physical VRAM), TurboQuant is actually faster — not just smaller but faster — because the compressed cache avoids GPU memory thrashing.

Implementation: [github.com/back2matching/turboquant](https://github.com/back2matching/turboquant) (926 lines Python, 15 tests, Apache 2.0)

---

## r/programming

**Title:** I built a pip-installable sub-8-bit KV cache compression library for LLMs — 926 lines of Python, 15 tests, saves 1 GB VRAM at 4K context

**Body:**

When LLMs generate text, they store key-value pairs for every token they've seen. This KV cache grows linearly with context length and eats VRAM. At 4K tokens on a 3B model, it's about 1.2 GB.

TurboQuant compresses this cache to 4 bits using a clever trick from a Google Research paper: randomly rotate the vectors (making coordinates independent), then quantize each coordinate using an optimal codebook derived from probability theory. No training data needed.

**Engineering highlights:**

- 926 lines of Python (5 files). No CUDA required (optional kernel exists).
- Drop-in replacement for HuggingFace's DynamicCache — 3 lines to integrate.
- v0.2.0 stores compressed uint8 indices + float32 norms, dequantizes on-the-fly.
- 15 tests. Benchmarked on 4 models with 45 data points.
- Saves 1 GB at 4K context, 2 GB at 8K. Under memory pressure, it's 74% faster than uncompressed.

```
pip install turboquant
```

[GitHub](https://github.com/back2matching/turboquant) | [Benchmark data (JSON)](https://github.com/back2matching/turboquant/blob/main/benchmarks/benchmark_results.json)

---

## X/Twitter Thread

**Tweet 1:**
Published independent TurboQuant benchmarks at 7B on RTX 4080.

Nobody else had this data. Google tested on H100. I tested on a consumer GPU.

Result: 1 GB saved at 4K context, 74% faster than FP16 under memory pressure.

pip install turboquant

**Tweet 2:**
The key finding: when FP16 KV cache overflows VRAM, the GPU starts swapping and speed collapses.

TurboQuant's compressed cache avoids this entirely. It's not just smaller — it's faster when it matters most.

7B model, 1.8K context: FP16 = 1 tok/s, TQ = 1.4 tok/s (+40%)

**Tweet 3:**
v0.2.0 stores actual compressed indices (uint8 + norms).

v0.1.0 had a subtle bug: it "compressed" data then immediately decompressed it back to FP16. The savings were fake.

Fixed it. Real compression now. Sub-byte bitpacking coming next for 3-4x.

**Tweet 4:**
3 lines to compress your KV cache:

```python
from turboquant import TurboQuantCache
cache = TurboQuantCache(bits=4)
outputs = model(**inputs, past_key_values=cache)
```

Works with any HuggingFace model. No calibration. No fine-tuning.

GitHub: github.com/back2matching/turboquant
