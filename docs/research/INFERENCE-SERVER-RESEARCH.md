# TurboQuant Inference Server Research

> Research date: 2026-03-24
> Goal: Fastest path to an inference server with TurboQuant KV cache compression, OpenAI-compatible API, running Qwen3.5-9B on RTX 4080 16GB.

## Table of Contents

1. [TurboQuant: What It Is and What Code Exists](#1-turboquant)
2. [QJL CUDA Kernels](#2-qjl-cuda-kernels)
3. [HuggingFace DynamicCache Subclassing](#3-dynamiccache-subclassing)
4. [Inference Server Options](#4-inference-server-options)
5. [Can Existing Engines Use Custom KV Cache?](#5-existing-engines)
6. [VRAM Budget for Qwen3.5-9B](#6-vram-budget)
7. [Comparison: TurboQuant vs Ollama q4_0 KV](#7-turboquant-vs-ollama)
8. [Fastest Path Forward](#8-fastest-path)
9. [Effort Estimates and Risks](#9-effort-and-risks)

---

## 1. TurboQuant

### What It Is

TurboQuant (ICLR 2026, Google Research) is a two-stage KV cache compression algorithm:

1. **PolarQuant step**: Randomly rotate data vectors using a random orthogonal matrix. This "concentrates" the coordinate distribution into a well-behaved Beta distribution, making each coordinate nearly independent. Then apply an optimal scalar quantizer per coordinate.
2. **QJL residual step**: Use 1 extra bit per element via the Quantized Johnson-Lindenstrauss (QJL) transform to correct the small remaining bias from step 1. This acts as a mathematical error-checker.

Result: 3-bit KV cache with zero accuracy loss on benchmarks (needle-in-a-haystack, downstream NLP). 5x+ memory reduction. Up to 8x attention speedup on H100 (4-bit variant).

### Authors

Praneeth Kacham (Google), Insu Han (KAIST), Majid Daliri (NYU), Lars Gottesbueren (Google), Rajesh Jayaram (Google).

### Code Availability

**No official public repository exists.** The paper is at [arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874) and the [ICLR 2026 proceedings](https://openreview.net/pdf/6593f484501e295cdbe7efcbc46d7f20fc7e741f.pdf). The Google Research blog post mentions the implementation "builds upon publicly available code with modifications to the dequantization module" but no repo link is given. The `github.com/turboquant` user is unrelated (a finance project).

**The "ygivenx/turboquant-a100" repo does not appear to exist.** Searched multiple times with different queries. Either it was removed, is private, or never existed publicly.

### What Would Need to Be Implemented

To build TurboQuant from scratch, you need:
- Random orthogonal rotation matrix generation (standard PyTorch: `torch.linalg.qr` on random Gaussian)
- Optimal scalar quantizer per coordinate (lookup table from Beta distribution CDF)
- QJL 1-bit residual sketch (random Gaussian projection + sign quantization)
- Dequantization: inverse rotation + QJL correction during attention computation
- CUDA kernels for the fused rotation + quantize on write, and dequantize + attention on read

---

## 2. QJL CUDA Kernels

The QJL component has a public repo: [github.com/amirzandieh/QJL](https://github.com/amirzandieh/QJL)

### What It Contains

- CUDA kernels for the QJL sketch (random Gaussian projection quantized to sign bit)
- Inner product estimator that works directly on the 1-bit quantized representation
- Tested on Llama-2 and Llama-3 across multiple NLP tasks
- Achieves 3-bit KV cache with 5x+ memory reduction

### Key Technical Details

- The JL transform is a random Gaussian projection. QJL quantizes the projected vectors to a single sign bit.
- Provides an **unbiased estimator** for inner products with minimal distortion.
- No need to store quantization constants (unlike per-channel or per-token quantization methods). This is the "zero overhead" claim.
- The CUDA kernels are lightweight. They handle the projection + sign quantization on write, and the approximate inner product on read.

### Relevance to TurboQuant

TurboQuant uses QJL as its second stage (1 bit for error correction on top of PolarQuant's 2-3 bit scalar quantization). The QJL repo provides the CUDA kernels for that second stage. The first stage (PolarQuant / random rotation + scalar quantization) would need to be implemented separately.

---

## 3. DynamicCache Subclassing

HuggingFace Transformers provides a clean cache abstraction in `transformers/cache_utils.py`.

### How It Works

- `DynamicCache` is the default. It stores `key_states` and `value_states` as lists of tensors, one per layer.
- The `update(key_states, value_states, layer_idx, cache_kwargs)` method is called by each attention layer.
- To add compression: subclass `DynamicCache`, override `update()` to quantize on write, and provide a custom `__getitem__` or attention-time dequantization.

### Existing Quantized Cache Examples

- `QuantoQuantizedCache`: Drop-in replacement using the Quanto library. Quantizes to INT4/INT8.
- `HQQQuantizedCache`: Uses Half-Quadratic Quantization.
- Both maintain a "residual buffer" at full precision for the most recent N tokens, and quantize older tokens.

### Pattern for Custom Implementation

```python
class TurboQuantCache(DynamicCache):
    def __init__(self, rotation_matrix, qjl_projection, ...):
        super().__init__()
        self.rotation_matrix = rotation_matrix  # Per-head random orthogonal
        self.qjl_projection = qjl_projection    # Random Gaussian for residual
        self.quantized_keys = []    # 3-bit quantized storage
        self.quantized_values = []  # 3-bit quantized storage

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # 1. Rotate: key_rotated = key_states @ self.rotation_matrix
        # 2. Scalar quantize to 2-3 bits (optimal quantizer from Beta CDF)
        # 3. Compute residual, apply QJL 1-bit sketch
        # 4. Store compressed representation
        # Return: (key_for_attention, value_for_attention)
        ...
```

### Complication: Qwen3.5 Hybrid Architecture

Qwen3.5-9B uses **Gated DeltaNet** (linear attention) in 75% of layers and standard attention in 25%. The linear attention layers maintain a fixed-size state matrix, NOT a growing KV cache. Only the 8 standard attention layers (out of 32) have a traditional KV cache. This means TurboQuant compression would only apply to those 8 layers.

---

## 4. Inference Server Options

### Option A: Custom FastAPI + HuggingFace Transformers (RECOMMENDED PATH)

**Architecture:**
- FastAPI server exposing `/v1/chat/completions` (OpenAI-compatible)
- Load model via `transformers.AutoModelForCausalLM`
- Pass custom `TurboQuantCache` as `past_key_values` to `model.generate()`
- SSE streaming for token-by-token output

**Pros:**
- Full control over cache implementation
- Can use `torch.compile()` for performance (like gpt-fast / Cold Compress)
- Simplest integration path. No engine internals to hack.
- Many tutorials and reference implementations exist

**Cons:**
- No PagedAttention, no continuous batching (single-request at a time)
- No speculative decoding without custom work
- Performance ceiling lower than vLLM/SGLang for concurrent requests
- Must implement chat template, tokenization, stopping criteria manually

**Reference projects:**
- [AnswerDotAI/cold-compress](https://github.com/AnswerDotAI/cold-compress) - Built on gpt-fast, hackable KV cache compression toolkit
- [ritun16/openai-compatible-fastapi](https://github.com/ritun16/openai-compatible-fastapi) - OpenAI API wrapper with FastAPI
- [Nayjest/lm-proxy](https://github.com/Nayjest/lm-proxy) - OpenAI-compatible proxy with local PyTorch inference

### Option B: Cold Compress (gpt-fast based)

**Architecture:**
- Built on PyTorch's gpt-fast (torch.compile optimized)
- All cache logic in a `KVCache` base class. Custom methods only need an eviction policy.
- Uses torch.compile for GPU efficiency without custom CUDA kernels

**Pros:**
- Purpose-built for KV cache compression research
- Already has heavy_hitter, streaming_llm, and other methods
- torch.compile makes pure-PyTorch code nearly as fast as custom CUDA
- Fixed cache at 4096 tokens maintains 70+ tok/s at 64k context

**Cons:**
- No OpenAI-compatible API (would need to add one)
- gpt-fast model loading is different from HuggingFace (needs conversion)
- May not support Qwen3.5's hybrid DeltaNet architecture out of the box
- Project last updated 2024, may need updates for recent transformers

### Option C: vLLM with Custom Quantization

**Feasibility: LOW**

- vLLM supports FP8 KV cache natively and has a `@register_quantization_config` decorator for custom weight quantization
- BUT: KV cache quantization is deeply integrated into the PagedAttention CUDA kernels. Adding a new KV cache dtype (3-bit TurboQuant) would require modifying the attention backend, the block manager, and the CUDA kernels
- The vLLM forum thread on "Custom KV cache implementation" confirms this is non-trivial
- Would essentially be forking vLLM. Months of work.

### Option D: SGLang with Custom Backend

**Feasibility: LOW**

- SGLang has HiCache (hierarchical caching) and RadixAttention, both sophisticated but rigid
- Custom storage backends (3FS, Mooncake, NIXL) are for LOCATION of cache, not FORMAT
- Quantized KV cache support exists but is limited to FP8
- Adding TurboQuant would require deep engine changes similar to vLLM

### Option E: Aphrodite Engine

**Feasibility: MEDIUM-LOW**

- vLLM fork with better quantization support (FP8 + custom INT8 with calibration)
- More amenable to custom KV cache quantization than upstream vLLM
- But still uses PagedAttention. 3-bit TurboQuant would need kernel modifications.
- Community-driven (PygmalionAI). Smaller team, less documentation.

### Option F: LiteLLM Proxy

**Feasibility: COMPLEMENT ONLY**

- LiteLLM is a proxy/gateway, NOT an inference engine
- Can sit in front of any OpenAI-compatible backend and add routing, rate limiting, cost tracking
- Useful as a layer on top of our custom server, but does not solve the inference problem itself

### Option G: TGI

**Feasibility: DEAD END**

- TGI entered maintenance mode December 11, 2025
- Only accepting bug fixes. No new features.
- Custom KV cache compression would be a major feature addition. Will not be accepted.

---

## 5. Can Existing Engines Use Custom KV Cache?

| Engine | Custom KV Cache Feasible? | Effort | Notes |
|--------|--------------------------|--------|-------|
| HuggingFace Transformers | **Yes** | Low | DynamicCache subclass. Clean API. |
| Cold Compress (gpt-fast) | **Yes** | Low-Medium | Purpose-built for this. Needs API wrapper. |
| vLLM | No (practical) | Very High | PagedAttention kernels are tightly coupled |
| SGLang | No (practical) | Very High | RadixAttention + HiCache are rigid |
| Aphrodite | Maybe | High | Better than vLLM but still kernel-level work |
| TGI | No | N/A | Maintenance mode |
| Ollama/llama.cpp | No | Very High | C++ codebase, GGML format, different paradigm |

**Verdict: HuggingFace Transformers is the only realistic path for custom KV cache compression without forking an engine.**

---

## 6. VRAM Budget for Qwen3.5-9B on RTX 4080 16GB

### Model Weights

- Qwen3.5-9B has 9.65B parameters
- FP16/BF16: 9.65B x 2 bytes = **~19.3 GB** -- DOES NOT FIT in 16GB
- INT8 (bitsandbytes): 9.65B x 1 byte = **~9.65 GB** -- fits with ~6GB headroom
- INT4 (GPTQ/AWQ): 9.65B x 0.5 bytes = **~4.8 GB** -- fits with ~11GB headroom

### KV Cache (Standard Attention Layers Only)

Qwen3.5-9B has only 8 standard attention layers (the other 24 use Gated DeltaNet with fixed-size state). For those 8 layers:

- Standard FP16 KV cache at 32K context: ~1.5-2 GB (much less than a pure transformer)
- With TurboQuant 3-bit: ~0.3-0.4 GB (5x reduction)
- Gated DeltaNet layers: ~12 MB fixed (negligible)

### VRAM Configurations

| Weights | KV Cache | Activations/Overhead | Total | Fits 16GB? |
|---------|----------|---------------------|-------|------------|
| FP16 (19.3 GB) | Any | Any | 19.3+ GB | **No** |
| INT8 (9.65 GB) | FP16 32K (~2 GB) | ~1.5 GB | ~13.2 GB | **Yes, tight** |
| INT8 (9.65 GB) | TurboQuant 3-bit 32K (~0.4 GB) | ~1.5 GB | ~11.6 GB | **Yes** |
| INT4 (4.8 GB) | FP16 32K (~2 GB) | ~1.5 GB | ~8.3 GB | **Yes, comfortable** |
| INT4 (4.8 GB) | TurboQuant 3-bit 32K (~0.4 GB) | ~1.5 GB | ~6.7 GB | **Yes** |
| INT4 (4.8 GB) | TurboQuant 3-bit 128K (~1.6 GB) | ~2 GB | ~8.4 GB | **Yes** |

**Key insight for Qwen3.5-9B specifically:** Because 75% of layers use linear attention (no KV cache), the KV cache savings from TurboQuant are much smaller than for a pure transformer. The main bottleneck is model weight memory, not KV cache.

### Implication

For Qwen3.5-9B on 16GB, the weight quantization matters far more than KV cache compression. You MUST use INT8 or INT4 weights. TurboQuant saves maybe 1-2 GB at 32K context. Not nothing, but not transformative either. TurboQuant becomes much more valuable at very long contexts (128K+) or with pure-transformer models.

---

## 7. TurboQuant vs Ollama q4_0 KV Cache

### Ollama KV Cache Quantization

Ollama supports KV cache quantization via `OLLAMA_KV_CACHE_TYPE`:
- **FP16** (default): Full precision
- **Q8_0**: ~50% memory reduction, negligible quality loss
- **Q4_0**: ~75% memory reduction, small-medium quality loss (K cache more sensitive)

### TurboQuant Advantages

| Metric | Ollama Q4_0 KV | TurboQuant 3-bit |
|--------|---------------|-----------------|
| Bits per element | 4 | 3 (2 scalar + 1 QJL) |
| Accuracy loss | Small-medium (noticeable at high context) | Zero (proven across benchmarks) |
| Theoretical guarantee | None | Yes (optimal distortion rate) |
| Calibration needed | No | No |
| Needle-in-haystack | Degrades | Perfect |
| Implementation | Built into llama.cpp | Must be custom-built |

### TurboQuant Disadvantages

- **No existing implementation** to plug into Ollama or any serving engine
- Ollama's Q4_0 KV is one environment variable. TurboQuant is months of engineering.
- For Qwen3.5-9B specifically, KV cache is already small (8/32 layers). Savings are marginal.
- The random rotation step adds latency per token (matrix multiply per head). On consumer GPUs without tensor cores optimized for this, overhead could be 5-15%.

### Performance Overhead of Random Rotation

The paper claims "negligible runtime overhead" and shows 8x speedup on H100 at 4-bit. But:
- H100 has specialized hardware (FP8 tensor cores, high memory bandwidth) that consumer GPUs lack
- RTX 4080 has FP16 tensor cores but no FP8. The rotation is an FP16 matmul per attention head per token.
- Per-head rotation matrix is `[head_dim, head_dim]` = `[128, 128]` for Qwen3.5. That is a 128x128 matmul per head, per token, per layer. With 8 attention layers x 32 heads = 256 matmuls per token.
- On RTX 4080 this should be microseconds each (tiny matrices), but it adds up. Realistic overhead: **5-10% on generation latency** at short contexts, decreasing at longer contexts where the attention computation itself dominates.

### Would It Be Faster Than Ollama?

**Almost certainly not for this specific model.** Here is why:

1. Ollama uses llama.cpp with heavily optimized GGML kernels, Flash Attention, and full GPU offload. A HuggingFace Transformers server with torch.compile will be 2-5x slower on raw token throughput.
2. Qwen3.5-9B's DeltaNet architecture already minimizes KV cache overhead. The 75% linear attention layers do not benefit from TurboQuant at all.
3. Ollama's Q8_0 KV cache already provides 50% reduction with negligible quality loss. Going from Q8_0 to TurboQuant 3-bit saves ~40% more on an already-small cache.
4. The engineering effort is orders of magnitude higher.

**Where TurboQuant WOULD win:** Pure transformer models (Llama, Mistral) at very long contexts (64K-128K+) where KV cache dominates VRAM. That is not the Qwen3.5-9B use case.

---

## 8. Fastest Path Forward

### If The Goal Is "Best Inference for Qwen3.5-9B on RTX 4080 16GB"

**Just use Ollama.** Specifically:
- Qwen3.5-9B at Q4_K_M quantization (~6.6 GB weights)
- `OLLAMA_KV_CACHE_TYPE=q8_0` for KV cache compression
- This gives you ~60-70 tok/s, 32K+ context, OpenAI-compatible API, and it works today
- Total engineering effort: 0 hours

### If The Goal Is "Build a TurboQuant Inference Server" (Research/Learning)

**Phase 1: Proof of Concept (2-3 days)**
1. Subclass HuggingFace `DynamicCache` with a simplified TurboQuant (rotation + uniform scalar quantizer, no QJL residual)
2. Test with a small model (Qwen3.5-0.8B) to validate the cache replacement works
3. Measure accuracy on a few benchmarks vs FP16 cache

**Phase 2: Full TurboQuant (1-2 weeks)**
1. Integrate QJL CUDA kernels from [amirzandieh/QJL](https://github.com/amirzandieh/QJL)
2. Implement the optimal scalar quantizer (Beta CDF lookup tables)
3. Add the full rotation + quantize + QJL pipeline
4. Benchmark accuracy and memory on Qwen3.5-9B

**Phase 3: OpenAI-Compatible Server (2-3 days)**
1. FastAPI server with `/v1/chat/completions` endpoint
2. Chat template handling for Qwen3.5
3. SSE streaming
4. Tool call support (critical for FlockRun)

**Phase 4: Optimization (1-2 weeks)**
1. torch.compile the model + cache operations
2. Fuse rotation + quantize into a single CUDA kernel (or rely on torch.compile fusion)
3. INT8 weight quantization (bitsandbytes) to fit in 16GB with FP16 attention
4. Benchmark vs Ollama Q4_K_M + Q8_0 KV

**Total: 3-5 weeks for a working, optimized server.**

### Alternative Fast Path: Cold Compress + API Wrapper (1 week)

1. Fork [AnswerDotAI/cold-compress](https://github.com/AnswerDotAI/cold-compress)
2. Add TurboQuant as a new cache strategy (the framework makes this easy)
3. Wrap in a FastAPI OpenAI-compatible endpoint
4. This gets you a working prototype faster but cold-compress may not support Qwen3.5's hybrid architecture

---

## 9. Effort Estimates and Risks

### Effort Breakdown

| Task | Estimate | Dependencies |
|------|----------|-------------|
| DynamicCache subclass (simplified TurboQuant) | 2-3 days | None |
| QJL CUDA kernel integration | 3-5 days | QJL repo compiles on Windows/CUDA |
| Optimal scalar quantizer | 1-2 days | Paper's Beta CDF tables |
| FastAPI OpenAI server | 2-3 days | None |
| Qwen3.5 hybrid architecture support | 2-3 days | Understanding DeltaNet layers |
| torch.compile optimization | 3-5 days | Model loads correctly |
| INT8 weight quantization | 1 day | bitsandbytes |
| Testing and benchmarking | 3-5 days | Everything above |
| **Total** | **3-5 weeks** | |

### Biggest Risks

1. **No reference implementation.** The TurboQuant paper has no public code. Implementing from the paper alone means potential bugs in the quantizer design, rotation matrix handling, or QJL integration that could take days to debug.

2. **Qwen3.5 hybrid architecture.** The Gated DeltaNet layers do not use standard attention. A custom cache must correctly handle the 3:1 linear-to-attention ratio. If transformers' generation loop does not cleanly separate these, significant patching may be needed.

3. **QJL CUDA kernel compatibility.** The QJL repo was built for Llama on A100s. It may need modifications for Qwen3.5's attention head dimensions, and CUDA kernel compilation on Windows with RTX 4080 (Ada Lovelace, sm_89) may have issues.

4. **Performance gap vs Ollama.** After all this work, the server will likely be slower than Ollama for single-request throughput due to lack of GGML-level optimization. The benefit is only realized at very long contexts where the KV cache savings matter.

5. **Marginal gains for Qwen3.5.** Because only 25% of layers use standard attention (and thus KV cache), TurboQuant saves much less memory than it would for a pure transformer. The ROI is questionable for this specific model.

6. **No continuous batching.** A HuggingFace Transformers-based server handles one request at a time. For FlockRun's use case (single agent talking to one model), this is fine. For multi-agent concurrent requests, it would be a bottleneck.

### Recommendation

**For FlockRun production: stick with Ollama.** The Q8_0 KV cache option is already good enough, Qwen3.5's architecture already minimizes KV cache overhead, and the engineering investment does not justify the marginal memory savings.

**TurboQuant becomes compelling when:**
- Using a pure-transformer model (Llama-4, Mistral) at 64K-256K context
- Running on a GPU where KV cache is the binding VRAM constraint
- Quality at long context matters (Ollama's Q4_0 degrades, TurboQuant does not)
- An official reference implementation is released, cutting the effort from weeks to days

---

## Sources

- [TurboQuant: Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [TurboQuant ICLR 2026 Paper](https://openreview.net/pdf/6593f484501e295cdbe7efcbc46d7f20fc7e741f.pdf)
- [TurboQuant arXiv](https://arxiv.org/abs/2504.19874)
- [QJL GitHub (amirzandieh/QJL)](https://github.com/amirzandieh/QJL)
- [QJL Paper (AAAI)](https://arxiv.org/abs/2406.03482)
- [PolarQuant Paper](https://arxiv.org/html/2502.02617)
- [HuggingFace Cache Strategies](https://huggingface.co/docs/transformers/en/kv_cache)
- [HuggingFace cache_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)
- [NVIDIA kvpress](https://github.com/NVIDIA/kvpress)
- [AnswerDotAI/cold-compress](https://github.com/AnswerDotAI/cold-compress)
- [Cold Compress Blog Post](https://www.answer.ai/posts/2024-08-01-cold-compress.html)
- [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [vLLM Custom KV Cache Forum](https://discuss.vllm.ai/t/custom-kv-cache-implementation/1124)
- [SGLang HiCache](https://lmsys.org/blog/2025-09-10-sglang-hicache/)
- [SGLang Quantized KV Cache](https://docs.sglang.io/advanced_features/quantized_kv_cache.html)
- [Aphrodite Engine](https://github.com/aphrodite-engine/aphrodite-engine)
- [Aphrodite KV Cache Quantization](https://aphrodite.pygmalion.chat/quantization/kv-cache/)
- [vLLM-kvcompress](https://github.com/IsaacRe/vllm-kvcompress)
- [Ollama KV Cache Quantization PR](https://github.com/ollama/ollama/pull/6279)
- [Ollama KV Cache Quantization Blog](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)
- [Qwen3.5-9B HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3.5 GPU Requirements](https://kaitchup.substack.com/p/qwen35-9b-4b-2b-and-08b-gpu-requirements)
- [Qwen3.5 VRAM Guide](https://apxml.com/posts/qwen-3-5-system-requirement-vram-guide)
- [How to Build an OpenAI-Compatible API (Towards Data Science)](https://towardsdatascience.com/how-to-build-an-openai-compatible-api-87c8edea2f06/)
- [LiteLLM Proxy](https://docs.litellm.ai/docs/proxy/configs)
- [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory)
- [Awesome KV Cache Compression](https://github.com/October2001/Awesome-KV-Cache-Compression)
