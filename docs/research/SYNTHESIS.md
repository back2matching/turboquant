# TurboQuant Research Synthesis

> Combined findings from 4 parallel research threads. Decision document for what to build.

**Date:** 2026-03-25 (TurboQuant blog post dropped ~24 hours ago)

---

## Executive Summary

TurboQuant is a real, well-proven algorithm (ICLR 2026, Google Research) that compresses LLM KV caches to 3 bits with zero accuracy loss and up to 8x attention speedup. **Nobody has built a usable implementation outside Google.** The QJL component has Apache-2.0 CUDA kernels. The field below FP8 KV cache in production inference is completely empty.

This is a genuine first-mover opportunity. The question is: which build path gives us the best ratio of impact to effort?

---

## What We Confirmed

### TurboQuant Is Real But Has No Usable Code
- Paper: solid (ICLR 2026, Google Research, Vahab Mirrokni)
- Algorithm: two-stage (PolarQuant rotation + QJL residual correction)
- Results: 6x memory reduction, 8x attention speedup on H100, zero accuracy loss at 3+ bits
- Code: **Google has NOT released TurboQuant code**
- Available pieces: QJL CUDA kernels (Apache-2.0, Llama-2/3), PolarQuant (custom Triton, A100 only)
- Community: 2-3 repos created in last 24 hours, all research-quality, no CUDA kernels

### Nobody Has Built What We Want To Build
- Zero TurboQuant inference servers exist
- Zero consumer GPU benchmarks (all testing on H100/A100)
- Zero llama.cpp/Ollama/vLLM integrations
- Zero multi-agent system integrations
- The entire space below FP8 KV cache in production serving is empty
- We would genuinely be first at any of these

### 27B on 16GB: Not Viable for Production
- Ollama's qwen3.5:27b (Q4_K_M, 17GB) doesn't fit in 16GB VRAM
- Real benchmark: 6.48 tok/s with CPU offloading (14x slower than 9B's 62 tok/s)
- Q3_K_M (13.85GB) possible with manual GGUF import but limited to 8-16K context
- Qwen3.5's hybrid DeltaNet architecture helps but doesn't overcome the weight size problem
- **Verdict: stick with qwen3.5:9b for production.** The 27B upgrade needs a GPU upgrade, not software tricks.

### Existing Ollama KV Cache Compression Already Works
- `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0` is available today
- q8_0: 2x KV reduction, +0.004 perplexity (essentially free)
- q4_0: 4x KV reduction, +0.2 perplexity (noticeable but acceptable)
- Known stability bugs with Qwen3.5 hybrid attention at 3+ parallel slots

---

## Build Paths Evaluated

### Path A: Port TurboQuant to llama.cpp

**What:** Add TurboQuant as a new ggml KV cache quant type (e.g., `tq3_0`). Submit as PR to llama.cpp. Auto-propagates to Ollama, LM Studio, KoboldCpp, etc.

**Effort:** 3-6 weeks. 8-15 files to change. Custom CUDA FlashAttention kernels required (the hard part). CPU fallback (scalar + AVX2 + ARM_NEON). Integration with ggml type system.

**Files to modify:**
- `ggml.h` -- add enum value + type traits
- `ggml-quants.c/h` -- quantize/dequantize reference implementations
- `ggml-cpu/` -- optimized CPU kernels (scalar, AVX2, NEON)
- `ggml-cuda/` -- CUDA dequantize + FlashAttention kernels (HARDEST PART)
- `tools/server/` -- expose new type in CLI flags
- Tests + benchmarks

**Technical challenges:**
1. TurboQuant uses random rotation (O(d^2) per vector). ggml types are block-based scalar quantizers. Fundamentally different approach.
2. FlashAttention CUDA kernels can't reuse existing dequant code (PR #7527 author confirmed this). Need custom kernels from scratch.
3. Rotation matrix storage -- needs per-layer persistent state, which ggml types don't have.
4. The rotation + dequantize cost during attention could negate the memory bandwidth savings at small context lengths.

**Risks:** HIGH. Deep ggml/CUDA expertise required. 793 open PRs on llama.cpp. Complex PRs take weeks-months to merge. The rotation-based approach may not fit ggml's block-based type system at all.

**Impact:** ENORMOUS if merged. Benefits every llama.cpp user worldwide. Front-page HN material.

**Verdict:** High-impact but high-risk. Needs deep C/CUDA skills. Multi-week effort with uncertain merge timeline.

### Path B: PyTorch TurboQuant Inference Server

**What:** Build a lightweight inference server using HuggingFace Transformers + custom TurboQuant DynamicCache subclass + QJL CUDA kernels. Expose OpenAI-compatible API. FlockRun's HTTP adapter connects unchanged.

**Effort:** 2-4 weeks.

**Architecture:**
```
FlockRun HTTP Adapter -> OpenAI API -> TurboQuant Server -> HuggingFace Model + TQ Cache -> GPU
```

**Technical challenges:**
1. HuggingFace models in FP16 are huge (qwen3.5:9b = 19.3 GB FP16). Must use bitsandbytes INT8/INT4 weight quantization alongside TurboQuant KV cache. Untested combination.
2. QJL CUDA kernels only support Llama-2/3 architecture. Need to adapt for Qwen3.5's hybrid attention.
3. Pure PyTorch TurboQuant (no custom CUDA kernels) will be SLOWER than FP16 baseline. The memory savings won't translate to speed gains without fused kernels.
4. PolarQuant (stage 1 of TurboQuant) requires custom Triton build. Fragile dependency.
5. Generation speed will be much slower than Ollama's optimized llama.cpp backend.

**Risks:** MEDIUM-HIGH. The server might work but be slower than Ollama, which defeats the purpose. CUDA kernel adaptation for Qwen3.5 is non-trivial.

**Impact:** MODERATE. Proves the concept. First TurboQuant inference server. But only useful for FlockRun users unless it generalizes.

**Verdict:** Feasible but risks being slower than Ollama. The value is in proving the concept, not production use.

### Path C: TurboQuant Benchmark Suite for Consumer GPUs

**What:** Be the first to benchmark TurboQuant (via PyTorch implementation) against q4_0/q8_0/FP16 on consumer GPUs (RTX 4080). Publish results. Use our existing 50-task harness to measure impact on tool calling accuracy.

**Effort:** 1-2 weeks.

**What we'd measure:**
1. VRAM savings: FP16 vs q8_0 vs q4_0 vs TurboQuant-3bit KV cache
2. Generation speed: tokens/sec for each
3. Tool calling accuracy: our 50-task benchmark harness
4. Context length limits: max context per method on 16GB
5. Multi-slot impact: 1 vs 2 vs 4 concurrent inference slots

**Technical challenges:**
1. Need a working TurboQuant PyTorch implementation (community repo or build from paper)
2. QJL CUDA kernels for Llama models exist. Qwen3.5 needs adaptation.
3. Fair comparison requires running same model, same prompts, same hardware across all methods.

**Risks:** LOW-MEDIUM. Even if TurboQuant underperforms on consumer GPUs, the benchmark data is valuable.

**Impact:** HIGH for visibility. "First TurboQuant benchmarks on consumer GPUs" is a compelling blog post / Reddit thread. Our existing benchmark harness gives us credibility.

**Verdict:** Best effort-to-impact ratio. Builds on our existing benchmark infrastructure. Publishable regardless of outcome.

### Path D: FlockRun Smart VRAM Optimizer

**What:** Build a FlockRun feature that dynamically manages KV cache compression across agents based on task type, available VRAM, and quality requirements. Starts with existing Ollama q4_0/q8_0, designed to support TurboQuant when available.

**Effort:** 1-2 weeks for the FlockRun integration. TurboQuant backend is separate.

**What it does:**
1. Per-agent KV cache compression settings (CEO can tolerate more compression, Eng Ops needs precision)
2. VRAM monitoring via `nvidia-smi` or NVML
3. Dynamic context adjustment: reduce context when VRAM is tight, expand when available
4. Agent scheduling aware of GPU memory (pause lower-priority agents when high-priority needs VRAM)
5. Dashboard VRAM monitoring widget

**Risks:** LOW. Uses existing Ollama capabilities. No CUDA kernel work.

**Impact:** MEDIUM. Practical for FlockRun users. Unique multi-agent VRAM management. Good differentiator.

**Verdict:** Solid FlockRun feature. Can be built independently of TurboQuant. Becomes more powerful when TurboQuant is available.

---

## Recommended Strategy

### Phase 1: Enable What Works Today (1 day)
- Enable `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0` on production
- Benchmark with our 50-task harness to verify no quality loss
- Document VRAM savings

### Phase 2: TurboQuant Benchmark Suite (1-2 weeks) -- THE MAIN PLAY
- Implement TurboQuant in PyTorch (from paper + QJL CUDA kernels)
- Benchmark against FP16/q8_0/q4_0 on RTX 4080 with our harness
- Publish results (blog post, Reddit r/LocalLLaMA, HN)
- First consumer GPU TurboQuant benchmarks anywhere
- This builds our credibility and gets attention

### Phase 3: FlockRun VRAM Optimizer (1-2 weeks, parallel with Phase 2)
- Per-agent KV cache config in adapter settings
- VRAM monitoring + dashboard widget
- Dynamic context/compression adjustment
- This is the FlockRun-specific differentiator

### Phase 4: llama.cpp Contribution (post-launch, 3-6 weeks)
- If TurboQuant proves valuable on consumer GPUs (Phase 2)
- Port the algorithm as a new ggml KV cache type
- Submit PR to llama.cpp
- This is the moonshot -- massive impact if accepted

---

## Why This Strategy Works

1. **Phase 1 is free** -- Ollama flags, zero code changes, immediate benefit
2. **Phase 2 gets attention** -- First TurboQuant benchmarks on consumer GPUs. Publishable regardless of results. Ties to today's Google announcement.
3. **Phase 3 builds product** -- FlockRun feature that nobody else has (multi-agent VRAM management)
4. **Phase 4 is the moonshot** -- llama.cpp PR would be front-page news, but only worth attempting if Phase 2 proves the algorithm works well on consumer GPUs

Each phase builds on the previous one. If Phase 2 shows TurboQuant doesn't help on consumer GPUs, we still have Phase 1 (existing compression) and Phase 3 (VRAM optimizer) as wins.

---

## Sources

Full source lists in companion docs:
- [TURBOQUANT-DEEP-DIVE.md](TURBOQUANT-DEEP-DIVE.md) -- Paper analysis, algorithm details, benchmarks
- [KV-CACHE-LANDSCAPE.md](KV-CACHE-LANDSCAPE.md) -- Full compression landscape, production engines, research methods
- [FLOCKRUN-GPU-ANALYSIS.md](FLOCKRUN-GPU-ANALYSIS.md) -- Our GPU VRAM calculations, 27B analysis, concurrent agent scenarios
