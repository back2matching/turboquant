# TurboQuant Combination Research

> How TurboQuant's random-rotation KV cache compression combines with other hot ML techniques.
> Research date: 2026-03-25

## Quick Reference: TurboQuant Core

TurboQuant (ICLR 2026, Google Research) compresses KV cache to 3 bits via two stages:
1. **PolarQuant**: Random orthogonal rotation makes coordinates near-independent Beta-distributed, enabling optimal per-coordinate scalar quantization
2. **QJL**: 1-bit residual correction eliminates inner-product bias

Key properties: data-oblivious (no calibration), O(d^2) rotation per vector, 6x memory reduction, 8x attention speedup on H100 with custom CUDA kernels.

---

## Ranked Combinations (by Impact x Feasibility)

| Rank | Combination | Impact | Feasibility | Difficulty | Priority |
|------|-------------|--------|-------------|------------|----------|
| 1 | TurboQuant + KV Cache Eviction | Very High | High | Medium | BUILD NOW |
| 2 | TurboQuant + PagedAttention (vLLM) | Very High | High | Medium-High | BUILD NOW |
| 3 | TurboQuant + Multi-modal VLMs | Very High | High | Medium | BUILD NEXT |
| 4 | TurboQuant + Speculative Decoding | High | High | Medium | BUILD NEXT |
| 5 | TurboQuant + Agent Frameworks | High | High | Low-Medium | BUILD NEXT |
| 6 | TurboQuant + MoE Models | Very High | Medium | Medium-High | RESEARCH MORE |
| 7 | TurboQuant + Ring Attention / Infinite Context | Very High | Medium | High | RESEARCH MORE |
| 8 | TurboQuant + LoRA/QLoRA | Medium-High | Medium | Medium | OPPORTUNISTIC |
| 9 | TurboQuant + QAT | Medium | Low-Medium | High | LONG-TERM |

---

## 1. TurboQuant + KV Cache Eviction (H2O, StreamingLLM)

**Rank: #1 -- BUILD NOW**

### Concept

Combine TurboQuant's quantization with token eviction strategies. Keep important tokens in the KV cache but store them at 3-4 bits via TurboQuant. Evict unimportant tokens entirely. Two layers of compression: fewer tokens AND smaller per-token storage.

### Existing Work

- **MiniKV** (ACL 2025): Already combines 2-bit quantization + eviction. Achieves 86% KV cache reduction, recovers 98.5% accuracy. Uses sub-channel key quantization + per-token value quantization + pyramid KV eviction. [Paper](https://aclanthology.org/2025.findings-acl.952/)
- **Q-Hitter**: Combines Heavy-Hitter Oracle (H2O) attention-score eviction with quantization.
- **ThinKV** (2025): Thought-adaptive compression for reasoning models. Different compression for "thinking" vs "output" tokens.
- **MagicDec** (ICLR 2025): Uses StreamingLLM (attention sink + sliding window) as a draft model for speculative decoding. Compressed KV cache for the draft, full cache for verification.
- **PagedEviction** (EACL 2026): Block-wise eviction aligned with vLLM's paged memory. Structured pruning compatible with PagedAttention.
- **No existing work combines TurboQuant specifically with eviction.**

### Technical Feasibility: HIGH

TurboQuant is orthogonal to eviction. Eviction decides WHICH tokens to keep. TurboQuant decides HOW to store them. The combination is straightforward:

```
1. Run attention, compute importance scores (H2O cumulative attention / StreamingLLM sink detection)
2. Evict low-importance tokens
3. Quantize remaining tokens with TurboQuant rotation + scalar quantization
4. Store: fewer tokens x 3-4 bits each = massive compression
```

The rotation matrix is shared across all tokens (generated once per head dimension), so evicting tokens doesn't affect the quantization quality of remaining tokens. This is a key advantage over calibration-based methods where eviction could change the calibration distribution.

### Impact Estimate

| Metric | Eviction Only (H2O) | Quant Only (TQ 3-bit) | Combined |
|--------|---------------------|----------------------|----------|
| Token reduction | 80-90% | 0% | 80-90% |
| Per-token compression | 1x | 5-6x | 5-6x |
| Total KV reduction | 80-90% | 80-83% | 96-98% |
| Quality loss | 2-5% on long ctx | <1% at 4-bit | 2-6% |

A 96-98% KV cache reduction means a 32K context window that normally costs 4.6 GB (Llama-3.1-8B, FP16) drops to ~100-200 MB. This enables 128K+ contexts on 16GB GPUs.

### Implementation Difficulty: MEDIUM

- Need to implement token importance scoring (H2O or simpler variants)
- TurboQuant quantize/dequantize on the retained subset
- Custom attention kernel that handles mixed sparse + quantized KV
- MiniKV's Triton kernel approach is a good template

### Why #1

- Highest total compression ratio of any combination
- Both techniques are independently proven
- Data-oblivious nature of TurboQuant means no interaction effects with eviction
- MiniKV already proved hybrid eviction+quantization works. TurboQuant's rotation-based approach should give better quality at same bit-width than MiniKV's scalar quantization

---

## 2. TurboQuant + PagedAttention (vLLM)

**Rank: #2 -- BUILD NOW**

### Concept

Store TurboQuant-compressed KV cache in vLLM's paged memory blocks. Each page holds quantized indices + norms instead of FP16/FP8 tensors. More sequences fit in the same GPU memory, higher throughput for serving.

### Existing Work

- **vLLM** currently supports FP8 E4M3, FP8 E5M2, and (on Blackwell) NVFP4 KV cache quantization. No sub-4-bit support.
- **PagedEviction** (EACL 2026): Block-wise eviction aligned with vLLM's PagedAttention. Proves paged memory is compatible with compression strategies.
- **NVFP4** (NVIDIA, 2025): 4-bit KV cache on Blackwell, doubles context budget, <1% accuracy loss. But hardware-specific (SM100 only).
- **No existing work integrates rotation-based quantization with PagedAttention.**

### Technical Feasibility: HIGH

vLLM's block table maps logical blocks to physical blocks. Each block stores `block_size` tokens of K and V. Currently stored as FP16 or FP8. Replacing with TurboQuant-compressed blocks:

```
Current FP8 block: block_size x num_heads x head_dim x 1 byte = e.g., 16 x 32 x 128 x 1 = 64 KB
TQ 4-bit block:    block_size x num_heads x (head_dim x 0.5 + 4) bytes = ~33 KB + norms
TQ 3-bit block:    block_size x num_heads x (head_dim x 0.375 + 4) bytes = ~25 KB + norms
```

Roughly 2x more blocks fit vs FP8, 4x more vs FP16. The rotation matrix is shared across all blocks (one per head dimension), stored once.

### Impact Estimate

| Metric | FP16 baseline | FP8 (current vLLM) | TQ 4-bit paged | TQ 3-bit paged |
|--------|-------------|-------------------|----------------|----------------|
| Bytes per KV element | 2 | 1 | 0.5 | 0.375 |
| Relative throughput | 1x | ~1.5-2x | ~3-4x | ~4-5x |
| Sequences in memory | N | ~2N | ~4N | ~5N |

For serving scenarios (vLLM targets this), throughput is the key metric. Fitting 4-5x more sequences means 4-5x higher batch throughput, which directly translates to cost savings.

### Implementation Difficulty: MEDIUM-HIGH

- Need to modify vLLM's cache engine to handle variable-width blocks
- Custom CUDA attention kernels that dequantize TQ blocks during attention computation
- The rotation step during dequantize adds compute. Need fused kernels to avoid memory bandwidth penalty.
- vLLM's existing FP8 integration is a good reference, but TQ requires matrix multiplication (rotation) not just scaling

### Why #2

- Massive throughput improvement for inference serving
- vLLM is THE dominant serving framework
- FP8 is already the floor for vLLM. Going below FP8 with quality preservation is the next frontier
- NVFP4 only works on Blackwell. TurboQuant could bring sub-8-bit to all GPUs

---

## 3. TurboQuant + Multi-modal VLMs

**Rank: #3 -- BUILD NEXT**

### Concept

Vision-language models encode images as hundreds/thousands of visual tokens, each generating KV cache entries. A single 1024x1024 image can produce 576+ tokens (LLaVA) to 2880+ tokens (high-res). Video models are worse: a 30-second clip can generate 10K+ tokens. TurboQuant compresses these massive visual KV caches.

### Existing Work

- **VL-Cache** (ICLR 2025, Amazon): Modality-aware KV compression for VLMs. Retaining 10% of KV achieves comparable accuracy. Up to 7x decoding speedup, 90% memory reduction. Key insight: visual tokens have different attention patterns than text tokens.
- **MBQ** (Tsinghua, 2025): Modality-balanced quantization for VLMs. Different quantization precision for visual vs text KV entries.
- **DyCoke** (CVPR 2025): Dynamic token compression for video LLMs.
- **MEDA** (NAACL 2025): Dynamic KV allocation for multimodal long-context.
- **No existing work applies rotation-based quantization to VLM KV caches.**

### Technical Feasibility: HIGH

Visual tokens have a specific property that makes them ideal for TurboQuant: they're highly redundant. Nearby image patches produce similar KV vectors. After TurboQuant's random rotation, this redundancy means the quantization error is even lower than for diverse text tokens.

The approach:
```
1. Process image through vision encoder -> visual tokens
2. Pass through LLM layers, generating KV cache
3. Apply TurboQuant to visual token KV entries at aggressive bit-width (2-3 bits)
4. Keep text token KV entries at higher precision (4-8 bits)
5. Modality-aware mixed precision: visual=TQ2, text=TQ4
```

### Impact Estimate

| Scenario | FP16 KV | TQ visual + FP16 text | TQ visual + TQ text |
|----------|---------|----------------------|---------------------|
| Single image (576 tokens) + 2K text | 370 MB | 125 MB | 85 MB |
| High-res image (2880 tokens) + 2K text | 700 MB | 175 MB | 115 MB |
| 30s video (10K tokens) + 2K text | 1.7 GB | 310 MB | 215 MB |

For video VLMs, TurboQuant could be the difference between "fits on consumer GPU" and "needs A100."

### Implementation Difficulty: MEDIUM

- Modality detection is straightforward (VLMs track which tokens are visual)
- Apply different TQ bit-widths per modality
- VL-Cache's sparsity analysis provides the importance scoring; TurboQuant provides the quantization
- Combining VL-Cache's eviction with TurboQuant's quantization = combo #1 applied to VLMs

### Why #3

- VLMs are the fastest-growing model category (GPT-4V, Claude 3, Gemini, LLaVA)
- Visual tokens dominate KV cache size and are highly compressible
- Running VLMs locally is currently impractical on consumer GPUs due to KV cache
- Nobody has applied rotation-based quantization to VLMs yet. Clear first-mover opportunity

---

## 4. TurboQuant + Speculative Decoding

**Rank: #4 -- BUILD NEXT**

### Concept

Two distinct applications:

**A) Compressed draft model KV cache**: The draft model in speculative decoding generates candidate tokens. Its KV cache is only needed for drafting, not final output. Aggressively compress the draft's KV cache with TurboQuant (2-3 bits) to free VRAM for the target model.

**B) Self-speculative decoding**: Use the same model with two KV cache qualities. Quantized KV (TurboQuant 4-bit) for fast draft generation. Full-precision KV for verification. The quantized version runs faster (less memory bandwidth) and serves as the "draft model."

### Existing Work

- **QuantSpec** (ICML 2025, Apple): Self-speculative decoding with hierarchical 4-bit quantized KV cache. Draft model shares architecture with target but uses 4-bit KV + 4-bit weights. Achieves >90% acceptance rate, up to 2.5x speedup. Bit-sharing between target and draft KV caches eliminates extra memory for the draft.
- **MagicDec** (ICLR 2025): Uses StreamingLLM (sliding window + attention sinks) as draft. Up to 2.51x speedup for batch sizes 32-256 on long contexts. Key insight: speculative decoding gets MORE effective with larger batches in the memory-bandwidth-bound regime.
- **LongSpec** (2025): Long-context lossless speculative decoding with efficient drafting.
- **SpecPV** (2025): Partial verification for long-context self-speculative decoding.
- **No existing work uses rotation-based quantization for speculative decoding.**

### Technical Feasibility: HIGH

QuantSpec already proved that quantized KV cache works for drafting with high acceptance rates. TurboQuant could replace QuantSpec's scalar 4-bit quantization with rotation-based 3-bit, potentially enabling:
- Longer draft sequences (more memory freed = more draft tokens before verification)
- Higher acceptance rates (lower quantization error = draft output closer to target)
- Running both draft and target simultaneously (TQ-compressed draft KV fits alongside target KV)

```
QuantSpec approach with TurboQuant:
1. Target model: full FP16 KV cache
2. Draft model (same weights): TQ 3-bit KV cache + 4-bit weights
3. Draft generates K candidate tokens using compressed KV
4. Target verifies all K tokens in one forward pass
5. Hierarchical bit-sharing: TQ indices are a subset of FP16 precision
```

### Impact Estimate

| Metric | No speculation | QuantSpec (4-bit) | TQ-Spec (3-bit) |
|--------|---------------|-------------------|-----------------|
| Draft KV size | N/A | 25% of target | 19% of target |
| Max draft length | N/A | K tokens | ~1.3K tokens (more memory) |
| Acceptance rate | N/A | >90% | ~85-92% (estimated) |
| Speedup | 1x | 2.0-2.5x | 2.0-2.8x (estimated) |

The speedup gain from TQ over QuantSpec is modest (maybe 10-15%) because the acceptance rate may slightly drop at 3-bit. The real win is enabling speculation on memory-constrained GPUs where both models wouldn't otherwise fit.

### Implementation Difficulty: MEDIUM

- QuantSpec provides the framework. Replace their scalar quantizer with TurboQuant.
- Need to implement hierarchical bit-sharing: TQ indices can be derived from higher-precision cache
- The rotation step adds latency to draft generation. Must be faster than the memory bandwidth savings.

### Why #4

- Speculative decoding is becoming standard (vLLM, TensorRT-LLM integrating it)
- QuantSpec from Apple proves quantized KV works for drafting
- TurboQuant's better quality-at-same-bitrate could enable lower bit-widths for drafting
- Good for consumer GPUs where you can't fit separate draft + target models

---

## 5. TurboQuant + Agent Frameworks

**Rank: #5 -- BUILD NEXT**

### Concept

Multi-agent systems where each agent maintains its own KV cache. With 5-10 agents on a single GPU, KV cache becomes the dominant memory cost. TurboQuant compresses each agent's cache, fitting more agents or longer contexts per agent.

### Existing Work

- **"Agent Memory Below the Prompt"** (2026): Persistent Q4 KV cache for multi-agent edge inference. On Apple M4 Pro, only 3 agents fit at 8K context in FP16. Q4 quantization fits 4x more agent contexts. Persisting compressed cache to disk eliminates 15.7s re-prefill per agent. Cache restoration reduces TTFT by up to 136x. [Paper](https://arxiv.org/abs/2603.04428), [Code](https://github.com/yshk-mxim/agent-memory)
- **LRAgent** (2026): Multi-LoRA agent KV cache sharing. Decomposes cache into shared base + per-adapter low-rank component. Flash-LoRA-Attention kernel for efficient reconstruction. [Paper](https://arxiv.org/abs/2602.01053)
- **DroidSpeak** (2024): Cross-LLM KV cache reuse for multi-agent serving.
- **Cache2Cache** (2025): Trainable projection to fuse one model's KV cache into another.
- **No existing work uses rotation-based quantization for multi-agent KV cache.**

### Technical Feasibility: HIGH

This is the most directly relevant combination for FlockRun. Current state:
- FlockRun runs 3 agents on a single RTX 4080 16GB via Ollama
- Each agent gets a KV cache slot. Ollama manages context via `num_ctx`
- With q8_0 KV: each slot uses ~72 KB/token (Llama-8B scale). 3 agents at 8K = ~1.7 GB KV
- With TQ 3-bit: drops to ~27 KB/token. Same 3 agents at 8K = ~650 MB KV
- Freed VRAM: ~1 GB, enough for a 4th agent or 2x longer context per agent

The "Agent Memory Below the Prompt" paper is particularly relevant -- it already does Q4 persistent cache for agents. Replacing their scalar Q4 with TurboQuant 3-bit would give ~25% more compression with equal or better quality.

### Impact Estimate

| Scenario (16GB GPU, 9B model) | FP16 KV | q8_0 KV | TQ 4-bit KV | TQ 3-bit KV |
|-------------------------------|---------|---------|-------------|-------------|
| 3 agents x 8K context | 3.5 GB | 1.7 GB | 0.9 GB | 0.65 GB |
| 5 agents x 8K context | 5.8 GB | 2.9 GB | 1.5 GB | 1.1 GB |
| 3 agents x 32K context | 14 GB | 7 GB | 3.5 GB | 2.6 GB |
| 10 agents x 4K context | 5.8 GB | 2.9 GB | 1.5 GB | 1.1 GB |

TQ 3-bit enables 3 agents at 32K context on 16GB (with ~12 GB for model weights in Q4). Without TQ, this is impossible.

### Implementation Difficulty: LOW-MEDIUM

- Agent KV cache persistence: serialize TQ indices + norms to disk (much smaller files)
- Fast cache swap: load compressed cache, dequantize on demand
- FlockRun integration: per-agent compression level setting in config YAML
- Can build on top of the "Agent Memory Below the Prompt" approach

### Why #5

- Directly applicable to FlockRun's production use case
- "Agent Memory Below the Prompt" paper validates the approach, published March 2026
- Low implementation difficulty because agent systems already handle cache lifecycle
- Enables capabilities not otherwise possible (10 agents on 16GB, or 3 agents at 32K context)

---

## 6. TurboQuant + MoE Models

**Rank: #6 -- RESEARCH MORE**

### Concept

MoE models (Mixtral, DBRX, DeepSeek-V3, Qwen-MoE) have the same KV cache problem as dense models -- KV cache size is determined by the attention layers, which are NOT sparse. A Mixtral 8x7B has 32 attention heads across 32 layers, same KV cache as a dense 7B model. The expert sparsity only helps with weight memory, not KV memory.

However, MoE models are memory-bound in a different way: expert weights dominate. The KV cache is a smaller fraction of total memory but still limits context length and batch size.

### Existing Work

- **DeepSeek-V2/V3 MLA**: Multi-Head Latent Attention reduces KV cache by 93.3% via low-rank factorization. This is an architectural change, not post-training quantization. 28x smaller KV cache inherently.
- **MoE-SpeQ** (2025): Speculative quantized decoding with expert prefetching/offloading for MoE.
- **KTransformers** (SOSP 2025): CPU/GPU hybrid inference for MoE. Specialized memory layout optimization. Runs Mixtral efficiently via expert offloading.
- **Joint MoE Scaling Laws** (2025): Proves MoE models can be memory-efficient with proper design.
- **TurboQuant has NOT been tested on MoE architectures.** The paper's "zero accuracy loss" claim is undemonstrated on MoE.

### Technical Feasibility: MEDIUM

The rotation-based approach should work identically on MoE KV caches because the attention mechanism is the same. The KV cache doesn't interact with the expert routing. However:

1. **MoE models with MLA (DeepSeek-V3)**: Already have tiny KV caches. TurboQuant adds little value. The KV cache is already compressed 28x by the architecture.
2. **MoE models without MLA (Mixtral, DBRX)**: Full KV cache. TurboQuant helps exactly as much as it helps dense models. The KV savings let you fit larger batches or longer contexts.
3. **Expert offloading + TQ KV**: The interesting combo. If experts are offloaded to CPU/SSD, GPU memory is almost entirely KV cache. Compressing it with TurboQuant frees GPU memory for... more KV cache (longer context) or higher batch sizes.

```
Mixtral 8x7B on 24GB GPU:
- Expert weights (top-2 active): ~24 GB full, or ~12 GB quantized
- KV cache at 8K context: ~1.8 GB (FP16) -> ~0.34 GB (TQ 3-bit)
- With offloading: experts on CPU, GPU holds only KV cache
- TQ enables: 8K -> 40K+ context on GPU (KV-only mode)
```

### Impact Estimate

| Scenario | Without TQ | With TQ 3-bit | Improvement |
|----------|-----------|---------------|-------------|
| Mixtral 8x7B, 8K ctx, 24GB | 12 GB model + 1.8 GB KV | 12 GB model + 0.34 GB KV | 1.5 GB saved, +4K ctx |
| Mixtral offloaded, GPU=KV only | 24 GB = ~107K ctx | 24 GB = ~530K ctx | 5x longer context |
| DeepSeek-V3 (MLA) | Tiny KV already | Marginal improvement | Not worth it |

### Implementation Difficulty: MEDIUM-HIGH

- For standard MoE (Mixtral): same as any transformer. No special work needed.
- For MLA-based MoE (DeepSeek-V3): TurboQuant on already-compressed latent vectors. Needs testing. May conflict with the low-rank structure.
- Expert offloading + TQ: need to coordinate expert scheduling with KV cache compression/decompression. Timing matters for overlapping compute and memory ops.

### Why #6

- MoE is the dominant architecture trend (DeepSeek, Qwen-MoE, Mixtral, Grok)
- For non-MLA MoE models, TurboQuant gives the same benefits as for dense models
- The "offloaded experts + compressed KV" scenario is uniquely powerful for consumer GPUs
- However, the trend is toward MLA-style architectures that make KV compression less needed
- Needs empirical validation on MoE models first

---

## 7. TurboQuant + Ring Attention / Infinite Context

**Rank: #7 -- RESEARCH MORE**

### Concept

Ring attention distributes a sequence across multiple GPUs, each holding a segment of the KV cache. Each GPU processes its local segment and passes partial results around the "ring." TurboQuant compresses each segment, reducing inter-GPU communication and per-GPU memory.

### Existing Work

- **KVQuant** (NeurIPS 2024, Berkeley): Enables 10 million token context on 8-GPU A100 system with 3-bit quantization. Per-channel keys, pre-RoPE quantization, non-uniform datatypes, outlier isolation. [Code](https://github.com/SqueezeAILab/KVQuant)
- **Inference-Time Hyper-Scaling** (NeurIPS 2025, NVIDIA): DMS (Dynamic Memory Sparsification) achieves 8x KV compression with 1K training steps. Improves inference-time compute scaling by fitting more "thinking" tokens. Qwen-R1 32B gains +9.1 on AIME 24.
- **HCAttention** (2025): Extreme KV cache compression via hierarchical coding.
- **Ring attention + KV quantization**: Not explicitly combined in published work.

### Technical Feasibility: MEDIUM

The rotation matrix is the challenge. In standard TurboQuant, one rotation matrix per head dimension is generated and shared across all tokens. In ring attention:

1. All GPUs must share the same rotation matrix (easy -- seed-based generation)
2. Each GPU quantizes/dequantizes its local KV segment independently (works because TurboQuant is per-vector)
3. When partial KV segments transfer between GPUs, they transfer as compressed indices (much less bandwidth)

```
Ring attention with TQ:
- GPU 0: tokens 0-8K, TQ-compressed KV for this segment
- GPU 1: tokens 8K-16K, TQ-compressed KV for this segment
- GPU 2: tokens 16K-24K, TQ-compressed KV for this segment
- Ring communication: pass compressed KV indices, not FP16 tensors
- Per-GPU memory: 5-6x less KV, enabling 5-6x longer segments per GPU
```

The bandwidth savings are significant. Inter-GPU communication (NVLink or PCIe) is often the bottleneck for ring attention. Sending 3-bit indices instead of 16-bit floats = 5x bandwidth reduction.

### Impact Estimate

| Metric | Ring + FP16 | Ring + FP8 | Ring + TQ 3-bit |
|--------|-----------|-----------|----------------|
| KV per GPU (64K segment) | 9.2 GB | 4.6 GB | 1.7 GB |
| Max sequence (8x A100 80GB) | ~2M tokens | ~4M tokens | ~10M+ tokens |
| Ring bandwidth per step | 100% | 50% | 19% |
| Scaling efficiency | Bandwidth-limited | Better | Near-linear |

### Implementation Difficulty: HIGH

- Need to integrate TurboQuant into ring attention kernel implementations
- Dequantization must happen at attention compute time, not separately
- Fused TQ-dequant + FlashAttention kernels needed for practical speed
- Cross-GPU synchronization of rotation matrices (trivially solved by seed sharing)
- Testing at scale requires multi-GPU hardware

### Why #7

- Infinite context is a major trend (Gemini 1M+, Claude 200K+)
- KV cache is THE bottleneck for long context
- KVQuant already demonstrated 10M tokens with 3-bit quantization. TurboQuant's better quality-per-bit could push this further
- However, the implementation requires deep CUDA expertise and multi-GPU access
- Best suited for cloud providers, not consumer GPU use cases

---

## 8. TurboQuant + LoRA/QLoRA

**Rank: #8 -- OPPORTUNISTIC**

### Concept

Three angles:
- **A) Compress LoRA adapter KV activations**: LoRA adapters modify the K and V projections. The adapter-specific component of the KV cache could be stored separately at low precision.
- **B) Fine-tune with compressed KV cache**: Train or fine-tune models where the KV cache is TurboQuant-compressed during training, so the model learns to work with quantized attention.
- **C) Rotation-aware fine-tuning**: Like RoLoRA, apply rotations before fine-tuning to make the model quantization-friendly.

### Existing Work

- **RoLoRA** (EMNLP 2024): Rotation-aware LoRA fine-tuning. Applies Hadamard rotation to eliminate outliers before fine-tuning. W4A4 quantization improves by up to 14.6 points on MMLU. [Code](https://github.com/HuangOwen/RoLoRA)
- **QA-LoRA**: Quantization-aware LoRA. Joint optimization of quantization and adaptation.
- **LRAgent** (2026): Decomposes multi-agent KV cache into shared base + per-LoRA adapter component. Flash-LoRA-Attention kernel.
- **HYC-LoRA** (2025): Memory-efficient LoRA fine-tuning with hybrid activation compression.
- **ROMA** (2025): Hardware accelerator with ROM for quantized base model, SRAM for LoRA weights and KV cache.

### Technical Feasibility: MEDIUM

**Angle A (compress LoRA KV)**: LRAgent already decomposes KV cache into base + adapter components. The adapter component is low-rank (by definition -- LoRA rank is typically 8-64). TurboQuant on a low-rank component doesn't help much because the vectors are already compact. The base component is where TurboQuant helps, and it's shared across LoRA adapters.

**Angle B (train with TQ cache)**: Requires differentiable TurboQuant. The argmin in quantization is non-differentiable. Would need straight-through estimator (STE) or Gumbel-softmax relaxation. Adds training complexity for uncertain benefit.

**Angle C (rotation-aware fine-tuning)**: RoLoRA already does this with Hadamard rotation for weight/activation quantization. Extending to KV cache rotation is natural: apply the same rotation used by TurboQuant during fine-tuning so the model adapts to the rotated distribution. This is the most promising angle.

### Impact Estimate

| Angle | Memory Savings | Quality Impact | Novelty |
|-------|---------------|----------------|---------|
| A: Compress LoRA KV | Modest (adapter KV is small) | Neutral | Low |
| B: Train with TQ cache | N/A (training technique) | Could improve TQ quality | Medium |
| C: Rotation-aware FT | Same as TQ + better quality | +5-15% accuracy at low bits | High |

### Implementation Difficulty: MEDIUM

- Angle A: Straightforward but limited impact
- Angle B: Research-level, requires training infrastructure modifications
- Angle C: Build on RoLoRA framework, add TurboQuant rotation to the fine-tuning pipeline

### Why #8

- The most promising angle (C) is essentially "QAT for TurboQuant" which overlaps with combo #9
- Angle A has limited impact because LoRA KV components are already small
- RoLoRA's existence proves rotation-aware training works, but it targets weight/activation quant, not KV cache quant
- Best as an enhancement to TurboQuant rather than a standalone product

---

## 9. TurboQuant + Quantization-Aware Training (QAT)

**Rank: #9 -- LONG-TERM**

### Concept

Train models that EXPECT TurboQuant-compressed KV cache. Instead of post-training quantization (where we hope the model tolerates compression), the model learns to produce KV vectors that quantize well under rotation.

### Existing Work

- **NVIDIA TensorRT Model Optimizer**: QAT support for FP8 and NVFP4 KV cache. Train the model with quantized KV, fine-tune for 1-2 epochs, model adapts. Used for Blackwell NVFP4 deployment.
- **CommVQ** (ICML 2025, Apple/UMass): Commutative vector quantization for KV cache. Trains a lightweight encoder and codebook that commutes with RoPE. 87.5% compression at 2-bit, enables 1-bit KV with minimal loss on RTX 4090. [Code](https://github.com/UMass-Embodied-AGI/CommVQ)
- **DMS** (NeurIPS 2025, NVIDIA): Dynamic Memory Sparsification. Only 1K training steps for 8x compression. Learns which tokens to keep vs evict.
- **No QAT-trained models for rotation-based KV quantization exist.**

### Technical Feasibility: LOW-MEDIUM

The challenge is making TurboQuant's quantization differentiable for backpropagation:

1. **Forward pass**: Apply rotation, quantize to indices, dequantize back (standard TQ pipeline)
2. **Backward pass**: The quantization step (argmin to nearest centroid) has zero gradient almost everywhere. Need STE (Straight-Through Estimator) to pass gradients through.
3. **What the model learns**: To produce KV vectors whose rotated coordinates land near centroid centers. Effectively, the model learns to "pre-quantize" its internal representations.

```
QAT pipeline:
1. Pre-train model normally (or start from pre-trained)
2. Insert TurboQuant into attention: KV -> rotate -> quantize -> STE -> dequantize -> rotate back
3. Fine-tune for 1-2 epochs on standard data
4. Result: model that produces KV cache naturally suited for TQ compression
```

CommVQ's approach is instructive: they train a lightweight codebook (not the full model) that commutes with RoPE. A TurboQuant-QAT approach could similarly fine-tune only the KV projection layers while keeping the rotation matrix fixed.

### Impact Estimate

| Approach | Training Cost | Quality at 3-bit | Quality at 2-bit |
|----------|-------------|-------------------|-------------------|
| Post-training TQ | Zero | Good (near-lossless) | Degraded |
| TQ + QAT (full fine-tune) | 1-2 epochs | Near-lossless | Good |
| TQ + QAT (KV projections only) | Hours | Near-lossless | Moderate |
| CommVQ (trained codebook) | Medium | N/A | Excellent (1-bit possible) |

QAT would primarily help at very low bit-widths (2-bit, 1-bit) where post-training TurboQuant degrades. At 3-4 bit, post-training TQ is already near-lossless, so QAT doesn't add much.

### Implementation Difficulty: HIGH

- Requires training infrastructure (GPUs, datasets, training loops)
- STE for quantization is well-understood but TurboQuant's rotation adds complexity
- Per-model fine-tuning -- not a general solution
- CommVQ's approach (train the codebook, not the model) is more practical

### Why #9

- Post-training TurboQuant already works well at 3-4 bits. QAT is only needed for extreme compression (2-bit, 1-bit)
- CommVQ already achieves 1-bit KV via trained codebook. Different approach, same goal.
- High training cost, per-model effort. Not a "drop-in" solution.
- Best as a long-term research direction after TurboQuant proves its value at 3-4 bits

---

## Cross-Cutting Observations

### 1. Eviction + Quantization is the Meta-Pattern

The highest-impact combinations (ranks #1, #3, #4) all involve combining TurboQuant with some form of token selection/eviction. This is the dominant trend in KV cache research:

- MiniKV: eviction + 2-bit scalar quant
- VL-Cache: modality-aware eviction
- MagicDec: sliding window eviction + speculative decoding
- ThinKV: thought-adaptive eviction for reasoning

**TurboQuant's unique advantage**: its data-oblivious nature means eviction doesn't affect quantization quality. Calibration-based methods (KVQuant, KIVI) may need re-calibration after eviction. TurboQuant's rotation-based approach doesn't care which tokens you keep.

### 2. The Rotation Matrix is Both Feature and Bug

The O(d^2) rotation matrix (128x128 = 16K elements for head_dim=128) is:
- **Feature**: Shared across all tokens and sequences. Generate once, reuse forever. Seed-based, so no storage needed.
- **Bug**: Matrix multiply during quantize/dequantize is expensive without custom CUDA kernels. This is what makes TurboQuant 30% slower in our CPU implementation.

Every combination needs fused CUDA kernels that apply rotation during the attention computation, not as a separate step. This is the common prerequisite for all high-impact combinations.

### 3. TurboQuant vs CommVQ vs VQKV

Three competing approaches for sub-8-bit KV cache, all published 2025-2026:

| Method | Approach | Bits | Trained? | Quality at 2-bit | Best For |
|--------|----------|------|----------|-------------------|----------|
| TurboQuant | Random rotation + scalar quant | 2.5-4 | No | Moderate | Drop-in compression, no training |
| CommVQ | Trained codebook + VQ | 1-2 | Yes (codebook) | Excellent | Maximum compression, willing to train |
| VQKV | Vector quantization + codebook | 2-4 | No (offline codebook) | Good | High compression ratio, training-free |

**TurboQuant's niche**: when you want drop-in, calibration-free, training-free compression that works on any model. CommVQ is better for maximum compression but requires per-model codebook training.

### 4. FlockRun-Specific Priority

For FlockRun's multi-agent use case on consumer GPUs:

1. **Immediate**: TurboQuant + Agent Frameworks (#5) -- directly improves FlockRun production
2. **Near-term**: TurboQuant + Eviction (#1) -- enables much longer agent context windows
3. **Medium-term**: TurboQuant + Speculative Decoding (#4) -- faster agent responses
4. **Long-term**: TurboQuant + vLLM (#2) -- if FlockRun moves to serving-scale deployment

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- Fused TurboQuant CUDA kernels (rotation + quantize + dequantize in one kernel)
- Benchmark on Llama-3.1-8B with standard perplexity/throughput tests
- This enables ALL downstream combinations

### Phase 2: Agent Integration (Weeks 2-3)
- TurboQuant persistent KV cache for multi-agent (combo #5)
- Per-agent compression settings in FlockRun config
- Benchmark: agents at 8K/16K/32K context on 16GB GPU

### Phase 3: Hybrid Compression (Weeks 3-5)
- TurboQuant + token eviction (combo #1)
- Modality-aware compression for VLMs (combo #3)
- Publish benchmarks. This is the "paper" opportunity.

### Phase 4: Serving Integration (Weeks 5-8)
- vLLM PagedAttention integration (combo #2)
- Speculative decoding with TQ draft (combo #4)

---

## Papers and Repos Referenced

### TurboQuant Family
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QJL CUDA kernels](https://github.com/amirzandieh/QJL) (Apache-2.0)
- [PolarQuant](https://github.com/ericshwu/PolarQuant) (custom Triton)
- [Google blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

### Speculative Decoding + KV Cache
- [QuantSpec](https://arxiv.org/abs/2502.10424) (ICML 2025, Apple)
- [MagicDec](https://arxiv.org/abs/2408.11049) (ICLR 2025)
- [LongSpec](https://arxiv.org/abs/2502.17421) (2025)

### Eviction + Quantization Hybrids
- [MiniKV](https://aclanthology.org/2025.findings-acl.952/) (ACL 2025)
- [H2O](https://arxiv.org/abs/2306.14048) (NeurIPS 2023)
- [StreamingLLM](https://arxiv.org/abs/2309.17453) (2023)
- [PagedEviction](https://arxiv.org/abs/2509.04377) (EACL 2026)
- [ThinKV](https://arxiv.org/abs/2510.01290) (2025)

### Vector Quantization for KV Cache
- [CommVQ](https://arxiv.org/abs/2506.18879) (ICML 2025, Apple/UMass)
- [VQKV](https://arxiv.org/abs/2603.16435) (2026)

### Multi-modal VLM Compression
- [VL-Cache](https://arxiv.org/abs/2410.23317) (ICLR 2025, Amazon)
- [MBQ](https://arxiv.org/abs/2412.19877) (Tsinghua, 2025)
- [DyCoke](https://arxiv.org/abs/2411.15024) (CVPR 2025)

### Long Context / Ring Attention
- [KVQuant](https://arxiv.org/abs/2401.18079) (NeurIPS 2024, Berkeley)
- [Inference-Time Hyper-Scaling / DMS](https://arxiv.org/abs/2506.05345) (NeurIPS 2025, NVIDIA)

### LoRA + Rotation
- [RoLoRA](https://arxiv.org/abs/2407.08044) (EMNLP 2024)
- [LRAgent](https://arxiv.org/abs/2602.01053) (2026)

### MoE Optimization
- [DeepSeek-V2 MLA](https://arxiv.org/abs/2405.04434) (2024)
- [KTransformers](https://arxiv.org/abs/2504.07320) (SOSP 2025)
- [MoE-SpeQ](https://arxiv.org/abs/2511.14102) (2025)

### Agent Memory
- [Agent Memory Below the Prompt](https://arxiv.org/abs/2603.04428) (2026)
- [agent-memory repo](https://github.com/yshk-mxim/agent-memory) (Apple Silicon)

### vLLM / Serving
- [PagedAttention](https://arxiv.org/abs/2309.06180) (SOSP 2023)
- [vLLM quantized KV docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [NVFP4 KV cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) (NVIDIA, 2025)
