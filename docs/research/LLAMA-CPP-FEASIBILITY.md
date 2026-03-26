# TurboQuant -> llama.cpp KV Cache Type: Feasibility Research

> Research date: 2026-03-25
> Goal: What would it take to port TurboQuant (random rotation + optimal scalar quantization) as a new KV cache type in llama.cpp?

---

## 1. TurboQuant: What It Is

Published at ICLR 2026 by Google Research. Paper: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874).

**Core algorithm:**
1. Randomly rotate input vectors using a randomized Hadamard transform (sign flip + Fast Hadamard Transform). This induces a concentrated Beta distribution on coordinates.
2. In high dimensions, coordinates become near-independent after rotation. Apply an optimal scalar quantizer to each coordinate independently.
3. Two-stage architecture: MSE-optimal quantizer first, then 1-bit QJL (Quantized Johnson-Lindenstrauss) transform on the residual for unbiased inner product estimation.

**Results for KV cache:**
- 3.5 bits/channel: quality-neutral (zero accuracy loss)
- 2.5 bits/channel: marginal degradation
- 4-bit TurboQuant: up to 8x speedup over FP32 keys on H100
- 6x minimum memory reduction on KV cache
- Data-oblivious (no calibration data, no fine-tuning needed)

**Open source status:** As of March 2026, NO public PyTorch/CUDA implementation released by Google. The paper references "publicly available code" for mixed-precision matmul kernels but no standalone repo found.

---

## 2. How llama.cpp KV Cache Quantization Works Today

### The Original PRs

The KV cache quantization feature was built across three PRs by JohannesGaessler:

| PR | Title | Purpose |
|----|-------|---------|
| [#7412](https://github.com/ggml-org/llama.cpp/pull/7412) | CUDA: quantized KV cache demo | Research prototype. Simple implementation to measure quality impact. |
| [#7527](https://github.com/ggml-org/llama.cpp/pull/7527) | CUDA: quantized KV support for FA vec | Production flash attention kernels for quantized KV. Only small-batch kernels. |
| #7681 | (follow-up) | Additional KV cache quantization support. |

**Key findings from JohannesGaessler's research:**
- K cache is much more sensitive to quantization than V cache
- Q8_0 KV: no significant quality loss (~0.002-0.05 perplexity increase)
- Q4_0 V + FP16 K: more precise than Q6_K weights with FP16 KV
- Q8_0 K + Q4_0 V (6.5 bpv average): more precise than Q6_K weights
- Quantizing K improves perf slightly; quantizing V hurts slightly

### Currently Supported KV Cache Types

```
f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
```

That's it. No sub-4-bit types. No rotation-based types. No K-quant types for KV cache.

### Flash Attention Kernels

Location: `ggml/src/ggml-cuda/` directory, specifically:
- `fattn.cu` / `fattn.cuh` - main FA implementation
- `fattn-mma-f16.cuh` - MMA (Turing+/Ampere+) kernels
- `fattn-wmma-f16.cuh` - WMMA (Volta; RDNA3) kernels
- `template-instances/fattn-tile*.cu` - compile-time template instances
- Compilation flag `GGML_CUDA_FA_ALL_QUANTS` enables all KV quant type combos (slow compilation)

FA kernel strategies by GPU:
- MMA: Turing+ / Ampere+
- WMMA: Volta, RDNA3 (via rocWMMA)
- Tile/vector fallback: everything else

### GGML Block Structure

All existing KV-compatible quant types use block-wise structure:
- **Q4_0**: block of 32 values. 1 FP16 scale + 16 bytes of 4-bit quants = 18 bytes/block
- **Q8_0**: block of 32 values. 1 FP16 scale + 32 bytes of 8-bit quants = 34 bytes/block
- K-quants use super-blocks of 256 values (16 sub-blocks of 16), but these are NOT supported for KV cache

---

## 3. What Adding a New GGML Type Requires

Based on analysis of the NVFP4 PR (#19769) and discussion #5063, here's the file checklist:

### Minimum Files to Touch

| File | What to add |
|------|-------------|
| `ggml/include/ggml.h` | New `GGML_TYPE_TQ3_0` (or similar) in the `ggml_type` enum |
| `ggml/src/ggml-quants.h` | Block struct definition (`block_tq3_0`) |
| `ggml/src/ggml-quants.c` | `quantize_row_tq3_0_ref()`, `dequantize_row_tq3_0()`, `quantize_row_tq3_0()` |
| `ggml/src/ggml.c` | Register in `type_traits` array: block size, type size, quant/dequant function pointers |
| `ggml/src/ggml-common.h` | Block struct (may need to go here for cross-backend visibility) |

### For KV Cache Support Specifically

| File | What to add |
|------|-------------|
| `src/llama-kv-cache.cpp` | Accept new type in `--cache-type-k` / `--cache-type-v` parsing |
| `ggml/src/ggml-cuda/fattn*.cu` | Flash attention dequant kernel for the new type |
| `ggml/src/ggml-cuda/template-instances/` | New template instantiations for FA with new type |
| `ggml/src/ggml-metal.m` | Metal shader support (Apple Silicon) |
| `ggml/src/ggml-cpu/` | CPU fallback implementation |

### For TurboQuant Specifically (Extra Complexity)

| Component | Challenge |
|-----------|-----------|
| **Hadamard transform** | GGML has NO `ggml_hadamard` op. Would need to add a new GGML operation OR bake the rotation into the quantize/dequantize functions. |
| **Random sign flips** | Need to store or deterministically regenerate the random sign vector per cache entry. |
| **Two-stage quant** | MSE quantizer + 1-bit QJL residual is non-standard. Block struct would be unusual. |
| **Rotation at write time** | KV cache quantization happens when tokens are written. Rotation must happen there. Dequantization during attention must include inverse rotation. |

---

## 4. The ikawrakow/ik_llama.cpp Fork Option

### What It Is

A fork by ikawrakow (the original creator of many GGML quant types) with:
- Custom quant types: IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K + _KS variants + Trellis types (IQ1_KT through IQ4_KT)
- Q8_KV: custom 8-bit KV cache type (added Feb 2025)
- Row-interleaved quant packing
- FlashMLA support
- Better CPU/hybrid GPU performance
- 1,038+ contributors (same base as llama.cpp)

### Fork Strategy

There is active community discussion about divergence from mainline llama.cpp ([Discussion #256](https://github.com/ikawrakow/ik_llama.cpp/discussions/256)). Key dynamics:
- ikawrakow has added many innovations that mainline hasn't adopted
- Community frustrated that improvements stay in the fork
- Some innovations (graph parallel / tensor parallel) were independently re-discovered by mainline ~6 weeks later
- The fork is diverging further, not converging

### Pros of Starting in ik_llama.cpp

1. **ikawrakow has already added custom quant types** - the pattern is proven and documented in his codebase
2. **Q8_KV already exists** - shows he's open to KV-specific quant types
3. **Less review friction** - smaller community, one primary maintainer
4. **Per-tensor row scales already supported** - IQ4_KS etc. break the standard block-wise rule
5. **Faster iteration** - can experiment without upstream politics

### Cons

1. **Smaller user base** - most llama.cpp users don't know about ik_llama.cpp
2. **Diverging codebase** - porting to mainline later means rewriting
3. **Single maintainer risk** - if ikawrakow loses interest, fork dies
4. **No guarantee of upstream acceptance** - mainline may never adopt it

### Verdict

**Start in ik_llama.cpp for proof-of-concept, then port to mainline.** ikawrakow's codebase already has the patterns for non-standard quant types. Get it working there first, measure quality, then submit a clean PR to upstream.

---

## 5. Related Work: Rotation-Based KV Quantization

### turboderp / ExLlamaV2

The original inspiration. turboderp's 4-bit KV cache in ExLlamaV2:
- Uses Hadamard transform to smooth KV distribution before quantization
- Q4 cache performs on par with FP16 (better than FP8)
- Per-layer parameter selection minimizing max quantization error
- Benchmarked on The Pile dataset
- Evaluation: [exllamav2/doc/qcache_eval.md](https://github.com/turboderp-org/exllamav2/blob/master/doc/qcache_eval.md)

### KVLinC (arXiv 2510.05373)

Combines Hadamard rotation (reduces V quantization error) with lightweight linear correction adapters that compensate for K quantization errors. More complex than TurboQuant (requires training adapters).

### RotateKV (arXiv 2501.16383, IJCAI 2025)

Achieves 2-bit KV cache quantization via:
1. Outlier-Aware Rotation: channel reordering + FWHT
2. Pre-RoPE Grouped-Head Rotation: mitigates RoPE disruption
3. Attention-Sink-Aware Quantization: protects sink tokens

Results: <0.3 PPL degradation at 2-bit on LLaMA-2-13B. More complex than TurboQuant but achieves lower bit-width.

### KVSplit

A llama.cpp patch for Apple Silicon: different bit-widths for K vs V (e.g., K8V4). 59% memory reduction, <1% quality loss. Validates the asymmetric K/V quantization approach that JohannesGaessler's research also found.

---

## 6. The Hard Parts

### 6a. No Hadamard Transform in GGML

GGML's operation set (documented in `docs/ops.md`) does NOT include a Hadamard transform. Options:

1. **Add a new GGML op** (`GGML_OP_HADAMARD`): Requires implementing across ALL backends (CPU, CUDA, Metal, Vulkan, SYCL, OpenCL). This is a massive undertaking.

2. **Bake rotation into quantize/dequantize**: Apply the Hadamard transform inside the `quantize_row_tq3_0()` function and inverse in `dequantize_row_tq3_0()`. Simpler but:
   - Dequantization happens inside hot FA kernels
   - Adding O(d log d) FHT to every dequant call would hurt throughput
   - The rotation is on the full head dimension vector, not per-block

3. **Fuse rotation into the FA kernel**: Write a custom flash attention kernel that applies inverse-Hadamard during the attention computation. This is what turboderp does in ExLlamaV2.

4. **Pre-rotate at cache write time, store rotated values**: Rotate K/V before quantizing into the cache. During attention, dequantize and apply inverse rotation. This keeps the block quant/dequant simple but adds a post-dequant step.

**Option 4 is most practical.** Rotate before quantize, inverse-rotate after dequantize. The FHT is O(d log d) where d = head_dim (typically 64-128), which is cheap per-token.

### 6b. Block Structure Mismatch

TurboQuant operates on entire vectors (head_dim = 64-128 values). GGML blocks are 32 values. Options:

1. Use block_size = head_dim (non-standard, breaks assumptions like `nb[0] == ggml_type_size(type)`)
2. Use standard 32-value blocks but apply rotation across the full vector before blocking
3. Use super-blocks (like K-quants use 256-value super-blocks)

**Option 2 is safest.** Rotate the full d-dimensional vector, then quantize in standard 32-value blocks using optimal scalar quantizers. The rotation makes each coordinate well-behaved, so per-block quantization works.

### 6c. The QJL Residual Stage

TurboQuant's two-stage approach (MSE quant + 1-bit QJL residual) is novel but complex:
- Doubles the dequant work
- Non-standard block structure (main quant + residual bits)
- May not be worth the complexity for a first PR

**Skip QJL for v1.** Pure rotation + optimal scalar quantization (stage 1 only) still beats naive quantization significantly. Add QJL in a follow-up.

### 6d. Optimal Scalar Quantizer

After rotation, coordinates follow a known Beta distribution. The optimal quantizer for this distribution has non-uniform thresholds (unlike Q4_0's uniform grid). Options:

1. **Lookup table**: Store optimal thresholds/centroids for each bit-width. Small constant memory.
2. **Approximate with uniform**: Rotation already helps a lot. Uniform quantization after rotation may be "good enough" and much simpler.

**Start with uniform post-rotation (basically Q4_0 after Hadamard rotation).** Measure quality. If not sufficient, upgrade to optimal non-uniform thresholds.

---

## 7. Minimum Viable Implementation Plan

### Phase 1: Proof of Concept in ik_llama.cpp (2-3 weeks)

**Goal:** Hadamard-rotated Q4_0 KV cache type, CPU only.

Files to touch:
- `ggml.h` - add `GGML_TYPE_HQ4_0` (Hadamard-Q4_0)
- `ggml-quants.h/c` - block struct (same as Q4_0 + stored sign vector), quantize/dequant with FHT
- `ggml.c` - type_traits registration
- `llama-kv-cache.cpp` - accept `hq4_0` as cache type
- CPU attention path - add dequant support

**What this proves:** Quality improvement from rotation alone, without CUDA kernels.

### Phase 2: CUDA Flash Attention Kernel (2-4 weeks)

**Goal:** Make it fast on GPU.

Files to touch:
- `ggml-cuda/fattn*.cu` - dequant kernel for HQ4_0 (dequant block + inverse FHT)
- Template instances for the new type
- Benchmark against Q4_0 and Q8_0

**The FHT kernel is well-documented.** Dao AI Lab's [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform) provides a reference CUDA implementation.

### Phase 3: Sub-4-bit Variant (2-3 weeks)

**Goal:** HQ3_0 or HQ2_0 - the actual TurboQuant sweet spot.

- New block struct for 3-bit or 2.5-bit per value
- Optimal non-uniform scalar quantizer (lookup table)
- Quality evaluation (perplexity, downstream tasks)

### Phase 4: Upstream PR to llama.cpp (timeline uncertain)

**Goal:** Clean, minimal PR to mainline.

Strategy based on lessons from NVFP4 PR (#19769) and PR rejection patterns:
1. **Separate backend PRs** - CPU first, then CUDA, then Metal
2. **Include KLD analysis + perplexity benchmarks**
3. **Test on small models** (not just 70B)
4. **Don't touch unrelated files**
5. **Expect 2-6 months for merge** based on community patterns

---

## 8. Realistic Timeline Assessment

| Phase | Duration | Blocker? |
|-------|----------|----------|
| Study ik_llama.cpp quant type pattern | 3-5 days | No |
| Implement FHT in C (CPU reference) | 2-3 days | No |
| HQ4_0 block struct + quant/dequant | 3-5 days | No |
| Wire into KV cache path | 2-3 days | No |
| Quality evaluation (perplexity) | 2-3 days | No |
| CUDA FA kernel | 2-4 weeks | Moderate - kernel dev is tricky |
| Sub-4-bit variant | 2-3 weeks | No |
| Upstream PR preparation | 1-2 weeks | No |
| **Upstream review + merge** | **2-6 months** | **Yes - maintainer bandwidth** |

**Total to working PoC: 4-6 weeks.**
**Total to upstream merge: 4-9 months (optimistic).**

---

## 9. Acceptance Risk Assessment

### What Would Get the PR Accepted

1. **Clear quality wins** - perplexity benchmarks showing HQ4_0 >> Q4_0 and HQ3_0 ~ Q4_0 at lower memory
2. **Minimal code surface** - don't touch 50 files. CPU-only first PR, CUDA follow-up
3. **No new GGML ops** - handle rotation inside quant/dequant, not as a new graph op
4. **Standard block structure** - 32-value blocks, familiar patterns
5. **Testing documentation** - KLD analysis, perplexity on multiple models, benchmark suite
6. **Small test models available** - don't require 70B downloads for review

### What Would Get It Rejected

1. **Too many files changed** - the NVFP4 PR got pushback for "ratio of verified information to maintainer-needed work"
2. **No benchmarks** - maintainers will not merge a quant type on theory alone
3. **Breaking changes** - touching core attention paths makes reviewers nervous
4. **Requiring a new GGML op** - adding `GGML_OP_HADAMARD` means 7+ backend implementations
5. **No upstream champion** - if JohannesGaessler or ggerganov aren't interested, it stalls

### Has Anyone Discussed Sub-4-bit KV Cache?

Yes, but primarily in the context of turboderp's work:
- [Discussion #5932](https://github.com/ggml-org/llama.cpp/discussions/5932) - "4-bit KV Cache" discussion references turboderp's Hadamard approach as the gold standard
- [Issue #6863](https://github.com/ggml-org/llama.cpp/issues/6863) - Feature request for 4-bit KV cache, citing UC Berkeley research
- No one has proposed sub-4-bit (2-bit, 3-bit) KV cache for llama.cpp
- The KVSplit project validates asymmetric K8/V4 but doesn't go below 4-bit

**This means TurboQuant's 2.5-3.5 bit KV cache would be genuinely novel in the llama.cpp ecosystem.**

---

## 10. Key Answers

**What's the minimum viable PR?**
CPU-only HQ4_0 (Hadamard-rotated Q4_0) for KV cache. ~8-10 files, no new GGML ops. Add CUDA FA kernel as a separate follow-up PR.

**Can we start with ikawrakow's fork?**
Yes, and we should. ik_llama.cpp already has Q8_KV, per-tensor row scales, and patterns for non-standard quant types. It's the ideal proving ground.

**Realistic timeline?**
Working PoC: 4-6 weeks. Upstream merge: 4-9 months. The kernel work and upstream review are the long poles.

**Has anyone discussed sub-4-bit KV cache?**
Only in reference to turboderp's ExLlamaV2 work. Nobody has proposed implementing it in llama.cpp. TurboQuant at 2.5-3.5 bits would be a first.

**What would make the PR get accepted vs rejected?**
Accepted: clean code, strong benchmarks, minimal file changes, separate PRs per backend. Rejected: monolithic PR, no benchmarks, new GGML ops, too many files.

---

## 11. What Would Make This Contribution Actually Worth Merging

> Added 2026-03-25 based on focused research into llama.cpp merge standards, competitor PRs, and benchmark expectations.

### 11a. llama.cpp Contribution Standards (from CONTRIBUTING.md)

The project has explicit rules:

1. **Code quality**: Follow existing code style. Use clang-format (clang-tools v15+). 4-space indentation, brackets on same line. Avoid fancy STL. Keep it simple.
2. **Testing**: Contributors MUST "execute the full CI locally on their machine before publishing" and "verify that perplexity and performance are not negatively affected."
3. **Perplexity + performance**: Use `llama-perplexity` (WikiText-2) and `llama-bench` as the standard tools.
4. **Review process**: Collaborators own code sections, Maintainers merge after code owner approval. New contributors need an existing collaborator willing to review and maintain the code long-term.
5. **AI-generated code**: Permitted, but must explicitly disclose AI usage and conduct thorough manual review before publishing.

**Takeaway:** A PR without WikiText-2 perplexity numbers and llama-bench throughput numbers will not be reviewed.

### 11b. Minimum Benchmark Evidence for a KV Cache Type PR

Based on how PR #7527 and the KV cache quantization work got merged:

**WikiText-2 perplexity (mandatory):**
- Standard command: `./llama-perplexity -m model.gguf -f wiki.test.raw --cache-type-k TYPE --cache-type-v TYPE`
- Convention: WikiText-2 test set via `scripts/get-wikitext-2.sh`
- Must show: f16 baseline vs new type at multiple model sizes
- Expected quality thresholds from JohannesGaessler's findings:
  - q8_0: +0.002-0.05 PPL (acceptable, near-zero quality loss)
  - q4_0: +0.206-0.25 PPL (noticeable but usable)
  - A new type must beat q4_0 at similar or lower bit-width to justify its existence

**llama-bench throughput (mandatory):**
- Token generation speed (tokens/sec) for prompt processing and generation
- Memory usage comparison (KV cache memory at various sequence lengths)

**What JohannesGaessler's merged work included:**
- Perplexity across multiple quant combinations (K type x V type matrix)
- The finding that K cache is more sensitive than V cache to quantization
- Specific numbers: "q4_0 V + FP16 K is more precise than q6_K weights with FP16 KV"
- No LongBench, no NIAH, no downstream task benchmarks were required for the initial merge

**What would be nice but isn't required for initial merge:**
- LongBench (standard for KV cache compression papers, but not standard for llama.cpp PRs)
- NIAH (Needle in a Haystack) - tests retrieval at various context depths
- SCBench (KV cache-centric long-context evaluation)
- Downstream task scores (GSM8K, BBH, etc.)

### 11c. Model Sizes to Test

Based on llama.cpp community conventions:

| Model Size | Required? | Notes |
|-----------|-----------|-------|
| 1B-3B | Nice to have | Fast iteration, good for development. Not sufficient alone. |
| **7B-8B** | **Yes, mandatory** | The standard benchmark size. Llama-3.1-8B-Instruct is the current gold standard test model. |
| 13B | Nice to have | Shows scaling behavior |
| 70B+ | Not required | Impractical for most reviewers to reproduce |

A recent paper ([arXiv 2601.14277](https://arxiv.org/html/2601.14277v1)) did a unified evaluation of llama.cpp quantization specifically on Llama-3.1-8B-Instruct. That's the benchmark model to beat.

### 11d. CPU-Only vs Full CUDA: What's Acceptable?

**CPU-only IS acceptable as a first PR.** Evidence:

- llama.cpp supports CPU-only builds (`-DGGML_CUDA=OFF`)
- The project explicitly supports multiple backends (CPU, CUDA, Metal, Vulkan, SYCL)
- ikawrakow's IQ types were merged as CPU-first implementations
- JohannesGaessler's PR #7412 was a CUDA-only research demo that got superseded by #7527 which added flash attention kernels

**Strategy that works:**
1. First PR: CPU implementation + quality benchmarks (proves the concept)
2. Second PR: CUDA flash attention kernels (proves the performance)
3. Third PR: Metal support (Apple Silicon reach)

Maintainers accept incremental PRs. A clean CPU-only implementation with strong quality numbers is mergeable.

### 11e. The Competitive Landscape (as of March 2026)

**Active TurboQuant integration efforts:**

1. **mudler's branch** ([github.com/mudler/llama.cpp/tree/feat/turbo-quant](https://github.com/mudler/llama.cpp/tree/feat/turbo-quant)) - builds and quantizes. Under evaluation.
2. **Discussion #20969** - community tracking thread, active interest
3. **Issue #20977** - formal feature request filed
4. **Issue #20979** - research tracking issue

**The race is on.** Multiple people are experimenting within hours of the Google announcement. The window for a first-mover contribution is weeks, not months.

**Key risk:** If mudler (who is a known LocalAI contributor) submits a working PR first, a second TurboQuant PR becomes redundant. The value proposition would shift from "add TurboQuant" to "add a better TurboQuant" (e.g., with optimal non-uniform quantizers, or the QJL residual stage).

### 11f. Has Anyone Submitted a Rotation-Based KV Cache Type Before?

**No.** Specifically in llama.cpp:
- turboderp's Hadamard rotation lives exclusively in ExLlamaV2 (Python/CUDA, completely separate codebase)
- KVLinC exists only as a paper (arXiv 2510.05373), no llama.cpp implementation
- RotateKV exists only as a paper (arXiv 2501.16383)
- KVSplit does asymmetric K8/V4 but with no rotation
- The existing q4_0/q8_0 KV cache types use plain scalar quantization, no preprocessing

**A Hadamard-rotated KV cache type would be a genuine first for llama.cpp.** That novelty is valuable.

### 11g. What Would the Maintainers Actually Want to See?

Based on analyzing merged PRs (especially ikawrakow's IQ types and JohannesGaessler's KV cache work):

**The ideal PR includes:**

1. **A clear problem statement**: "Current q4_0 KV cache adds +0.2 PPL. Hadamard rotation before quantization reduces this to +0.05 PPL at the same bit-width, or achieves q4_0-level quality at 3 bits."

2. **Perplexity table** on Llama-3.1-8B with WikiText-2:
   ```
   Cache Type  | Bits/value | PPL    | Delta vs f16
   f16         | 16.0       | X.XXX  | baseline
   q8_0        | 8.0        | X.XXX  | +0.0XX
   q4_0        | 4.0        | X.XXX  | +0.2XX
   hq4_0       | 4.0        | X.XXX  | +0.0XX  <-- the win
   hq3_0       | 3.0        | X.XXX  | +0.1XX  <-- the real win
   ```

3. **Memory savings table**: KV cache size at 4K/8K/16K/32K context for each type

4. **llama-bench numbers**: tokens/sec comparison showing no throughput regression (or ideally improvement from smaller cache)

5. **Minimal diff**: <500 lines for the core implementation, clean separation of files

6. **Reproducible**: Reviewer can test with a 7B model on a single GPU

**What NOT to include in a first PR:**
- QJL residual stage (too complex, save for follow-up)
- Optimal non-uniform quantizers (uniform-after-rotation is good enough to prove the concept)
- Multi-backend support (CPU first, CUDA later)
- New GGML ops (bake rotation into quant/dequant)

### 11h. The Honest Assessment: Is This Worth Doing?

**Arguments for:**
- Genuine novelty in the llama.cpp ecosystem (first rotation-based KV cache type)
- Strong theoretical backing (ICLR 2026 paper, proven in ExLlamaV2)
- Clear practical value (2x+ context length at same memory, or same context at half memory)
- Community demand is high (3 issues/discussions filed within hours of Google announcement)
- The window is open right now

**Arguments against:**
- mudler is already experimenting with a branch. May beat you to it.
- Google may release official code in Q2 2026, making community implementations redundant
- The maintainers might prefer to wait for Google's official code
- KV cache quantization requires Flash Attention to be enabled (limits usage)
- Compilation time is already a pain point for KV quant type combos
- You need a collaborator willing to own and maintain the code long-term

**Bottom line:** A well-benchmarked HQ4_0 (Hadamard + Q4_0) implementation with strong perplexity numbers on Llama-3.1-8B is the minimum viable contribution. If it clearly beats q4_0 by 0.1+ PPL at the same bit-width, it has a real shot at getting merged. Without those numbers, it's dead on arrival.

---

## Sources

- [TurboQuant paper (arXiv 2504.19874)](https://arxiv.org/abs/2504.19874)
- [TurboQuant ICLR 2026 (OpenReview)](https://openreview.net/forum?id=tO3ASKZlok)
- [Google Research blog: TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [llama.cpp Discussion #5932: 4-bit KV Cache](https://github.com/ggml-org/llama.cpp/discussions/5932)
- [llama.cpp Issue #6863: 4-bit KV cache feature request](https://github.com/ggml-org/llama.cpp/issues/6863)
- [llama.cpp Discussion #5063: Even more quantization types?](https://github.com/ggml-org/llama.cpp/discussions/5063)
- [llama.cpp PR #7527: CUDA quantized KV support for FA](https://github.com/ggml-org/llama.cpp/pull/7527)
- [llama.cpp PR #7412: CUDA quantized KV cache demo](https://github.com/ggml-org/llama.cpp/pull/7412)
- [llama.cpp PR #19769: NVFP4 quantization type](https://github.com/ggml-org/llama.cpp/pull/19769)
- [HuggingFace Discussion: PR#7527 GGUF Quantized KV Support](https://huggingface.co/AetherArchitectural/Community-Discussions/discussions/15)
- [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp/)
- [ik_llama.cpp Discussion #8: New IQn_K quantization types](https://github.com/ikawrakow/ik_llama.cpp/discussions/8)
- [ik_llama.cpp Discussion #256: Diverging from llama.cpp](https://github.com/ikawrakow/ik_llama.cpp/discussions/256)
- [ExLlamaV2 KV cache evaluation](https://github.com/turboderp-org/exllamav2/blob/master/doc/qcache_eval.md)
- [RotateKV paper (arXiv 2501.16383)](https://arxiv.org/abs/2501.16383)
- [KVLinC paper (arXiv 2510.05373)](https://arxiv.org/abs/2510.05373)
- [KVSplit - differentiated KV precision on Apple Silicon](https://github.com/dipampaul17/KVSplit)
- [Dao AI Lab fast-hadamard-transform CUDA](https://github.com/Dao-AILab/fast-hadamard-transform)
- [HadaCore: Tensor Core Accelerated Hadamard Transform Kernel](https://arxiv.org/html/2412.08832v1)
- [Flash Attention and Optimizations (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/7.4-flash-attention-and-optimizations)
- [Quantization Techniques (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/7.3-distributed-inference-and-rpc)
- [Bringing K/V Context Quantisation to Ollama](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)
- [KVSplit on Hacker News](https://news.ycombinator.com/item?id=44009321)
- [llama.cpp CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- [llama.cpp perplexity tool README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)
- [Which Quantization Should I Use? (arXiv 2601.14277)](https://arxiv.org/html/2601.14277v1) - Unified evaluation on Llama-3.1-8B-Instruct
- [PolarQuant paper (arXiv 2502.02617)](https://arxiv.org/abs/2502.02617) - Foundation for TurboQuant's polar coordinate approach
- [TurboQuant paper (arXiv 2504.19874)](https://arxiv.org/abs/2504.19874) - Full paper with pseudocode
- [KVQuant: Towards 10M Context (NeurIPS 2024)](https://github.com/SqueezeAILab/KVQuant) - KIVI baseline for KV quantization
- [HuggingFace blog: KV Cache Quantization](https://huggingface.co/blog/kv-cache-quantization) - Practical quality comparison of int4/int2 KV
- [SCBench: KV Cache-Centric Analysis](https://openreview.net/forum?id=gkUyYcY1W9) - Comprehensive KV evaluation benchmark
- [mudler/llama.cpp feat/turbo-quant branch](https://github.com/mudler/llama.cpp/tree/feat/turbo-quant) - Active experimental implementation
- [llama.cpp Issue #20977: TurboQuant feature request](https://github.com/ggml-org/llama.cpp/issues/20977)
- [llama.cpp Issue #20979: TurboQuant research tracking](https://github.com/ggml-org/llama.cpp/issues/20979)
- [IQ*_K/IQ*_KS merged into llama.cpp mainline (PR #19726)](https://jangwook.net/en/blog/en/llama-cpp-iq-quantization-merge/)
- [Ollama K/V Context Quantisation blog](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)
- [JohannesGaessler's llama.cpp performance testing](https://johannesgaessler.github.io/llamacpp_performance)
- [CommVQ: 87.5% KV cache reduction with 2-bit quantization](https://arxiv.org/html/2506.18879v1)
- [NVIDIA NVFP4 KV cache blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
