# ExecPlan: TurboQuant llama.cpp Integration

> Get TQ4_0 merged into llama.cpp with real evidence. No rushing. No premature submissions.

**Created:** 2026-03-25 | **Revised:** 2026-03-25 (post-mortem rewrite)
**Research:** [docs/research/kv-cache-compression/](../research/kv-cache-compression/)

---

## Lesson Learned

PR #20995 was submitted prematurely (30% slower, tiny models, no WikiText-2, no llama-bench). Closed. We will not submit again until:
1. Real benchmarks on Llama-3.1-8B with WikiText-2 perplexity
2. llama-bench throughput comparison
3. Agent team review confirms results are significant
4. User approves the submission

---

## What We Have (Verified Working)

- TQ4_0 type registered in upstream llama.cpp (15 files, 782 lines)
- CPU quantize/dequant with per-head rotation (calls real Q4_0 functions)
- Correct text output with real TurboQuant rotation
- Perplexity: TQ4_0 better than Q4_0 on TinyLlama (2.739 vs 2.989) and Llama-3.2-1B (1.728 vs 1.754)
- Built with CUDA (model on GPU, KV on CPU via --no-kv-offload)
- Branch: `back2matching/llama.cpp:tq4_0-kv-cache`

## What We Don't Have (Must Get Before Resubmitting)

- ⬜ WikiText-2 perplexity on Llama-3.1-8B (the standard llama.cpp benchmark)
- ⬜ llama-bench throughput numbers
- ⬜ Memory footprint comparison at multiple context lengths
- ⬜ Agent team verification that results are significant
- ⬜ User approval to submit

---

## Phase 1: Get Real Benchmarks (2-3 days)

### 1.1 Download Standard Test Data + Model

- ⬜ Download WikiText-2 raw test set (standard for llama.cpp perplexity)
- ⬜ Download Llama-3.1-8B-Instruct Q4_0 GGUF (~4.7 GB)
- ⬜ Verify both work with our build (`--cache-type-k f16` baseline)

### 1.2 WikiText-2 Perplexity Matrix

Run on Llama-3.1-8B with ctx=512 and ctx=2048:

| Config | What It Proves |
|--------|---------------|
| `--cache-type-k f16 --cache-type-v f16` | Baseline |
| `--cache-type-k q8_0 --cache-type-v f16` | Current best practice |
| `--cache-type-k q4_0 --cache-type-v f16` | Existing 4-bit baseline |
| `--cache-type-k tq4_0 --cache-type-v f16` | Our contribution |

All with `--no-kv-offload -ngl 99` (model on GPU, KV on CPU).

- ⬜ Run all 4 configs at ctx=512
- ⬜ Run all 4 configs at ctx=2048
- ⬜ Record exact perplexity numbers + standard deviations

### 1.3 llama-bench Throughput

- ⬜ Build llama-bench tool
- ⬜ Run throughput comparison for same 4 configs
- ⬜ Record: prompt processing tok/s, generation tok/s, peak memory

### 1.4 Memory Footprint

- ⬜ Record VRAM usage at 4K, 8K, 16K context for each config
- ⬜ Calculate KV cache size savings

### Success Criteria
- TQ4_0 perplexity is measurably better than Q4_0 on Llama-3.1-8B
- Speed penalty documented honestly (expected ~30% slower due to CPU rotation)
- All numbers reproducible with exact commands

---

## Phase 2: Agent Team Review (1 day)

### 2.1 Present Results to Eng Ops

- ⬜ Send full benchmark data to Engineering Ops agent
- ⬜ Ask: "Are these results significant enough to submit a PR?"
- ⬜ Ask: "What are the dealbreakers?"
- ⬜ Ask: "How does this compare to what other KV cache PRs showed?"

### 2.2 Competitive Check

- ⬜ Search for any new TurboQuant implementations since our last check
- ⬜ Check mudler's branch status (github.com/mudler/llama.cpp/tree/feat/turbo-quant)
- ⬜ Check if Google released code
- ⬜ Verify we're still first with working rotation-based KV cache

### 2.3 User Review

- ⬜ Present synthesized findings to user
- ⬜ Get explicit approval before submitting

### Gate: DO NOT PROCEED TO PHASE 3 WITHOUT USER APPROVAL

---

## Phase 3: Clean PR Submission (1-2 days)

### 3.1 Code Cleanup

- ⬜ Remove debug logging from tq_quants.c
- ⬜ Remove unused CUDA files (tq4-rotate.cu, tq4-set-rows.cu) — these are for future CUDA-native PR
- ⬜ Add code comments explaining the rotation algorithm
- ⬜ Ensure minimal diff (target <500 lines core)

### 3.2 Write PR Description

- ⬜ Problem statement with numbers
- ⬜ Perplexity table (Llama-3.1-8B, WikiText-2)
- ⬜ Throughput table (llama-bench)
- ⬜ Memory table
- ⬜ Usage example
- ⬜ Limitations section (honest)
- ⬜ Paper reference

### 3.3 Submit

- ⬜ Push clean branch
- ⬜ Create PR to ggml-org/llama.cpp
- ⬜ Frame as: "CPU-first, CUDA kernel in follow-up PR"

---

## Phase 4: Follow-up (Post-Merge)

Only after Phase 3 PR is merged or accepted:

- ⬜ CUDA rotation kernel PR (eliminates CPU fallback overhead)
- ⬜ Flash Attention support for TQ4_0
- ⬜ Ollama integration (expose tq4_0 via OLLAMA_KV_CACHE_TYPE)
- ⬜ Update turboquant PyPI package with llama.cpp integration guide
- ⬜ Blog post with benchmark data

---

## Existing Assets (From Previous Work)

| Asset | Status | Notes |
|-------|--------|-------|
| turboquant PyPI | Published (0.1.0) | Thin, 13 tests. HF DynamicCache + server. |
| kvcache-bench PyPI | Published (0.1.0) | Zero tests. Charts work. Needs cleanup. |
| FlockRun GPU monitoring | Shipped | gpu.ts + API + VRAM-aware heartbeat |
| Production q8_0 KV | Running | 88.8% harness accuracy |
| llama-cpp-upstream/ | Local | TQ4_0 branch with working code |
| Research docs (6 files) | Reference quality | SYNTHESIS, DEEP-DIVE, LANDSCAPE, GPU-ANALYSIS, PORT-DESIGN, INFERENCE-SERVER |

---

## Timeline

| Phase | Est. Time | Blocked On |
|-------|-----------|-----------|
| Phase 1 (benchmarks) | 2-3 days | Llama-3.1-8B download (~4.7 GB) |
| Phase 2 (review) | 1 day | Phase 1 results |
| Phase 3 (submit) | 1-2 days | User approval |
| Phase 4 (follow-up) | Weeks | PR merge |
