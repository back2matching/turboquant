# ACTIVE PLAN: Produce 8B-Scale TurboQuant Benchmark Data

> Created: 2026-03-26
> Status: COMPLETE — all success criteria met, data published
> Hardware: RTX 4080 16GB, Windows 10

## Progress

- [x] (2026-03-26 12:00) Environment verified: RTX 4080 16GB, PyTorch 2.5.1, CUDA 12.1
- [x] (2026-03-26 12:02) Sanity test passed: 0.5B model, all 3 modes work
- [x] (2026-03-26 12:15) **7B benchmark completed** (460 + 1860 tokens). FP16 at 1.8K exceeds 16GB, TQ-4bit 40% faster under pressure
- [x] (2026-03-26 12:30) **3B context sweep completed** (460, 930, 1860, 3720 tokens). VRAM savings: 42 MB → 955 MB scaling linearly
- [x] (2026-03-26 12:35) benchmark_kv.py upgraded: --context flag, per-model JSON output, combined results
- [x] (2026-03-26 12:36) README.md updated with real benchmark tables and key takeaways
- [x] (2026-03-26 12:45) **Cross-architecture benchmark completed** — StableLM-2-1.6B (Llama gated, Phi-3.5 incompatible cache API). Finding: TQ uses MORE VRAM on StableLM due to dequantized storage overhead. Validates known gotcha.
- [x] (2026-03-26 12:55) **Long-context sweep completed** — 0.5B model, 512-8K tokens. TQ-4bit saves 2 GB at 8K context and is 11% faster. 16K OOM'd for all modes.
- [x] (2026-03-26 13:05) **All results committed.** 4 models, 45 data points. README updated with full tables + cross-architecture analysis.

## Why This Matters

Nobody has published independent TurboQuant KV cache benchmark data at 7-8B model scale. Google's paper tested 8B but that's their data. tonbistudio/turboquant-pytorch (253 stars) tested 3B. Community members on Twitter are asking for independent validation at scale. We have the implementation and the hardware. Data is the deliverable, not code.

## Ecosystem Context

This repo (`turboquant`) is one of 4 related projects under back2matching:

| Repo | What | Status |
|------|------|--------|
| **turboquant** (THIS) | KV cache compression library, HuggingFace drop-in | ACTIVE — run benchmarks |
| kvcache-bench | Ollama KV cache benchmarking (f16/q8_0/q4_0) | PARKED — has RTX 4080 data but only native Ollama types |
| turboquant-vectors | Embedding compression + privacy for vector DBs | PARKED — different domain entirely |
| quant-sim | Model weight quantization benchmarks | DONE |

## Competitive Landscape

- **tonbistudio/turboquant-pytorch**: 253 stars, tested Qwen2.5-3B on RTX 3060. We need to beat this with 7B+ data.
- **NVIDIA/kvpress**: 984 stars, 30+ methods. Research framework, not comparable.
- **Google official code**: Expected Q2 2026. Will make all community implementations secondary. We have a narrow window.
- **turboquant-torch (PyPI)**: Another community impl by nxank4. No published benchmarks.
- **llama.cpp**: Multiple TQ implementations in progress (discussion #20969), none merged. Not our fight.

## What This Repo Actually Has (Verified 2026-03-26)

### Working Code
- `turboquant/core.py` (344 lines) — TurboQuantMSE (Algorithm 1) + TurboQuantIP (Algorithm 2). Correct implementation. Passes MSE bound tests and unbiasedness tests.
- `turboquant/cache.py` (166 lines) — TurboQuantCache, subclasses HuggingFace DynamicCache. Works for manual generation loops. Uses residual window pattern (128 tokens FP16, older tokens quantized).
- `turboquant/cuda_accel.py` (76 lines) — CUDA wrapper with PyTorch fallback. NOT wired into main classes.
- `turboquant/server.py` (263 lines) — OpenAI-compatible HTTP server. Single-threaded stdlib.
- `cuda/turboquant_kernel.cu` (194 lines) — Fused quantize/dequantize CUDA kernels. Must build locally.
- `benchmarks/benchmark_kv.py` (254 lines) — THE BENCHMARK SCRIPT. Tests FP16 vs TQ-4bit vs TQ-3bit.
- `tests/` — 13 tests (8 core + 5 cache)
- Published on PyPI as `turboquant 0.1.0`

### Known Issues That May Affect Benchmarks
1. **Quantized buffer stores dequantized FP16** — cache.py stores the lossy FP16 roundtrip output, not compressed indices. VRAM savings won't be as dramatic as theoretical max. The ARCHITECTURE.md acknowledges this.
2. **CUDA kernel disconnected** — cuda_accel.py exists but core.py always uses PyTorch path. Speed may be slower than necessary.
3. **Global quantizer registry hardcodes seed=42** — if multiple caches created, they share quantizers. Not a benchmark issue.
4. **model.generate() may not work** — benchmark_kv.py already uses manual loop, so this is handled.

### Dependencies
```
torch, numpy, scipy, transformers >= 4.40.0
```

## Phase 1: Run the Benchmark

### Step 1: Verify Environment
```bash
cd C:\Users\PC\Documents\GitHub\turboquant
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.mem_get_info()[1]/1024**3:.1f} GB total')"
```
Expected: RTX 4080, 16.0 GB total

### Step 2: Quick Sanity Test
```bash
python benchmarks/benchmark_kv.py --model Qwen/Qwen2.5-0.5B-Instruct --quick
```
This should complete in <2 minutes. Verify all 3 modes run (FP16, TQ-4bit, TQ-3bit).

### Step 3: Run on 7B Model
```bash
python benchmarks/benchmark_kv.py --model Qwen/Qwen2.5-7B-Instruct
```

**If OOM:** Qwen2.5-7B-Instruct in FP16 is ~14GB. With KV cache overhead it may not fit in 16GB.
- Fallback 1: `--model Qwen/Qwen2.5-3B-Instruct` (safer, still larger than what tonbistudio tested)
- Fallback 2: `--model meta-llama/Llama-3.2-3B-Instruct` (different architecture = more interesting data)
- Fallback 3: Try loading model in bfloat16 instead of float16 (edit benchmark_kv.py line ~145)

**If it works:** Let it run. Full benchmark takes 5-15 minutes depending on model size and prompt length.

### Step 4: Validate Results
Check `benchmarks/benchmark_results.json`:
- All 3 modes should have results (fp16, turboquant-4bit, turboquant-3bit)
- TQ modes should show lower VRAM peak than FP16
- TQ modes should produce coherent (not garbage) output text
- Speed (tok/s) will be slower for TQ due to rotation overhead — that's expected

## Phase 2: Extended Benchmarks (if Phase 1 succeeds)

### Context Length Sweep
Modify `benchmark_kv.py` to test multiple context lengths. The current script uses fixed prompts. Changes needed:
- Add a `--context` flag or modify the prompt list to include a padding prompt that fills to 2K/4K/8K/16K tokens
- For each context length, run FP16 vs TQ-4bit vs TQ-3bit
- This shows WHERE compression matters (it matters more at longer context)

### Second Model Architecture
Run the same benchmark on a different model family:
- If Phase 1 used Qwen, also try Llama-3.2-3B
- Cross-architecture data is more valuable than same-architecture data

### Perplexity Measurement (stretch goal)
The gold standard for KV cache quality measurement:
- Generate text with each cache type
- Measure cross-entropy loss against ground truth
- The paper uses WikiText-2 as the standard

This requires adding a perplexity function to the benchmark script. Estimated ~50 lines of code.

## Phase 3: Publish Results

### In This Repo
1. Commit `benchmarks/benchmark_results.json` (and any extended results)
2. Add a `## Benchmarks` section to README.md with:
   - Hardware specs
   - Results table (VRAM, speed, output quality per mode)
   - Context length sweep chart if available
3. Commit any benchmark script modifications

### Comparison Data
Create a comparison document or JSON that combines:
- TurboQuant results (from this repo's benchmark)
- Ollama native q4_0/q8_0 results (from kvcache-bench repo's existing JSON)
- This becomes the shareable dataset

## What NOT To Do

- **Don't refactor core.py or cache.py.** The code works. Ship data, not code quality.
- **Don't wire the CUDA kernel into main classes** unless benchmarks show unacceptable speed.
- **Don't fix the FP16 buffer issue** unless VRAM data is completely uninteresting.
- **Don't add features to the server.** It's a demo.
- **Don't submit another llama.cpp PR.** Multiple competing implementations exist and Google's code drops soon.
- **Don't try to run 27B+ models.** RTX 4080 maxes out around 7-8B in FP16. That's fine — 8B data is what's missing.
- **Don't compete with tonbistudio on GitHub stars.** Compete on DATA.

## Success Criteria

The benchmark is successful if:
- [x] At least one 3B+ model benchmarked with all 3 modes (FP16, TQ-4bit, TQ-3bit) — **DONE: 7B + 3B**
- [x] TQ modes produce coherent output (not garbage text) — **DONE: coherent code output across all runs**
- [x] VRAM savings are measurable (even if buffer stores FP16, the peak should be lower) — **DONE: 42-955 MB savings**
- [x] Results JSON committed to repo — **DONE: 4 model files + combined JSON**
- [x] README updated with benchmark table — **DONE: two model tables + key takeaways**

The benchmark is worth sharing externally if:
- [x] TQ-3bit or TQ-4bit shows measurable VRAM reduction — **DONE: 955 MB at 4K context (3B), 444 MB at 1.8K (7B)**
- [x] Output quality is comparable to FP16 (not degraded to garbage) — **DONE: coherent across all runs**
- [x] Speed is within 3x of FP16 (rotation overhead is expected but shouldn't be catastrophic) — **DONE: 65-89% of FP16 speed**
- [x] Data exists for at least 2 context lengths showing where compression helps — **DONE: 4 context lengths on 3B**
