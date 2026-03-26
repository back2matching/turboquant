# TurboQuant Deep Dive

> Google's KV cache compression algorithm. ICLR 2026, released 2026-03-24.

## Paper

- **arXiv:** [2504.19874](https://arxiv.org/abs/2504.19874)
- **Authors:** Zandieh, Daliri, Hadian, Mirrokni (Google Research / NYU)
- **Three-paper family:** QJL (AAAI 2025) + PolarQuant (AISTATS 2026) + TurboQuant (ICLR 2026)

## Algorithm

**Stage 1 (PolarQuant):** Random rotation via QR decomposition makes each coordinate of a unit vector follow Beta((d-1)/2, (d-1)/2). Near-independent coordinates enable optimal scalar quantization per coordinate.

**Stage 2 (QJL):** 1-bit Quantized Johnson-Lindenstrauss on the residual eliminates quantization bias for inner products. Adds 1 bit per dimension.

## Key Claims vs Our Findings (Updated 2026-03-26 with real data)

| Claim | Paper | Our RTX 4080 Results (45 data points, 4 models) |
|-------|-------|---------------------|
| 6x memory reduction | At 2.5-3 bits on H100 | **479 MB saved at 1.8K ctx on 3B** (6.5%). **2 GB saved at 8K on 0.5B** (15.6%). Scales linearly with context. Note: current impl stores dequantized FP16, not indices — real compression would be larger. |
| 8x attention speedup | H100 with custom CUDA kernels | N/A — no CUDA kernels in our impl. ~30% slower at short contexts, but **11% faster at 8K** and **40% faster at 7B/1.8K** when FP16 hits VRAM pressure. |
| Zero accuracy loss at 3-bit | Llama-3.1-8B on LongBench | **FALSE for small models.** 3-bit degrades on 0.5B (filler repetition). 4-bit is coherent on 3B+. |
| Data-oblivious (no calibration) | Confirmed | Confirmed. Works on Qwen, StableLM — any HF model out of the box. |
| Architecture-agnostic | Implied | **Partially.** Works on Qwen (savings) but StableLM shows TQ uses MORE VRAM due to quantizer overhead. Architecture-dependent allocation patterns matter. |

## Code Availability (as of 2026-03-25)

| Component | Code? | License |
|-----------|-------|---------|
| QJL | [github.com/amirzandieh/QJL](https://github.com/amirzandieh/QJL) | Apache-2.0 |
| PolarQuant | [github.com/ericshwu/PolarQuant](https://github.com/ericshwu/PolarQuant) | No license, requires custom Triton |
| TurboQuant | None from Google | N/A |
| Community PyTorch | Repos < 24hrs old, no CUDA kernels | Unlicensed |
| **Our implementation** | [github.com/back2matching/turboquant](https://github.com/back2matching/turboquant) | Apache-2.0, PyPI `turboquant` |

## Theoretical Bounds

MSE distortion: D_mse <= sqrt(3)*pi/2 * (1/4^b)
- b=1: 0.680, b=2: 0.170, b=3: 0.043, b=4: 0.011

Inner product: unbiased (E[<y, dequant(quant(x))>] = <y, x>)

## Bottom Line

TurboQuant is real and works. VRAM savings are meaningful at 4K+ context (1 GB on 3B, 2 GB on 0.5B at 8K). TQ becomes compelling for standard transformer models at long context on VRAM-constrained GPUs. At 7B scale, FP16 exceeds 16 GB VRAM at 1.8K context while TQ stays under — making TQ the only option for longer contexts without model offloading.

See [SYNTHESIS.md](SYNTHESIS.md) for the full strategy.
