# TurboQuant Deep Dive

> Google's KV cache compression algorithm. ICLR 2026, released 2026-03-24.

## Paper

- **arXiv:** [2504.19874](https://arxiv.org/abs/2504.19874)
- **Authors:** Zandieh, Daliri, Hadian, Mirrokni (Google Research / NYU)
- **Three-paper family:** QJL (AAAI 2025) + PolarQuant (AISTATS 2026) + TurboQuant (ICLR 2026)

## Algorithm

**Stage 1 (PolarQuant):** Random rotation via QR decomposition makes each coordinate of a unit vector follow Beta((d-1)/2, (d-1)/2). Near-independent coordinates enable optimal scalar quantization per coordinate.

**Stage 2 (QJL):** 1-bit Quantized Johnson-Lindenstrauss on the residual eliminates quantization bias for inner products. Adds 1 bit per dimension.

## Key Claims vs Our Findings

| Claim | Paper | Our RTX 4080 Results |
|-------|-------|---------------------|
| 6x memory reduction | At 2.5-3 bits on H100 | 474MB saved at 1.5K ctx on 3B (6.8% reduction). Scales with context. |
| 8x attention speedup | H100 with custom CUDA kernels | N/A — no CUDA kernels in our impl. 30% slower due to PyTorch rotation overhead. |
| Zero accuracy loss at 3-bit | Llama-3.1-8B on LongBench | **FALSE for small models.** 3-bit degrades quality on 0.5B and 3B. 4-bit is minimum safe threshold. |
| Data-oblivious (no calibration) | Confirmed | Confirmed. Works out of the box on any model. |

## Code Availability (as of 2026-03-25)

| Component | Code? | License |
|-----------|-------|---------|
| QJL | [github.com/amirzandieh/QJL](https://github.com/amirzandieh/QJL) | Apache-2.0 |
| PolarQuant | [github.com/ericshwu/PolarQuant](https://github.com/ericshwu/PolarQuant) | No license, requires custom Triton |
| TurboQuant | None from Google | N/A |
| Community PyTorch | Repos < 24hrs old, no CUDA kernels | Unlicensed |
| **Our implementation** | `tests/kv-compression/turboquant.py` | FlockRun project |

## Theoretical Bounds

MSE distortion: D_mse <= sqrt(3)*pi/2 * (1/4^b)
- b=1: 0.680, b=2: 0.170, b=3: 0.043, b=4: 0.011

Inner product: unbiased (E[<y, dequant(quant(x))>] = <y, x>)

## Bottom Line

TurboQuant is real and works. The VRAM savings are meaningful at long contexts (32K+). But for our specific setup (Qwen3.5 hybrid arch with tiny KV cache), existing Ollama q8_0 gives most of the benefit for zero engineering effort. TurboQuant becomes compelling for standard transformer models at long context on VRAM-constrained GPUs.

See [SYNTHESIS.md](SYNTHESIS.md) for the full strategy.
