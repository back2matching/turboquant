# FlockRun GPU Analysis

> Concrete VRAM math for our RTX 4080 16GB.

## Current Setup

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 4080 16GB |
| Model | qwen3.5:9b (Q4_K_M, 6.6GB) |
| Ollama | FLASH_ATTENTION=1, KV_CACHE_TYPE=q8_0, NUM_PARALLEL=2 |
| VRAM used (model loaded) | 9.1 GB |
| VRAM free | 6.5 GB |
| Agents | 3 (CEO + Eng Ops + Research), 2 on Ollama, 1 on OpenAI OAuth |

## Measured VRAM (Production)

| Config | VRAM | Notes |
|--------|------|-------|
| qwen3.5:9b + FP16 KV | ~8.0 GB | Previous default |
| qwen3.5:9b + q8_0 KV | ~9.1 GB | Current production (includes model + overhead) |
| qwen3.5:27b Q4_K_M | 17+ GB | Does NOT fit. 6.48 tok/s with CPU offload. |

## Measured Benchmarks (TurboQuant)

### Qwen2.5-3B on RTX 4080

| Mode | 13 tok ctx | 1456 tok ctx | Speed |
|------|-----------|-------------|-------|
| FP16 | 5906 MB | 6922 MB | 28-29 tok/s |
| TQ 4-bit | 5901 MB | 6448 MB (-474 MB) | 19-21 tok/s |
| TQ 3-bit | 5901 MB | 6448 MB (-474 MB) | 20-21 tok/s |

### FlockRun Harness (qwen3.5:9b via Ollama)

| KV Config | FlockRun Score | Raw Score | Regression? |
|-----------|---------------|-----------|-------------|
| FP16 (baseline) | 87.4% | 64.6% | — |
| q8_0 (production) | 88.8% | 72.1% | **No regression** |

## 27B Verdict: Not Viable on 16GB

- Ollama's qwen3.5:27b (Q4_K_M, 17GB) exceeds VRAM
- Q3_K_M (13.85GB) fits but leaves only 2GB for KV cache (8-16K context max)
- Real benchmark: 6.48 tok/s with CPU offload (14x slower than 9B)
- TurboQuant can't fix this — the bottleneck is model weight size, not KV cache

## Where TurboQuant Helps Most

1. **Standard transformers (Llama, Mistral)** — 100% attention layers, large KV cache
2. **Long context (32K-128K)** — KV cache dominates VRAM, savings scale linearly
3. **Multi-user serving** — multiple concurrent KV caches on same GPU
4. **FlockRun multi-agent** — 2-4 agents sharing one GPU, each with separate KV cache

For our specific setup (Qwen3.5 hybrid, 20K context, 2 agents), existing Ollama q8_0 is sufficient. TurboQuant adds value when we hit VRAM limits.
