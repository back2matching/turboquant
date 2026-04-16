"""
Benchmark: TurboQuant KV Cache on Real Models

Compares FP16 baseline vs TurboQuant 3-bit and 4-bit KV cache compression
on actual model inference. Measures VRAM, speed, and output quality.

Usage:
    python benchmark_kv.py                    # Full benchmark
    python benchmark_kv.py --quick            # Quick test (fewer tokens)
    python benchmark_kv.py --model qwen2.5-0.5b  # Specific model
"""

import torch
import time
import gc
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# GPU memory tracking
def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def gpu_mem_reserved_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0

def reset_gpu():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@dataclass
class BenchmarkResult:
    model: str
    kv_mode: str  # "fp16", "turboquant-3bit", "turboquant-4bit"
    context_length: int
    generated_tokens: int
    vram_model_mb: float
    vram_peak_mb: float
    vram_kv_estimate_mb: float  # kept for backwards compat; prefer kv_vram_* below
    prefill_time_s: float
    generation_time_s: float
    tokens_per_sec: float
    output_text: str
    perplexity: Optional[float] = None
    kv_vram_fp16_mb: Optional[float] = None        # reference FP16 KV cost
    kv_vram_compressed_mb: Optional[float] = None  # actual quantized-buffer cost
    kv_vram_savings_ratio: Optional[float] = None  # fp16 / compressed (1.00 for fp16 baseline)


def fp16_kv_bytes_from_config(model, seq_len: int) -> int:
    """Compute FP16 KV cache size from model config: L * 2 (K+V) * T * H * d * 2 bytes."""
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", 0)
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", 0)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", 0)
    num_heads = getattr(cfg, "num_attention_heads", num_kv_heads)
    head_dim = hidden_size // num_heads if num_heads else 0
    return num_layers * 2 * seq_len * num_kv_heads * head_dim * 2


def benchmark_fp16(model, tokenizer, prompt: str, max_new_tokens: int, context_length: int) -> BenchmarkResult:
    """Baseline: standard FP16 KV cache."""
    reset_gpu()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_length).to(model.device)
    input_len = inputs['input_ids'].shape[1]
    vram_before = gpu_mem_mb()

    # Prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    t_prefill = time.perf_counter() - t0

    vram_after_prefill = gpu_mem_mb()

    # Generation
    t0 = time.perf_counter()
    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    t_gen = time.perf_counter() - t0

    vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    output_text = tokenizer.decode(gen_outputs[0][input_len:], skip_special_tokens=True)
    gen_tokens = gen_outputs.shape[1] - input_len

    fp16_kv_mb = fp16_kv_bytes_from_config(model, input_len + gen_tokens) / 1e6

    return BenchmarkResult(
        model=model.config._name_or_path,
        kv_mode="fp16",
        context_length=input_len,
        generated_tokens=gen_tokens,
        vram_model_mb=vram_before,
        vram_peak_mb=vram_peak,
        vram_kv_estimate_mb=vram_after_prefill - vram_before,
        prefill_time_s=t_prefill,
        generation_time_s=t_gen,
        tokens_per_sec=gen_tokens / t_gen if t_gen > 0 else 0,
        output_text=output_text[:200],
        kv_vram_fp16_mb=fp16_kv_mb,
        kv_vram_compressed_mb=fp16_kv_mb,
        kv_vram_savings_ratio=1.0,
    )


def benchmark_turboquant(model, tokenizer, prompt: str, max_new_tokens: int, context_length: int, bits: int) -> BenchmarkResult:
    """TurboQuant compressed KV cache."""
    from turboquant import TurboQuantCache

    reset_gpu()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_length).to(model.device)
    input_len = inputs['input_ids'].shape[1]
    vram_before = gpu_mem_mb()

    cache = TurboQuantCache(bits=bits)

    # Prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, past_key_values=cache)
    t_prefill = time.perf_counter() - t0

    vram_after_prefill = gpu_mem_mb()

    # Generation (manual loop since generate() may not work with custom cache)
    t0 = time.perf_counter()
    generated_ids = []
    past = outputs.past_key_values  # This is our TurboQuantCache

    next_token_logits = outputs.logits[:, -1, :]
    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
    generated_ids.append(next_token.item())

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
        past = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    t_gen = time.perf_counter() - t0

    vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Get cache memory stats (exact, from TurboQuantCache tensor storage)
    mem_stats = cache.memory_usage_bytes() if hasattr(cache, 'memory_usage_bytes') else {}
    kv_compressed_mb = mem_stats.get("total_bytes", 0) / 1e6 if mem_stats else None
    kv_fp16_mb = mem_stats.get("fp16_equivalent_bytes", 0) / 1e6 if mem_stats else None
    kv_savings = mem_stats.get("savings_ratio") if mem_stats else None

    return BenchmarkResult(
        model=model.config._name_or_path,
        kv_mode=f"turboquant-{bits}bit",
        context_length=input_len,
        generated_tokens=len(generated_ids),
        vram_model_mb=vram_before,
        vram_peak_mb=vram_peak,
        vram_kv_estimate_mb=vram_after_prefill - vram_before,
        prefill_time_s=t_prefill,
        generation_time_s=t_gen,
        tokens_per_sec=len(generated_ids) / t_gen if t_gen > 0 else 0,
        output_text=output_text[:200],
        kv_vram_fp16_mb=kv_fp16_mb,
        kv_vram_compressed_mb=kv_compressed_mb,
        kv_vram_savings_ratio=kv_savings,
    )


def make_prompt(tokenizer, target_tokens: int) -> str:
    """Build a prompt that tokenizes to approximately target_tokens."""
    question = "\n\nWrite a function to check if a number is prime in Python. Include docstring and examples."
    filler = "The quick brown fox jumps over the lazy dog. "
    q_len = len(tokenizer.encode(question))
    filler_len = len(tokenizer.encode(filler))
    repeats = max(1, (target_tokens - q_len) // filler_len)
    return filler * repeats + question


def run_single_context(model, tokenizer, context_length: int, max_tokens: int, results: list):
    """Run FP16, TQ-4bit, TQ-3bit for a single context length."""
    prompt = make_prompt(tokenizer, context_length)

    print(f"\n{'='*60}")
    print(f"Context target: {context_length} tokens")
    print(f"{'='*60}")

    for mode_name, run_fn in [
        ("FP16 baseline", lambda: benchmark_fp16(model, tokenizer, prompt, max_tokens, context_length)),
        ("TurboQuant 4-bit", lambda: benchmark_turboquant(model, tokenizer, prompt, max_tokens, context_length, bits=4)),
        ("TurboQuant 3-bit", lambda: benchmark_turboquant(model, tokenizer, prompt, max_tokens, context_length, bits=3)),
    ]:
        print(f"\n[{mode_name}]")
        try:
            r = run_fn()
            results.append(r)
            kv_line = f"KV FP16: {r.kv_vram_fp16_mb:.1f} MB" if r.kv_vram_fp16_mb is not None else "KV FP16: -"
            if r.kv_vram_compressed_mb is not None and r.kv_vram_savings_ratio is not None:
                kv_line += f", KV TQ: {r.kv_vram_compressed_mb:.1f} MB ({r.kv_vram_savings_ratio:.2f}x)"
            print(f"  VRAM peak: {r.vram_peak_mb:.0f} MB; {kv_line}")
            print(f"  Speed: {r.tokens_per_sec:.1f} tok/s, Prefill: {r.prefill_time_s:.3f}s")
            print(f"  Output: {r.output_text[:100]}...")
        except Exception as e:
            print(f"  ERROR: {e}")


def run_benchmarks(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", quick: bool = False,
                   context_lengths: list = None):
    """Run full benchmark suite."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    print(f"Model loaded. VRAM: {gpu_mem_mb():.0f} MB")

    max_tokens = 50 if quick else 100
    results = []

    if context_lengths is None:
        context_lengths = [512] if quick else [512, 2048]

    for ctx_len in context_lengths:
        run_single_context(model, tokenizer, ctx_len, max_tokens, results)

    # Save results per model
    model_slug = model_name.split("/")[-1].lower()
    output_path = Path(__file__).parent / f"results_{model_slug}.json"
    with open(output_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also append to combined results file
    combined_path = Path(__file__).parent / "benchmark_results.json"
    existing = []
    if combined_path.exists():
        with open(combined_path) as f:
            existing = json.load(f)
    # Remove old results for this model
    existing = [r for r in existing if r.get('model') != model_name]
    existing.extend([asdict(r) for r in results])
    with open(combined_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Combined results updated: {combined_path}")

    # Summary table — KV FP16 / KV TQ / Savings are the issue-#4 headline columns
    print(f"\n{'='*100}")
    print(f"{'Mode':<22} {'Context':<8} {'Peak VRAM':<11} {'KV FP16':<10} {'KV TQ':<9} {'Savings':<10} {'Speed':<11} {'Prefill':<8}")
    print(f"{'='*100}")
    for r in results:
        fp16 = f"{r.kv_vram_fp16_mb:.1f}" if r.kv_vram_fp16_mb is not None else "-"
        tq = f"{r.kv_vram_compressed_mb:.1f}" if r.kv_vram_compressed_mb is not None else "-"
        savings = f"{r.kv_vram_savings_ratio:.2f}x" if r.kv_vram_savings_ratio is not None else "-"
        print(f"{r.kv_mode:<22} {r.context_length:<8} {r.vram_peak_mb:<11.0f} {fp16:<10} {tq:<9} {savings:<10} {r.tokens_per_sec:<11.1f} {r.prefill_time_s:<8.3f}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--context', type=str, default=None,
                        help='Comma-separated context lengths, e.g. "512,1024,2048,4096"')
    args = parser.parse_args()

    ctx = None
    if args.context:
        ctx = [int(x.strip()) for x in args.context.split(',')]

    run_benchmarks(model_name=args.model, quick=args.quick, context_lengths=ctx)
