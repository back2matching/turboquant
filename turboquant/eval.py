"""
TurboQuant evaluation CLI — perplexity on WikiText-2.

Measures the quality cost of TurboQuant KV-cache compression vs FP16 baseline.
Uses a sliding-window NLL loop (the HuggingFace canonical recipe) with
past_key_values threaded through so the custom cache is actually exercised.

Usage:
    turboquant-eval --model Qwen/Qwen2.5-0.5B-Instruct --no-quant
    turboquant-eval --model Qwen/Qwen2.5-3B-Instruct --bits 4
    turboquant-eval --model Qwen/Qwen2.5-3B-Instruct --bits 4 --max-length 2048 --stride 512

Output: JSON file with token_ppl, word_ppl, timing, environment metadata.
"""

import argparse
import json
import math
import os
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class EvalResult:
    model: str
    config: dict
    results: dict
    timing: dict
    environment: dict


def wikitext_detokenize(text: str) -> str:
    """Undo WikiText BPE-style detokenization artifacts.

    Mirrors lm_eval/tasks/wikitext/preprocess_wikitext.py. Necessary to match
    published word-PPL numbers; skipping this inflates word count by ~3-5%.
    """
    text = text.replace("s '", "s'")
    text = text.replace(" @-@ ", "-")
    text = text.replace(" @,@ ", ",")
    text = text.replace(" @.@ ", ".")
    text = text.replace(" : ", ": ")
    text = text.replace(" ; ", "; ")
    text = text.replace(" . ", ". ")
    text = text.replace(" , ", ", ")
    text = text.replace(" ? ", "? ")
    text = text.replace(" ! ", "! ")
    text = text.replace(" 's", "'s")
    text = text.replace(" 'm", "'m")
    text = text.replace(" 're", "'re")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'll", "'ll")
    text = text.replace(" 'd", "'d")
    text = text.replace(" n't", "n't")
    return text


def load_wikitext2(split: str = "test") -> str:
    """Load WikiText-2 raw text and concatenate all docs in the split."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    return wikitext_detokenize(text)


def _make_cache(use_tq: bool, bits: int, key_bits: Optional[int], value_bits: Optional[int]):
    """Return a fresh cache instance per sliding window (prevents token leakage)."""
    if use_tq:
        from turboquant import TurboQuantCache
        if key_bits is not None or value_bits is not None:
            return TurboQuantCache(bits=bits, key_bits=key_bits, value_bits=value_bits)
        return TurboQuantCache(bits=bits)
    from transformers import DynamicCache
    return DynamicCache()


def evaluate_ppl(model, tokenizer, text: str, max_length: int, stride: int,
                 use_tq: bool, bits: int, key_bits: Optional[int] = None,
                 value_bits: Optional[int] = None, device: str = "cuda") -> dict:
    """Sliding-window NLL loop. Returns {token_ppl, word_ppl, n_tokens, n_words, sum_nll}."""
    import torch

    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    word_count = len(text.split())

    nll_sum = 0.0
    n_tokens_scored = 0
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # the portion not masked out

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # mask the overlap

        cache = _make_cache(use_tq, bits, key_bits, value_bits)
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids, use_cache=True, past_key_values=cache)
            neg_log_likelihood = outputs.loss

        # Count valid target tokens (non -100) for correct averaging
        num_valid = int((target_ids != -100).sum().item())
        nll_sum += neg_log_likelihood.item() * num_valid
        n_tokens_scored += num_valid

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / max(n_tokens_scored, 1)
    token_ppl = math.exp(avg_nll)
    # Word PPL: exp(total_nll / word_count) — published numbers typically use this
    word_ppl = math.exp(nll_sum / max(word_count, 1))

    return {
        "token_ppl": token_ppl,
        "word_ppl": word_ppl,
        "n_tokens": n_tokens_scored,
        "n_words": word_count,
        "sum_nll": nll_sum,
    }


def main():
    import torch

    parser = argparse.ArgumentParser(description="TurboQuant perplexity evaluation")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--bits", type=int, default=4, help="TurboQuant KV bits (ignored if --no-quant)")
    parser.add_argument("--key-bits", type=int, default=None, help="Asymmetric: bits for keys (overrides --bits)")
    parser.add_argument("--value-bits", type=int, default=None, help="Asymmetric: bits for values (overrides --bits)")
    parser.add_argument("--no-quant", action="store_true", help="FP16 baseline — use stock DynamicCache")
    parser.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2"],
                        help="Currently only wikitext-2 supported")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-length", type=int, default=2048, help="Context window for each forward pass")
    parser.add_argument("--stride", type=int, default=512, help="Sliding-window stride (smaller = more overlap = tighter PPL)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Output JSON path (default: results_<model>_<bits>_<dataset>.json)")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_tq = not args.no_quant
    bits = args.bits if use_tq else None
    mode = "turboquant" if use_tq else "baseline"
    bits_desc = (
        f"k{args.key_bits}v{args.value_bits}" if args.key_bits or args.value_bits else
        f"{args.bits}bit" if use_tq else "fp16"
    )

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map=args.device, trust_remote_code=True,
    )
    model.eval()

    print(f"Loading {args.dataset} / {args.split}...")
    text = load_wikitext2(args.split)
    print(f"  {len(text):,} chars, {len(text.split()):,} words")

    print(f"Running PPL — mode={mode}, bits={bits_desc}, max_length={args.max_length}, stride={args.stride}")
    t0 = time.perf_counter()
    results = evaluate_ppl(
        model, tokenizer, text,
        max_length=args.max_length, stride=args.stride,
        use_tq=use_tq, bits=args.bits,
        key_bits=args.key_bits, value_bits=args.value_bits,
        device=args.device,
    )
    wall_s = time.perf_counter() - t0
    results_with_timing = {
        **results,
    }

    # Environment snapshot
    from turboquant import __version__ as tq_version
    env = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "turboquant": tq_version,
        "python": platform.python_version(),
    }

    out = EvalResult(
        model=args.model,
        config={
            "mode": mode,
            "bits": bits,
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "dataset": args.dataset,
            "split": args.split,
            "max_length": args.max_length,
            "stride": args.stride,
        },
        results=results_with_timing,
        timing={
            "wall_seconds": round(wall_s, 2),
            "tokens_per_second": round(results["n_tokens"] / wall_s, 1) if wall_s > 0 else 0,
        },
        environment=env,
    )

    # Default output path
    if args.output is None:
        slug = args.model.replace("/", "-")
        args.output = f"results_{slug}_{bits_desc}_{args.dataset}.json"

    with open(args.output, "w") as f:
        json.dump(asdict(out), f, indent=2)

    print(f"\nResults:")
    print(f"  token PPL: {out.results['token_ppl']:.3f}")
    print(f"  word PPL:  {out.results['word_ppl']:.3f}")
    print(f"  n_tokens:  {out.results['n_tokens']:,}")
    print(f"  wall:      {out.timing['wall_seconds']}s ({out.timing['tokens_per_second']} tok/s)")
    print(f"  saved to:  {args.output}")


if __name__ == "__main__":
    main()
