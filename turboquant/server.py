"""
TurboQuant Inference Server — First TurboQuant-powered inference server anywhere.

OpenAI-compatible API that loads a HuggingFace model with TurboQuant KV cache compression.
FlockRun's HTTP adapter connects unchanged (just point at this server).

Usage:
    python server.py                                          # Default: Qwen2.5-3B-Instruct
    python server.py --model Qwen/Qwen2.5-7B-Instruct        # Custom model
    python server.py --bits 4 --port 8000                     # 4-bit KV, custom port
    python server.py --quantize int8                          # INT8 weight quantization

Endpoints:
    POST /v1/chat/completions    — OpenAI-compatible chat
    GET  /v1/models              — List loaded models
    GET  /health                 — Health check + GPU stats
"""

import torch
import json
import time
import argparse
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Optional
from dataclasses import dataclass

# Local imports
from turboquant.cache import TurboQuantCache

# Lazy-loaded to speed up startup
_model = None
_tokenizer = None
_model_name = ""
_tq_bits = 3
_device = "cuda"


def load_model(model_name: str, quantize: Optional[str] = None):
    """Load model + tokenizer. Supports INT8/INT4 weight quantization."""
    global _model, _tokenizer, _model_name

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if quantize == "int8":
        load_kwargs["load_in_8bit"] = True
        print("  Using INT8 weight quantization (bitsandbytes)")
    elif quantize == "int4":
        load_kwargs["load_in_4bit"] = True
        print("  Using INT4 weight quantization (bitsandbytes)")
    else:
        load_kwargs["dtype"] = torch.float16

    _model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    _model_name = model_name

    vram = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print(f"  Loaded. VRAM: {vram:.0f} MB")


def generate_response(messages: list, max_tokens: int = 512, temperature: float = 0.7,
                       tools: Optional[list] = None, stream: bool = False) -> dict:
    """Generate a chat completion with TurboQuant KV cache."""
    global _model, _tokenizer, _tq_bits

    # Build prompt from messages
    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(_model.device)
    input_len = inputs["input_ids"].shape[1]

    # Create TurboQuant cache
    cache = TurboQuantCache(bits=_tq_bits)

    t0 = time.perf_counter()

    # Generate with TurboQuant KV cache
    with torch.no_grad():
        # Prefill
        outputs = _model(**inputs, use_cache=True, past_key_values=cache)
        past = outputs.past_key_values

        generated_ids = []
        next_logits = outputs.logits[:, -1, :]

        # Apply temperature
        if temperature > 0 and temperature != 1.0:
            next_logits = next_logits / temperature

        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

        # Autoregressive generation
        for _ in range(max_tokens - 1):
            outputs = _model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature

            next_token = next_logits.argmax(dim=-1, keepdim=True)
            token_id = next_token.item()
            generated_ids.append(token_id)

            if token_id == _tokenizer.eos_token_id:
                break

    duration = time.perf_counter() - t0
    output_text = _tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Build OpenAI-format response
    return {
        "id": f"chatcmpl-tq-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output_text,
            },
            "finish_reason": "stop" if generated_ids[-1] == _tokenizer.eos_token_id else "length",
        }],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": len(generated_ids),
            "total_tokens": input_len + len(generated_ids),
        },
        "turboquant": {
            "kv_bits": _tq_bits,
            "generation_time_s": round(duration, 3),
            "tokens_per_sec": round(len(generated_ids) / duration, 1) if duration > 0 else 0,
            "vram_mb": round(torch.cuda.memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0,
        },
    }


class TurboQuantHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the TurboQuant inference server."""

    def log_message(self, fmt, *args):
        """Suppress default logging (too noisy)."""
        pass

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu": torch.cuda.get_device_name(0),
                    "vram_used_mb": round(torch.cuda.memory_allocated() / 1024 / 1024),
                    "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024),
                }
            self._send_json({
                "status": "ok",
                "model": _model_name,
                "kv_bits": _tq_bits,
                **gpu_info,
            })

        elif self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": _model_name,
                    "object": "model",
                    "owned_by": "turboquant-server",
                }],
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json({"error": "not found"}, 404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
        tools = body.get("tools")
        stream = body.get("stream", False)

        if not messages:
            self._send_json({"error": "messages required"}, 400)
            return

        try:
            t0 = time.perf_counter()
            result = generate_response(messages, max_tokens, temperature, tools, stream)
            elapsed = time.perf_counter() - t0
            print(f"  [{result['usage']['completion_tokens']} tok, {elapsed:.1f}s, {result['turboquant']['tokens_per_sec']} tok/s]")
            self._send_json(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            self._send_json({"error": str(e)}, 500)


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Inference Server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--bits", type=int, default=4, help="TurboQuant KV cache bits (3 or 4)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--quantize", choices=["none", "int8", "int4"], default="none", help="Weight quantization")
    args = parser.parse_args()

    global _tq_bits
    _tq_bits = args.bits

    # Load model
    load_model(args.model, quantize=args.quantize if args.quantize != "none" else None)

    # Start server
    server = HTTPServer(("0.0.0.0", args.port), TurboQuantHandler)
    print(f"\nTurboQuant Server running at http://localhost:{args.port}")
    print(f"  Model: {args.model}")
    print(f"  KV cache: TurboQuant {args.bits}-bit")
    print(f"  Endpoints:")
    print(f"    POST /v1/chat/completions  — OpenAI-compatible chat")
    print(f"    GET  /v1/models            — List models")
    print(f"    GET  /health               — Health + GPU stats")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
