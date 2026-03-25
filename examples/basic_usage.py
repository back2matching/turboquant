"""
Basic usage: Drop TurboQuant into any HuggingFace model.

This example loads a model, creates a TurboQuantCache, and generates text
with compressed KV cache. That's it. Three lines to add TurboQuant.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantCache
import torch

# 1. Load any HuggingFace model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, device_map="auto"
)

# 2. Create TurboQuant cache (this is the only change!)
cache = TurboQuantCache(bits=4)  # 4-bit KV compression

# 3. Generate with compressed KV cache
messages = [{"role": "user", "content": "Write a haiku about GPU memory."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, past_key_values=cache, use_cache=True)
    # Continue generating token by token
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]

    for _ in range(50):
        outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(generated, skip_special_tokens=True))
