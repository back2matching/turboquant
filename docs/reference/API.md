# API Reference

> All public classes and functions in the `turboquant` package.

## Package Exports

```python
from turboquant import TurboQuantMSE, TurboQuantIP, TurboQuantCache
```

Version: `turboquant.__version__` = `"0.1.0"`

---

## turboquant.core

### class TurboQuantMSE

MSE-optimal vector quantization (Algorithm 1 from the paper).

```python
TurboQuantMSE(dim: int, bits: int = 3, device: str = 'cuda', seed: int = 42)
```

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | int | required | Vector dimension. Typically `head_dim` (64, 128, 256). |
| `bits` | int | 3 | Bits per coordinate. Valid: 1, 2, 3, 4. |
| `device` | str | `'cuda'` | `'cuda'` or `'cpu'`. Rotation matrix and codebook live here. |
| `seed` | int | 42 | Random seed for the rotation matrix. Same seed = same rotation. |

**Attributes:**

| Attr | Type | Shape | Description |
|------|------|-------|-------------|
| `rotation` | Tensor | (dim, dim) | Orthogonal rotation matrix Pi |
| `rotation_t` | Tensor | (dim, dim) | Pi transposed (for quantization) |
| `codebook` | Tensor | (2^bits,) | Optimal centroid values on [-1, 1] |
| `dim` | int | -- | Vector dimension |
| `bits` | int | -- | Bits per coordinate |
| `num_centroids` | int | -- | 2^bits |

#### quantize()

```python
quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Quantize vectors. Input need not be unit norm (norms are stored separately).

**Args:**
- `x`: Input vectors, shape `(..., dim)`. Any batch dimensions.

**Returns:**
- `indices`: Quantization indices, shape `(..., dim)`, dtype `uint8` (bits <= 8) or `int16`
- `norms`: Vector norms, shape `(..., 1)`, dtype `float32`

#### dequantize()

```python
dequantize(indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor
```

Reconstruct vectors from quantized representation.

**Args:**
- `indices`: From `quantize()`, shape `(..., dim)`
- `norms`: From `quantize()`, shape `(..., 1)`

**Returns:**
- Reconstructed vectors, shape `(..., dim)`, dtype `float32`

---

### class TurboQuantIP

Inner-product optimal quantization (Algorithm 2). Subclasses `TurboQuantMSE`.

```python
TurboQuantIP(dim: int, bits: int = 3, device: str = 'cuda', seed: int = 42)
```

Same constructor parameters as `TurboQuantMSE`. Internally uses `(bits-1)` bits for MSE stage and 1 bit for QJL correction.

**Additional attributes:**

| Attr | Type | Shape | Description |
|------|------|-------|-------------|
| `total_bits` | int | -- | Total bits per dimension (same as `bits` arg) |
| `S` | Tensor | (dim, dim) | Random Gaussian matrix for JL projection, scaled by 1/sqrt(dim) |

#### quantize()

```python
quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

Two-stage quantization.

**Args:**
- `x`: Input vectors, shape `(..., dim)`

**Returns:**
- `mse_indices`: Stage 1 indices, shape `(..., dim)`, dtype `uint8`
- `norms`: Vector norms, shape `(..., 1)`
- `qjl_signs`: Stage 2 QJL sign bits, shape `(..., dim)`, dtype `uint8` (0 or 1)
- `residual_norms`: Residual norms, shape `(..., 1)`

#### dequantize()

```python
dequantize(mse_indices: torch.Tensor, norms: torch.Tensor,
           qjl_signs: torch.Tensor, residual_norms: torch.Tensor) -> torch.Tensor
```

Two-stage dequantization. Note: signature differs from `TurboQuantMSE.dequantize()`.

**Args:**
- `mse_indices`, `norms`, `qjl_signs`, `residual_norms`: All from `quantize()`

**Returns:**
- Reconstructed vectors, shape `(..., dim)`, dtype `float32`

---

### compute_memory_bytes()

```python
compute_memory_bytes(dim: int, bits: int, n_vectors: int, two_stage: bool = False) -> dict
```

Calculate memory usage for TurboQuant-compressed vectors.

**Args:**

| Param | Type | Description |
|-------|------|-------------|
| `dim` | int | Vector dimension |
| `bits` | int | Bits per element |
| `n_vectors` | int | Number of vectors |
| `two_stage` | bool | If True, compute for TurboQuantIP (MSE + QJL) |

**Returns dict with:**

| Key | Type | Description |
|-----|------|-------------|
| `index_bytes` | float | Bytes for quantization indices |
| `qjl_bytes` | float | Bytes for QJL signs (two_stage only) |
| `norm_bytes` | float | Bytes for vector norms |
| `residual_norm_bytes` | float | Bytes for residual norms (two_stage only) |
| `total_bytes` | float | Total bytes |
| `bits_per_element` | float | Average bits per scalar element |
| `compression_ratio` | float | Compression vs FP16 (16 / bits_per_element) |

---

## turboquant.cache

### class TurboQuantLayer

A cache layer that quantizes KV states with a residual window. Subclasses `transformers.cache_utils.DynamicLayer`.

```python
TurboQuantLayer(bits: int = 3, residual_len: int = 128)
```

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `bits` | int | 3 | Bits per element for quantization |
| `residual_len` | int | 128 | Number of recent tokens to keep in FP16 |

#### update()

```python
update(key_states: torch.Tensor, value_states: torch.Tensor,
       cache_kwargs: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]
```

Add new KV states to the layer. Handles quantization of overflow automatically.

**Args:**
- `key_states`: Shape `(batch, num_heads, seq_len, head_dim)`
- `value_states`: Same shape as `key_states`
- `cache_kwargs`: Unused, for API compatibility

**Returns:**
- `keys`: Full key sequence (quantized old + FP16 recent), same shape as input with accumulated seq_len
- `values`: Same structure as keys

#### get_seq_length()

```python
get_seq_length() -> int
```

Returns total number of cached tokens (quantized + residual).

---

### class TurboQuantCache

Drop-in replacement for HuggingFace's `DynamicCache`. Subclasses `transformers.cache_utils.DynamicCache`.

```python
TurboQuantCache(bits: int = 3, **kwargs)
```

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `bits` | int | 3 | Bits per element, passed to each TurboQuantLayer |
| `**kwargs` | -- | -- | Forwarded to DynamicCache.__init__() |

#### update()

```python
update(key_states: torch.Tensor, value_states: torch.Tensor,
       layer_idx: int, cache_kwargs: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]
```

Update cache for a specific transformer layer. Creates new TurboQuantLayer instances as needed.

**Args:**
- `key_states`: Shape `(batch, num_heads, seq_len, head_dim)`
- `value_states`: Same shape
- `layer_idx`: Transformer layer index (0-based)
- `cache_kwargs`: Forwarded to layer

**Returns:**
- `keys`, `values`: Full accumulated sequences for this layer

**Usage:**

```python
from turboquant import TurboQuantCache

cache = TurboQuantCache(bits=4)
outputs = model(**inputs, past_key_values=cache, use_cache=True)
# cache is updated in-place by the model
```

---

## turboquant.cuda_accel

### is_cuda_available()

```python
is_cuda_available() -> bool
```

Returns `True` if the `cuda_turboquant` extension is built and importable.

### cuda_quantize()

```python
cuda_quantize(x: torch.Tensor, rotation_t: torch.Tensor, codebook: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor]
```

CUDA-accelerated quantization. Falls back to PyTorch if CUDA extension unavailable or input is on CPU.

**Args:**
- `x`: Input vectors, shape `(N, D)`
- `rotation_t`: Rotation matrix transposed, shape `(D, D)`
- `codebook`: Centroid values, shape `(C,)`

**Returns:**
- `indices`: Shape `(N, D)`, dtype `uint8`
- `norms`: Shape `(N, 1)`, dtype `float32`

### cuda_dequantize()

```python
cuda_dequantize(indices: torch.Tensor, norms: torch.Tensor,
                rotation: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor
```

CUDA-accelerated dequantization. Falls back to PyTorch if unavailable.

**Args:**
- `indices`: Shape `(N, D)`, dtype `uint8`
- `norms`: Shape `(N, 1)`
- `rotation`: Rotation matrix (not transposed), shape `(D, D)`
- `codebook`: Centroid values, shape `(C,)`

**Returns:**
- Reconstructed vectors, shape `(N, D)`, dtype `float32`

---

## turboquant.server

### load_model()

```python
load_model(model_name: str, quantize: Optional[str] = None) -> None
```

Load a HuggingFace model and tokenizer into global state.

**Args:**
- `model_name`: HuggingFace model ID (e.g., `"Qwen/Qwen2.5-3B-Instruct"`)
- `quantize`: `"int8"`, `"int4"`, or `None` for FP16

### generate_response()

```python
generate_response(messages: list, max_tokens: int = 512, temperature: float = 0.7,
                   tools: Optional[list] = None, stream: bool = False) -> dict
```

Generate a chat completion using TurboQuant KV cache.

**Args:**
- `messages`: OpenAI-format message list (`[{"role": "user", "content": "..."}]`)
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (applied as logit division)
- `tools`: Accepted but not executed
- `stream`: Accepted but not implemented (always returns full response)

**Returns:** OpenAI-format completion dict with extra `turboquant` field:

```json
{
  "id": "chatcmpl-tq-...",
  "choices": [{"message": {"role": "assistant", "content": "..."}}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
  "turboquant": {"kv_bits": 4, "generation_time_s": 1.2, "tokens_per_sec": 41.7, "vram_mb": 3200}
}
```

### class TurboQuantHandler

HTTP request handler. Subclasses `http.server.BaseHTTPRequestHandler`.

Routes: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `OPTIONS *`.

### main()

```python
main() -> None
```

CLI entry point. Parses args and starts the server.

**CLI args:**
- `--model`: HuggingFace model ID (default: `Qwen/Qwen2.5-3B-Instruct`)
- `--bits`: KV cache bit width (default: 4)
- `--port`: Server port (default: 8000)
- `--quantize`: Weight quantization (`none`, `int8`, `int4`)
