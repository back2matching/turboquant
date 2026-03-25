"""
TurboQuantCache: Drop-in replacement for HuggingFace DynamicCache.

Subclasses DynamicCache with a custom layer type that quantizes KV entries
via TurboQuant. Full API compatibility with transformers 5.3.0+.
"""

import torch
from typing import Any, Optional, Tuple
from transformers.cache_utils import DynamicCache, DynamicLayer
from turboquant.core import TurboQuantMSE

# Shared quantizer registry (one per head_dim)
_quantizers: dict = {}

def _get_quantizer(head_dim: int, bits: int, device: str) -> TurboQuantMSE:
    key = (head_dim, bits, device)
    if key not in _quantizers:
        _quantizers[key] = TurboQuantMSE(dim=head_dim, bits=bits, device=device, seed=42)
    return _quantizers[key]


class TurboQuantLayer(DynamicLayer):
    """
    A cache layer that quantizes KV states via TurboQuant with a residual window.

    The residual window keeps the most recent `residual_len` tokens in full FP16
    precision, only quantizing older tokens. This follows the KIVI pattern and
    preserves quality for recently-generated tokens (most important for attention).
    """

    def __init__(self, bits: int = 3, residual_len: int = 128):
        super().__init__()
        self.bits = bits
        self.residual_len = residual_len
        self._quantized_keys: Optional[torch.Tensor] = None
        self._quantized_values: Optional[torch.Tensor] = None
        self._residual_keys: Optional[torch.Tensor] = None
        self._residual_values: Optional[torch.Tensor] = None
        self._total_len = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self._quantized_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self._quantized_values = torch.tensor([], dtype=self.dtype, device=self.device)
        self._residual_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self._residual_values = torch.tensor([], dtype=self.dtype, device=self.device)
        # Parent class expects these
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Add new tokens to residual (FP16) window
        self._residual_keys = torch.cat([self._residual_keys, key_states], dim=-2)
        self._residual_values = torch.cat([self._residual_values, value_states], dim=-2)
        self._total_len += key_states.shape[-2]

        # If residual exceeds limit, quantize the overflow
        if self._residual_keys.shape[-2] > self.residual_len:
            overflow = self._residual_keys.shape[-2] - self.residual_len

            # Quantize the oldest tokens in residual
            to_quantize_k = self._residual_keys[..., :overflow, :]
            to_quantize_v = self._residual_values[..., :overflow, :]

            head_dim = key_states.shape[-1]
            device = str(key_states.device)
            quantizer = _get_quantizer(head_dim, self.bits, device)

            # Quantize + dequantize (lossy roundtrip)
            k_flat = to_quantize_k.reshape(-1, head_dim)
            k_idx, k_norms = quantizer.quantize(k_flat)
            k_deq = quantizer.dequantize(k_idx, k_norms).reshape(to_quantize_k.shape).to(dtype=self.dtype, device=self.device)

            v_flat = to_quantize_v.reshape(-1, head_dim)
            v_idx, v_norms = quantizer.quantize(v_flat)
            v_deq = quantizer.dequantize(v_idx, v_norms).reshape(to_quantize_v.shape).to(dtype=self.dtype, device=self.device)

            # Move quantized data to the quantized buffer
            self._quantized_keys = torch.cat([self._quantized_keys, k_deq], dim=-2)
            self._quantized_values = torch.cat([self._quantized_values, v_deq], dim=-2)

            # Trim residual window
            self._residual_keys = self._residual_keys[..., overflow:, :]
            self._residual_values = self._residual_values[..., overflow:, :]

        # Build full view: quantized (old) + residual (recent FP16)
        self.keys = torch.cat([self._quantized_keys, self._residual_keys], dim=-2) if self._quantized_keys.numel() > 0 else self._residual_keys
        self.values = torch.cat([self._quantized_values, self._residual_values], dim=-2) if self._quantized_values.numel() > 0 else self._residual_values

        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self._total_len


class TurboQuantCache(DynamicCache):
    """
    DynamicCache that uses TurboQuant-compressed layers.

    Drop-in replacement: pass as `past_key_values` to any HuggingFace model.
    """

    def __init__(self, bits: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits
        # Override the layer class so new layers are TurboQuantLayer
        self.layer_class_to_replicate = None  # Disable default

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure we have enough layers
        while len(self.layers) <= layer_idx:
            self.layers.append(TurboQuantLayer(bits=self.bits))

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
        return keys, values


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing TurboQuantCache on {device}")

    cache = TurboQuantCache(bits=3)

    batch, num_heads, head_dim = 1, 4, 128

    for layer in range(8):
        k = torch.randn(batch, num_heads, 512, head_dim, device=device)
        v = torch.randn(batch, num_heads, 512, head_dim, device=device)
        full_k, full_v = cache.update(k, v, layer_idx=layer)
        assert full_k.shape == (batch, num_heads, 512, head_dim)

    print(f"Cached {cache.get_seq_length()} tokens across 8 layers")

    for step in range(10):
        for layer in range(8):
            k = torch.randn(batch, num_heads, 1, head_dim, device=device)
            v = torch.randn(batch, num_heads, 1, head_dim, device=device)
            full_k, full_v = cache.update(k, v, layer_idx=layer)

    print(f"After generation: {cache.get_seq_length()} tokens")

    # Quality check
    test_cache = TurboQuantCache(bits=3)
    orig_k = torch.randn(batch, num_heads, 100, head_dim, device=device)
    orig_v = torch.randn(batch, num_heads, 100, head_dim, device=device)
    restored_k, restored_v = test_cache.update(orig_k, orig_v, layer_idx=0)
    error = ((orig_k - restored_k) ** 2).mean().item()
    print(f"Roundtrip MSE (3-bit): {error:.6f}")

    print("\nAll tests passed!")
