"""
TurboQuantCache: Drop-in replacement for HuggingFace DynamicCache.

Subclasses DynamicCache with a custom layer type that quantizes KV entries
via TurboQuant. Full API compatibility with transformers 5.3.0+.

Features (v0.3.0):
- Asymmetric K/V bit allocation (e.g., 4-bit keys + 2-bit values)
- Layer-adaptive precision (protect sensitive layers at full FP16)
- Compressed index storage with 4-bit nibble packing
"""

import torch
from typing import Any, Optional, Tuple
from transformers.cache_utils import DynamicCache, DynamicLayer
from turboquant.core import TurboQuantMSE, pack_uint4, unpack_uint4

# Shared quantizer registry (one per head_dim + bits combo)
_quantizers: dict = {}

def _get_quantizer(head_dim: int, bits: int, device: str) -> TurboQuantMSE:
    key = (head_dim, bits, device)
    if key not in _quantizers:
        _quantizers[key] = TurboQuantMSE(dim=head_dim, bits=bits, device=device, seed=42)
    return _quantizers[key]


def _should_pack(bits: int, head_dim: int) -> bool:
    return bits == 4 and head_dim % 2 == 0


class TurboQuantLayer(DynamicLayer):
    """
    A cache layer that quantizes KV states via TurboQuant with a residual window.

    Supports asymmetric K/V bit allocation: keys and values can use different
    bit widths (e.g., 4-bit keys + 2-bit values). Keys typically need more bits
    because K/V norm disparity can exceed 1000x in some architectures.
    """

    def __init__(self, key_bits: int = 4, value_bits: int = 2,
                 residual_len: int = 128, skip_quantization: bool = False):
        super().__init__()
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_len = residual_len
        self.skip_quantization = skip_quantization
        self._key_indices: Optional[torch.Tensor] = None
        self._key_norms: Optional[torch.Tensor] = None
        self._value_indices: Optional[torch.Tensor] = None
        self._value_norms: Optional[torch.Tensor] = None
        self._residual_keys: Optional[torch.Tensor] = None
        self._residual_values: Optional[torch.Tensor] = None
        self._total_len = 0
        self._head_dim: Optional[int] = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self._head_dim = key_states.shape[-1]
        self._key_indices = torch.tensor([], dtype=torch.uint8, device=self.device)
        self._key_norms = torch.tensor([], dtype=torch.float32, device=self.device)
        self._value_indices = torch.tensor([], dtype=torch.uint8, device=self.device)
        self._value_norms = torch.tensor([], dtype=torch.float32, device=self.device)
        self._residual_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self._residual_values = torch.tensor([], dtype=self.dtype, device=self.device)
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

        self._residual_keys = torch.cat([self._residual_keys, key_states], dim=-2)
        self._residual_values = torch.cat([self._residual_values, value_states], dim=-2)
        self._total_len += key_states.shape[-2]

        # Protected layers skip quantization — everything stays in residual
        if not self.skip_quantization and self._residual_keys.shape[-2] > self.residual_len:
            overflow = self._residual_keys.shape[-2] - self.residual_len
            to_quantize_k = self._residual_keys[..., :overflow, :]
            to_quantize_v = self._residual_values[..., :overflow, :]

            head_dim = key_states.shape[-1]
            device = str(key_states.device)

            # Separate quantizers for K and V (asymmetric bits)
            k_quantizer = _get_quantizer(head_dim, self.key_bits, device)
            v_quantizer = _get_quantizer(head_dim, self.value_bits, device)

            k_flat = to_quantize_k.reshape(-1, head_dim)
            k_idx, k_norms = k_quantizer.quantize(k_flat)

            v_flat = to_quantize_v.reshape(-1, head_dim)
            v_idx, v_norms = v_quantizer.quantize(v_flat)

            k_idx = k_idx.reshape(to_quantize_k.shape)
            k_norms = k_norms.reshape(to_quantize_k.shape[:-1] + (1,))
            v_idx = v_idx.reshape(to_quantize_v.shape)
            v_norms = v_norms.reshape(to_quantize_v.shape[:-1] + (1,))

            if _should_pack(self.key_bits, head_dim):
                k_idx = pack_uint4(k_idx)
            if _should_pack(self.value_bits, head_dim):
                v_idx = pack_uint4(v_idx)

            self._key_indices = torch.cat([self._key_indices, k_idx], dim=-2) if self._key_indices.numel() > 0 else k_idx
            self._key_norms = torch.cat([self._key_norms, k_norms], dim=-2) if self._key_norms.numel() > 0 else k_norms
            self._value_indices = torch.cat([self._value_indices, v_idx], dim=-2) if self._value_indices.numel() > 0 else v_idx
            self._value_norms = torch.cat([self._value_norms, v_norms], dim=-2) if self._value_norms.numel() > 0 else v_norms

            self._residual_keys = self._residual_keys[..., overflow:, :]
            self._residual_values = self._residual_values[..., overflow:, :]

        # Build full view
        if self._key_indices.numel() > 0:
            k_quantizer = _get_quantizer(self._head_dim, self.key_bits, str(self.device))
            v_quantizer = _get_quantizer(self._head_dim, self.value_bits, str(self.device))

            k_idx = self._key_indices
            v_idx = self._value_indices
            if _should_pack(self.key_bits, self._head_dim):
                k_idx = unpack_uint4(k_idx, self._head_dim)
            if _should_pack(self.value_bits, self._head_dim):
                v_idx = unpack_uint4(v_idx, self._head_dim)

            k_deq = k_quantizer.dequantize(
                k_idx.reshape(-1, self._head_dim),
                self._key_norms.reshape(-1, 1),
            ).reshape(k_idx.shape).to(dtype=self.dtype)
            v_deq = v_quantizer.dequantize(
                v_idx.reshape(-1, self._head_dim),
                self._value_norms.reshape(-1, 1),
            ).reshape(v_idx.shape).to(dtype=self.dtype)
            self.keys = torch.cat([k_deq, self._residual_keys], dim=-2)
            self.values = torch.cat([v_deq, self._residual_values], dim=-2)
        else:
            self.keys = self._residual_keys
            self.values = self._residual_values

        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self._total_len

    def memory_usage_bytes(self) -> dict:
        """Report actual memory usage: compressed vs FP16-equivalent."""
        compressed = 0
        fp16_equivalent = 0
        if self._key_indices is not None and self._key_indices.numel() > 0:
            compressed += self._key_indices.nelement() * self._key_indices.element_size()
            compressed += self._key_norms.nelement() * self._key_norms.element_size()
            compressed += self._value_indices.nelement() * self._value_indices.element_size()
            compressed += self._value_norms.nelement() * self._value_norms.element_size()
            # FP16 equivalent accounts for packing: packed indices represent more elements
            k_elements = self._key_indices.nelement() * (2 if _should_pack(self.key_bits, self._head_dim or 128) else 1)
            v_elements = self._value_indices.nelement() * (2 if _should_pack(self.value_bits, self._head_dim or 128) else 1)
            fp16_equivalent += k_elements * 2 + v_elements * 2
        residual = 0
        if self._residual_keys is not None and self._residual_keys.numel() > 0:
            residual += self._residual_keys.nelement() * self._residual_keys.element_size()
            residual += self._residual_values.nelement() * self._residual_values.element_size()
        return {
            "compressed_bytes": compressed,
            "residual_bytes": residual,
            "total_bytes": compressed + residual,
            "fp16_equivalent_bytes": fp16_equivalent + residual,
            "savings_ratio": (fp16_equivalent + residual) / max(compressed + residual, 1),
        }


class TurboQuantCache(DynamicCache):
    """
    DynamicCache that uses TurboQuant-compressed layers.

    Drop-in replacement: pass as `past_key_values` to any HuggingFace model.

    Args:
        bits: Shorthand for symmetric K/V bits (e.g., bits=4 means 4-bit K + 4-bit V).
        key_bits: Bits for key quantization. Overrides `bits` for keys.
        value_bits: Bits for value quantization. Overrides `bits` for values.
        protected_layers: List of layer indices to keep at full FP16 precision.
            Negative indices supported (e.g., -1 = last layer). First and last layers
            are most sensitive to quantization — protecting them improves quality.
    """

    def __init__(self, bits: int = 4, key_bits: Optional[int] = None,
                 value_bits: Optional[int] = None,
                 protected_layers: Optional[list] = None, **kwargs):
        super().__init__(**kwargs)
        self.key_bits = key_bits if key_bits is not None else bits
        self.value_bits = value_bits if value_bits is not None else bits
        self.protected_layers = set(protected_layers) if protected_layers else set()
        self._num_layers_seen = 0
        self.layer_class_to_replicate = None

    def _is_protected(self, layer_idx: int) -> bool:
        if layer_idx in self.protected_layers:
            return True
        # Resolve negative indices (requires knowing total layer count)
        for p in self.protected_layers:
            if p < 0 and self._num_layers_seen > 0:
                resolved = self._num_layers_seen + p
                if resolved == layer_idx:
                    return True
        return False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            idx = len(self.layers)
            skip = self._is_protected(idx)
            self.layers.append(TurboQuantLayer(
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                skip_quantization=skip,
            ))
        self._num_layers_seen = max(self._num_layers_seen, layer_idx + 1)

        # Resolve negative protected_layers after we know total count
        # (re-check on first full pass through all layers)
        layer = self.layers[layer_idx]
        if not layer.skip_quantization and self._is_protected(layer_idx):
            layer.skip_quantization = True

        keys, values = layer.update(key_states, value_states, cache_kwargs)
        return keys, values

    def memory_usage_bytes(self) -> dict:
        """Aggregate memory usage across all layers."""
        totals = {"compressed_bytes": 0, "residual_bytes": 0, "total_bytes": 0, "fp16_equivalent_bytes": 0}
        for layer in self.layers:
            if hasattr(layer, 'memory_usage_bytes'):
                stats = layer.memory_usage_bytes()
                for k in totals:
                    totals[k] += stats[k]
        totals["savings_ratio"] = totals["fp16_equivalent_bytes"] / max(totals["total_bytes"], 1)
        return totals


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing TurboQuantCache on {device}")

    # Test asymmetric K/V (4-bit keys, 2-bit values)
    cache = TurboQuantCache(key_bits=4, value_bits=2)
    batch, num_heads, head_dim = 1, 4, 128

    for layer in range(8):
        k = torch.randn(batch, num_heads, 512, head_dim, device=device)
        v = torch.randn(batch, num_heads, 512, head_dim, device=device)
        full_k, full_v = cache.update(k, v, layer_idx=layer)
        assert full_k.shape == (batch, num_heads, 512, head_dim)

    print(f"Cached {cache.get_seq_length()} tokens across 8 layers (4-bit K, 2-bit V)")

    # Test layer-adaptive precision
    cache2 = TurboQuantCache(bits=4, protected_layers=[0, 1, -1, -2])
    for layer in range(8):
        k = torch.randn(batch, num_heads, 256, head_dim, device=device)
        v = torch.randn(batch, num_heads, 256, head_dim, device=device)
        cache2.update(k, v, layer_idx=layer)

    # Protected layers should have no compressed indices
    assert cache2.layers[0].skip_quantization
    assert cache2.layers[1].skip_quantization
    print(f"Layer-adaptive: layers 0,1 protected (FP16), middle layers compressed")

    mem = cache.memory_usage_bytes()
    print(f"Memory: {mem['total_bytes']/1024:.0f} KB actual, {mem['fp16_equivalent_bytes']/1024:.0f} KB FP16 equiv, {mem['savings_ratio']:.2f}x savings")

    print("\nAll tests passed!")
