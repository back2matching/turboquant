"""
TurboQuantStaticCache: Static variant of TurboQuantCache.

Pre-allocates all buffers on creation. No memory growth during generation.
Based on transformers StaticCache: fixed-size tensors, in-place updates.

Usage:
    from turboquant import TurboQuantStaticCache
    cache = TurboQuantStaticCache(config=model.config, max_cache_len=2048, bits=4)
    outputs = model.generate(..., past_key_values=cache)
"""

import torch
from typing import Optional, Tuple
from transformers.cache_utils import StaticCache, Cache, CacheLayerMixin
from turboquant.core import pack_uint4, unpack_uint4
from turboquant.cache import _get_quantizer


class TurboQuantStaticLayer(CacheLayerMixin):
    """
    Static cache layer with pre-allocated TurboQuant compressed storage.

    All buffers are allocated once during lazy initialization and never grow.
    Returns full max_cache_len-sized tensors (zero-padded beyond current length),
    matching StaticLayer's contract for attention mask compatibility.

    Internal layout:
        [quantized tokens (dequantized into output) | residual FP16 tokens | zeros...]
    """

    is_compileable = False
    is_sliding = False

    def __init__(self, max_cache_len: int, bits: int = 4, residual_len: int = 128):
        super().__init__()
        self.max_cache_len = max_cache_len
        self.bits = bits
        self.residual_len = min(residual_len, max_cache_len)
        self.max_quantized_len = max_cache_len - self.residual_len
        self.cumulative_length = torch.tensor([0], dtype=torch.int64)
        self._quantized_count = 0
        self._residual_count = 0
        self._head_dim: Optional[int] = None
        self._packed_dim: Optional[int] = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.max_batch_size, self.num_heads = key_states.shape[:2]
        self._head_dim = key_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]
        self.v_head_dim = value_states.shape[-1]
        self._packed_dim = self._head_dim // 2 if (self.bits == 4 and self._head_dim % 2 == 0) else self._head_dim

        # Compressed index + norm storage (pre-allocated at max quantized capacity)
        if self.max_quantized_len > 0:
            self._key_indices = torch.zeros(
                self.max_batch_size, self.num_heads, self.max_quantized_len, self._packed_dim,
                dtype=torch.uint8, device=self.device,
            )
            self._key_norms = torch.zeros(
                self.max_batch_size, self.num_heads, self.max_quantized_len, 1,
                dtype=torch.float32, device=self.device,
            )
            self._value_indices = torch.zeros(
                self.max_batch_size, self.num_heads, self.max_quantized_len, self._packed_dim,
                dtype=torch.uint8, device=self.device,
            )
            self._value_norms = torch.zeros(
                self.max_batch_size, self.num_heads, self.max_quantized_len, 1,
                dtype=torch.float32, device=self.device,
            )
        else:
            self._key_indices = None
            self._key_norms = None
            self._value_indices = None
            self._value_norms = None

        # Residual FP16 buffer (pre-allocated at residual capacity)
        if self.residual_len > 0:
            self._residual_keys = torch.zeros(
                self.max_batch_size, self.num_heads, self.residual_len, self._head_dim,
                dtype=self.dtype, device=self.device,
            )
            self._residual_values = torch.zeros(
                self.max_batch_size, self.num_heads, self.residual_len, self._head_dim,
                dtype=self.dtype, device=self.device,
            )
        else:
            self._residual_keys = None
            self._residual_values = None

        # Output scratch buffers (what attention sees — full max_cache_len size)
        self.keys = torch.zeros(
            self.max_batch_size, self.num_heads, self.max_cache_len, self._head_dim,
            dtype=self.dtype, device=self.device,
        )
        self.values = torch.zeros(
            self.max_batch_size, self.num_heads, self.max_cache_len, self._head_dim,
            dtype=self.dtype, device=self.device,
        )

        self.cumulative_length = self.cumulative_length.to(self.device)
        self.is_initialized = True

    def _quantize_and_store(self, keys: torch.Tensor, values: torch.Tensor) -> int:
        """Quantize tokens and write into pre-allocated compressed buffers."""
        count = keys.shape[-2]
        if count == 0 or self._key_indices is None:
            return 0

        quantizer = _get_quantizer(self._head_dim, self.bits, str(self.device))

        k_flat = keys.reshape(-1, self._head_dim)
        k_idx, k_norms = quantizer.quantize(k_flat)
        k_idx = k_idx.reshape(keys.shape)
        k_norms = k_norms.reshape(keys.shape[:-1] + (1,))

        v_flat = values.reshape(-1, self._head_dim)
        v_idx, v_norms = quantizer.quantize(v_flat)
        v_idx = v_idx.reshape(values.shape)
        v_norms = v_norms.reshape(values.shape[:-1] + (1,))

        if self.bits == 4 and self._head_dim % 2 == 0:
            k_idx = pack_uint4(k_idx)
            v_idx = pack_uint4(v_idx)

        start = self._quantized_count
        end = start + count
        self._key_indices[:, :, start:end, :] = k_idx
        self._key_norms[:, :, start:end, :] = k_norms
        self._value_indices[:, :, start:end, :] = v_idx
        self._value_norms[:, :, start:end, :] = v_norms

        self._quantized_count += count
        return count

    def _dequantize_range_to_output(self, start: int, end: int) -> None:
        """Dequantize a range of compressed tokens into the output buffer."""
        if start >= end or self._key_indices is None:
            return

        quantizer = _get_quantizer(self._head_dim, self.bits, str(self.device))

        k_idx = self._key_indices[:, :, start:end, :]
        v_idx = self._value_indices[:, :, start:end, :]

        if self.bits == 4 and self._head_dim % 2 == 0:
            k_idx = unpack_uint4(k_idx, self._head_dim)
            v_idx = unpack_uint4(v_idx, self._head_dim)

        k_deq = quantizer.dequantize(
            k_idx.reshape(-1, self._head_dim),
            self._key_norms[:, :, start:end, :].reshape(-1, 1),
        ).reshape(k_idx.shape[:-1] + (self._head_dim,)).to(self.dtype)

        v_deq = quantizer.dequantize(
            v_idx.reshape(-1, self._head_dim),
            self._value_norms[:, :, start:end, :].reshape(-1, 1),
        ).reshape(v_idx.shape[:-1] + (self._head_dim,)).to(self.dtype)

        self.keys[:, :, start:end, :] = k_deq
        self.values[:, :, start:end, :] = v_deq

    def _write_residual_to_output(self) -> None:
        """Write current residual tokens into output buffer, zero-pad the rest."""
        # Clear everything from quantized_count onwards (residual region + padding)
        self.keys[:, :, self._quantized_count:, :] = 0
        self.values[:, :, self._quantized_count:, :] = 0

        if self._residual_count > 0:
            res_start = self._quantized_count
            res_end = res_start + self._residual_count
            self.keys[:, :, res_start:res_end, :] = self._residual_keys[:, :, :self._residual_count, :]
            self.values[:, :, res_start:res_end, :] = self._residual_values[:, :, :self._residual_count, :]

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        kv_len = key_states.shape[-2]
        total = self._residual_count + kv_len

        if total <= self.residual_len:
            # All new tokens fit in residual — no quantization needed
            dst_start = self._residual_count
            dst_end = dst_start + kv_len
            self._residual_keys[:, :, dst_start:dst_end, :] = key_states
            self._residual_values[:, :, dst_start:dst_end, :] = value_states

            # Write directly to output buffer (append after existing content)
            out_start = self._quantized_count + self._residual_count
            self.keys[:, :, out_start:out_start + kv_len, :] = key_states
            self.values[:, :, out_start:out_start + kv_len, :] = value_states

            self._residual_count = total
        else:
            # Overflow: quantize oldest tokens to make room in residual
            overflow = total - self.residual_len
            from_old = min(self._residual_count, overflow)
            from_new = overflow - from_old

            quantize_start = self._quantized_count

            # Quantize oldest from existing residual
            if from_old > 0:
                self._quantize_and_store(
                    self._residual_keys[:, :, :from_old, :],
                    self._residual_values[:, :, :from_old, :],
                )

            # Quantize oldest from new input
            if from_new > 0:
                self._quantize_and_store(
                    key_states[:, :, :from_new, :],
                    value_states[:, :, :from_new, :],
                )

            # Rebuild residual: [leftover old | leftover new]
            leftover_old = self._residual_count - from_old
            leftover_new = kv_len - from_new

            if leftover_old > 0:
                self._residual_keys[:, :, :leftover_old, :] = \
                    self._residual_keys[:, :, from_old:self._residual_count, :].clone()
                self._residual_values[:, :, :leftover_old, :] = \
                    self._residual_values[:, :, from_old:self._residual_count, :].clone()

            if leftover_new > 0:
                self._residual_keys[:, :, leftover_old:leftover_old + leftover_new, :] = \
                    key_states[:, :, from_new:, :]
                self._residual_values[:, :, leftover_old:leftover_old + leftover_new, :] = \
                    value_states[:, :, from_new:, :]

            self._residual_count = self.residual_len

            # Update output: dequantize only new compressed tokens, rewrite residual
            self._dequantize_range_to_output(quantize_start, self._quantized_count)
            self._write_residual_to_output()

        self.cumulative_length.add_(kv_len)
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        return self.max_cache_len, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length if self.is_initialized else 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        super().reset()
        if self.is_initialized:
            if self._key_indices is not None:
                self._key_indices.zero_()
                self._key_norms.zero_()
                self._value_indices.zero_()
                self._value_norms.zero_()
            if self._residual_keys is not None:
                self._residual_keys.zero_()
                self._residual_values.zero_()
            self._quantized_count = 0
            self._residual_count = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder all internal buffers for beam search."""
        if not self.is_initialized:
            return
        self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
        self.values = self.values.index_select(0, beam_idx.to(self.values.device))
        if self._key_indices is not None:
            self._key_indices = self._key_indices.index_select(0, beam_idx.to(self._key_indices.device))
            self._key_norms = self._key_norms.index_select(0, beam_idx.to(self._key_norms.device))
            self._value_indices = self._value_indices.index_select(0, beam_idx.to(self._value_indices.device))
            self._value_norms = self._value_norms.index_select(0, beam_idx.to(self._value_norms.device))
        if self._residual_keys is not None:
            self._residual_keys = self._residual_keys.index_select(0, beam_idx.to(self._residual_keys.device))
            self._residual_values = self._residual_values.index_select(0, beam_idx.to(self._residual_values.device))

    def memory_usage_bytes(self) -> dict:
        """Report allocated memory vs StaticCache equivalent."""
        if not self.is_initialized:
            return {"allocated_bytes": 0, "compressed_backing_bytes": 0,
                    "output_buffer_bytes": 0, "static_cache_equivalent_bytes": 0}

        idx_bytes = 0
        norm_bytes = 0
        if self._key_indices is not None:
            idx_bytes = (self._key_indices.nelement() + self._value_indices.nelement())
            norm_bytes = (self._key_norms.nelement() + self._value_norms.nelement()) * 4

        res_bytes = 0
        if self._residual_keys is not None:
            res_bytes = (self._residual_keys.nelement() + self._residual_values.nelement()) * self._residual_keys.element_size()

        out_bytes = (self.keys.nelement() + self.values.nelement()) * self.keys.element_size()
        compressed_backing = idx_bytes + norm_bytes + res_bytes
        total = compressed_backing + out_bytes
        static_equiv = out_bytes  # vanilla StaticCache only has the output buffers

        return {
            "allocated_bytes": total,
            "compressed_backing_bytes": compressed_backing,
            "output_buffer_bytes": out_bytes,
            "static_cache_equivalent_bytes": static_equiv,
        }


class TurboQuantStaticCache(StaticCache):
    """
    StaticCache with TurboQuant compression for older tokens.

    All memory is allocated once at initialization (lazily on first update).
    No dynamic growth during generation — predictable VRAM budget.

    Drop-in replacement: pass as `past_key_values` to any HuggingFace model.

    Args:
        config: Model config (PreTrainedConfig). Required if num_layers is not set.
        max_cache_len: Maximum number of tokens the cache can hold.
        bits: Quantization bits per coordinate (2, 3, or 4).
        residual_len: Number of most recent tokens kept in full FP16 precision.
        num_layers: Number of model layers. Alternative to passing config.

    Example:
        >>> cache = TurboQuantStaticCache(config=model.config, max_cache_len=2048, bits=4)
        >>> outputs = model.generate(inputs, past_key_values=cache)

        >>> # Or without a model config:
        >>> cache = TurboQuantStaticCache(max_cache_len=2048, bits=4, num_layers=32)
    """

    def __init__(
        self,
        config=None,
        max_cache_len: int = 1024,
        bits: int = 4,
        residual_len: int = 128,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        if config is not None:
            config_text = config.get_text_config(decoder=True)
            n_layers = config_text.num_hidden_layers
            if hasattr(config_text, "num_kv_shared_layers"):
                n_layers -= config_text.num_kv_shared_layers
        elif num_layers is not None:
            n_layers = num_layers
        else:
            raise ValueError("Either config or num_layers must be provided")

        layers = [
            TurboQuantStaticLayer(max_cache_len=max_cache_len, bits=bits, residual_len=residual_len)
            for _ in range(n_layers)
        ]
        # Call Cache.__init__ directly (skip StaticCache's layer construction)
        Cache.__init__(self, layers=layers)
        self.bits = bits

    def memory_usage_bytes(self) -> dict:
        """Aggregate memory usage across all layers."""
        totals = {"allocated_bytes": 0, "compressed_backing_bytes": 0,
                  "output_buffer_bytes": 0, "static_cache_equivalent_bytes": 0}
        for layer in self.layers:
            if hasattr(layer, 'memory_usage_bytes'):
                stats = layer.memory_usage_bytes()
                for k in totals:
                    totals[k] += stats[k]
        return totals


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing TurboQuantStaticCache on {device}")

    batch, num_heads, head_dim = 1, 4, 128
    max_len = 1024
    cache = TurboQuantStaticCache(max_cache_len=max_len, bits=4, num_layers=8)

    # Prefill
    for layer in range(8):
        k = torch.randn(batch, num_heads, 512, head_dim, device=device)
        v = torch.randn(batch, num_heads, 512, head_dim, device=device)
        full_k, full_v = cache.update(k, v, layer_idx=layer)
        assert full_k.shape == (batch, num_heads, max_len, head_dim)

    print(f"After prefill: {cache.get_seq_length()} tokens cached")

    # Incremental generation
    for step in range(10):
        for layer in range(8):
            k = torch.randn(batch, num_heads, 1, head_dim, device=device)
            v = torch.randn(batch, num_heads, 1, head_dim, device=device)
            full_k, full_v = cache.update(k, v, layer_idx=layer)
            assert full_k.shape == (batch, num_heads, max_len, head_dim)

    print(f"After generation: {cache.get_seq_length()} tokens cached")
    print(f"Memory: {cache.memory_usage_bytes()}")
    print("All checks passed!")
