"""Tests for TurboQuantStaticCache (StaticCache-based variant)."""

import torch
import pytest
from turboquant import TurboQuantStaticCache
from turboquant.static_cache import TurboQuantStaticLayer


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


BATCH, HEADS, HEAD_DIM = 1, 4, 128


class TestTurboQuantStaticLayer:
    def test_basic_update(self, device):
        """Layer returns full max_cache_len-sized tensors."""
        layer = TurboQuantStaticLayer(max_cache_len=512, bits=4)
        k = torch.randn(BATCH, HEADS, 100, HEAD_DIM, device=device)
        v = torch.randn(BATCH, HEADS, 100, HEAD_DIM, device=device)
        full_k, full_v = layer.update(k, v)
        assert full_k.shape == (BATCH, HEADS, 512, HEAD_DIM)
        assert full_v.shape == (BATCH, HEADS, 512, HEAD_DIM)

    def test_zero_padding(self, device):
        """Positions beyond current length should be zero."""
        layer = TurboQuantStaticLayer(max_cache_len=512, bits=4, residual_len=128)
        k = torch.randn(BATCH, HEADS, 50, HEAD_DIM, device=device)
        v = torch.randn(BATCH, HEADS, 50, HEAD_DIM, device=device)
        full_k, full_v = layer.update(k, v)
        # Positions 50..511 should be all zeros
        assert torch.all(full_k[:, :, 50:, :] == 0)
        assert torch.all(full_v[:, :, 50:, :] == 0)

    def test_residual_exact(self, device):
        """Tokens within residual window should be FP16-exact."""
        layer = TurboQuantStaticLayer(max_cache_len=512, bits=4, residual_len=128)
        k = torch.randn(BATCH, HEADS, 50, HEAD_DIM, device=device, dtype=torch.float16)
        v = torch.randn(BATCH, HEADS, 50, HEAD_DIM, device=device, dtype=torch.float16)
        full_k, full_v = layer.update(k, v)
        assert torch.allclose(k, full_k[:, :, :50, :], atol=1e-6)
        assert torch.allclose(v, full_v[:, :, :50, :], atol=1e-6)

    def test_incremental_generation(self, device):
        """Token-by-token generation should work correctly."""
        layer = TurboQuantStaticLayer(max_cache_len=512, bits=4, residual_len=128)
        # Prefill 100 tokens
        k = torch.randn(BATCH, HEADS, 100, HEAD_DIM, device=device)
        v = torch.randn(BATCH, HEADS, 100, HEAD_DIM, device=device)
        layer.update(k, v)

        # Generate 50 tokens one at a time
        for _ in range(50):
            k = torch.randn(BATCH, HEADS, 1, HEAD_DIM, device=device)
            v = torch.randn(BATCH, HEADS, 1, HEAD_DIM, device=device)
            full_k, full_v = layer.update(k, v)

        assert layer.get_seq_length().item() == 150
        # Non-zero content at positions 0..149, zeros at 150..511
        assert full_k[:, :, :150, :].abs().sum() > 0
        assert torch.all(full_k[:, :, 150:, :] == 0)

    def test_overflow_triggers_quantization(self, device):
        """Exceeding residual_len should quantize older tokens."""
        layer = TurboQuantStaticLayer(max_cache_len=512, bits=4, residual_len=64)
        # 200 tokens > residual_len=64 → quantization occurs
        k = torch.randn(BATCH, HEADS, 200, HEAD_DIM, device=device)
        v = torch.randn(BATCH, HEADS, 200, HEAD_DIM, device=device)
        layer.update(k, v)

        assert layer._quantized_count == 136  # 200 - 64
        assert layer._residual_count == 64
        assert layer._key_indices is not None
        assert layer._key_indices.dtype == torch.uint8

    def test_no_growth_during_generation(self, device):
        """Verify no tensor reallocation occurs after init (static contract)."""
        layer = TurboQuantStaticLayer(max_cache_len=256, bits=4, residual_len=64)
        # Trigger lazy init
        k = torch.randn(BATCH, HEADS, 10, HEAD_DIM, device=device)
        v = torch.randn(BATCH, HEADS, 10, HEAD_DIM, device=device)
        layer.update(k, v)

        # Capture tensor data_ptrs
        key_ptr = layer.keys.data_ptr()
        val_ptr = layer.values.data_ptr()
        idx_ptr = layer._key_indices.data_ptr() if layer._key_indices is not None else None
        res_ptr = layer._residual_keys.data_ptr() if layer._residual_keys is not None else None

        # Generate more tokens
        for _ in range(200):
            k = torch.randn(BATCH, HEADS, 1, HEAD_DIM, device=device)
            v = torch.randn(BATCH, HEADS, 1, HEAD_DIM, device=device)
            layer.update(k, v)

        # Same tensor objects (no reallocation)
        assert layer.keys.data_ptr() == key_ptr
        assert layer.values.data_ptr() == val_ptr
        if idx_ptr is not None:
            assert layer._key_indices.data_ptr() == idx_ptr
        if res_ptr is not None:
            assert layer._residual_keys.data_ptr() == res_ptr

    def test_reset(self, device):
        """Reset should zero all buffers and counters."""
        layer = TurboQuantStaticLayer(max_cache_len=256, bits=4, residual_len=64)
        k = torch.randn(BATCH, HEADS, 200, HEAD_DIM, device=device)
        v = torch.randn(BATCH, HEADS, 200, HEAD_DIM, device=device)
        layer.update(k, v)
        assert layer._quantized_count > 0

        layer.reset()
        assert layer._quantized_count == 0
        assert layer._residual_count == 0
        assert torch.all(layer.keys == 0)
        assert torch.all(layer.values == 0)

    def test_mask_sizes(self, device):
        """get_mask_sizes should return full max_cache_len."""
        layer = TurboQuantStaticLayer(max_cache_len=512, bits=4)
        k = torch.randn(BATCH, HEADS, 10, HEAD_DIM, device=device)
        v = torch.randn(BATCH, HEADS, 10, HEAD_DIM, device=device)
        layer.update(k, v)
        kv_length, kv_offset = layer.get_mask_sizes(query_length=1)
        assert kv_length == 512
        assert kv_offset == 0

    def test_large_prefill(self, device):
        """Large prefill that exceeds residual by a lot."""
        layer = TurboQuantStaticLayer(max_cache_len=1024, bits=4, residual_len=128)
        k = torch.randn(BATCH, HEADS, 900, HEAD_DIM, device=device)
        v = torch.randn(BATCH, HEADS, 900, HEAD_DIM, device=device)
        full_k, full_v = layer.update(k, v)

        assert layer._quantized_count == 772  # 900 - 128
        assert layer._residual_count == 128
        assert full_k.shape == (BATCH, HEADS, 1024, HEAD_DIM)
        # Content at 0..899, zeros at 900..1023
        assert full_k[:, :, :900, :].abs().sum() > 0
        assert torch.all(full_k[:, :, 900:, :] == 0)


class TestTurboQuantStaticCache:
    def test_basic_creation(self, device):
        """Cache can be created with num_layers."""
        cache = TurboQuantStaticCache(max_cache_len=512, bits=4, num_layers=8)
        assert len(cache.layers) == 8

    def test_multi_layer_update(self, device):
        """Update across multiple layers."""
        cache = TurboQuantStaticCache(max_cache_len=512, bits=4, num_layers=4)
        for layer_idx in range(4):
            k = torch.randn(BATCH, HEADS, 100, HEAD_DIM, device=device)
            v = torch.randn(BATCH, HEADS, 100, HEAD_DIM, device=device)
            full_k, full_v = cache.update(k, v, layer_idx=layer_idx)
            assert full_k.shape == (BATCH, HEADS, 512, HEAD_DIM)
        assert cache.get_seq_length() == 100

    def test_seq_length_tracking(self, device):
        """Sequence length should track across prefill + generation."""
        cache = TurboQuantStaticCache(max_cache_len=1024, bits=4, num_layers=2)
        # Prefill
        for layer_idx in range(2):
            k = torch.randn(BATCH, HEADS, 200, HEAD_DIM, device=device)
            v = torch.randn(BATCH, HEADS, 200, HEAD_DIM, device=device)
            cache.update(k, v, layer_idx=layer_idx)
        assert cache.get_seq_length().item() == 200

        # Generate
        for _ in range(50):
            for layer_idx in range(2):
                k = torch.randn(BATCH, HEADS, 1, HEAD_DIM, device=device)
                v = torch.randn(BATCH, HEADS, 1, HEAD_DIM, device=device)
                cache.update(k, v, layer_idx=layer_idx)
        assert cache.get_seq_length().item() == 250

    def test_different_bit_widths(self, device):
        """All supported bit widths should work."""
        for bits in [2, 3, 4]:
            cache = TurboQuantStaticCache(max_cache_len=512, bits=bits, num_layers=2)
            k = torch.randn(BATCH, HEADS, 200, HEAD_DIM, device=device)
            v = torch.randn(BATCH, HEADS, 200, HEAD_DIM, device=device)
            full_k, full_v = cache.update(k, v, layer_idx=0)
            assert full_k.shape == (BATCH, HEADS, 512, HEAD_DIM)

    def test_memory_usage(self, device):
        """Memory tracking should report non-zero after updates."""
        cache = TurboQuantStaticCache(max_cache_len=512, bits=4, num_layers=4)
        for layer_idx in range(4):
            k = torch.randn(BATCH, HEADS, 300, HEAD_DIM, device=device)
            v = torch.randn(BATCH, HEADS, 300, HEAD_DIM, device=device)
            cache.update(k, v, layer_idx=layer_idx)

        stats = cache.memory_usage_bytes()
        assert stats["allocated_bytes"] > 0
        assert stats["compressed_backing_bytes"] > 0
        assert stats["output_buffer_bytes"] > 0

    def test_requires_config_or_num_layers(self):
        """Should raise if neither config nor num_layers is provided."""
        with pytest.raises(ValueError, match="Either config or num_layers"):
            TurboQuantStaticCache(max_cache_len=512, bits=4)

    def test_small_residual_len(self, device):
        """Edge case: residual_len=0 means everything gets quantized."""
        cache = TurboQuantStaticCache(max_cache_len=256, bits=4, num_layers=1, residual_len=0)
        k = torch.randn(BATCH, HEADS, 100, HEAD_DIM, device=device)
        v = torch.randn(BATCH, HEADS, 100, HEAD_DIM, device=device)
        full_k, full_v = cache.update(k, v, layer_idx=0)
        assert full_k.shape == (BATCH, HEADS, 256, HEAD_DIM)
        layer = cache.layers[0]
        assert layer._quantized_count == 100
        assert layer._residual_count == 0

    def test_residual_equals_max_len(self, device):
        """Edge case: residual_len >= max_cache_len means no quantization."""
        cache = TurboQuantStaticCache(max_cache_len=256, bits=4, num_layers=1, residual_len=256)
        k = torch.randn(BATCH, HEADS, 100, HEAD_DIM, device=device)
        v = torch.randn(BATCH, HEADS, 100, HEAD_DIM, device=device)
        full_k, full_v = cache.update(k, v, layer_idx=0)
        layer = cache.layers[0]
        assert layer._quantized_count == 0
        assert layer._residual_count == 100
