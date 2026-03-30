"""Tests for TurboQuantCache (HuggingFace DynamicCache integration)."""

import torch
import pytest
from turboquant import TurboQuantCache
from turboquant.core import pack_uint4, unpack_uint4


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestTurboQuantCache:
    def test_basic_update(self, device):
        cache = TurboQuantCache(bits=4)
        k = torch.randn(1, 4, 512, 128, device=device)
        v = torch.randn(1, 4, 512, 128, device=device)
        full_k, full_v = cache.update(k, v, layer_idx=0)
        assert full_k.shape == (1, 4, 512, 128)
        assert full_v.shape == (1, 4, 512, 128)

    def test_incremental_generation(self, device):
        cache = TurboQuantCache(bits=4)
        # Prefill
        k = torch.randn(1, 4, 100, 128, device=device)
        v = torch.randn(1, 4, 100, 128, device=device)
        cache.update(k, v, layer_idx=0)
        assert cache.get_seq_length() == 100

        # Generate 10 tokens
        for _ in range(10):
            k = torch.randn(1, 4, 1, 128, device=device)
            v = torch.randn(1, 4, 1, 128, device=device)
            full_k, full_v = cache.update(k, v, layer_idx=0)

        assert cache.get_seq_length() == 110
        assert full_k.shape == (1, 4, 110, 128)

    def test_multi_layer(self, device):
        cache = TurboQuantCache(bits=3)
        for layer in range(8):
            k = torch.randn(1, 4, 50, 128, device=device)
            v = torch.randn(1, 4, 50, 128, device=device)
            cache.update(k, v, layer_idx=layer)
        assert len(cache) == 8

    def test_residual_window_quality(self, device):
        """Recent tokens (in residual window) should be FP16-exact."""
        cache = TurboQuantCache(bits=4)
        # Only 50 tokens — should all be in residual window (128 default)
        k = torch.randn(1, 4, 50, 128, device=device, dtype=torch.float16)
        v = torch.randn(1, 4, 50, 128, device=device, dtype=torch.float16)
        full_k, full_v = cache.update(k, v, layer_idx=0)
        # Should be exact (no quantization applied yet, all in residual)
        assert torch.allclose(k, full_k, atol=1e-3)

    def test_different_bit_widths(self, device):
        for bits in [2, 3, 4]:
            cache = TurboQuantCache(bits=bits)
            k = torch.randn(1, 4, 200, 128, device=device)
            v = torch.randn(1, 4, 200, 128, device=device)
            full_k, full_v = cache.update(k, v, layer_idx=0)
            assert full_k.shape == (1, 4, 200, 128)

    def test_compressed_index_storage(self, device):
        """Verify internal storage uses uint8 indices, not dequantized FP16."""
        cache = TurboQuantCache(bits=4)
        # 256 tokens: 128 will overflow residual and get compressed
        k = torch.randn(1, 4, 256, 128, device=device, dtype=torch.float16)
        v = torch.randn(1, 4, 256, 128, device=device, dtype=torch.float16)
        cache.update(k, v, layer_idx=0)

        layer = cache.layers[0]
        # Indices should be uint8
        assert layer._key_indices.dtype == torch.uint8
        assert layer._value_indices.dtype == torch.uint8
        # Norms should be float32
        assert layer._key_norms.dtype == torch.float32
        # 128 tokens should be in compressed buffer (256 - 128 residual)
        assert layer._key_indices.shape[-2] == 128
        assert layer._residual_keys.shape[-2] == 128

    def test_memory_usage_bytes(self, device):
        """Verify compressed storage uses less memory than FP16 equivalent."""
        cache = TurboQuantCache(bits=4)
        k = torch.randn(1, 4, 512, 128, device=device)
        v = torch.randn(1, 4, 512, 128, device=device)
        cache.update(k, v, layer_idx=0)

        stats = cache.memory_usage_bytes()
        assert stats["compressed_bytes"] > 0
        assert stats["total_bytes"] < stats["fp16_equivalent_bytes"]
        assert stats["savings_ratio"] > 1.0

    def test_nibble_packing_roundtrip(self, device):
        """Verify pack_uint4/unpack_uint4 is lossless."""
        indices = torch.randint(0, 16, (2, 4, 10, 128), dtype=torch.uint8, device=device)
        packed = pack_uint4(indices)
        assert packed.shape[-1] == 64  # halved
        unpacked = unpack_uint4(packed, 128)
        assert torch.equal(indices, unpacked)

    def test_4bit_uses_packed_storage(self, device):
        """Verify 4-bit cache stores nibble-packed indices (half the size of uint8)."""
        cache = TurboQuantCache(bits=4)
        k = torch.randn(1, 4, 256, 128, device=device, dtype=torch.float16)
        v = torch.randn(1, 4, 256, 128, device=device, dtype=torch.float16)
        cache.update(k, v, layer_idx=0)

        layer = cache.layers[0]
        # 128 tokens quantized, 128 in residual
        # Packed: last dim should be 64 (128 / 2) not 128
        assert layer._key_indices.shape[-1] == 64
        assert layer._key_indices.dtype == torch.uint8

    def test_asymmetric_kv_bits(self, device):
        """Verify asymmetric K/V allocation uses different quantizers."""
        cache = TurboQuantCache(key_bits=4, value_bits=2)
        k = torch.randn(1, 4, 256, 128, device=device)
        v = torch.randn(1, 4, 256, 128, device=device)
        cache.update(k, v, layer_idx=0)

        layer = cache.layers[0]
        assert layer.key_bits == 4
        assert layer.value_bits == 2
        # 4-bit keys should be packed (last dim 64), 2-bit values should not (last dim 128)
        assert layer._key_indices.shape[-1] == 64   # packed
        assert layer._value_indices.shape[-1] == 128 # not packed (2-bit uses uint8)

    def test_protected_layers(self, device):
        """Protected layers should keep everything in FP16, no quantization."""
        cache = TurboQuantCache(bits=4, protected_layers=[0, 1])
        for layer_idx in range(4):
            k = torch.randn(1, 4, 256, 128, device=device, dtype=torch.float16)
            v = torch.randn(1, 4, 256, 128, device=device, dtype=torch.float16)
            full_k, full_v = cache.update(k, v, layer_idx=layer_idx)

        # Protected layers: no compressed indices, all in residual
        assert cache.layers[0].skip_quantization
        assert cache.layers[0]._key_indices.numel() == 0
        assert cache.layers[0]._residual_keys.shape[-2] == 256
        # Unprotected layers: should have compressed indices
        assert not cache.layers[2].skip_quantization
        assert cache.layers[2]._key_indices.numel() > 0

    def test_protected_layer_exact_output(self, device):
        """Protected layer output should be bit-exact with input (no quantization loss)."""
        cache = TurboQuantCache(bits=4, protected_layers=[0])
        k = torch.randn(1, 4, 50, 128, device=device, dtype=torch.float16)
        v = torch.randn(1, 4, 50, 128, device=device, dtype=torch.float16)
        full_k, full_v = cache.update(k, v, layer_idx=0)
        assert torch.equal(k, full_k)
        assert torch.equal(v, full_v)

    def test_bits_shorthand_backward_compat(self, device):
        """bits=N should work as shorthand for key_bits=N, value_bits=N."""
        cache = TurboQuantCache(bits=3)
        assert cache.key_bits == 3
        assert cache.value_bits == 3
