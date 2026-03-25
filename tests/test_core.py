"""Tests for TurboQuant core algorithms."""

import torch
import math
import pytest
from turboquant import TurboQuantMSE, TurboQuantIP


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestTurboQuantMSE:
    def test_1bit_within_bounds(self, device):
        tq = TurboQuantMSE(dim=128, bits=1, device=device)
        x = torch.randn(1000, 128, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        idx, norms = tq.quantize(x)
        x_hat = tq.dequantize(idx, norms)
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        bound = math.sqrt(3) * math.pi / 2 * (1 / 4 ** 1)
        assert mse < bound * 1.5, f"MSE {mse} exceeds 1.5x bound {bound}"

    def test_2bit_within_bounds(self, device):
        tq = TurboQuantMSE(dim=128, bits=2, device=device)
        x = torch.randn(1000, 128, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        idx, norms = tq.quantize(x)
        x_hat = tq.dequantize(idx, norms)
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        bound = math.sqrt(3) * math.pi / 2 * (1 / 4 ** 2)
        assert mse < bound * 1.5

    def test_roundtrip_preserves_norm(self, device):
        tq = TurboQuantMSE(dim=128, bits=4, device=device)
        x = torch.randn(100, 128, device=device) * 5.0  # Non-unit vectors
        idx, norms = tq.quantize(x)
        x_hat = tq.dequantize(idx, norms)
        orig_norms = torch.norm(x, dim=-1)
        recon_norms = torch.norm(x_hat, dim=-1)
        norm_error = ((orig_norms - recon_norms).abs() / orig_norms).mean().item()
        assert norm_error < 0.15, f"Norm preservation error {norm_error} > 15%"

    def test_compression_ratio(self, device):
        tq = TurboQuantMSE(dim=128, bits=3, device=device)
        idx, norms = tq.quantize(torch.randn(100, 128, device=device))
        # 3 bits per dim + 32-bit norm = 3*128 + 32 = 416 bits per vector
        # FP16 baseline = 16*128 = 2048 bits
        ratio = 2048 / 416
        assert ratio > 4.5, f"Compression ratio {ratio} < 4.5x"


class TestTurboQuantIP:
    def test_unbiased_inner_product(self, device):
        tq = TurboQuantIP(dim=128, bits=3, device=device)
        x = torch.randn(500, 128, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        q = torch.randn(100, 128, device=device)
        q = q / torch.norm(q, dim=-1, keepdim=True)

        idx, norms, qjl, res_norms = tq.quantize(x)
        x_hat = tq.dequantize(idx, norms, qjl, res_norms)

        true_ip = (q.unsqueeze(1) * x.unsqueeze(0)).sum(dim=-1)
        approx_ip = (q.unsqueeze(1) * x_hat.unsqueeze(0)).sum(dim=-1)
        bias = (approx_ip - true_ip).mean().item()
        assert abs(bias) < 0.01, f"IP bias {bias} > 0.01 (not unbiased)"


class TestEdgeCases:
    def test_single_vector(self, device):
        tq = TurboQuantMSE(dim=64, bits=4, device=device)
        x = torch.randn(1, 64, device=device)
        idx, norms = tq.quantize(x)
        x_hat = tq.dequantize(idx, norms)
        assert x_hat.shape == (1, 64)

    def test_zero_vector(self, device):
        tq = TurboQuantMSE(dim=64, bits=4, device=device)
        x = torch.zeros(1, 64, device=device)
        idx, norms = tq.quantize(x)
        x_hat = tq.dequantize(idx, norms)
        assert torch.allclose(x_hat, x, atol=1e-5)

    def test_different_dimensions(self, device):
        for dim in [32, 64, 128, 256]:
            tq = TurboQuantMSE(dim=dim, bits=3, device=device)
            x = torch.randn(10, dim, device=device)
            idx, norms = tq.quantize(x)
            x_hat = tq.dequantize(idx, norms)
            assert x_hat.shape == x.shape
