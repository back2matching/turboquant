"""
Optional CUDA acceleration for TurboQuant.

If the cuda_turboquant extension is built and available, uses fused
rotation+quantize CUDA kernels for 2.4x speedup over pure PyTorch.

Build the extension:
    cd cuda/ && python setup.py build_ext --inplace

Falls back to pure PyTorch if not available.
"""

import torch
from typing import Optional, Tuple

_cuda_available = False
_cuda_module = None

try:
    import cuda_turboquant
    _cuda_available = True
    _cuda_module = cuda_turboquant
except ImportError:
    pass


def is_cuda_available() -> bool:
    """Check if CUDA acceleration is available."""
    return _cuda_available


def cuda_quantize(
    x: torch.Tensor,
    rotation_t: torch.Tensor,
    codebook: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-accelerated TurboQuant quantization.

    Falls back to PyTorch if CUDA extension not available.
    """
    if _cuda_available and x.is_cuda:
        result = _cuda_module.quantize(x, rotation_t, codebook)
        dim = x.shape[-1]
        indices = result[:, :dim].to(torch.uint8)
        norms = result[:, dim]
        return indices, norms.unsqueeze(-1)

    # PyTorch fallback
    x_f32 = x.float()
    norms = torch.norm(x_f32, dim=-1, keepdim=True)
    x_unit = x_f32 / (norms + 1e-10)
    y = x_unit @ rotation_t
    dists = (y.unsqueeze(-1) - codebook.unsqueeze(0)).abs()
    indices = dists.argmin(dim=-1).to(torch.uint8)
    return indices, norms


def cuda_dequantize(
    indices: torch.Tensor,
    norms: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA-accelerated TurboQuant dequantization.

    Falls back to PyTorch if CUDA extension not available.
    """
    if _cuda_available and indices.is_cuda:
        return _cuda_module.dequantize(indices, norms.squeeze(-1), rotation, codebook)

    # PyTorch fallback
    y_hat = codebook[indices.long()]
    x_hat = y_hat @ rotation
    return x_hat * norms
