"""
TurboQuant: KV cache compression for LLM inference.

First open-source implementation of Google's TurboQuant (ICLR 2026).
Compresses KV cache to 3-4 bits with minimal quality loss.

Quick start:
    from turboquant import TurboQuantCache
    cache = TurboQuantCache(bits=4)
    outputs = model.generate(..., past_key_values=cache)

Paper: https://arxiv.org/abs/2504.19874
"""

from turboquant.core import TurboQuantMSE, TurboQuantIP
from turboquant.cache import TurboQuantCache

__version__ = "0.3.0"
__all__ = ["TurboQuantMSE", "TurboQuantIP", "TurboQuantCache"]
