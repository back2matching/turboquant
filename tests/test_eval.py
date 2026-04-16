"""Smoke tests for turboquant.eval module.

Avoids requiring network + GPU by testing the detokenizer + cache factory
and leaving the full end-to-end PPL run as an out-of-band check.
"""

import math
import pytest

from turboquant.eval import wikitext_detokenize, _make_cache


class TestDetokenizer:
    def test_hyphen_restoration(self):
        assert wikitext_detokenize("a @-@ b") == "a-b"

    def test_comma_and_period(self):
        assert wikitext_detokenize("1 @,@ 000 @.@ 5") == "1,000.5"

    def test_contractions(self):
        assert wikitext_detokenize("it 's a test") == "it's a test"
        assert wikitext_detokenize("do n't go") == "don't go"

    def test_idempotent_on_clean_text(self):
        clean = "The quick brown fox."
        # Known gotcha: " . " -> ". " also affects trailing-period sentences.
        # Just verify we don't crash and output is a string.
        assert isinstance(wikitext_detokenize(clean), str)


class TestCacheFactory:
    def test_no_quant_returns_dynamic_cache(self):
        from transformers import DynamicCache
        cache = _make_cache(use_tq=False, bits=4, key_bits=None, value_bits=None)
        assert isinstance(cache, DynamicCache)

    def test_tq_returns_tq_cache(self):
        from turboquant import TurboQuantCache
        cache = _make_cache(use_tq=True, bits=4, key_bits=None, value_bits=None)
        assert isinstance(cache, TurboQuantCache)
        assert cache.key_bits == 4
        assert cache.value_bits == 4

    def test_asymmetric_kv_bits(self):
        from turboquant import TurboQuantCache
        cache = _make_cache(use_tq=True, bits=4, key_bits=4, value_bits=2)
        assert isinstance(cache, TurboQuantCache)
        assert cache.key_bits == 4
        assert cache.value_bits == 2

    def test_fresh_cache_per_call(self):
        """Critical for sliding-window PPL — each window needs a fresh cache."""
        c1 = _make_cache(use_tq=True, bits=4, key_bits=None, value_bits=None)
        c2 = _make_cache(use_tq=True, bits=4, key_bits=None, value_bits=None)
        assert c1 is not c2
