"""Pricing-math tests for `src.pricing`.

Thin by design — the module itself is thin. These tests exist as a canary:
if a provider's pricing changes and a dev updates one number, the test
suite catches a stale assertion before it ships. The deliberately
hard-coded expected values are the whole point.
"""

from __future__ import annotations

import pytest

from src.pricing import (
    ANTHROPIC_PRICING,
    COHERE_PRICING,
    VOYAGE_PRICING,
    anthropic_cost,
    cohere_cost,
    voyage_cost,
)


class TestVoyagePricing:
    def test_known_model(self) -> None:
        # voyage-3-large: $0.18 / 1M tokens
        assert voyage_cost("voyage-3-large", 1_000_000) == pytest.approx(0.18)
        assert voyage_cost("voyage-3-large", 500_000) == pytest.approx(0.09)
        assert voyage_cost("voyage-3-large", 0) == pytest.approx(0.0)

    def test_unknown_model_is_none(self) -> None:
        assert voyage_cost("voyage-made-up", 1000) is None

    def test_dict_entry_present(self) -> None:
        # Guard against an accidental deletion of the single known model.
        assert "voyage-3-large" in VOYAGE_PRICING


class TestAnthropicPricing:
    def test_known_model_input_and_output(self) -> None:
        # claude-sonnet-4-6: $3 input / $15 output per 1M tokens
        assert anthropic_cost("claude-sonnet-4-6", 1_000_000, 0) == pytest.approx(3.0)
        assert anthropic_cost("claude-sonnet-4-6", 0, 1_000_000) == pytest.approx(15.0)
        assert anthropic_cost("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(18.0)

    def test_unknown_model_is_none(self) -> None:
        assert anthropic_cost("claude-made-up", 100, 100) is None

    def test_dict_entry_present(self) -> None:
        assert "claude-sonnet-4-6" in ANTHROPIC_PRICING


class TestCoherePricing:
    def test_known_model_single_search(self) -> None:
        # rerank-english-v3.0: $2 / 1000 searches = $0.002 / search
        assert cohere_cost("rerank-english-v3.0") == pytest.approx(0.002)
        assert cohere_cost("rerank-english-v3.0", 1) == pytest.approx(0.002)

    def test_known_model_multi_search_scales_linearly(self) -> None:
        # 100 Q&As × 1 rerank each = $0.20 — the "cost per eval run" story.
        assert cohere_cost("rerank-english-v3.0", 100) == pytest.approx(0.20)
        assert cohere_cost("rerank-english-v3.0", 1000) == pytest.approx(2.0)

    def test_zero_searches_is_zero(self) -> None:
        # Edge case the eval loop never hits, but worth asserting for the
        # "shouldn't crash on an unused reranker" claim.
        assert cohere_cost("rerank-english-v3.0", 0) == pytest.approx(0.0)

    def test_unknown_model_is_none(self) -> None:
        assert cohere_cost("rerank-made-up") is None
        assert cohere_cost("rerank-made-up", 100) is None

    def test_dict_entry_present(self) -> None:
        # Guard against an accidental deletion. The CohereReranker's
        # __init__ raises if its configured model is missing from this
        # dict — this test fails before __init__ gets a chance to.
        assert "rerank-english-v3.0" in COHERE_PRICING
