"""Centralized pricing for every LLM provider used by the pipeline."""

from __future__ import annotations

VOYAGE_PRICING: dict[str, float] = {
    "voyage-3-large": 0.18,
}

ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
}

COHERE_PRICING: dict[str, float] = {
    "rerank-english-v3.0": 0.002,  # USD per rerank call, not per token.
}


def voyage_cost(model: str, input_tokens: int) -> float | None:
    """Return USD cost for a Voyage embed call, or None if the model is unknown."""
    per_million = VOYAGE_PRICING.get(model)
    if per_million is None:
        return None
    return (input_tokens / 1_000_000) * per_million


def anthropic_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return USD cost for an Anthropic messages call, or None if the model is unknown."""
    rates = ANTHROPIC_PRICING.get(model)
    if rates is None:
        return None
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates[
        "output"
    ]


def cohere_cost(model: str, num_searches: int) -> float | None:
    """Return USD cost for `num_searches` Cohere Rerank calls, or None if the model is unknown."""
    per_search = COHERE_PRICING.get(model)
    if per_search is None:
        return None
    return num_searches * per_search
