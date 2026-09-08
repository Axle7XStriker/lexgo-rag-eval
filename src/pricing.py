"""Centralized pricing for every LLM provider used by the pipeline.

One place to keep the numbers so a provider price change is a one-line edit
and the cost story in the blog stays accurate across every module that logs
a call. Provider modules (embed / generate / future rerank / judge) import
from here rather than owning their own PRICING dicts.

Voyage embeddings are input-only (the response is a vector, not generated
text). Anthropic messages have distinct input and output rates. The two
shapes are exposed as separate constants so downstream callers don't have
to branch on provider — each just consumes the shape it needs.

USD per 1M tokens throughout. Tracked for periodic verification against
each provider's public pricing page; see the follow-up issue.
"""

from __future__ import annotations

VOYAGE_PRICING: dict[str, float] = {
    "voyage-3-large": 0.18,
}

ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
}


def voyage_cost(model: str, input_tokens: int) -> float | None:
    """Return USD cost for a Voyage embed call, or None if the model is unknown.

    None (not 0.0) is the "unknown model" sentinel so callers can decide the
    policy: fail fast at init, log-then-attribute-zero at call time, etc.
    """
    per_million = VOYAGE_PRICING.get(model)
    if per_million is None:
        return None
    return (input_tokens / 1_000_000) * per_million


def anthropic_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return USD cost for an Anthropic messages call, or None if the model is unknown.

    None (not 0.0) is the "unknown model" sentinel — same convention as
    `voyage_cost` so provider modules handle unknowns uniformly.
    """
    rates = ANTHROPIC_PRICING.get(model)
    if rates is None:
        return None
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates[
        "output"
    ]
