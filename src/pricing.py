"""Centralized pricing for every LLM provider used by the pipeline.

One place to keep the numbers so a provider price change is a one-line edit
and the cost story in the blog stays accurate across every module that logs
a call. Provider modules (embed / generate / future rerank / judge) import
from here rather than owning their own PRICING dicts.

Voyage embeddings are input-only (the response is a vector, not generated
text). Anthropic messages have distinct input and output rates. Cohere
Rerank bills per search (one API call), independent of query or document
token counts — so its unit is per-call USD, not per-million-tokens. The
three shapes are exposed as separate constants so downstream callers
don't have to branch on provider — each just consumes the shape it needs.

Units: VOYAGE_PRICING and ANTHROPIC_PRICING are USD per 1M tokens;
COHERE_PRICING is USD per rerank search. Tracked for periodic verification
against each provider's public pricing page.
"""

from __future__ import annotations

VOYAGE_PRICING: dict[str, float] = {
    "voyage-3-large": 0.18,
}

ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
}

# Cohere Rerank bills per API invocation, not per token. One "search" =
# one `client.rerank(...)` call (up to 1000 documents per call per Cohere's
# API limits — well above our top_k=20 pool). Rerank 3 tier is $2.00 per
# 1000 searches = $0.002 per call.
COHERE_PRICING: dict[str, float] = {
    "rerank-english-v3.0": 0.002,
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


def cohere_cost(model: str, num_searches: int = 1) -> float | None:
    """Return USD cost for `num_searches` Cohere Rerank calls, or None if the model is unknown.

    `num_searches` is almost always 1 — one rerank call per query. Kept as
    a parameter so a future batch-rerank path can account cleanly without
    the caller duplicating pricing math.
    """
    per_search = COHERE_PRICING.get(model)
    if per_search is None:
        return None
    return num_searches * per_search
