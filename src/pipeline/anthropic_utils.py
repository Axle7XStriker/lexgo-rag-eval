"""Shared helpers for Anthropic client wrappers (ClaudeGenerator, ClaudeJudge).

Both wrappers share the same retry policy, cost lookup, provider label, and
default temperature. Kept here so a change (new retryable exception, price
adjustment, provider rename) is a one-file edit that both wrappers pick up.
"""

from __future__ import annotations

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from src.observability import get_logger
from src.pricing import ANTHROPIC_PRICING as PRICING
from src.pricing import anthropic_cost

_logger = get_logger("anthropic")

PROVIDER = "anthropic"
DEFAULT_TEMPERATURE = 0.0

__all__ = [
    "DEFAULT_TEMPERATURE",
    "PRICING",
    "PROVIDER",
    "cost_for",
    "is_retryable",
]


def is_retryable(exc: BaseException) -> bool:
    """True for transient failures; False for deterministic ones.

    RateLimitError, APIConnectionError, APITimeoutError are always transient.
    APIStatusError catches the raw HTTP surface — we retry only 5xx server
    errors, not 4xx client errors (BadRequestError, AuthenticationError,
    NotFoundError, PermissionDeniedError all inherit from APIStatusError but
    correspond to deterministic mistakes we should surface immediately).
    """
    if isinstance(exc, RateLimitError | APIConnectionError | APITimeoutError):
        return True
    if isinstance(exc, APIStatusError):
        # `status_code` is set on typed APIStatusError subclasses; guard with
        # getattr in case a subclass without one slips through.
        code = getattr(exc, "status_code", None)
        return isinstance(code, int) and code >= 500
    return False


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for a call at `model`'s pricing. Missing model → 0.0."""
    cost = anthropic_cost(model, input_tokens, output_tokens)
    if cost is None:
        _logger.warning(
            "anthropic_pricing_missing",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return 0.0
    return cost
