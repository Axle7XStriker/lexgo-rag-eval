"""Shared helpers for Cohere client wrappers (currently CohereReranker).

Mirror of `src.pipeline.anthropic_utils`. Lives here so a change (new
retryable exception class, pricing adjustment, provider rename) is a
one-file edit that every Cohere callsite picks up.

Cohere's typed exception hierarchy (as of cohere 7.x): all 4xx/5xx errors
subclass `cohere.core.api_error.ApiError`, which carries a `status_code`
attribute. `is_retryable` prefers the typed subclass check (`TooManyRequestsError`,
`InternalServerError`, `ServiceUnavailableError`, `GatewayTimeoutError`)
and falls back to a 5xx `status_code` check for any future server-side
subclass the SDK adds. Everything 4xx (auth, bad request, 404, 422) is
treated as deterministic and fails fast — these are author errors, not
transient conditions.
"""

from __future__ import annotations

import httpx
from cohere.core.api_error import ApiError
from cohere.errors import (
    GatewayTimeoutError,
    InternalServerError,
    ServiceUnavailableError,
    TooManyRequestsError,
)

from src.observability import get_logger
from src.pricing import COHERE_PRICING as PRICING
from src.pricing import cohere_cost

_logger = get_logger("cohere")

PROVIDER = "cohere"

__all__ = [
    "PRICING",
    "PROVIDER",
    "cost_for",
    "is_retryable",
]


def is_retryable(exc: BaseException) -> bool:
    """True for transient failures; False for deterministic ones.

    Retryable: Cohere's rate-limit + 5xx typed exceptions, and httpx
    transport failures (DNS, connection reset, read timeout). The typed
    check comes first because it's cheaper and doesn't require the SDK
    to have populated `status_code` on the exception.

    Not retryable: 4xx author errors (`BadRequestError`, `UnauthorizedError`,
    `ForbiddenError`, `NotFoundError`, `UnprocessableEntityError`,
    `ClientClosedRequestError`, `InvalidTokenError`, `NotImplementedError`).
    Retrying these would just burn budget on the same deterministic failure.
    """
    if isinstance(
        exc,
        TooManyRequestsError | InternalServerError | ServiceUnavailableError | GatewayTimeoutError,
    ):
        return True
    if isinstance(exc, ApiError):
        # Catches any future 5xx subclass the SDK adds without needing a
        # code change here. 4xx stays non-retryable.
        code = exc.status_code
        return isinstance(code, int) and code >= 500
    # httpx errors surface when Cohere's SDK fails before getting a response
    # (DNS, TCP reset, read timeout). All transient by definition.
    return isinstance(exc, httpx.ConnectError | httpx.ReadTimeout | httpx.RemoteProtocolError)


def cost_for(model: str, num_searches: int = 1) -> float:
    """Cost in USD for `num_searches` Cohere rerank calls. Missing model → 0.0."""
    cost = cohere_cost(model, num_searches)
    if cost is None:
        _logger.warning("cohere_pricing_missing", model=model, num_searches=num_searches)
        return 0.0
    return cost
