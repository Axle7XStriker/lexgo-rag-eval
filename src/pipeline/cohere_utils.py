"""Shared helpers for Cohere client wrappers (currently CohereReranker)."""

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
    transport failures (DNS, connection reset, timeouts on connect/read/
    write/pool, read/write/close on an established socket, remote protocol
    errors). The typed check comes first because it's cheaper and doesn't
    require the SDK to have populated `status_code` on the exception.

    Not retryable: 4xx author errors (`BadRequestError`, `UnauthorizedError`,
    `ForbiddenError`, `NotFoundError`, `UnprocessableEntityError`,
    `ClientClosedRequestError`, `InvalidTokenError`, `NotImplementedError`),
    and httpx errors that signal a programming/config bug rather than a
    transient condition (`LocalProtocolError`, `ProxyError`,
    `UnsupportedProtocol`). Retrying any of these would just burn budget on
    the same deterministic failure.
    """
    if isinstance(
        exc,
        TooManyRequestsError | InternalServerError | ServiceUnavailableError | GatewayTimeoutError,
    ):
        return True
    if isinstance(exc, ApiError):
        code = exc.status_code
        return isinstance(code, int) and code >= 500
    # Author-side protocol and proxy errors stay out of this union.
    return isinstance(
        exc,
        httpx.TimeoutException | httpx.NetworkError | httpx.RemoteProtocolError,
    )


def cost_for(model: str, num_searches: int) -> float:
    """Cost in USD for `num_searches` Cohere rerank calls. Missing model → 0.0."""
    cost = cohere_cost(model, num_searches)
    if cost is None:
        _logger.warning("cohere_pricing_missing", model=model, num_searches=num_searches)
        return 0.0
    return cost
