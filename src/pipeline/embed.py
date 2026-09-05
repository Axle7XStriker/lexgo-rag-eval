"""Voyage embedding client with retry, batching, and per-call cost logging.

One class, `VoyageEmbedder`. Two methods:
  - `embed_documents(texts)` — batched, `input_type="document"`.
  - `embed_query(text)` — single input, `input_type="query"`.

Design notes worth remembering:
  - Voyage's SDK is sync and its batch limit is 128 inputs per request. We
    default to 64 to leave headroom for the token-per-request cap and to
    keep any single failure/refire cost small.
  - Every successful call writes one `log_llm_call` record. `input_tokens`
    is Voyage's reported `total_tokens`. Voyage's SDK does NOT return a
    per-call cost — cost is computed locally from the `PRICING` dict, so
    that dict is the only place that needs updating if Voyage's prices
    change.
  - Retries are tenacity, targeted at transient failures only (rate-limit,
    5xx, connection). Deterministic errors (bad API key, invalid model)
    fail fast.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from voyageai import Client as VoyageClient
from voyageai.error import RateLimitError, ServerError

from src.observability import get_logger, log_llm_call

_logger = get_logger("embed")

# USD per 1M input tokens, keyed on Voyage model id. Voyage's SDK does not
# return a per-call cost, so we compute it locally from this table. Tracked
# for periodic verification against Voyage's docs — see the follow-up issue.
PRICING: dict[str, float] = {
    "voyage-3-large": 0.18,
    # Add other models here as they get exercised.
}

# Voyage's max inputs per request is 128; leaving headroom keeps any single
# refire cheap and stays comfortably under the 120k-token-per-request cap for
# 500-token chunks (64 * 500 = 32k tokens per request).
DEFAULT_BATCH_SIZE = 64
MAX_BATCH_SIZE = 128

InputType = Literal["document", "query"]

# tenacity: retry on the two Voyage exception classes plus connection-level
# TimeoutError. Deterministic exceptions (bad key, unknown model) are not
# retryable and surface immediately.
_RETRYABLE = (RateLimitError, ServerError, TimeoutError, ConnectionError)


def _cost_for(model: str, input_tokens: int) -> float:
    """Cost in USD for `input_tokens` at `model`'s pricing. Missing model → 0.0."""
    per_million = PRICING.get(model)
    if per_million is None:
        _logger.warning("voyage_pricing_missing", model=model, input_tokens=input_tokens)
        return 0.0
    return (input_tokens / 1_000_000) * per_million


class VoyageEmbedder:
    """Sync Voyage embedding client with retry, batching, and per-call cost logging.

    Not thread-safe (the underlying `voyageai.Client` isn't documented as
    such). Fine for the batch ingest job and single-request Streamlit path.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        log_path: Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
        client: VoyageClient | None = None,
    ) -> None:
        if batch_size <= 0 or batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}], got {batch_size}")
        self._model = model
        self._log_path = log_path
        self._batch_size = batch_size
        # `client` injection is the seam tests use — no HTTP mock required.
        self._client = client or VoyageClient(api_key=api_key.get_secret_value())

    @property
    def model(self) -> str:
        return self._model

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def embed_documents(
        self,
        texts: list[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]:
        """Embed a list of document chunks. Batches internally. Preserves input order."""
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            out.extend(self._embed_batch(batch, input_type="document", run_id=run_id))
        return out

    def embed_query(
        self,
        text: str,
        *,
        run_id: str | None = None,
    ) -> list[float]:
        """Embed a single query string. `input_type="query"` per Voyage docs."""
        result = self._embed_batch([text], input_type="query", run_id=run_id)
        return result[0]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def _embed_batch(
        self,
        batch: list[str],
        *,
        input_type: InputType,
        run_id: str | None,
    ) -> list[list[float]]:
        """One Voyage API call. Wrapped in tenacity — do not call directly from tests."""
        started = time.perf_counter()
        result = self._client.embed(
            texts=batch,
            model=self._model,
            input_type=input_type,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        # Voyage's response exposes `total_tokens` (int) and `embeddings`
        # (list[list[float]]). Both are load-bearing here.
        total_tokens = int(getattr(result, "total_tokens", 0) or 0)
        embeddings: list[list[float]] = list(result.embeddings)

        # Explicit branch instead of f"embed_{input_type}s" to avoid an
        # `embed_querys` typo; distinct constants also make log filters trivial.
        operation = "embed_documents" if input_type == "document" else "embed_query"
        log_llm_call(
            self._log_path,
            provider="voyage",
            model=self._model,
            operation=operation,
            input_tokens=total_tokens,
            # Embedding calls have no output tokens — the response is a vector,
            # not generated text. Kept in the shared log schema for uniformity
            # with chat/judge calls that DO emit output tokens.
            output_tokens=0,
            cost_usd=_cost_for(self._model, total_tokens),
            latency_ms=elapsed_ms,
            run_id=run_id,
            # Embeddings don't use a prompt template, so no prompt_version to
            # record. Kept as an explicit None to match the log_llm_call kwargs.
            prompt_version=None,
            extra={"batch_size": len(batch), "input_type": input_type},
        )
        return embeddings


def make_embedder(
    *,
    api_key: SecretStr,
    model: str,
    log_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    client: Any | None = None,
) -> VoyageEmbedder:
    """Convenience factory. Kept thin — most callers instantiate directly."""
    return VoyageEmbedder(
        api_key=api_key,
        model=model,
        log_path=log_path,
        batch_size=batch_size,
        client=client,
    )
