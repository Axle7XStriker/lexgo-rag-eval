"""Voyage embedding client with retry, batching, and per-call cost logging."""

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
from voyageai.error import (
    APIConnectionError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    Timeout,
    TryAgain,
)

from src.observability import get_logger, log_llm_call
from src.pricing import VOYAGE_PRICING as PRICING
from src.pricing import voyage_cost

_logger = get_logger("embed")

DEFAULT_BATCH_SIZE = 64  # Headroom under MAX_BATCH_SIZE.
MAX_BATCH_SIZE = 128

InputType = Literal["document", "query"]

_RETRYABLE = (
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    APIConnectionError,
    Timeout,
    TryAgain,
)


def _cost_for(model: str, input_tokens: int) -> float:
    """Cost in USD for `input_tokens` at `model`'s pricing. Missing model → 0.0."""
    cost = voyage_cost(model, input_tokens)
    if cost is None:
        _logger.warning("voyage_pricing_missing", model=model, input_tokens=input_tokens)
        return 0.0
    return cost


class VoyageEmbedder:
    """Sync Voyage embedding client with retry, batching, and per-call cost logging."""

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
        if model not in PRICING:
            raise ValueError(
                f"unknown Voyage model {model!r}; add its price to PRICING "
                f"in src/pipeline/embed.py before use. Known: {sorted(PRICING)}"
            )
        self._model = model
        self._log_path = log_path
        self._batch_size = batch_size
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
        """Embed one retrieval query."""
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
        started = time.perf_counter()
        result = self._client.embed(
            texts=batch,
            model=self._model,
            input_type=input_type,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        total_tokens = int(getattr(result, "total_tokens", 0) or 0)
        embeddings: list[list[float]] = list(result.embeddings)

        operation = "embed_documents" if input_type == "document" else "embed_query"
        log_llm_call(
            self._log_path,
            provider="voyage",
            model=self._model,
            operation=operation,
            input_tokens=total_tokens,
            output_tokens=0,
            cost_usd=_cost_for(self._model, total_tokens),
            latency_ms=elapsed_ms,
            run_id=run_id,
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
    return VoyageEmbedder(
        api_key=api_key,
        model=model,
        log_path=log_path,
        batch_size=batch_size,
        client=client,
    )
