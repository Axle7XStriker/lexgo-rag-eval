"""Cohere rerank client with retry and per-call cost logging.

Retries, pricing, and the provider name live in `src.pipeline.cohere_utils`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cohere
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.observability import log_llm_call
from src.pipeline.cohere_utils import (
    PRICING,
    PROVIDER,
    cost_for,
    is_retryable,
)
from src.pipeline.store import RetrievedChunk


@dataclass(frozen=True)
class RerankResult:
    """Return shape of `CohereReranker.rerank` — the pipeline reads all three.

    `chunks` is a NEW list in reranked order; its `score` field holds the
    Cohere relevance score (in [0, 1], higher = more relevant), not the
    original retriever score. `cost_usd` + `latency_ms` are per-call —
    `answer_question` surfaces them separately from generator cost so the
    rerank stage cost can be attributed separately.
    """

    chunks: list[RetrievedChunk]
    cost_usd: float
    latency_ms: float


class RerankExecutionError(RuntimeError):
    """A billed rerank response that could not be parsed.

    `.cause` is the parse error and `.cost_usd` is the charge already paid.
    """

    def __init__(self, cause: Exception, *, cost_usd: float) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.cost_usd = cost_usd


class CohereReranker:
    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        log_path: Path,
        client: cohere.ClientV2 | None = None,
    ) -> None:
        if model not in PRICING:
            raise ValueError(
                f"unknown Cohere model {model!r}; add its price to "
                f"COHERE_PRICING in src/pricing.py before use. "
                f"Known: {sorted(PRICING)}"
            )
        self._model = model
        self._log_path = log_path
        self._client = client or cohere.ClientV2(
            api_key=api_key.get_secret_value(),
            max_retries=0,  # This client already retries. The SDK default would retry again.
        )

    @property
    def model(self) -> str:
        return self._model

    def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
        run_id: str | None = None,
    ) -> RerankResult:
        """Rerank `chunks` by Cohere relevance to `query`; return top-`top_n`.

        Empty `chunks` → empty result, zero cost, zero latency, no API call.
        `top_n` is clamped to the candidate count.
        """
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        if not chunks:
            return RerankResult(chunks=[], cost_usd=0.0, latency_ms=0.0)
        return self._call(query=query, chunks=chunks, top_n=top_n, run_id=run_id)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception(is_retryable),
        reraise=True,
    )
    def _call(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
        run_id: str | None,
    ) -> RerankResult:
        effective_top_n = min(top_n, len(chunks))

        started = time.perf_counter()
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=[c.text for c in chunks],
            top_n=effective_top_n,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        cost_usd = cost_for(self._model, num_searches=1)

        reranked: list[RetrievedChunk] = []
        try:
            for r in response.results:
                if not 0 <= r.index < len(chunks):
                    raise IndexError(
                        f"Cohere rerank result index {r.index} outside [0, {len(chunks)})"
                    )
                source = chunks[r.index]
                reranked.append(
                    RetrievedChunk(
                        chunk_id=source.chunk_id,
                        document_id=source.document_id,
                        doc_path=source.doc_path,
                        source_id=source.source_id,
                        chunk_set_id=source.chunk_set_id,
                        chunk_index=source.chunk_index,
                        text=source.text,
                        page_start=source.page_start,
                        page_end=source.page_end,
                        score=r.relevance_score,
                    )
                )
        except Exception as parse_exc:
            # Record paid calls even when their response cannot be consumed.
            log_llm_call(
                self._log_path,
                provider=PROVIDER,
                model=self._model,
                operation="rerank_chunks",
                input_tokens=0,
                output_tokens=0,
                cost_usd=cost_usd,
                latency_ms=elapsed_ms,
                run_id=run_id,
                prompt_version=None,
                extra={
                    "num_candidates": len(chunks),
                    "top_n": effective_top_n,
                    "returned": len(reranked),
                    "parse_error": type(parse_exc).__name__,
                },
            )
            raise RerankExecutionError(parse_exc, cost_usd=cost_usd) from parse_exc

        log_llm_call(
            self._log_path,
            provider=PROVIDER,
            model=self._model,
            operation="rerank_chunks",
            input_tokens=0,
            output_tokens=0,
            cost_usd=cost_usd,
            latency_ms=elapsed_ms,
            run_id=run_id,
            prompt_version=None,
            extra={
                "num_candidates": len(chunks),
                "top_n": effective_top_n,
                "returned": len(reranked),
            },
        )
        return RerankResult(chunks=reranked, cost_usd=cost_usd, latency_ms=elapsed_ms)


def make_reranker(
    *,
    api_key: SecretStr,
    model: str,
    log_path: Path,
    client: cohere.ClientV2 | None = None,
) -> CohereReranker:
    return CohereReranker(api_key=api_key, model=model, log_path=log_path, client=client)
