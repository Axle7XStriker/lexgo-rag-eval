"""Cohere Rerank 3 client with retry, cost logging, and per-call accounting.

One class, `CohereReranker`. One public method: `rerank(query, chunks, top_n)`.

Design notes worth remembering:
  - Mirrors `VoyageEmbedder` / `ClaudeGenerator` line-for-line: sync client,
    `SecretStr` key, `tenacity` retries on transient exceptions only,
    fail-fast on unknown model in `__init__`, exactly one `log_llm_call` per
    successful API return (including the case where response-parsing then
    raises — we were billed, we log), dependency-injection seam via
    `client=` for tests.
  - Shared retry / cost / provider helpers live in
    `src.pipeline.cohere_utils` so a future second Cohere callsite picks up
    changes without duplication.
  - Tenacity is the single retry layer. The SDK's own `max_retries` is
    disabled (`max_retries=0`) at client construction — otherwise a
    transient 429 would be retried 2x by the SDK + 4x by tenacity = up to
    15 compounded attempts per rerank call, each with its own backoff.
  - Cohere's SDK is sync. One rerank call per query, no batching — the eval
    loop reranks 100 queries sequentially; parallelism is complexity without
    payoff at this scale.
  - Pricing is PER-SEARCH ($0.002 per API call), independent of query or
    document token counts. We log `input_tokens=0` + `output_tokens=0` to
    keep the `logs/llm_calls.jsonl` schema uniform across providers; the
    `cost_usd` field is the honest per-call charge.
  - Returns a NEW list of `RetrievedChunk` in reranked order with the
    Cohere relevance score (in [0, 1]) overwriting the retriever's score.
    Downstream renders this verbatim; document the semantic shift in any
    UI that surfaces the number.
  - Empty-chunks short-circuit: `rerank([])` returns an empty result with
    zero cost and zero latency, no API call. Guards against spending on a
    doomed call when upstream retrieval returned nothing (and matches the
    `answer_question` empty-retrieval short-circuit semantic).
  - `top_n` is clamped to `len(chunks)` before the API call — Cohere errors
    `BadRequestError` if `top_n > len(documents)`, which would waste a
    retry cycle on a deterministic client bug.
  - If the API call succeeds but response-parsing raises (malformed
    `index` or missing field — rare SDK regression), the call is logged
    with its honest per-search cost, then the caller sees a
    `RerankExecutionError` carrying both the underlying cause and
    `cost_usd` so the paid charge isn't silently lost from accounting.
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
    """A rerank API call that was BILLED but whose response couldn't be parsed.

    Raised only after Cohere returned successfully (HTTP 2xx + a response
    object) but subsequent parsing raised — e.g. a `results[i].index`
    outside `[0, len(chunks))` from an SDK regression. The underlying
    exception is preserved on `.cause` and the per-call charge on
    `.cost_usd` so the callers can preserve the paid cost rather than
    silently losing it.

    Transient API-level failures (429, 5xx, httpx transport errors) go
    through tenacity and reraise unchanged — they're not billed and
    don't need this wrapper.
    """

    def __init__(self, cause: Exception, *, cost_usd: float) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.cost_usd = cost_usd


class CohereReranker:
    """Sync Cohere Rerank client wrapper with retry, cost accounting, and per-call logging.

    Not thread-safe (the underlying `cohere.ClientV2` is not documented as
    such). Fine for the sequential eval loop and single-request Streamlit path.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        log_path: Path,
        client: cohere.ClientV2 | None = None,
    ) -> None:
        # Fail fast on unknown models. If a new Cohere model is being trialed,
        # add it to PRICING first.
        if model not in PRICING:
            raise ValueError(
                f"unknown Cohere model {model!r}; add its price to "
                f"COHERE_PRICING in src/pricing.py before use. "
                f"Known: {sorted(PRICING)}"
            )
        self._model = model
        self._log_path = log_path
        # `client` injection is the seam tests use — no HTTP mock required.
        # `max_retries=0` disables the SDK's own retry layer: tenacity owns
        # retries here. One retry layer, not two.
        self._client = client or cohere.ClientV2(
            api_key=api_key.get_secret_value(),
            max_retries=0,
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
        `top_n` is clamped to `len(chunks)` to avoid a Cohere 400 error on
        the trivially-oversized request.
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
        """One Cohere Rerank API call. Wrapped in tenacity — do not call directly from tests."""
        # Clamp before the API call: `top_n > len(documents)` is a
        # Cohere BadRequestError (4xx, non-retryable), which would surface
        # as a mysterious skip in the eval loop. Clamping here keeps the
        # caller's `top_n` as a UPPER bound rather than a strict count.
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
            # V2RerankResponse.results is sorted by relevance descending;
            # each entry has `index` (into `chunks`) and `relevance_score`
            # (in [0, 1]). Build the reranked list by mapping back through
            # `chunks` and overwriting score.
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
            # Response-parse errors (bad index, unexpected shape from an SDK
            # regression). Log the call before raising so logs/llm_calls.jsonl
            # matches the actual invoice, and wrap the exception so callers
            # see the paid cost (if any).
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
            # Rerank bills per-call; token counts don't apply. Kept in the
            # shared log schema for uniformity with chat/embed/judge entries.
            input_tokens=0,
            output_tokens=0,
            cost_usd=cost_usd,
            latency_ms=elapsed_ms,
            run_id=run_id,
            # Rerank doesn't use a prompt template, so no prompt_version.
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
    """Convenience factory. Kept thin — most callers instantiate directly."""
    return CohereReranker(api_key=api_key, model=model, log_path=log_path, client=client)
