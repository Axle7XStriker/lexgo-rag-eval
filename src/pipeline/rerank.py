"""Cohere Rerank 3 client with retry, cost logging, and per-call accounting.

One class, `CohereReranker`. One public method: `rerank(query, chunks, top_n)`.

Design notes worth remembering:
  - Mirrors `VoyageEmbedder` / `ClaudeGenerator` line-for-line: sync client,
    `SecretStr` key, `tenacity` retries on transient exceptions only,
    fail-fast on unknown model in `__init__`, exactly one `log_llm_call` per
    API call, dependency-injection seam via `client=` for tests.
  - Shared retry / cost / provider helpers live in
    `src.pipeline.cohere_utils` so a future second Cohere callsite picks up
    changes without duplication.
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
    blog's cost story can attribute spend to the rerank stage specifically.
    """

    chunks: list[RetrievedChunk]
    cost_usd: float
    latency_ms: float


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
        # Fail fast on unknown models: a silent $0 per-call cost would poison
        # the blog's cost story with no visible signal. If a new Cohere model
        # is being trialed, add it to PRICING first.
        if model not in PRICING:
            raise ValueError(
                f"unknown Cohere model {model!r}; add its price to "
                f"COHERE_PRICING in src/pricing.py before use. "
                f"Known: {sorted(PRICING)}"
            )
        self._model = model
        self._log_path = log_path
        # `client` injection is the seam tests use — no HTTP mock required.
        self._client = client or cohere.ClientV2(api_key=api_key.get_secret_value())

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

        # Cohere's V2RerankResponse.results is already sorted by relevance
        # descending; each entry has `index` (into our `chunks` list) and
        # `relevance_score` (float in [0, 1]). Build the reranked list by
        # mapping back through `chunks` and overwriting score.
        reranked: list[RetrievedChunk] = []
        for r in response.results:
            source = chunks[r.index]
            reranked.append(
                RetrievedChunk(
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    doc_path=source.doc_path,
                    source_id=source.source_id,
                    pipeline=source.pipeline,
                    chunk_index=source.chunk_index,
                    text=source.text,
                    page_start=source.page_start,
                    page_end=source.page_end,
                    score=r.relevance_score,
                )
            )

        cost_usd = cost_for(self._model, num_searches=1)
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
