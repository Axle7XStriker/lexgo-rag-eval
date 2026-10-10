"""Query pipeline — query → embed → dense retrieve → generate answer with citations.

One public entrypoint: `answer_question(...)`. Takes already-built dependencies
(embedder, store, generator) so tests can inject fakes and Streamlit / eval
callers can share connections.

Design notes worth remembering:
  - No client construction here. All I/O flows through the injected
    dependencies, so this module is pure orchestration + parsing.
  - Empty retrieval short-circuits to the out-of-corpus sentinel — no
    generator call, cost 0. Guards against wrong `pipeline_tag`, empty DB,
    or a degenerate filter and keeps the eval loop honest.
  - Citation parsing is a regex over `[N]` markers, deduped in first-mention
    order, with out-of-range markers dropped and logged. Claude occasionally
    invents markers past `top_k`; we don't propagate them into the eval.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from src.observability import get_logger
from src.pipeline.embed import VoyageEmbedder
from src.pipeline.generate import ClaudeGenerator
from src.pipeline.pipeline_config import RerankerConfig, RetrieverConfig
from src.pipeline.prompts import OUT_OF_CORPUS_SENTINEL, load_prompt, render_user_template
from src.pipeline.rerank import CohereReranker
from src.pipeline.store import RetrievedChunk, VectorStore

_logger = get_logger("query")

# `role` + `version` locate the prompt file at prompts/<role>/<version>.md.
# When we author a v2 answer prompt, bump PROMPT_VERSION here. Any change to
# the prompt file's semantics MUST come with a version bump.
PROMPT_ROLE = "answer"
PROMPT_VERSION = "v1"

# Placeholders the answer user template MUST contain — enforced at load time.
_REQUIRED_USER_PLACEHOLDERS: tuple[str, ...] = ("{question}", "{context}")

# Match [N] citation brackets. Anchored with `[` and `]` — Claude occasionally
# emits `(1)` or bare `1.` prose references; those are intentionally ignored.
_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class RetrievedCitation:
    """A citation the model actually made — bracket marker → chunk provenance.

    Only chunks Claude cited land here; the full top-k retrieval is preserved
    separately in `QueryResult.retrieved_chunks` for UI display and eval
    metrics (retrieval recall@5 needs all of them, not just the cited subset).
    """

    marker: int  # 1..k, as it appeared in the answer text
    chunk_id: int
    doc_path: str
    source_id: str
    page_start: int
    page_end: int
    score: float


@dataclass(frozen=True)
class QueryResult:
    """Full return shape of `answer_question`. Consumed by Streamlit + eval loop.

    Cost fields are per-stage so the eval + UI can attribute spend
    separately: `cost_usd` is generation-only (Claude), `rerank_cost_usd`
    is Cohere Rerank (0.0 when the pipeline has no reranker). Voyage embed
    cost lives per-call in `logs/llm_calls.jsonl` and is not duplicated here.
    """

    query: str
    answer: str
    citations: list[RetrievedCitation]  # first-mention order, deduped
    retrieved_chunks: list[RetrievedChunk]  # final top-k seen by the LLM
    prompt_version: str
    latency_ms: float  # end-to-end wall time (embed + retrieve + rerank + generate)
    tokens_input: int  # generation only; embed is logged separately
    tokens_output: int
    cost_usd: float  # generation only; embed cost logged separately
    rerank_cost_usd: float = 0.0  # Cohere rerank only; 0.0 when no reranker


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Enumerate chunks as `[N] source_id doc_path (pages P-Q)\\nTEXT`.

    The header line is what Claude reads to know which bracket to cite; the
    text below is what it grounds the answer in. Newline between chunks so a
    citation on one chunk can't accidentally get glued to the next chunk's
    header in the prompt.

    Score is intentionally omitted — it's not useful signal to the model,
    and it carries different semantics across pipelines (cosine for dense,
    small RRF value for hybrid, Cohere relevance for rerank-active P4), so
    surfacing it to Claude would at best be ignored and at worst be misread
    as a confidence hint. UI consumers still see the score via
    `RetrievedChunk.score` directly.
    """
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] {c.source_id} {c.doc_path} (pages {c.page_start}–{c.page_end})"
        lines.append(f"{header}\n{c.text}")
    return "\n\n".join(lines)


def _parse_citations(
    answer: str,
    retrieved: list[RetrievedChunk],
) -> list[RetrievedCitation]:
    """Extract `[N]` markers from `answer`, dedup preserving first-mention order.

    Out-of-range markers (Claude occasionally invents `[9]` when only 3 chunks
    were retrieved) are dropped and logged. We prefer honest citation
    precision numbers over pretending invented markers exist.
    """
    seen: set[int] = set()
    ordered_markers: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        n = int(match.group(1))
        if n in seen:
            continue
        seen.add(n)
        ordered_markers.append(n)

    citations: list[RetrievedCitation] = []
    for n in ordered_markers:
        if not 1 <= n <= len(retrieved):
            _logger.warning(
                "citation_out_of_range",
                marker=n,
                top_k=len(retrieved),
            )
            continue
        chunk = retrieved[n - 1]
        citations.append(
            RetrievedCitation(
                marker=n,
                chunk_id=chunk.chunk_id,
                doc_path=chunk.doc_path,
                source_id=chunk.source_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                score=chunk.score,
            )
        )
    return citations


def _validate_retriever_config(retriever_config: RetrieverConfig) -> None:
    """Reject unsupported or incomplete retriever configurations.

    Hybrid-only: `rrf_k` must be set and positive. Catching `rrf_k <= 0`
    here (not deeper in `_rrf_fuse`) means a misconfigured pipeline fails
    before `embed_query` spends a Voyage call — the pipeline embed cost
    for a run that was always going to fail is strictly waste.
    """
    if retriever_config.kind not in ("dense", "hybrid"):
        raise ValueError(f"unknown retriever.kind: {retriever_config.kind!r}")
    if retriever_config.kind == "hybrid":
        if retriever_config.rrf_k is None:
            raise ValueError("hybrid retriever requires rrf_k to be set on RetrieverConfig")
        if retriever_config.rrf_k <= 0:
            raise ValueError(
                f"hybrid retriever rrf_k must be positive, got {retriever_config.rrf_k}"
            )


def _retrieve(
    *,
    store: VectorStore,
    retriever_config: RetrieverConfig,
    pipeline_tag: str,
    query: str,
    query_embedding: list[float],
) -> list[RetrievedChunk]:
    """Retrieve chunks closely related to the query."""
    _validate_retriever_config(retriever_config)
    if retriever_config.kind == "dense":
        return store.dense_search(pipeline_tag, query_embedding, k=retriever_config.top_k)

    assert retriever_config.rrf_k is not None
    return store.hybrid_search(
        pipeline_tag,
        query_embedding,
        query_text=query,
        k=retriever_config.top_k,
        rrf_k=retriever_config.rrf_k,
    )


def _validate_reranker_pair(
    reranker: CohereReranker | None,
    reranker_config: RerankerConfig | None,
) -> None:
    """Enforce the both-or-neither invariant on the reranker/config pair.

    Having one without the other is a caller bug — a reranker with no
    `top_n` has no way to know how many results to return, and a config
    with no client has no way to actually rerank. Catching this at the
    boundary beats a confusing AttributeError five frames deep.
    """
    if (reranker is None) != (reranker_config is None):
        raise ValueError(
            "reranker and reranker_config must both be provided or both omitted; "
            f"got reranker={type(reranker).__name__ if reranker else None}, "
            f"reranker_config={reranker_config}"
        )


def answer_question(
    *,
    query: str,
    embedder: VoyageEmbedder,
    store: VectorStore,
    generator: ClaudeGenerator,
    pipeline_tag: str,
    retriever: RetrieverConfig,
    reranker: CohereReranker | None = None,
    reranker_config: RerankerConfig | None = None,
    run_id: str | None = None,
) -> QueryResult:
    """Run one query through the selected pipeline. Never raises for empty retrieval.

    Steps:
      1. Validate the retriever + reranker configurations.
      2. Load prompt v1 (cached).
      3. Embed `query` with Voyage.
      4. Retrieve top-k via `retriever.kind` (dense — pgvector cosine;
         hybrid — dense + BM25 fused via RRF).
      5. If retrieval is empty: short-circuit with the out-of-corpus sentinel;
         no generator call, no rerank call, cost 0.
      6. If reranker set: call Cohere Rerank; replace `retrieved` with the
         top-`reranker_config.top_n` reranked list (score field now holds
         Cohere relevance in [0, 1]).
      7. Format enumerated context block, substitute into the user template.
      8. Call Claude for the answer.
      9. Parse `[N]` citations, map to `RetrievedChunk` provenance, dedup.

    `reranker` and `reranker_config` must both be provided or both omitted
    (invariant enforced in `_validate_reranker_pair`). For P1-P3 both stay
    None; P4 passes both.

    Wall-clock `latency_ms` covers the whole run (steps 1-9, whichever ran);
    individual provider tokens/cost land in `logs/llm_calls.jsonl` per each
    client's own bookkeeping. `QueryResult.rerank_cost_usd` is populated
    from the Cohere call when the reranker ran (0.0 otherwise).
    """
    started = time.perf_counter()
    _validate_retriever_config(retriever)
    _validate_reranker_pair(reranker, reranker_config)
    system_body, user_template = load_prompt(
        PROMPT_ROLE,
        PROMPT_VERSION,
        required_user_placeholders=_REQUIRED_USER_PLACEHOLDERS,
    )

    query_embedding = embedder.embed_query(query, run_id=run_id)
    retrieved = _retrieve(
        store=store,
        retriever_config=retriever,
        pipeline_tag=pipeline_tag,
        query=query,
        query_embedding=query_embedding,
    )

    if not retrieved:
        # Empty retrieval → the prompt would have Claude respond with the
        # sentinel anyway; skipping the call saves the token spend and keeps
        # cost accounting clean for the degenerate case (wrong pipeline_tag,
        # empty DB, over-filtering). Reranker is also skipped — no candidates
        # to re-order means no reason to spend $0.002.
        _logger.info(
            "empty_retrieval",
            query=query,
            pipeline_tag=pipeline_tag,
            retriever_kind=retriever.kind,
            top_k=retriever.top_k,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return QueryResult(
            query=query,
            answer=OUT_OF_CORPUS_SENTINEL,
            citations=[],
            retrieved_chunks=[],
            prompt_version=PROMPT_VERSION,
            latency_ms=elapsed_ms,
            tokens_input=0,
            tokens_output=0,
            cost_usd=0.0,
            rerank_cost_usd=0.0,
        )

    rerank_cost_usd = 0.0
    if reranker is not None:
        # `_validate_reranker_pair` already enforced that reranker_config
        # is non-None when reranker is non-None; assert for the type checker.
        assert reranker_config is not None
        rr = reranker.rerank(
            query=query,
            chunks=retrieved,
            top_n=reranker_config.top_n,
            run_id=run_id,
        )
        retrieved = rr.chunks
        rerank_cost_usd = rr.cost_usd

    context_block = _format_context(retrieved)
    user_text = render_user_template(
        user_template,
        {"question": query, "context": context_block},
    )
    gen = generator.generate(
        system=system_body,
        user=user_text,
        prompt_version=PROMPT_VERSION,
        run_id=run_id,
    )

    citations = _parse_citations(gen.text, retrieved)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return QueryResult(
        query=query,
        answer=gen.text,
        citations=citations,
        retrieved_chunks=retrieved,
        prompt_version=PROMPT_VERSION,
        latency_ms=elapsed_ms,
        tokens_input=gen.input_tokens,
        tokens_output=gen.output_tokens,
        cost_usd=gen.cost_usd,
        rerank_cost_usd=rerank_cost_usd,
    )
