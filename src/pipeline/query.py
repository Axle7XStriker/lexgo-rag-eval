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
from src.pipeline.pipeline_config import RetrieverConfig
from src.pipeline.prompts import OUT_OF_CORPUS_SENTINEL, load_prompt, render_user_template
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
    """Full return shape of `answer_question`. Consumed by Streamlit + eval loop."""

    query: str
    answer: str
    citations: list[RetrievedCitation]  # first-mention order, deduped
    retrieved_chunks: list[RetrievedChunk]  # full top-k, for UI + eval
    prompt_version: str
    latency_ms: float  # end-to-end wall time (embed + retrieve + generate)
    tokens_input: int  # generation only; embed is logged separately
    tokens_output: int
    cost_usd: float  # generation only; embed cost logged separately


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Enumerate chunks as `[N] source_id doc_path (pages P-Q, score S.SS)\\nTEXT`.

    The header line is what Claude reads to know which bracket to cite; the
    text below is what it grounds the answer in. Newline between chunks so a
    citation on one chunk can't accidentally get glued to the next chunk's
    header in the prompt.
    """
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = (
            f"[{i}] {c.source_id} {c.doc_path} "
            f"(pages {c.page_start}–{c.page_end}, score {c.score:.2f})"
        )
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
    """Reject unsupported or incomplete retriever configurations."""
    if retriever_config.kind not in ("dense", "hybrid"):
        raise ValueError(f"unknown retriever.kind: {retriever_config.kind!r}")
    if retriever_config.kind == "hybrid" and retriever_config.rrf_k is None:
        raise ValueError("hybrid retriever requires rrf_k to be set on RetrieverConfig")


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


def answer_question(
    *,
    query: str,
    embedder: VoyageEmbedder,
    store: VectorStore,
    generator: ClaudeGenerator,
    pipeline_tag: str,
    retriever: RetrieverConfig,
    run_id: str | None = None,
) -> QueryResult:
    """Run one query through the selected pipeline. Never raises for empty retrieval.

    Steps:
      1. Validate the retriever configuration.
      2. Load prompt v1 (cached).
      3. Embed `query` with Voyage.
      4. Retrieve top-k via `retriever.kind` (dense — pgvector cosine;
         hybrid — dense + BM25 fused via RRF).
      5. If retrieval is empty: short-circuit with the out-of-corpus sentinel;
         no generator call, cost 0.
      6. Format enumerated context block, substitute into the user template.
      7. Call Claude for the answer.
      8. Parse `[N]` citations, map to `RetrievedChunk` provenance, dedup.

    Wall-clock `latency_ms` covers the whole run (steps 1-8, whichever ran);
    individual provider tokens/cost land in `logs/llm_calls.jsonl` per the
    embedder and generator's own bookkeeping.
    """
    started = time.perf_counter()
    _validate_retriever_config(retriever)
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
        # empty DB, over-filtering).
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
        )

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
    )
