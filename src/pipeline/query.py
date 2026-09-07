"""P1 query pipeline — query → embed → dense retrieve → generate answer with citations.

One public entrypoint: `answer_question(...)`. Takes already-built dependencies
(embedder, store, generator) so tests can inject fakes and Streamlit / eval
callers can share connections.

Design notes worth remembering:
  - No client construction here. All I/O flows through the injected
    dependencies, so this module is pure orchestration + parsing.
  - Prompt lives at prompts/answer/v1.md per `prompts/README.md`. Two sections
    separated by `# System` / `# User template` markdown headings. The version
    string is the caller's (matches the filename) and is what gets logged and
    later captured in the eval run manifest — the front-matter is human-facing
    metadata, not the source of truth.
  - Empty retrieval short-circuits to the out-of-corpus sentinel — no
    generator call, cost 0. Guards against wrong `pipeline_tag`, empty DB,
    or a degenerate filter and keeps the eval loop honest.
  - Citation parsing is a regex over `[N]` markers, deduped in first-mention
    order, with out-of-range markers dropped and logged. Claude occasionally
    invents markers past `top_k`; we don't propagate them into the eval.
"""

from __future__ import annotations

import functools
import re
import time
from dataclasses import dataclass
from pathlib import Path

from src.observability import get_logger
from src.pipeline.chunk import PIPELINE_TAG
from src.pipeline.embed import VoyageEmbedder
from src.pipeline.generate import ClaudeGenerator
from src.pipeline.store import RetrievedChunk, VectorStore

_logger = get_logger("query")

# P1 matrix from CLAUDE.md — dense top-10.
DEFAULT_TOP_K = 10

# `role` + `version` locate the prompt file at prompts/<role>/<version>.md.
# When we author a v2 answer prompt, bump PROMPT_VERSION here. Any change to
# the prompt file's semantics MUST come with a version bump (see prompts/README.md).
PROMPT_ROLE = "answer"
PROMPT_VERSION = "v1"

# Exact sentinel string the answer prompt tells Claude to return when the
# context doesn't cover the question. Kept in code (not just the prompt) so
# the empty-retrieval short-circuit produces identical output.
OUT_OF_CORPUS_SENTINEL = "This isn't covered in the provided course materials."

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# Section markers inside a prompt file. Markdown headings so the file also
# renders well in a browser / editor preview.
_SYSTEM_MARKER = "# System"
_USER_TEMPLATE_MARKER = "# User template"

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


@functools.cache
def _load_prompt(role: str, version: str) -> tuple[str, str]:
    """Load a prompt file, return (system_body, user_template).

    Cached per (role, version) — prompt files are immutable once shipped
    (see prompts/README.md), so a one-time read is safe.

    Raises:
      FileNotFoundError — no file at prompts/<role>/<version>.md.
      ValueError — file missing YAML front-matter terminator, missing
        `# System` / `# User template` sections, or template lacks the
        required `{question}` / `{context}` placeholders.
    """
    path = PROMPTS_DIR / role / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    text = path.read_text(encoding="utf-8")

    # Strip YAML front-matter if present. We don't need any of its fields at
    # runtime — the caller-supplied `version` is the source of truth. The
    # front-matter is metadata for humans opening the file.
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end == -1:
            raise ValueError(f"{path}: unterminated YAML front-matter")
        text = text[end + len("\n---\n") :]

    sys_idx = text.find(_SYSTEM_MARKER)
    user_idx = text.find(_USER_TEMPLATE_MARKER)
    if sys_idx == -1 or user_idx == -1 or user_idx <= sys_idx:
        raise ValueError(
            f"{path}: expected '{_SYSTEM_MARKER}' then '{_USER_TEMPLATE_MARKER}' sections"
        )
    system_body = text[sys_idx + len(_SYSTEM_MARKER) : user_idx].strip()
    user_template = text[user_idx + len(_USER_TEMPLATE_MARKER) :].strip()

    # Fail fast on template drift: the user template MUST contain both
    # placeholders. A silent missing placeholder would substitute nothing and
    # ship the raw literal to Claude — hard to notice, easy to prevent here.
    missing = [p for p in ("{question}", "{context}") if p not in user_template]
    if missing:
        raise ValueError(
            f"{path}: user template missing required placeholder(s): {', '.join(missing)}"
        )
    return system_body, user_template


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


def answer_question(
    *,
    query: str,
    embedder: VoyageEmbedder,
    store: VectorStore,
    generator: ClaudeGenerator,
    pipeline_tag: str = PIPELINE_TAG,
    top_k: int = DEFAULT_TOP_K,
    run_id: str | None = None,
) -> QueryResult:
    """Run one query through the P1 pipeline. Never raises for empty retrieval.

    Steps:
      1. Load prompt v1 (cached).
      2. Embed `query` with Voyage.
      3. Dense top-k retrieve from pgvector.
      4. If retrieval is empty: short-circuit with the out-of-corpus sentinel;
         no generator call, cost 0.
      5. Format enumerated context block, substitute into the user template.
      6. Call Claude for the answer.
      7. Parse `[N]` citations, map to `RetrievedChunk` provenance, dedup.

    Wall-clock `latency_ms` covers steps 2-6 (whichever ran); individual
    provider tokens/cost land in `logs/llm_calls.jsonl` per the embedder and
    generator's own bookkeeping.
    """
    started = time.perf_counter()
    system_body, user_template = _load_prompt(PROMPT_ROLE, PROMPT_VERSION)

    query_embedding = embedder.embed_query(query, run_id=run_id)
    retrieved = store.dense_search(pipeline_tag, query_embedding, k=top_k)

    if not retrieved:
        # Empty retrieval → the prompt would have Claude respond with the
        # sentinel anyway; skipping the call saves the token spend and keeps
        # cost accounting clean for the degenerate case (wrong pipeline_tag,
        # empty DB, over-filtering).
        _logger.info(
            "empty_retrieval",
            query=query,
            pipeline_tag=pipeline_tag,
            top_k=top_k,
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
    user_text = user_template.format(question=query, context=context_block)
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
