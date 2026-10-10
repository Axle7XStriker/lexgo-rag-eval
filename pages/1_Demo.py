"""Streamlit demo page — query → answer + citations + retrieved chunks.

All four pipelines are wired end-to-end via the pipeline_config registry:
pick one from the sidebar and the same `answer_question` code path runs,
dispatching on the pipeline's `RetrieverConfig.kind` and (for P4) its
`RerankerConfig`. The Demo UI only ever shows the FINAL chunk set the LLM
saw — for P4 that's the post-rerank 5, not the pre-rerank 20 (consistent
with how recall@5 is measured in the eval loop).
"""

from __future__ import annotations

import threading
import uuid

import psycopg
import streamlit as st

from src.config import Settings
from src.observability import get_logger
from src.pipeline.embed import VoyageEmbedder
from src.pipeline.generate import ClaudeGenerator
from src.pipeline.pipeline_config import get_pipeline
from src.pipeline.prompts import OUT_OF_CORPUS_SENTINEL
from src.pipeline.query import QueryResult, answer_question
from src.pipeline.rerank import CohereReranker
from src.pipeline.store import VectorStore
from src.ui_helpers import load_settings_or_stop, render_page_header, render_sidebar

# Selector labels → pipeline_config keys. Order matches the P1..P4 progression
# in the blog-post story.
_PIPELINE_OPTIONS: dict[str, str] = {
    "P1 baseline (dense)": "p1",
    "P2 semantic (dense)": "p2",
    "P3 hybrid (BM25 + dense, RRF)": "p3",
    "P4 hybrid + Cohere rerank": "p4",
}

st.set_page_config(page_title="lexgo — demo", page_icon="📚", layout="wide")

_logger = get_logger("demo_page")

# VoyageEmbedder, ClaudeGenerator, and VectorStore all document "not thread-safe"
# in their module docstrings, and @st.cache_resource shares them across every
# browser session. Serialize the whole query flow behind one lock — for a
# portfolio demo the cost of one-at-a-time queries beats a concurrency bug in
# the blog-post's live demo. Swap for psycopg_pool + per-session SDK clients if
# real traffic ever shows up.
_query_lock = threading.Lock()

settings = load_settings_or_stop()
render_sidebar(settings)

render_page_header(
    "Demo",
    "Rigorously-evaluated RAG over MIT 6.006 (Algorithms) + MIT 6.830 (Databases). "
    "Query → answer + citations + retrieved chunks. Four pipelines wired: P1 (dense), "
    "P2 (semantic + dense), P3 (hybrid BM25 + dense, RRF-fused), P4 (hybrid + Cohere rerank).",
)


# The leading underscore on `_settings` tells @st.cache_resource to skip hashing
# it (Settings holds SecretStr fields that shouldn't be hashed anyway). Load-
# bearing per Streamlit's cache-key rules — do not rename.
@st.cache_resource
def _get_embedder(_settings: Settings) -> VoyageEmbedder:
    return VoyageEmbedder(
        api_key=_settings.voyage_api_key,
        model=_settings.embedding_model,
        log_path=_settings.llm_call_log,
    )


@st.cache_resource
def _get_generator(_settings: Settings) -> ClaudeGenerator:
    return ClaudeGenerator(
        api_key=_settings.anthropic_api_key,
        model=_settings.chat_model,
        log_path=_settings.llm_call_log,
    )


@st.cache_resource
def _get_reranker(_settings: Settings) -> CohereReranker:
    # Built eagerly on first P4 selection; cached for the process lifetime.
    # Settings already requires COHERE_API_KEY, so no "missing key" branch here.
    return CohereReranker(
        api_key=_settings.cohere_api_key,
        model=_settings.rerank_model,
        log_path=_settings.llm_call_log,
    )


@st.cache_resource
def _get_store(database_url: str) -> VectorStore:
    # VectorStore is a context manager (opens the psycopg connection + registers
    # pgvector on __enter__). Streamlit reruns the script per interaction, so a
    # `with` block would reconnect per query; enter once and let the process
    # shutdown close the socket. No atexit — it would pin dead stores forever
    # across cache.clear() cycles.
    store = VectorStore(database_url)
    store.__enter__()
    return store


with st.sidebar:
    st.subheader("Pipeline variant")
    selected_label = st.selectbox(
        "Variant",
        list(_PIPELINE_OPTIONS.keys()),
        index=0,
        help="Select a retrieval pipeline. Each uses the same answer prompt + judge.",
    )
    selected_key = _PIPELINE_OPTIONS[selected_label]
    active_cfg = get_pipeline(selected_key)
    caption_parts = [
        f"tag: `{active_cfg.tag}`",
        f"kind: `{active_cfg.retriever.kind}`",
        f"top_k: {active_cfg.retriever.top_k}",
    ]
    if active_cfg.reranker is not None:
        caption_parts.append(
            f"rerank: `{active_cfg.reranker.provider}/{active_cfg.reranker.model}`, "
            f"top_n={active_cfg.reranker.top_n}"
        )
    st.caption("  ·  ".join(caption_parts))

with st.form("query_form", clear_on_submit=False):
    question = st.text_area(
        "Ask a question about MIT 6.006 or 6.830",
        placeholder=(
            "e.g. What's the worst-case complexity of quicksort with median-of-medians pivot?"
        ),
        key="demo_question",
    )
    submitted = st.form_submit_button("Answer")

if submitted:
    if not question.strip():
        st.warning("Enter a question before submitting.")
    else:
        # Per-query run_id: groups the (embed, generate) pair for one question
        # in logs/llm_calls.jsonl. Session-wide would conflate every query.
        run_id = uuid.uuid4().hex
        try:
            embedder = _get_embedder(settings)
            generator = _get_generator(settings)
            store = _get_store(settings.database_url)
            # Reranker only built for pipelines that configure one (P4).
            # answer_question's both-or-neither invariant means we pass
            # BOTH reranker + reranker_config here, or NEITHER.
            reranker = _get_reranker(settings) if active_cfg.reranker is not None else None
            with st.spinner("Retrieving and generating…"), _query_lock:
                result = answer_question(
                    query=question,
                    embedder=embedder,
                    store=store,
                    generator=generator,
                    pipeline_tag=active_cfg.tag,
                    retriever=active_cfg.retriever,
                    reranker=reranker,
                    reranker_config=active_cfg.reranker,
                    run_id=run_id,
                )
            st.session_state["last_result"] = result
        except psycopg.Error as exc:
            # psycopg's OperationalError message can include the full DSN with
            # credentials on connection failure; never render it verbatim. The
            # broken connection is also unusable — drop the cached store so the
            # next attempt reconnects.
            _get_store.clear()
            _logger.exception("demo_db_error", run_id=run_id)
            st.error(f"Database error ({type(exc).__name__}).")
            st.info("Run `make db-up && make ingest`, then retry.")
        except Exception as exc:
            # Provider errors (anthropic / voyageai) after tenacity retries, or
            # anything else. Show the type but not the message — provider
            # messages are usually safe, but "usually" isn't a security stance.
            _logger.exception("demo_query_error", run_id=run_id)
            st.error(f"Query failed ({type(exc).__name__}).")
            st.info("Check your API keys and try again; see server logs for details.")

result: QueryResult | None = st.session_state.get("last_result")


def _render_run_details(r: QueryResult) -> None:
    with st.expander("Run details"):
        st.write(f"**Latency:** {r.latency_ms:.0f} ms")
        st.write(f"**Tokens in / out:** {r.tokens_input} / {r.tokens_output}")
        # Generation cost only — Voyage embed cost is logged per-call to
        # logs/llm_calls.jsonl, not summed here.
        st.write(f"**Generation cost:** ${r.cost_usd:.4f}")
        # Rerank cost only shown when it ran (P4). Hiding the $0.0000 row
        # for P1-P3 keeps the panel compact and signals "this pipeline
        # didn't do a rerank step" by its absence.
        if r.rerank_cost_usd > 0:
            st.write(f"**Rerank cost:** ${r.rerank_cost_usd:.4f}")
        st.write(f"**Prompt version:** {r.prompt_version}")


st.divider()

col_answer, col_chunks = st.columns([2, 1])

with col_answer:
    st.subheader("Answer")
    if result is None:
        st.info("Ask a question above to see an answer with `[N]` citations.")
    elif result.answer == OUT_OF_CORPUS_SENTINEL:
        st.info(result.answer)
        _render_run_details(result)
    else:
        st.markdown(result.answer)
        if result.citations:
            st.markdown("**Citations**")
            # Numerical order reads more naturally in a UI list than the
            # first-mention order query.py preserves on RetrievedCitation. The
            # data model still carries first-mention order for eval consumers.
            for cit in sorted(result.citations, key=lambda c: c.marker):
                st.markdown(
                    f"- **[{cit.marker}]** `{cit.source_id}` {cit.doc_path} — "
                    f"pages {cit.page_start}–{cit.page_end} (score {cit.score:.2f})"
                )
        else:
            st.caption("Model didn't emit any `[N]` citations.")
        _render_run_details(result)

with col_chunks:
    if result is None:
        st.subheader("Retrieved chunks")
        st.info("Top-k retrieved chunks (with scores) render here.")
    else:
        st.subheader(f"Retrieved chunks (top {len(result.retrieved_chunks)})")
        for i, chunk in enumerate(result.retrieved_chunks, start=1):
            with st.container(border=True):
                st.markdown(
                    f"**[{i}]** `{chunk.source_id}` · {chunk.doc_path} · "
                    f"pp {chunk.page_start}–{chunk.page_end} · score {chunk.score:.2f}"
                )
                with st.expander("Text"):
                    st.write(chunk.text)
