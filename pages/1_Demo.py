"""Streamlit demo page — query → answer + citations + retrieved chunks."""

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

_PIPELINE_OPTIONS: dict[str, str] = {
    "P1 baseline (dense)": "p1",
    "P2 semantic (dense)": "p2",
    "P3 hybrid (BM25 + dense, RRF)": "p3",
    "P4 hybrid + Cohere rerank": "p4",
}

st.set_page_config(page_title="lexgo — demo", page_icon="📚", layout="wide")

_logger = get_logger("demo_page")

_query_lock = threading.Lock()  # Cached clients are shared and not thread-safe.

settings = load_settings_or_stop()
render_sidebar(settings)

render_page_header(
    "Demo",
    "Rigorously-evaluated RAG over MIT 6.006 (Algorithms) + MIT 6.830 (Databases). "
    "Query → answer + citations + retrieved chunks. Four pipelines wired: P1 (dense), "
    "P2 (semantic + dense), P3 (hybrid BM25 + dense, RRF-fused), P4 (hybrid + Cohere rerank).",
)


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
def _get_reranker(_settings: Settings, model: str) -> CohereReranker:
    return CohereReranker(
        api_key=_settings.cohere_api_key,
        model=model,
        log_path=_settings.llm_call_log,
    )


@st.cache_resource
def _get_store(database_url: str) -> VectorStore:
    """Open the store once. The cached resource outlives a single query."""
    store = VectorStore(database_url)
    store.__enter__()
    return store


with st.sidebar:
    st.subheader("Pipeline variant")
    selected_label = st.selectbox(
        "Variant",
        list(_PIPELINE_OPTIONS.keys()),
        index=0,
        help="Select a retrieval pipeline.",
    )
    selected_key = _PIPELINE_OPTIONS[selected_label]
    active_cfg = get_pipeline(selected_key)
    if active_cfg.reranker is not None:
        rerank_chip = (
            f"rerank: `{active_cfg.reranker.provider}/{active_cfg.reranker.model}`, "
            f"top_n={active_cfg.reranker.top_n}"
        )
    else:
        rerank_chip = "rerank: n/a"
    caption_parts = [
        f"tag: `{active_cfg.tag}`",
        f"chunk_set: `{active_cfg.chunk_set_id}`",
        f"kind: `{active_cfg.retriever.kind}`",
        f"top_k: {active_cfg.retriever.top_k}",
        rerank_chip,
    ]
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
        run_id = uuid.uuid4().hex  # Correlates this question's provider calls.
        try:
            embedder = _get_embedder(settings)
            generator = _get_generator(settings)
            store = _get_store(settings.database_url)
            reranker = (
                _get_reranker(settings, active_cfg.reranker.model)
                if active_cfg.reranker is not None
                else None
            )
            with st.spinner("Retrieving and generating…"), _query_lock:
                result = answer_question(
                    query=question,
                    embedder=embedder,
                    store=store,
                    generator=generator,
                    chunk_set_id=active_cfg.chunk_set_id,
                    retriever=active_cfg.retriever,
                    reranker=reranker,
                    reranker_config=active_cfg.reranker,
                    run_id=run_id,
                )
            st.session_state["last_result"] = result
        except psycopg.Error as exc:
            _get_store.clear()
            _logger.exception("demo_db_error", run_id=run_id)
            st.error(f"Database error ({type(exc).__name__}).")
            st.info("Run `make db-up && make ingest`, then retry.")
        except Exception as exc:
            _logger.exception("demo_query_error", run_id=run_id)
            st.error(f"Query failed ({type(exc).__name__}).")
            st.info("Check your API keys and try again; see server logs for details.")

result: QueryResult | None = st.session_state.get("last_result")


def _render_run_details(r: QueryResult) -> None:
    with st.expander("Run details"):
        st.write(f"**Latency:** {r.latency_ms:.0f} ms")
        st.write(f"**Tokens in / out:** {r.tokens_input} / {r.tokens_output}")
        st.write(f"**Generation cost:** ${r.cost_usd:.4f}")
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
