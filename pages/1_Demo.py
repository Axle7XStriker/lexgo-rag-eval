"""Streamlit demo page — query → answer + citations + retrieved chunks.

P1 baseline wired end-to-end: dense Voyage top-10 -> Claude answer with `[N]`
citations. P2-P4 pipelines are not built yet, so the sidebar selector stays
disabled with a tooltip.
"""

from __future__ import annotations

import threading
import uuid

import psycopg
import streamlit as st

from src.config import Settings
from src.observability import get_logger
from src.pipeline.chunk import PIPELINE_TAG
from src.pipeline.embed import VoyageEmbedder
from src.pipeline.generate import ClaudeGenerator
from src.pipeline.prompts import OUT_OF_CORPUS_SENTINEL
from src.pipeline.query import DEFAULT_TOP_K, QueryResult, answer_question
from src.pipeline.store import VectorStore
from src.ui_helpers import load_settings_or_stop, render_page_header, render_sidebar

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
    "Query → answer + citations + retrieved chunks. P1 baseline is wired; P2–P4 land later.",
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
    st.selectbox(
        "Variant",
        ["P1 baseline", "P2 semantic", "P3 hybrid", "P4 hybrid+rerank"],
        index=0,
        disabled=True,
        help="Only P1 is wired today; P2–P4 land later in W3.",
    )

with st.form("query_form", clear_on_submit=False):
    question = st.text_area(
        "Ask a question about MIT 6.006 or 6.830",
        placeholder=(
            "e.g. What's the worst-case complexity of quicksort "
            "with median-of-medians pivot?"
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
            with st.spinner("Retrieving and generating…"), _query_lock:
                result = answer_question(
                    query=question,
                    embedder=embedder,
                    store=store,
                    generator=generator,
                    pipeline_tag=PIPELINE_TAG,
                    top_k=DEFAULT_TOP_K,
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
