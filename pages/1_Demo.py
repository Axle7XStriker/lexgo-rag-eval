"""Streamlit demo page — query → answer + citations + retrieved chunks.

P1 (dense), P2 (semantic + dense), and P3 (hybrid BM25 + dense, RRF-fused) are
wired end-to-end via the pipeline_config registry: pick one from the sidebar
and the same `answer_question` code path runs, dispatching on the pipeline's
`RetrieverConfig.kind`. P4 (hybrid + Cohere rerank) is not built yet — its
selector option stays disabled.
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
from src.pipeline.store import VectorStore
from src.ui_helpers import load_settings_or_stop, render_page_header, render_sidebar

# Selector labels → pipeline_config keys. Order matches the P1..P4 progression
# in the blog-post story. P4 is disabled in the selectbox until it lands.
_PIPELINE_OPTIONS: dict[str, str | None] = {
    "P1 baseline (dense)": "p1",
    "P2 semantic (dense)": "p2",
    "P3 hybrid (BM25 + dense, RRF)": "p3",
    "P4 hybrid + rerank": None,  # placeholder, disabled
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
    # Streamlit's selectbox doesn't natively disable individual options,
    # so we render the disabled P4 entry via `format_func` and re-filter
    # after selection — if the user somehow lands on P4, we surface a
    # notice and fall back to P1 rather than crashing on a None key.
    selected_label = st.selectbox(
        "Variant",
        list(_PIPELINE_OPTIONS.keys()),
        index=0,
        format_func=lambda label: (
            f"{label}  (coming in W3)" if _PIPELINE_OPTIONS[label] is None else label
        ),
        help="Select a retrieval pipeline. P4 lands later in W3.",
    )
    selected_key = _PIPELINE_OPTIONS[selected_label]
    if selected_key is None:
        st.info("P4 isn't wired yet — using P1 baseline.")
        selected_key = "p1"
    active_cfg = get_pipeline(selected_key)
    st.caption(
        f"tag: `{active_cfg.tag}`  ·  kind: `{active_cfg.retriever.kind}`  ·  "
        f"top_k: {active_cfg.retriever.top_k}"
    )

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
            with st.spinner("Retrieving and generating…"), _query_lock:
                result = answer_question(
                    query=question,
                    embedder=embedder,
                    store=store,
                    generator=generator,
                    pipeline_tag=active_cfg.tag,
                    retriever=active_cfg.retriever,
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
