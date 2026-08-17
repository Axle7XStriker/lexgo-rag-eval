"""Streamlit demo — query → answer + citations + retrieved chunks.

Skeleton only. The retrieval + generation pipeline lands in W3; this file
proves the app boots, config loads, and the UI shell is laid out.
"""

import streamlit as st
from pydantic import ValidationError

from src.config import get_settings
from src.observability import configure_logging

st.set_page_config(
    page_title="lexgo — RAG eval demo",
    page_icon="📚",
    layout="wide",
)


def _load_settings_or_stop():
    try:
        settings = get_settings()
    except ValidationError as e:
        st.error("Config failed to load — missing or invalid environment variables.")
        st.code(str(e))
        st.info("Copy `.env.example` to `.env` and fill in the required API keys.")
        st.stop()
    configure_logging(settings.log_level)
    return settings


settings = _load_settings_or_stop()

st.title("lexgo — RAG eval demo")
st.caption(
    "Rigorously-evaluated RAG over MIT 6.006 (Algorithms) + MIT 6.830 (Databases). "
    "Query → answer + citations + retrieved chunks. Pipeline lands W3."
)

with st.sidebar:
    st.subheader("Pipeline variant")
    st.selectbox(
        "Variant",
        ["V1 baseline", "V2 semantic", "V3 hybrid", "V4 hybrid+rerank"],
        index=0,
        disabled=True,
        help="Selector enabled once the retrieval pipeline lands (W3).",
    )

    st.subheader("Config")
    st.write(f"**Chat:** `{settings.chat_model}`")
    st.write(f"**Embed:** `{settings.embedding_model}`")
    st.write(f"**Rerank:** `{settings.rerank_model}`")

    st.subheader("Health")
    st.success("Config loaded — all required keys present.")

query = st.text_area(
    "Ask a question about MIT 6.006 or 6.830",
    placeholder="e.g. What's the worst-case complexity of quicksort with median-of-medians pivot?",
    disabled=True,
    help="Enabled once the pipeline lands (W3).",
)
st.button("Answer", disabled=True)

st.divider()

col_answer, col_chunks = st.columns([2, 1])
with col_answer:
    st.subheader("Answer")
    st.info("Answer + inline citations render here once V1 lands.")
with col_chunks:
    st.subheader("Retrieved chunks")
    st.info("Top-k retrieved chunks (with scores) render here.")
