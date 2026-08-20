"""Streamlit demo page — query → answer + citations + retrieved chunks.

Skeleton only. The retrieval + generation pipeline lands in W3; this page
proves the app boots, config loads, and the UI shell is laid out.
"""

import streamlit as st

from src.ui_helpers import load_settings_or_stop, render_page_header, render_sidebar

st.set_page_config(page_title="lexgo — demo", page_icon="📚", layout="wide")

settings = load_settings_or_stop()
render_sidebar(settings)

render_page_header(
    "Demo",
    "Rigorously-evaluated RAG over MIT 6.006 (Algorithms) + MIT 6.830 (Databases). "
    "Query → answer + citations + retrieved chunks. Pipeline lands W3.",
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

st.text_area(
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
