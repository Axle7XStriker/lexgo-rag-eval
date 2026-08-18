"""lexgo — RAG eval demo (Streamlit entrypoint).

This file is the landing page. Two subpages live in `pages/`:
  - 📚 Demo — query the RAG pipeline (lands W3)
  - ✍️ Author — hand-curate the 100-Q&A golden set

Streamlit auto-discovers the `pages/` directory and renders navigation in
the sidebar. Booting via `make app` opens all three pages.
"""

import streamlit as st

from src.ui_helpers import load_settings_or_stop, render_sidebar

st.set_page_config(
    page_title="lexgo — RAG eval",
    page_icon="🔎",
    layout="wide",
)

settings = load_settings_or_stop()
render_sidebar(settings)

st.title("lexgo — RAG eval")
st.caption(
    "Rigorously-evaluated multi-source RAG over MIT 6.006 (Algorithms) + "
    "MIT 6.830 (Databases). 4 pipeline variants, 100 hand-authored golden Q&As, "
    "honest numbers."
)

st.markdown(
    """
    ### Pages

    - **📚 Demo** — ask the RAG pipeline a question, see the answer + citations
      + retrieved chunks. Pipeline itself lands in W3; the shell is up now.
    - **✍️ Author** — the hand-curation UI for the 100-Q&A golden set. Add,
      browse, edit, delete records against `evals/golden/qa.jsonl` with live
      distribution tracking (40 factual / 25 cross-source synthesis / 20
      paraphrase / 10 out-of-corpus / 5 adversarial).

    Both pages are in the sidebar.
    """
)

st.divider()

st.markdown(
    """
    ### Project context

    Full scoping, corpus breakdown, working conventions, and 4-week schedule
    live in [`CLAUDE.md`](https://github.com/Axle7XStriker/lexgo-rag-eval/blob/main/CLAUDE.md).

    Repo: [github.com/Axle7XStriker/lexgo-rag-eval](https://github.com/Axle7XStriker/lexgo-rag-eval)
    """
)
