"""Shared Streamlit widgets and startup helpers.

Both `pages/1_Demo.py` and `pages/2_Author.py` need the same settings
bootstrap and sidebar. Putting them here avoids two page files drifting apart.
Per-page browser-tab icons live on each page's `st.set_page_config(page_icon=...)`
call — filenames stay plain so `git`, `ls`, and shell completion stay ergonomic.
"""

from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

from src.config import Settings, get_settings
from src.observability import configure_logging


def load_settings_or_stop() -> Settings:
    """Validate config once per Streamlit run. On failure, render an actionable
    error and st.stop() — a missing API key should never yield an obscure trace
    deep inside a downstream call.
    """
    try:
        settings = get_settings()
    except ValidationError as e:
        st.error("Config failed to load — missing or invalid environment variables.")
        st.code(str(e))
        st.info("Copy `.env.example` to `.env` and fill in the required API keys.")
        st.stop()
    configure_logging(settings.log_level)
    return settings


def render_sidebar(settings: Settings) -> None:
    """Consistent sidebar across all pages: model IDs + config health."""
    with st.sidebar:
        st.subheader("Config")
        st.write(f"**Chat:** `{settings.chat_model}`")
        st.write(f"**Embed:** `{settings.embedding_model}`")
        st.write(f"**Rerank:** `{settings.rerank_model}`")
        st.subheader("Health")
        st.success("Config loaded — all required keys present.")


def render_page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
