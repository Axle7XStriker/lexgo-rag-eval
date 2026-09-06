"""Corpus hash primitives — the ingest idempotency key lives here.

Two invariants held in one place so `extract.py` and `chunk.py` can't drift:
  - `PAGE_JOIN` — the canonical page separator. Every persisted
    `documents.content_hash` and `chunks.content_hash` was computed against
    a string joined with this exact separator. Changing it invalidates
    every hash in the DB (schema break).
  - `sha256_utf8` — the hash function. Same rule.

Keeping both here means "how do we canonicalize + hash text" is defined
once. A drift on either would silently break the "unchanged corpus → zero
API calls" short-circuit in `scripts/ingest.py`.
"""

from __future__ import annotations

import hashlib

# Canonical page separator. Used when joining PageText into one string for
# hashing or tokenization. Changing this string is a schema break.
PAGE_JOIN = "\n\n"


def join_pages(page_texts: list[str]) -> str:
    """Deterministic page-join used everywhere the ingest pipeline concatenates pages."""
    return PAGE_JOIN.join(page_texts)


def sha256_utf8(text: str) -> str:
    """sha256 of `text` as utf-8 bytes, hex digest — the corpus hash primitive."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
