"""Text utilities shared across the ingest pipeline.

Kept in one place so the join separator + hash function are defined once —
a drift between `extract.py` (whole-document hash, idempotency key) and
`chunk.py` (per-chunk hash) would silently invalidate re-run behavior.

Not a public API. Underscore prefix marks it as internal to `src/pipeline`.
"""

from __future__ import annotations

import hashlib

# Canonical page separator. Used when joining PageText into one string for
# hashing or tokenization. Changing this string is a schema break — every
# persisted `documents.content_hash` and `chunks.content_hash` rehashes.
PAGE_JOIN = "\n\n"


def join_pages(page_texts: list[str]) -> str:
    """Deterministic page-join used everywhere the ingest pipeline concatenates pages."""
    return PAGE_JOIN.join(page_texts)


def sha256_utf8(text: str) -> str:
    """sha256 of `text` as utf-8 bytes, hex digest — the corpus hash primitive."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
