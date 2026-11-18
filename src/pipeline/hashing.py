"""Corpus hash primitives

Page separator and hash used as the ingest idempotency key.
"""

from __future__ import annotations

import hashlib

PAGE_JOIN = "\n\n"  # Changing this invalidates every stored content hash.


def join_pages(page_texts: list[str]) -> str:
    return PAGE_JOIN.join(page_texts)


def sha256_utf8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
