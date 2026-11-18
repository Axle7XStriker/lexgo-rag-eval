"""PDF text extraction"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from src.pipeline.hashing import join_pages, sha256_utf8

PDF_MAGIC = b"%PDF"


@dataclass(frozen=True)
class PageText:
    """One page of extracted text. `page_number` is 1-indexed to match PDF viewers."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedDoc:
    """Full extraction result for one PDF.

    `content_hash` is the sha256 of the page-joined text — the ingest
    idempotency key. Callers compare it against the row in `documents`
    to decide whether to re-embed.
    """

    pages: list[PageText]
    num_pages: int
    title: str | None
    content_hash: str


def _strip_null(text: str) -> str:
    """Drop `0x00`. Postgres text and tsvector reject it; PyMuPDF sometimes emits it."""
    if "\x00" not in text:
        return text
    return text.replace("\x00", "")


def extract_pdf(path: Path) -> ExtractedDoc:
    """Extract text from `path` and return an `ExtractedDoc`.

    Raises:
      FileNotFoundError — path does not exist.
      ValueError — path is not a PDF (magic bytes don't match), the PDF is
        encrypted, or every page comes back whitespace-only. Ingest catches
        and skips these; the whole run does not fail on one bad doc.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    with path.open("rb") as f:
        head = f.read(len(PDF_MAGIC))
    if head != PDF_MAGIC:
        raise ValueError(f"{path} is not a PDF (first bytes: {head!r})")

    doc = pymupdf.open(path, filetype="pdf")
    try:
        if doc.is_encrypted:
            raise ValueError(f"{path} is encrypted; ingest does not decrypt PDFs")

        pages: list[PageText] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = _strip_null(page.get_text("text"))
            pages.append(PageText(page_number=i + 1, text=text))

        if not any(p.text.strip() for p in pages):
            raise ValueError(f"{path} yielded no extractable text on any page")

        raw_title = (doc.metadata or {}).get("title") or None
        title = _strip_null(raw_title).strip() if raw_title else None
        title = title or None

        content_hash = sha256_utf8(join_pages([p.text for p in pages]))
        return ExtractedDoc(
            pages=pages,
            num_pages=doc.page_count,
            title=title,
            content_hash=content_hash,
        )
    finally:
        doc.close()
