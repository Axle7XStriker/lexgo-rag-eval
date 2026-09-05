"""PDF text extraction — the front door to the ingest pipeline.

One public function: `extract_pdf(path) -> ExtractedDoc`. Every downstream
concern (chunking, hashing, page-range citations) reads from `ExtractedDoc`,
so keeping this module small + strict pays for itself.

Design notes worth remembering:
  - PyMuPDF is used for extraction. It is dual-licensed AGPL-3 / commercial;
    flagged in the pyproject dep comment so the license question is a
    conscious choice.
  - `content_hash` is sha256 over the page-joined text (`_text.join_pages`).
    That's the idempotency key ingest uses to decide "same doc, skip embed".
    Changing the join separator or the hash function is a schema break.
  - Whitespace-only pages are kept in `pages` (empty `text`) so 1-indexed
    page numbers stay honest — a chunk that spans pages 4-5 while page 4
    is blank still gets `page_start=4` truthfully. Page-range fidelity is
    the load-bearing infrastructure behind citations.
  - Encrypted / empty / non-PDF inputs raise `ValueError`. Ingest catches
    and logs — one bad doc must never wedge the whole run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from src.pipeline._text import join_pages, sha256_utf8

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

    # Magic-byte sniff before handing to pymupdf — pymupdf raises on non-PDFs
    # but its error message is less useful for a corpus health check.
    with path.open("rb") as f:
        head = f.read(len(PDF_MAGIC))
    if head != PDF_MAGIC:
        raise ValueError(f"{path} is not a PDF (first bytes: {head!r})")

    # `filetype="pdf"` forces PyMuPDF to treat the input as PDF and skip its
    # own magic sniff, giving us predictable failure modes.
    doc = pymupdf.open(path, filetype="pdf")
    try:
        if doc.is_encrypted:
            raise ValueError(f"{path} is encrypted; ingest does not decrypt PDFs")

        pages: list[PageText] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            # get_text("text") returns the layout-flow text — good enough for
            # OCW lecture PDFs and papers. Alternatives ("blocks", "words",
            # "dict") are richer but noisier for downstream chunking.
            text = page.get_text("text")
            pages.append(PageText(page_number=i + 1, text=text))

        # Reject only when EVERY page is whitespace-only — a doc with a blank
        # cover page but real content elsewhere is still a valid extraction.
        if not any(p.text.strip() for p in pages):
            raise ValueError(f"{path} yielded no extractable text on any page")

        # Metadata title falls back to None so downstream doesn't have to
        # sniff empty strings.
        raw_title = (doc.metadata or {}).get("title") or None
        title = raw_title.strip() if raw_title else None
        title = title or None  # empty-after-strip → None

        content_hash = sha256_utf8(join_pages([p.text for p in pages]))
        return ExtractedDoc(
            pages=pages,
            num_pages=doc.page_count,
            title=title,
            content_hash=content_hash,
        )
    finally:
        doc.close()
