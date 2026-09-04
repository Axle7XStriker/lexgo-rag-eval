"""PDF text extraction — the front door to the ingest pipeline.

One public function: `extract_pdf(path) -> ExtractedDoc`. Every downstream
concern (chunking, hashing, page-range citations) reads from `ExtractedDoc`,
so keeping this module small + strict pays for itself.

Design notes worth remembering:
  - PyMuPDF (`fitz`) is used for extraction. It is dual-licensed AGPL-3 /
    commercial — flagged in the ingest plan; not litigated here.
  - `content_hash` is sha256 over the `\\n\\n`-joined page text. That's the
    idempotency key ingest uses to decide "same doc, skip embed". Any
    change to the join separator is a schema break (rehashes everything).
  - Whitespace-only pages are kept in `pages` (empty `text`) so 1-indexed
    page numbers stay honest — a chunk that spans pages 4-5 while page 4
    is blank still gets `page_start=4` truthfully.
  - Encrypted / empty / non-PDF inputs raise `ValueError`. Ingest catches
    and logs — one bad doc must never wedge the whole run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pymupdf

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


def _joined_text(pages: list[PageText]) -> str:
    """Canonical page-joined text — the input to `content_hash`.

    Kept as a private helper so tests and callers agree on the exact
    join separator. Changing this string is a schema break.
    """
    return "\n\n".join(p.text for p in pages)


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

    # Magic-byte sniff before handing to fitz — fitz raises on non-PDFs but
    # its error message is less useful for a corpus health check.
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

        if not any(p.text.strip() for p in pages):
            raise ValueError(f"{path} yielded no extractable text on any page")

        # Metadata title falls back to None so downstream doesn't have to
        # sniff empty strings.
        raw_title = (doc.metadata or {}).get("title") or None
        title = raw_title.strip() if raw_title else None
        title = title or None  # empty-after-strip → None

        content_hash = hashlib.sha256(_joined_text(pages).encode("utf-8")).hexdigest()
        return ExtractedDoc(
            pages=pages,
            num_pages=doc.page_count,
            title=title,
            content_hash=content_hash,
        )
    finally:
        doc.close()
