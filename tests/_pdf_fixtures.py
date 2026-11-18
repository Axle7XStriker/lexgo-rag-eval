"""Shared PDF-generation helper for tests that need a synthetic corpus file."""

from __future__ import annotations

from pathlib import Path

import pymupdf


def write_pdf(path: Path, page_texts: list[str], *, encrypt: bool = False) -> None:
    """Write a fresh PDF at `path` with one page per string in `page_texts`.

    Text is placed inside a wrapping textbox that fills the printable area.
    Empty strings produce blank pages (kept in the doc so page numbering stays
    honest). If `encrypt` is True the PDF is written with AES-256 + owner/user
    passwords — useful for the encrypted-input failure-mode test.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    try:
        for txt in page_texts:
            page = doc.new_page()
            if txt:
                rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
                page.insert_textbox(rect, txt, fontsize=11)
        if encrypt:
            doc.save(
                str(path),
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner",
                user_pw="user",
            )
        else:
            doc.save(str(path))
    finally:
        doc.close()
