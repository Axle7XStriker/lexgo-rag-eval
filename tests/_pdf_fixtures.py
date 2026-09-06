"""Shared PDF-generation helper for tests that need a synthetic corpus file.

Both `test_extract.py` (extraction contract) and `test_ingest.py` (end-to-end
orchestrator against a real Postgres) need to write a tiny PDF to a `tmp_path`
and hand its path to the code under test. Kept in one place so the fixture
shape stays consistent — a drift here would mean the two test suites are
exercising subtly different inputs.

Not a public API. Underscore prefix marks it as test-internal.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf


def write_pdf(path: Path, page_texts: list[str], *, encrypt: bool = False) -> None:
    """Write a fresh PDF at `path` with one page per string in `page_texts`.

    Text is placed inside a wrapping textbox that fills the printable area,
    so long inputs get onto the page in full rather than being clipped at
    the right margin (PyMuPDF's `insert_text` is single-line). This lets
    tests pass "word " * N patterns without having to know page geometry.

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
                # Full printable area with a 1-inch margin on every side.
                rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
                page.insert_textbox(rect, txt, fontsize=11)
        if encrypt:
            # Owner + user passwords + AES-256 → is_encrypted True on reopen.
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
