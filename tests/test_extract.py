"""PDF extraction tests.

Fully offline — every fixture PDF is synthesized in-memory via PyMuPDF so
the test suite carries no binary blobs. Covers: page count / order,
whitespace-only pages preserved, content_hash stability + sensitivity,
and the three failure modes extract_pdf must raise on.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from src.pipeline.extract import ExtractedDoc, PageText, extract_pdf


def _write_pdf(path: Path, page_texts: list[str], *, encrypt: bool = False) -> None:
    """Write a fresh PDF at `path` with one page per string in `page_texts`.

    Kept as a helper so tests can construct pathological inputs (blank page,
    unicode page, encrypted doc) with one line each.
    """
    doc = pymupdf.open()
    try:
        for txt in page_texts:
            page = doc.new_page()
            if txt:
                page.insert_text((72, 72), txt, fontsize=11)
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


class TestExtractHappyPath:
    """Page count, order, and content_hash properties on a well-formed PDF."""

    def test_three_page_extract(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        _write_pdf(path, ["Page one text.", "Page two text.", "Page three text."])
        doc = extract_pdf(path)
        assert isinstance(doc, ExtractedDoc)
        assert doc.num_pages == 3
        assert len(doc.pages) == 3
        assert all(isinstance(p, PageText) for p in doc.pages)
        assert [p.page_number for p in doc.pages] == [1, 2, 3]
        for idx, expected in enumerate(["Page one", "Page two", "Page three"]):
            assert expected in doc.pages[idx].text, doc.pages[idx].text

    def test_content_hash_stable_across_calls(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        _write_pdf(path, ["Alpha.", "Beta."])
        h1 = extract_pdf(path).content_hash
        h2 = extract_pdf(path).content_hash
        assert h1 == h2

    def test_content_hash_sensitive_to_text(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.pdf"
        p2 = tmp_path / "b.pdf"
        _write_pdf(p1, ["Alpha."])
        _write_pdf(p2, ["Bravo."])
        assert extract_pdf(p1).content_hash != extract_pdf(p2).content_hash

    def test_whitespace_only_page_preserves_page_number(self, tmp_path: Path) -> None:
        # Middle page blank — page numbers must remain 1/2/3, not collapse to 1/2.
        path = tmp_path / "gap.pdf"
        _write_pdf(path, ["First.", "", "Third."])
        doc = extract_pdf(path)
        assert [p.page_number for p in doc.pages] == [1, 2, 3]
        assert doc.pages[1].text.strip() == ""
        assert "First" in doc.pages[0].text
        assert "Third" in doc.pages[2].text


class TestExtractFailureModes:
    """extract_pdf must raise on missing / non-PDF / encrypted / empty inputs."""

    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_pdf(tmp_path / "nope.pdf")

    def test_non_pdf_file_raises_valueerror(self, tmp_path: Path) -> None:
        path = tmp_path / "text.pdf"
        path.write_bytes(b"<html>not a pdf</html>")
        with pytest.raises(ValueError, match="not a PDF"):
            extract_pdf(path)

    def test_encrypted_pdf_raises_valueerror(self, tmp_path: Path) -> None:
        path = tmp_path / "locked.pdf"
        _write_pdf(path, ["Secret."], encrypt=True)
        with pytest.raises(ValueError, match="encrypted"):
            extract_pdf(path)

    def test_all_blank_pages_raises_valueerror(self, tmp_path: Path) -> None:
        path = tmp_path / "blank.pdf"
        _write_pdf(path, ["", "", ""])
        with pytest.raises(ValueError, match="no extractable text"):
            extract_pdf(path)
