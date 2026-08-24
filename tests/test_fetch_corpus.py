"""Tests for the corpus fetcher — narrowly scoped to the manual-placement
short-circuit and the CLI argument surface. The HTTP/parsing paths are
exercised by real runs against OCW; recreating them here would be a
mock-per-call exercise with low signal.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import fetch_corpus
from scripts.corpus_manifest import ManifestEntry
from src.qa_schema import SourceId


@pytest.fixture
def textbook_entry() -> ManifestEntry:
    """A copyrighted-textbook entry — publisher URL that will not return a PDF.

    Semantically identical to the real A4/B4 entries: `optional=True` so the
    fetcher's inevitable %PDF-magic failure reports `missing_optional`, not
    `failed`. A human dropping a legally-obtained PDF at `dest_path` opts
    it into retrieval via the standard `dest.exists()` short-circuit.
    """
    return ManifestEntry(
        source_id=SourceId.A4,
        description="test-only CLRS reference",
        kind="direct_pdf",
        urls=("https://example.invalid/clrs.pdf",),
        dest_path="6.006/textbook/A4_test_ref.pdf",
        optional=True,
    )


def test_manual_placement_yields_skipped_present(
    textbook_entry: ManifestEntry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The manual-placement contract: if a human drops a legally-obtained PDF
    # at dest_path, the fetcher must NOT hit the network — the standard
    # `dest.exists() && size > 0` short-circuit wins. This is what lets a
    # textbook entry participate in retrieval without a downloadable URL.
    monkeypatch.setattr(fetch_corpus, "CORPUS_ROOT", tmp_path)
    dest = tmp_path / textbook_entry.dest_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"%PDF-1.4\n... fake but non-empty ...")

    # Patch the module-under-test's own attribute (not urllib.request.urlopen)
    # so a future `from urllib.request import urlopen` at the top of
    # fetch_corpus.py doesn't silently defeat this guard.
    def _boom(url: str) -> bytes:
        raise AssertionError("_http_get must not be called when dest exists")

    monkeypatch.setattr(fetch_corpus, "_http_get", _boom)

    result = fetch_corpus.fetch_entry(textbook_entry)
    assert result.status == "skipped_present"


def test_optional_textbook_missing_reports_missing_optional(
    textbook_entry: ManifestEntry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no dest on disk, the fetcher attempts each URL; because publisher
    # pages return HTML (not PDF bytes), the %PDF magic check fails. With
    # `optional=True`, that failure must convert to `missing_optional`,
    # not `failed` — otherwise CI breaks on every cold run.
    monkeypatch.setattr(fetch_corpus, "CORPUS_ROOT", tmp_path)

    def _fake_get(url: str) -> bytes:
        return b"<!DOCTYPE html><p>Publisher landing page, not a PDF.</p>"

    monkeypatch.setattr(fetch_corpus, "_http_get", _fake_get)

    result = fetch_corpus.fetch_entry(textbook_entry)
    assert result.status == "missing_optional"
    assert result.error is not None  # error string preserved for the summary


def test_only_flag_accepts_new_source_ids() -> None:
    # Argparse-level guard: --only must accept every current SourceId value.
    # A drift (new enum member without a matching --only choice) would make
    # the entire bucket unfilterable from the CLI.
    for source_id in SourceId:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.fetch_corpus",
                "--dry-run",
                "--only",
                source_id.value,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, (
            f"--only {source_id.value} exited {result.returncode}\n"
            f"stderr: {result.stderr}"
        )
