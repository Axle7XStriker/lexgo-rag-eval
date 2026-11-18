"""Tests for the corpus fetcher — narrowly scoped to the manual-placement
short-circuit and the CLI argument surface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import fetch_corpus
from scripts.corpus_manifest import ManifestEntry
from src.qa_schema import SourceId


@pytest.fixture
def optional_entry() -> ManifestEntry:
    return ManifestEntry(
        source_id=SourceId.A1,
        description="test fixture — optional entry",
        kind="direct_pdf",
        urls=("https://example.invalid/fixture.pdf",),
        dest_path="test/fixture.pdf",
        optional=True,
    )


def test_manual_placement_yields_skipped_present(
    optional_entry: ManifestEntry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch_corpus, "CORPUS_ROOT", tmp_path)
    dest = tmp_path / optional_entry.dest_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"%PDF-1.4\n... non-empty stub ...")

    def _boom(url: str) -> bytes:
        raise AssertionError("_http_get must not be called when dest exists")

    monkeypatch.setattr(fetch_corpus, "_http_get", _boom)

    result = fetch_corpus.fetch_entry(optional_entry)
    assert result.status == "skipped_present"


def test_optional_fetch_failure_reports_missing_optional(
    optional_entry: ManifestEntry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch_corpus, "CORPUS_ROOT", tmp_path)

    def _fake_get(url: str) -> bytes:
        return b"<!DOCTYPE html><p>Not a PDF.</p>"

    monkeypatch.setattr(fetch_corpus, "_http_get", _fake_get)

    result = fetch_corpus.fetch_entry(optional_entry)
    assert result.status == "missing_optional"
    assert result.error is not None


def test_only_flag_accepts_every_source_id() -> None:
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
            timeout=30,
        )
        assert result.returncode == 0, (
            f"--only {source_id.value} exited {result.returncode}\nstderr: {result.stderr}"
        )
