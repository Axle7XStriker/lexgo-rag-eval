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
    # If a non-empty file already exists at `dest_path`, the fetcher must
    # short-circuit before any network call — regardless of the entry's URL
    # reachability. Any HTTP call in this branch is a bug.
    monkeypatch.setattr(fetch_corpus, "CORPUS_ROOT", tmp_path)
    dest = tmp_path / optional_entry.dest_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"%PDF-1.4\n... non-empty stub ...")

    # Patch the module-under-test's own attribute (not urllib.request.urlopen)
    # so a future `from urllib.request import urlopen` at the top of
    # fetch_corpus.py doesn't silently defeat this guard.
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
    # With no dest on disk, the fetcher attempts each URL. When every URL
    # yields bytes that fail the %PDF magic check, and the entry is
    # `optional=True`, the outcome must be `missing_optional` — not
    # `failed` — so the run exits 0.
    monkeypatch.setattr(fetch_corpus, "CORPUS_ROOT", tmp_path)

    def _fake_get(url: str) -> bytes:
        return b"<!DOCTYPE html><p>Not a PDF.</p>"

    monkeypatch.setattr(fetch_corpus, "_http_get", _fake_get)

    result = fetch_corpus.fetch_entry(optional_entry)
    assert result.status == "missing_optional"
    assert result.error is not None  # error string preserved for the summary


def test_only_flag_accepts_every_source_id() -> None:
    # The --only argparse choice list must stay in sync with SourceId. A
    # drift (new enum member without a matching --only choice) would make
    # the entire bucket unfilterable from the CLI.
    #
    # NOTE: subprocess-per-value is a coarse but robust check — it exercises
    # the same argparse instance the real CLI uses. A cheaper in-process
    # equivalent would require factoring the parser out of main(); doing so
    # is on the follow-up list.
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
            # Fail-fast if an interpreter hangs (import-time blocking IO,
            # infinite loop in a future refactor) so CI doesn't stall.
            timeout=30,
        )
        assert result.returncode == 0, (
            f"--only {source_id.value} exited {result.returncode}\n"
            f"stderr: {result.stderr}"
        )
