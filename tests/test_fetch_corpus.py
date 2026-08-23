"""Tests for the corpus fetcher — narrowly scoped to the reference-only
short-circuit and CLI argument surface. The HTTP/parsing paths are
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
def reference_only_entry() -> ManifestEntry:
    return ManifestEntry(
        source_id=SourceId.A4,
        description="test-only CLRS reference",
        kind="direct_pdf",
        urls=("https://example.invalid/clrs.pdf",),
        dest_path="6.006/textbook/A4_test_ref.pdf",
        reference_only=True,
    )


def test_reference_only_returns_status_without_http(
    reference_only_entry: ManifestEntry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Redirect CORPUS_ROOT so we don't touch the real corpus/ tree.
    monkeypatch.setattr(fetch_corpus, "CORPUS_ROOT", tmp_path)

    # Any HTTP call is a bug: reference_only must short-circuit before urlopen.
    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("urlopen must not be called for reference_only entries")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    result = fetch_corpus.fetch_entry(reference_only_entry)
    assert result.status == "reference_only"
    assert result.error is None
    # Dest must NOT be created — the entry is a citation placeholder only.
    assert not (tmp_path / reference_only_entry.dest_path).exists()


def test_reference_only_yields_skipped_present_when_dest_exists(
    reference_only_entry: ManifestEntry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If a human manually places a legally-obtained PDF at dest_path, the
    # existing `dest.exists() && size > 0` short-circuit must win over the
    # reference_only branch — this is what makes the "manual placement"
    # path work with zero code changes downstream.
    monkeypatch.setattr(fetch_corpus, "CORPUS_ROOT", tmp_path)
    dest = tmp_path / reference_only_entry.dest_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"%PDF-1.4\n... fake but non-empty ...")

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("urlopen must not be called when dest exists")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    result = fetch_corpus.fetch_entry(reference_only_entry)
    assert result.status == "skipped_present"


def test_only_flag_accepts_new_source_ids() -> None:
    # Argparse-level check: --only must accept every current SourceId value.
    # A drift here (new enum member without a matching --only choice) would
    # make the entire bucket unfilterable from the CLI.
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
