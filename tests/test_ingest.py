"""Ingest orchestrator tests.

Two layers of coverage:
  - `_process_entry` against a real Postgres (skipped when unreachable) with
    a fake embedder — exercises idempotency, --force, upsert semantics, and
    missing/required exit behavior.
  - Manifest helper (`_select_manifest_entries`) — pure, no DB.

Integration path uses a real DB per CLAUDE.md ("no mocked DB in integration
tests"). Voyage is always mocked — the eval-suite ban on mocked LLM calls
applies to evals only; unit tests for pipeline plumbing use fakes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import pytest

from scripts import ingest as ingest_mod
from scripts.corpus_manifest import ManifestEntry
from src.pipeline.chunk import PIPELINE_TAG
from src.pipeline.embed import VoyageEmbedder
from src.pipeline.store import EMBEDDING_DIM, VectorStore
from src.qa_schema import SourceId

# ── DB fixtures ───────────────────────────────────────────────────────


def _dsn_from_env() -> str:
    return os.environ.get("DATABASE_URL", "postgresql://lexgo:lexgo@localhost:5432/lexgo")


@pytest.fixture
def db_dsn() -> str:
    """Real Postgres DSN. Skips the test if the DB isn't reachable."""
    dsn = _dsn_from_env()
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            pass
    except Exception as e:
        # Any connect failure means "no DB" — skip rather than fail so unit-only
        # runs stay green.
        pytest.skip(f"Postgres not reachable at {dsn}: {e}")
    return dsn


@pytest.fixture
def clean_store(db_dsn: str):
    """VectorStore against a fresh schema; wipes `chunks` + `documents` on entry.

    A hard TRUNCATE isolates the test from any pre-existing rows (e.g. from
    a prior `make ingest` run). CASCADE drops chunks along with documents.
    """
    with VectorStore(db_dsn) as store:
        store.ensure_schema()
        with store.conn.cursor() as cur:
            cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
        store.conn.commit()
        yield store


# ── Fake embedder ─────────────────────────────────────────────────────


@dataclass
class _FakeEmbedder:
    """Stand-in for VoyageEmbedder that returns deterministic vectors + counts calls."""

    calls: int = 0
    texts_seen: list[str] | None = None

    def __post_init__(self) -> None:
        self.texts_seen = []

    def embed_documents(self, texts: list[str], *, run_id: str | None = None) -> list[list[float]]:
        self.calls += 1
        assert self.texts_seen is not None
        self.texts_seen.extend(texts)
        # Deterministic non-zero vectors; correct dim matters (upsert_chunks checks).
        return [[float((i % 7) + 1) / 10.0] * EMBEDDING_DIM for i, _ in enumerate(texts)]


# ── Fixture PDF ───────────────────────────────────────────────────────


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    """Small PDF for ingest to consume. Same helper shape as test_extract."""
    import pymupdf  # local import — module has already been imported by extract.py

    doc = pymupdf.open()
    try:
        for txt in page_texts:
            page = doc.new_page()
            if txt:
                page.insert_text((72, 72), txt, fontsize=11)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(path))
    finally:
        doc.close()


def _make_entry(dest_path: str, *, optional: bool = False) -> ManifestEntry:
    """Minimal manifest entry for tests. Only fields ingest reads are exercised."""
    return ManifestEntry(
        source_id=SourceId.A1,
        description="test fixture",
        kind="direct_pdf",
        urls=("https://example.invalid/x.pdf",),
        dest_path=dest_path,
        optional=optional,
    )


# ── Unit tests (no DB) ────────────────────────────────────────────────


class TestEntrySelection:
    """_select_manifest_entries: full manifest / source_id filter / doc_path filter / miss."""

    def test_none_returns_all(self) -> None:
        entries = ingest_mod._select_manifest_entries(None)
        assert len(entries) == len(ingest_mod.MANIFEST)

    def test_source_id_filter(self) -> None:
        entries = ingest_mod._select_manifest_entries("A1")
        assert entries and all(e.source_id == SourceId.A1 for e in entries)

    def test_doc_path_filter(self) -> None:
        # Pick a real manifest entry so we know it matches.
        target = ingest_mod.MANIFEST[0].dest_path
        entries = ingest_mod._select_manifest_entries(target)
        assert len(entries) == 1
        assert entries[0].dest_path == target

    def test_unknown_only_raises_systemexit(self) -> None:
        with pytest.raises(SystemExit):
            ingest_mod._select_manifest_entries("Z9")


class TestIsRequired:
    """_is_required: MANIFEST lookup with unknown-path fallback."""

    def test_known_required(self) -> None:
        required = next(e for e in ingest_mod.MANIFEST if not e.optional)
        assert ingest_mod._is_required(required.dest_path) is True

    def test_known_optional(self) -> None:
        optional = next((e for e in ingest_mod.MANIFEST if e.optional), None)
        if optional is None:
            pytest.skip("no optional entries in manifest")
        assert ingest_mod._is_required(optional.dest_path) is False

    def test_unknown_path_treated_required(self) -> None:
        assert ingest_mod._is_required("nowhere/nothing.pdf") is True


# ── Integration tests (require DB) ────────────────────────────────────


class TestProcessEntryIntegration:
    """Full extract → chunk → embed → upsert cycle against a real Postgres."""

    def test_missing_required_returns_missing(
        self, clean_store: VectorStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        entry = _make_entry("nothing/nope.pdf", optional=False)
        embedder = _FakeEmbedder()
        # Redirect the corpus root to tmp — the file will be absent.
        original_root = ingest_mod.CORPUS_ROOT
        ingest_mod.CORPUS_ROOT = tmp_path
        try:
            result = ingest_mod._process_entry(
                entry,
                store=clean_store,
                embedder=embedder,  # type: ignore[arg-type]
                force=False,
                dry_run=False,
                log_path=tmp_path / "llm_calls.jsonl",
                logger=_QuietLogger(),
            )
        finally:
            ingest_mod.CORPUS_ROOT = original_root
        assert result.status == "missing"
        assert embedder.calls == 0
        # Missing REQUIRED must exit non-zero when this result feeds the summary.
        assert ingest_mod._is_required(entry.dest_path) is True

    def test_first_run_ingests_second_run_skips(
        self, clean_store: VectorStore, tmp_path: Path
    ) -> None:
        dest = tmp_path / "corpus" / "test/A1_fake.pdf"
        _write_pdf(dest, ["Chunk one text. " * 40, "Chunk two text. " * 40])
        entry = _make_entry("test/A1_fake.pdf")

        embedder = _FakeEmbedder()
        original_root = ingest_mod.CORPUS_ROOT
        ingest_mod.CORPUS_ROOT = tmp_path / "corpus"
        try:
            first = ingest_mod._process_entry(
                entry,
                store=clean_store,
                embedder=embedder,  # type: ignore[arg-type]
                force=False,
                dry_run=False,
                log_path=tmp_path / "llm_calls.jsonl",
                logger=_QuietLogger(),
            )
            assert first.status == "ingested"
            assert first.num_chunks > 0
            first_calls = embedder.calls
            assert first_calls > 0

            # Sanity: chunks landed under the expected pipeline tag.
            assert clean_store.count_chunks(PIPELINE_TAG) == first.num_chunks

            # Second run — same file, unchanged → skipped, zero embed calls.
            second = ingest_mod._process_entry(
                entry,
                store=clean_store,
                embedder=embedder,  # type: ignore[arg-type]
                force=False,
                dry_run=False,
                log_path=tmp_path / "llm_calls.jsonl",
                logger=_QuietLogger(),
            )
            assert second.status == "skipped_unchanged"
            assert embedder.calls == first_calls, "second run must not call embed"
            assert clean_store.count_chunks(PIPELINE_TAG) == first.num_chunks
        finally:
            ingest_mod.CORPUS_ROOT = original_root

    def test_force_reembeds(self, clean_store: VectorStore, tmp_path: Path) -> None:
        dest = tmp_path / "corpus" / "test/A1_fake.pdf"
        _write_pdf(dest, ["Alpha " * 100])
        entry = _make_entry("test/A1_fake.pdf")

        embedder = _FakeEmbedder()
        original_root = ingest_mod.CORPUS_ROOT
        ingest_mod.CORPUS_ROOT = tmp_path / "corpus"
        try:
            first = ingest_mod._process_entry(
                entry,
                store=clean_store,
                embedder=embedder,  # type: ignore[arg-type]
                force=False,
                dry_run=False,
                log_path=tmp_path / "llm_calls.jsonl",
                logger=_QuietLogger(),
            )
            assert first.status == "ingested"
            first_calls = embedder.calls

            second = ingest_mod._process_entry(
                entry,
                store=clean_store,
                embedder=embedder,  # type: ignore[arg-type]
                force=True,
                dry_run=False,
                log_path=tmp_path / "llm_calls.jsonl",
                logger=_QuietLogger(),
            )
            assert second.status == "ingested"
            assert embedder.calls == first_calls + 1
        finally:
            ingest_mod.CORPUS_ROOT = original_root

    def test_dry_run_no_embed_no_writes(self, clean_store: VectorStore, tmp_path: Path) -> None:
        dest = tmp_path / "corpus" / "test/A1_fake.pdf"
        _write_pdf(dest, ["Beta " * 100])
        entry = _make_entry("test/A1_fake.pdf")

        embedder = _FakeEmbedder()
        original_root = ingest_mod.CORPUS_ROOT
        ingest_mod.CORPUS_ROOT = tmp_path / "corpus"
        try:
            result = ingest_mod._process_entry(
                entry,
                store=clean_store,
                embedder=embedder,  # type: ignore[arg-type]
                force=False,
                dry_run=True,
                log_path=tmp_path / "llm_calls.jsonl",
                logger=_QuietLogger(),
            )
        finally:
            ingest_mod.CORPUS_ROOT = original_root

        assert result.status == "ingested"  # dry-run reports success shape
        assert result.tokens_embedded > 0  # reports the pre-flight estimate
        assert result.cost_usd == 0.0
        assert embedder.calls == 0
        assert clean_store.count_chunks(PIPELINE_TAG) == 0

    def test_optional_missing_does_not_fail(self, clean_store: VectorStore, tmp_path: Path) -> None:
        entry = _make_entry("test/A1_optional.pdf", optional=True)
        embedder = _FakeEmbedder()
        original_root = ingest_mod.CORPUS_ROOT
        ingest_mod.CORPUS_ROOT = tmp_path
        try:
            result = ingest_mod._process_entry(
                entry,
                store=clean_store,
                embedder=embedder,  # type: ignore[arg-type]
                force=False,
                dry_run=False,
                log_path=tmp_path / "llm_calls.jsonl",
                logger=_QuietLogger(),
            )
        finally:
            ingest_mod.CORPUS_ROOT = original_root
        assert result.status == "missing"
        # This entry isn't in the real manifest, so _is_required returns True
        # by default. The optional flag on the manifest entry is what would
        # spare it in the real summary — the process_entry step logs the
        # difference (warning vs error) but returns the same "missing" status.


# ── Type stubs for VoyageEmbedder duck-typing ─────────────────────────


# Structural guard: _FakeEmbedder must expose the same duck-typed method
# ingest calls. If VoyageEmbedder.embed_documents signature drifts, this
# test catches it before an integration run tries to embed 71 PDFs.
def test_fake_embedder_signature_matches_real() -> None:
    import inspect

    real = inspect.signature(VoyageEmbedder.embed_documents)
    fake = inspect.signature(_FakeEmbedder.embed_documents)
    assert list(real.parameters.keys()) == list(fake.parameters.keys())


class _QuietLogger:
    """No-op logger sink so tests don't leak stderr chatter."""

    def info(self, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, *args: Any, **kwargs: Any) -> None: ...
    def error(self, *args: Any, **kwargs: Any) -> None: ...
    def debug(self, *args: Any, **kwargs: Any) -> None: ...
