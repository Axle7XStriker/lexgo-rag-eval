"""Ingest orchestrator tests.

Two layers of coverage:
  - `_process_entry` against a real Postgres (skipped when unreachable) with
    a fake embedder — exercises idempotency, --force, upsert semantics, and
    missing/required behavior.
  - Manifest helpers (`_select_manifest_entries`, `_is_required`) — pure,
    no DB.

Voyage is always mocked; ingest tests are not part of the eval suite, so no
live embedding API is needed. Postgres is real — mocking the DB would defeat
the invariants ingest relies on (ON CONFLICT, transaction commits, unique
constraint enforcement).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import pytest

from scripts import ingest as ingest_mod
from scripts.corpus_manifest import MANIFEST, ManifestEntry
from src.pipeline.chunk import PIPELINE_TAG
from src.pipeline.embed import VoyageEmbedder
from src.pipeline.store import EMBEDDING_DIM, VectorStore
from src.qa_schema import SourceId
from tests._pdf_fixtures import write_pdf


class _QuietLogger:
    """No-op logger sink so tests don't leak stderr chatter."""

    def info(self, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, *args: Any, **kwargs: Any) -> None: ...
    def error(self, *args: Any, **kwargs: Any) -> None: ...
    def debug(self, *args: Any, **kwargs: Any) -> None: ...


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
    a prior `make ingest` run).
      - TRUNCATE removes all rows in one shot (faster than DELETE and doesn't
        write per-row WAL).
      - RESTART IDENTITY resets the SERIAL sequences so `documents.id` starts
        at 1 again — makes test assertions on ids stable across runs.
      - CASCADE follows the chunks→documents foreign key; without it the
        TRUNCATE on `documents` would be refused while chunks reference it.
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
        assert len(entries) == len(MANIFEST)

    def test_source_id_filter(self) -> None:
        # Compare against the expected slice from MANIFEST directly — asserting
        # `all(e.source_id == A1)` would pass even if we silently dropped most
        # A1 entries, since it doesn't verify we got them all.
        expected = [e for e in MANIFEST if e.source_id == SourceId.A1]
        entries = ingest_mod._select_manifest_entries("A1")
        assert entries == expected

    def test_doc_path_filter(self) -> None:
        # Pick a real manifest entry so we know it matches.
        target = MANIFEST[0].dest_path
        entries = ingest_mod._select_manifest_entries(target)
        assert len(entries) == 1
        assert entries[0].dest_path == target

    def test_unknown_only_raises_systemexit(self) -> None:
        with pytest.raises(SystemExit):
            ingest_mod._select_manifest_entries("Z9")


class TestIsRequired:
    """_is_required: MANIFEST lookup with unknown-path fallback."""

    def test_known_required(self) -> None:
        required = next(e for e in MANIFEST if not e.optional)
        assert ingest_mod._is_required(required.dest_path) is True

    def test_known_optional(self) -> None:
        optional = next((e for e in MANIFEST if e.optional), None)
        if optional is None:
            pytest.skip("no optional entries in manifest")
        assert ingest_mod._is_required(optional.dest_path) is False

    def test_unknown_path_treated_required(self) -> None:
        assert ingest_mod._is_required("nowhere/nothing.pdf") is True


class TestExitCodeLogic:
    """_any_required_failed: what actually decides the process exit code."""

    def _result(self, dest_path: str, status: ingest_mod.DocStatus) -> ingest_mod.DocResult:
        return ingest_mod.DocResult(source_id="A1", doc_path=dest_path, status=status)

    def test_all_ingested_exits_clean(self) -> None:
        required = next(e for e in MANIFEST if not e.optional)
        assert not ingest_mod._any_required_failed([self._result(required.dest_path, "ingested")])

    def test_required_missing_fails(self) -> None:
        required = next(e for e in MANIFEST if not e.optional)
        assert ingest_mod._any_required_failed([self._result(required.dest_path, "missing")])

    def test_required_extract_failed_fails(self) -> None:
        required = next(e for e in MANIFEST if not e.optional)
        assert ingest_mod._any_required_failed([self._result(required.dest_path, "extract_failed")])

    def test_optional_extract_failed_does_not_fail(self) -> None:
        # The specific regression: previously any extract_failed row exited 1
        # regardless of optional status, contradicting the required-vs-optional
        # policy applied to `missing`.
        optional = next((e for e in MANIFEST if e.optional), None)
        if optional is None:
            pytest.skip("no optional entries in manifest")
        assert not ingest_mod._any_required_failed(
            [self._result(optional.dest_path, "extract_failed")]
        )

    def test_optional_missing_does_not_fail(self) -> None:
        optional = next((e for e in MANIFEST if e.optional), None)
        if optional is None:
            pytest.skip("no optional entries in manifest")
        assert not ingest_mod._any_required_failed([self._result(optional.dest_path, "missing")])


# ── Integration tests (require DB) ────────────────────────────────────


class TestProcessEntryIntegration:
    """Full extract → chunk → embed → upsert cycle against a real Postgres."""

    def test_missing_required_returns_missing(
        self,
        clean_store: VectorStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # `monkeypatch.setattr` auto-restores `CORPUS_ROOT` at test teardown —
        # no manual save/restore dance needed and no leak if the test errors.
        monkeypatch.setattr(ingest_mod, "CORPUS_ROOT", tmp_path)
        entry = _make_entry("nothing/nope.pdf", optional=False)
        embedder = _FakeEmbedder()
        result = ingest_mod._process_entry(
            entry,
            store=clean_store,
            embedder=embedder,  # type: ignore[arg-type]
            force=False,
            dry_run=False,
            log_path=tmp_path / "llm_calls.jsonl",
            logger=_QuietLogger(),
        )
        assert result.status == "missing"
        assert embedder.calls == 0
        # Missing REQUIRED must exit non-zero when this result feeds the summary.
        assert ingest_mod._is_required(entry.dest_path) is True

    def test_first_run_ingests_second_run_skips(
        self,
        clean_store: VectorStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ingest_mod, "CORPUS_ROOT", tmp_path / "corpus")
        dest = tmp_path / "corpus" / "test/A1_fake.pdf"
        write_pdf(dest, ["Chunk one text. " * 40, "Chunk two text. " * 40])
        entry = _make_entry("test/A1_fake.pdf")

        embedder = _FakeEmbedder()
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

    def test_force_reembeds(
        self,
        clean_store: VectorStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ingest_mod, "CORPUS_ROOT", tmp_path / "corpus")
        dest = tmp_path / "corpus" / "test/A1_fake.pdf"
        write_pdf(dest, ["Alpha " * 100])
        entry = _make_entry("test/A1_fake.pdf")

        embedder = _FakeEmbedder()
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

    def test_dry_run_no_embed_no_writes(
        self,
        clean_store: VectorStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ingest_mod, "CORPUS_ROOT", tmp_path / "corpus")
        dest = tmp_path / "corpus" / "test/A1_fake.pdf"
        write_pdf(dest, ["Beta " * 100])
        entry = _make_entry("test/A1_fake.pdf")

        embedder = _FakeEmbedder()
        result = ingest_mod._process_entry(
            entry,
            store=clean_store,
            embedder=embedder,  # type: ignore[arg-type]
            force=False,
            dry_run=True,
            log_path=tmp_path / "llm_calls.jsonl",
            logger=_QuietLogger(),
        )
        assert result.status == "ingested"  # dry-run reports success shape
        assert result.tokens_embedded > 0  # reports the pre-flight estimate
        assert result.cost_usd == 0.0
        assert embedder.calls == 0
        assert clean_store.count_chunks(PIPELINE_TAG) == 0

    def test_dry_run_works_without_store(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The M1 contract: --dry-run must be usable on a laptop with no
        # docker up. store=None mirrors what main() passes in that case.
        monkeypatch.setattr(ingest_mod, "CORPUS_ROOT", tmp_path / "corpus")
        dest = tmp_path / "corpus" / "test/A1_fake.pdf"
        write_pdf(dest, ["Delta " * 100])
        entry = _make_entry("test/A1_fake.pdf")

        embedder = _FakeEmbedder()
        result = ingest_mod._process_entry(
            entry,
            store=None,
            embedder=embedder,  # type: ignore[arg-type]
            force=False,
            dry_run=True,
            log_path=tmp_path / "llm_calls.jsonl",
            logger=_QuietLogger(),
        )
        assert result.status == "ingested"
        assert result.tokens_embedded > 0
        assert result.cost_usd == 0.0
        assert embedder.calls == 0

    def test_halfingested_doc_reembeds_not_skips(
        self,
        clean_store: VectorStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Guards C3: a doc row whose content_hash matches but has ZERO chunks
        # for PIPELINE_TAG must NOT be skipped. This is the partial-failure
        # recovery path AND the future P2/P3/P4-after-P1 path.
        monkeypatch.setattr(ingest_mod, "CORPUS_ROOT", tmp_path / "corpus")
        dest = tmp_path / "corpus" / "test/A1_fake.pdf"
        write_pdf(dest, ["Gamma " * 100])
        entry = _make_entry("test/A1_fake.pdf")

        # Seed the docs table with a matching content_hash but zero chunks —
        # simulates a prior partial failure. Use extract_pdf to get the
        # canonical hash so we're not duplicating the hashing logic here.
        from src.pipeline.extract import extract_pdf

        doc = extract_pdf(dest)
        clean_store.upsert_document(
            ingest_mod.DocumentRow(
                source_id="A1",
                doc_path="test/A1_fake.pdf",
                title=doc.title,
                num_pages=doc.num_pages,
                content_hash=doc.content_hash,
            )
        )
        clean_store.conn.commit()
        assert clean_store.count_chunks(PIPELINE_TAG) == 0

        embedder = _FakeEmbedder()
        result = ingest_mod._process_entry(
            entry,
            store=clean_store,
            embedder=embedder,  # type: ignore[arg-type]
            force=False,
            dry_run=False,
            log_path=tmp_path / "llm_calls.jsonl",
            logger=_QuietLogger(),
        )
        assert result.status == "ingested"  # NOT skipped_unchanged
        assert embedder.calls == 1
        assert clean_store.count_chunks(PIPELINE_TAG) == result.num_chunks

    def test_shrinking_chunkset_no_orphans(
        self,
        clean_store: VectorStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Guards C2: v1 of a doc ingests as N chunks, v2 as fewer. The DB
        # must NOT retain the tail chunks from v1 — retrieval would surface
        # them as stale hits.
        monkeypatch.setattr(ingest_mod, "CORPUS_ROOT", tmp_path / "corpus")
        dest = tmp_path / "corpus" / "test/A1_fake.pdf"
        # v1: big doc → many chunks.
        write_pdf(dest, ["Big content page. " * 400] * 3)
        entry = _make_entry("test/A1_fake.pdf")
        embedder = _FakeEmbedder()
        v1 = ingest_mod._process_entry(
            entry,
            store=clean_store,
            embedder=embedder,  # type: ignore[arg-type]
            force=False,
            dry_run=False,
            log_path=tmp_path / "llm_calls.jsonl",
            logger=_QuietLogger(),
        )
        assert v1.status == "ingested"
        assert v1.num_chunks > 1
        v1_count = clean_store.count_chunks(PIPELINE_TAG)
        assert v1_count == v1.num_chunks

        # v2: same path, much smaller content → fewer chunks.
        write_pdf(dest, ["Tiny."])
        v2 = ingest_mod._process_entry(
            entry,
            store=clean_store,
            embedder=embedder,  # type: ignore[arg-type]
            force=False,
            dry_run=False,
            log_path=tmp_path / "llm_calls.jsonl",
            logger=_QuietLogger(),
        )
        assert v2.status == "ingested"
        assert v2.num_chunks < v1_count, "test setup should produce fewer v2 chunks"
        # The load-bearing assertion: total chunks == v2's chunks, not v1_count.
        assert clean_store.count_chunks(PIPELINE_TAG) == v2.num_chunks

    def test_partial_failure_leaves_docrow_absent(
        self,
        clean_store: VectorStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Guards C1: if the chunk write path fails, the document row's new
        # content_hash must NOT be committed. Otherwise the next run's
        # idempotency check would silently skip a doc with no chunks.
        monkeypatch.setattr(ingest_mod, "CORPUS_ROOT", tmp_path / "corpus")
        dest = tmp_path / "corpus" / "test/A1_fake.pdf"
        write_pdf(dest, ["Zeta " * 100])
        entry = _make_entry("test/A1_fake.pdf")

        # Embedder that produces WRONG-DIMENSION vectors — upsert_chunks
        # raises ValueError at the dim guard, mid-transaction.
        class _BadDimEmbedder:
            calls = 0

            def embed_documents(
                self, texts: list[str], *, run_id: str | None = None
            ) -> list[list[float]]:
                self.calls += 1
                return [[0.1] * 4 for _ in texts]  # dim=4, not EMBEDDING_DIM

        embedder = _BadDimEmbedder()
        result = ingest_mod._process_entry(
            entry,
            store=clean_store,
            embedder=embedder,  # type: ignore[arg-type]
            force=False,
            dry_run=False,
            log_path=tmp_path / "llm_calls.jsonl",
            logger=_QuietLogger(),
        )
        assert result.status == "error"

        # Neither the doc row NOR any chunks should be committed.
        with clean_store.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents WHERE doc_path = %s", (entry.dest_path,))
            row = cur.fetchone()
            assert row is not None
            assert row[0] == 0, "partial-failure must not commit the doc row"
        assert clean_store.count_chunks(PIPELINE_TAG) == 0

    def test_optional_missing_does_not_fail(
        self,
        clean_store: VectorStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ingest_mod, "CORPUS_ROOT", tmp_path)
        entry = _make_entry("test/A1_optional.pdf", optional=True)
        embedder = _FakeEmbedder()
        result = ingest_mod._process_entry(
            entry,
            store=clean_store,
            embedder=embedder,  # type: ignore[arg-type]
            force=False,
            dry_run=False,
            log_path=tmp_path / "llm_calls.jsonl",
            logger=_QuietLogger(),
        )
        assert result.status == "missing"
        # This entry isn't in the real manifest, so _is_required returns True
        # by default. The optional flag on the manifest entry is what would
        # spare it in the real summary — the process_entry step logs the
        # difference (warning vs error) but returns the same "missing" status.


# ── Structural guard ──────────────────────────────────────────────────


# Structural guard: _FakeEmbedder must expose the same duck-typed method
# ingest calls. If VoyageEmbedder.embed_documents signature drifts, this
# test catches it before an integration run tries to embed 71 PDFs.
def test_fake_embedder_signature_matches_real() -> None:
    import inspect

    real = inspect.signature(VoyageEmbedder.embed_documents)
    fake = inspect.signature(_FakeEmbedder.embed_documents)
    assert list(real.parameters.keys()) == list(fake.parameters.keys())
