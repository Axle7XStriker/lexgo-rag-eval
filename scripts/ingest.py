"""P1 ingest orchestrator — corpus PDFs → pgvector.

For each present PDF in the manifest: extract text → chunk (fixed 500/50) →
embed via Voyage → upsert into the `documents` and `chunks` tables.
Idempotent: unchanged documents are skipped by content-hash short-circuit,
so re-running is cheap.

CLI::

  python -m scripts.ingest [--only DOC_PATH] [--force] [--dry-run]
                           [--batch-size N] [--log-level LEVEL]

Design notes worth remembering:
  - Idempotency key is `(documents.content_hash, chunks-exist-for-pipeline)`.
    Match on BOTH → skip embed + upsert; miss on either (or --force) →
    re-embed all the doc's chunks. Requiring the chunks-exist half means a
    prior partial-failure ingest recovers on the next run instead of being
    permanently short-circuited by a stale hash — and P2..P4 don't silently
    no-op on docs P1 already touched.
  - Per-doc write is one transaction: upsert `documents` + DELETE stale
    chunks for `(doc_id, pipeline)` + insert new chunks + commit. Chunk-
    write failures roll back the document row's new hash too, so the
    invariant above holds.
  - One VectorStore, one VoyageEmbedder, one loop. No parallelism — the
    Voyage SDK is sync, the DB writes are cheap, and the total budget is
    a few hundred chunks. Parallelism would be complexity without payoff.
  - Missing REQUIRED corpus files fail-fast at doc time (log + surface in
    summary + non-zero exit). Missing OPTIONAL files (per MANIFEST.optional)
    are warned + skipped; optional extract_failed / error are also demoted.
  - Per-doc summary rows land in `logs/ingest_<utc_iso>.jsonl` — one JSONL
    record per document with num_chunks, tokens, cost, elapsed. Per Voyage
    call, one line lands in `logs/llm_calls.jsonl` tagged with the same
    run_id so cost can be attributed to a specific ingest run.
  - --dry-run opens NO database connection — extract + chunk + estimate
    only. Useful for pre-flight cost checks on a laptop with no docker up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import psycopg

from scripts.corpus_manifest import MANIFEST, ManifestEntry
from src.config import get_settings
from src.observability import configure_logging, get_logger
from src.pipeline.chunk import PIPELINE_TAG, chunk_fixed
from src.pipeline.embed import VoyageEmbedder
from src.pipeline.extract import extract_pdf
from src.pipeline.store import ChunkRow, DocumentRow, VectorStore

# Finite outcome set for one document's ingest attempt. Kept as a Literal so
# the type checker catches typos across `_process_entry` and `_print_summary`.
DocStatus = Literal["ingested", "skipped_unchanged", "missing", "extract_failed", "error"]

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "corpus"
# Minimum extracted-text length below which a doc is treated as extraction
# failure (log + skip, don't fail the run). Guards against a scanned PDF
# with unreadable text yielding a spurious 0-chunk document row.
MIN_EXTRACTED_CHARS = 200


@dataclass
class DocResult:
    """Per-document outcome. Written to the run log; aggregated into the summary."""

    source_id: str
    doc_path: str
    status: DocStatus
    num_pages: int = 0
    num_chunks: int = 0
    tokens_embedded: int = 0
    cost_usd: float = 0.0
    elapsed_ms: float = 0.0
    error: str | None = None


def _select_manifest_entries(only: str | None) -> list[ManifestEntry]:
    """Return the manifest entries to process — the full manifest when `only` is None,
    otherwise the subset matching either a full `dest_path` or a `source_id`."""
    if not only:
        return list(MANIFEST)
    matches = [e for e in MANIFEST if e.dest_path == only or e.source_id == only]
    if not matches:
        raise SystemExit(
            f"--only {only!r} matched no manifest entries. "
            f"Give a full dest_path (e.g. '6.006/lectures/A1_lec03.pdf') "
            f"or a source_id (A1..B5)."
        )
    return matches


def _write_run_log(path: Path, results: list[DocResult]) -> None:
    """One JSON object per doc, appended to the per-run log."""
    # Ensure the log directory exists — first-time runs on a fresh checkout
    # hit this before anything else has written under logs/.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "source_id": r.source_id,
                        "doc_path": r.doc_path,
                        "status": r.status,
                        "num_pages": r.num_pages,
                        "num_chunks": r.num_chunks,
                        "tokens_embedded": r.tokens_embedded,
                        "cost_usd": round(r.cost_usd, 6),
                        "elapsed_ms": round(r.elapsed_ms, 2),
                        "error": r.error,
                    }
                )
                + "\n"
            )


def _fetch_existing(store: VectorStore, doc_path: str, pipeline: str) -> tuple[str, bool] | None:
    """Read `(content_hash, has_chunks_for_pipeline)` for `doc_path`, or None if unseen.

    Combined into one call because both pieces are load-bearing for the
    idempotency short-circuit: same hash AND chunks present = safe skip.
    Same hash but zero chunks means a prior ingest half-failed (or a
    different pipeline populated the doc row), and we must re-embed.
    """
    with store.conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                d.content_hash,
                EXISTS(
                    SELECT 1 FROM chunks c
                    WHERE c.document_id = d.id AND c.pipeline = %s
                )
            FROM documents d
            WHERE d.doc_path = %s
            """,
            (pipeline, doc_path),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return row[0], bool(row[1])


def _tokens_since(log_path: Path, offset_bytes: int) -> tuple[int, float]:
    """Sum input_tokens and cost_usd from `log_path` starting at `offset_bytes`.

    Ingest snapshots the log file size before each doc's embed call; after
    the call, everything appended since is attributed to that doc. This is
    simpler and more accurate than duplicating token accounting in embed.py.
    """
    if not log_path.exists():
        return 0, 0.0
    tokens = 0
    cost = 0.0
    with log_path.open("rb") as f:
        f.seek(offset_bytes)
        for raw in f:
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            tokens += int(rec.get("input_tokens") or 0)
            cost += float(rec.get("cost_usd") or 0.0)
    return tokens, cost


def _process_entry(
    entry: ManifestEntry,
    *,
    store: VectorStore | None,
    embedder: VoyageEmbedder,
    force: bool,
    dry_run: bool,
    log_path: Path,
    logger,
    run_id: str | None = None,
) -> DocResult:
    """Extract + chunk + embed + upsert one manifest entry. Never raises.

    `store` may be None in --dry-run (no DB is opened at all). Real runs
    must pass a live VectorStore.
    """
    dest = CORPUS_ROOT / entry.dest_path
    result = DocResult(source_id=entry.source_id, doc_path=entry.dest_path, status="error")
    started = time.perf_counter()

    if not dest.exists() or dest.stat().st_size == 0:
        result.status = "missing"
        result.error = (
            "PDF not present in corpus/ — run `make corpus` "
            "(or place manually for optional entries)"
        )
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        if entry.optional:
            logger.warning("optional_missing", doc_path=entry.dest_path, source_id=entry.source_id)
        else:
            logger.error("required_missing", doc_path=entry.dest_path, source_id=entry.source_id)
        return result

    # Extract. `OSError` catches PermissionError, corrupt-file I/O errors,
    # and anything pymupdf raises through the filesystem layer — without
    # this, one unreadable PDF kills the whole run.
    try:
        doc = extract_pdf(dest)
    except (OSError, ValueError) as e:
        result.status = "extract_failed"
        result.error = str(e)
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning("extract_failed", doc_path=entry.dest_path, error=str(e))
        return result

    result.num_pages = doc.num_pages
    joined_chars = sum(len(p.text) for p in doc.pages)
    if joined_chars < MIN_EXTRACTED_CHARS:
        result.status = "extract_failed"
        result.error = f"extracted only {joined_chars} chars (< {MIN_EXTRACTED_CHARS})"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning(
            "extract_too_short",
            doc_path=entry.dest_path,
            extracted_chars=joined_chars,
        )
        return result

    # Idempotency short-circuit — same content hash AND chunks exist for
    # THIS pipeline, and not forced → skip. The pipeline check is what
    # keeps a partially-failed prior ingest (doc row committed, chunks
    # empty) from being permanently skipped, and what keeps P2..P4 from
    # silently no-op'ing on docs P1 already touched.
    if store is not None:
        try:
            existing = _fetch_existing(store, entry.dest_path, PIPELINE_TAG)
        except psycopg.Error as e:
            # A failed SELECT poisons the connection until rolled back —
            # otherwise every subsequent doc's SELECT fails with
            # `current transaction is aborted`. Rollback + surface as a
            # doc-level error and continue.
            store.conn.rollback()
            result.status = "error"
            result.error = f"idempotency lookup failed: {type(e).__name__}: {e}"
            result.elapsed_ms = (time.perf_counter() - started) * 1000
            logger.error("idempotency_lookup_failed", doc_path=entry.dest_path, error=str(e))
            return result

        if existing is not None and existing[0] == doc.content_hash and existing[1] and not force:
            result.status = "skipped_unchanged"
            result.elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "skipped_unchanged",
                doc_path=entry.dest_path,
                content_hash=doc.content_hash,
                pipeline=PIPELINE_TAG,
            )
            return result

    # Chunk.
    chunks = chunk_fixed(doc)
    result.num_chunks = len(chunks)
    if not chunks:
        result.status = "extract_failed"
        result.error = "chunker returned zero chunks"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning("zero_chunks", doc_path=entry.dest_path)
        return result

    if dry_run:
        # Report tokens as the sum of per-chunk num_tokens — an estimate, not
        # a Voyage-billed number. Useful for a pre-flight cost check.
        estimated_tokens = sum(c.num_tokens for c in chunks)
        result.status = "ingested"  # dry-run success shape
        result.tokens_embedded = estimated_tokens
        result.cost_usd = 0.0
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "dry_run_ok",
            doc_path=entry.dest_path,
            num_chunks=result.num_chunks,
            estimated_tokens=estimated_tokens,
        )
        return result

    assert store is not None, "non-dry-run requires a live VectorStore"

    # Embed. Snapshot log size so we can attribute per-doc token counts.
    # If a batch fails partway through, tokens for earlier successful
    # batches are still in the log — attribute what actually landed so
    # the summary reflects real spend even on partial-failure docs.
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    try:
        embeddings = embedder.embed_documents([c.text for c in chunks], run_id=run_id)
    except Exception as e:
        # Third-party SDK exceptions surface into the run summary; we cannot
        # anticipate every subclass but must not let one bad doc wedge the run.
        # Attribute any partial-batch cost to this doc even though it failed.
        partial_tokens, partial_cost = _tokens_since(log_path, log_offset)
        result.tokens_embedded = partial_tokens
        result.cost_usd = partial_cost
        result.status = "error"
        result.error = f"embed failed: {type(e).__name__}: {e}"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "embed_failed",
            doc_path=entry.dest_path,
            error=str(e),
            partial_tokens=partial_tokens,
            partial_cost_usd=round(partial_cost, 4),
        )
        return result

    tokens_billed, cost = _tokens_since(log_path, log_offset)
    result.tokens_embedded = tokens_billed
    result.cost_usd = cost

    # Atomic write: upsert doc + wipe stale chunks for this pipeline +
    # insert new chunks, all in one transaction. If chunks fail, the doc
    # row rolls back too — so the next run's hash check won't skip a
    # half-ingested doc, and shrinking chunk sets don't leave orphans.
    doc_row = DocumentRow(
        source_id=entry.source_id,
        doc_path=entry.dest_path,
        title=doc.title,
        num_pages=doc.num_pages,
        content_hash=doc.content_hash,
    )
    chunk_rows = [
        ChunkRow(
            pipeline=PIPELINE_TAG,
            chunk_index=c.chunk_index,
            text=c.text,
            num_tokens=c.num_tokens,
            page_start=c.page_start,
            page_end=c.page_end,
            content_hash=c.content_hash,
            embedding=e,
        )
        for c, e in zip(chunks, embeddings, strict=True)
    ]
    try:
        store.replace_document_chunks(doc_row, chunk_rows, pipeline=PIPELINE_TAG)
    except Exception as e:
        # DB errors surface into the summary rather than aborting the whole
        # run — the outer main() exits non-zero if any required entry failed.
        result.status = "error"
        result.error = f"upsert failed: {type(e).__name__}: {e}"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        logger.error("upsert_failed", doc_path=entry.dest_path, error=str(e))
        return result

    result.status = "ingested"
    result.elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "ingested",
        doc_path=entry.dest_path,
        num_chunks=result.num_chunks,
        tokens_billed=tokens_billed,
        cost_usd=round(cost, 4),
    )
    return result


def _print_summary(results: list[DocResult]) -> None:
    """Print per-status counts, failing entries, and total chunks / tokens / cost."""
    buckets: dict[str, list[DocResult]] = {}
    for r in results:
        buckets.setdefault(r.status, []).append(r)

    n_ingested = len(buckets.get("ingested", []))
    n_skipped = len(buckets.get("skipped_unchanged", []))
    n_missing = len(buckets.get("missing", []))
    n_extract_failed = len(buckets.get("extract_failed", []))
    n_error = len(buckets.get("error", []))
    total_chunks = sum(r.num_chunks for r in results)
    total_tokens = sum(r.tokens_embedded for r in results)
    total_cost = sum(r.cost_usd for r in results)

    print("\n── ingest summary ─────────────────────────────────────────")
    print(
        f"  ingested: {n_ingested:3d}    skipped: {n_skipped:3d}"
        f"    missing: {n_missing:3d}    extract_failed: {n_extract_failed:3d}"
        f"    error: {n_error:3d}"
    )
    print(
        f"  chunks_written: {total_chunks}    "
        f"tokens_embedded: {total_tokens}    cost_usd: ${total_cost:.4f}"
    )
    for status in ("missing", "extract_failed", "error"):
        rows = buckets.get(status, [])
        # `missing` is the only status that splits required vs. optional —
        # optional-missing is demoted to a low-noise line, required-missing
        # (and every extract_failed / error) stays loud.
        is_missing_status = status == "missing"
        if is_missing_status:
            required_rows = [r for r in rows if _is_required(r.doc_path)]
            optional_rows = [r for r in rows if not _is_required(r.doc_path)]
            if required_rows:
                print(f"\n  REQUIRED {status}:")
                for r in required_rows:
                    print(f"    - {r.source_id}/{r.doc_path} :: {r.error}")
            if optional_rows:
                print(f"\n  optional {status} (source manually if needed):")
                for r in optional_rows:
                    print(f"    - {r.source_id}/{r.doc_path}")
        elif rows:
            print(f"\n  {status}:")
            for r in rows:
                print(f"    - {r.source_id}/{r.doc_path} :: {r.error}")
    print("───────────────────────────────────────────────────────────\n")


def _is_required(doc_path: str) -> bool:
    """Look up whether MANIFEST marks `doc_path` as required (not optional)."""
    for e in MANIFEST:
        if e.dest_path == doc_path:
            return not e.optional
    return True  # unknown path → treat as required so it stays loud


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        help="Restrict to one doc_path (full manifest dest_path) or one source_id (A1..B5).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed even if the content_hash matches an existing document row.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract + chunk only. No Voyage calls, no DB open, no DB writes. "
        "Reports estimated tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override Voyage embed batch size (default: 64).",
    )
    parser.add_argument(
        # Default reads settings.log_level (which honors .env) rather than
        # os.environ directly — otherwise a .env-only LOG_LEVEL override is
        # silently ignored for ingest but honored everywhere else.
        "--log-level",
        default=settings.log_level,
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger = get_logger("ingest")

    entries = _select_manifest_entries(args.only)
    logger.info(
        "ingest_start",
        n_entries=len(entries),
        pipeline=PIPELINE_TAG,
        dry_run=args.dry_run,
        force=args.force,
    )

    embedder_kwargs = {}
    if args.batch_size is not None:
        embedder_kwargs["batch_size"] = args.batch_size

    # run_id ties per-doc rows in `ingest_run_<ts>.jsonl` to per-call rows in
    # `llm_calls.jsonl` so a cost regression can be attributed to a specific
    # run without timestamp-window heuristics.
    run_started = datetime.now(UTC)
    run_id = f"ingest_{run_started.strftime('%Y%m%dT%H%M%SZ')}"
    run_log = settings.log_dir / f"{run_id}.jsonl"

    embedder = VoyageEmbedder(
        api_key=settings.voyage_api_key,
        model=settings.embedding_model,
        log_path=settings.llm_call_log,
        **embedder_kwargs,
    )

    results: list[DocResult] = []
    # --dry-run runs entirely without a DB — the point of dry-run is to
    # pre-flight the corpus + estimate cost on a laptop with no docker up.
    store_ctx = _NullContext() if args.dry_run else VectorStore(settings.database_url)
    with store_ctx as store:
        if store is not None:
            store.ensure_schema()

        for i, entry in enumerate(entries, start=1):
            result = _process_entry(
                entry,
                store=store,
                embedder=embedder,
                force=args.force,
                dry_run=args.dry_run,
                log_path=settings.llm_call_log,
                logger=logger,
                run_id=run_id,
            )
            results.append(result)
            marker = {
                "ingested": "✓",
                "skipped_unchanged": "·",
                "missing": "?" if entry.optional else "✗",
                "extract_failed": "✗",
                "error": "✗",
            }[result.status]
            print(
                f"  {marker} [{entry.source_id}] {entry.dest_path}  "
                f"({result.status}, chunks={result.num_chunks}, "
                f"${result.cost_usd:.4f}, {result.elapsed_ms:.0f}ms)  "
                f"[{i}/{len(entries)}]"
            )

    _write_run_log(run_log, results)
    _print_summary(results)
    return 1 if _any_required_failed(results) else 0


def _any_required_failed(results: list[DocResult]) -> bool:
    """True iff any REQUIRED entry ended in a failure status.

    Failure statuses are missing, extract_failed, and error. Optional
    entries in ANY of those states are demoted to non-failure — an
    operator-placed optional PDF that is corrupt should not be louder
    than one that is simply absent.
    """
    return any(
        r.status in {"error", "extract_failed", "missing"} and _is_required(r.doc_path)
        for r in results
    )


class _NullContext:
    """Context manager that yields None. Used to skip DB open in --dry-run."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


if __name__ == "__main__":
    sys.exit(main())
