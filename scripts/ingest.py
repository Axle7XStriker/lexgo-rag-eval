"""P1 ingest orchestrator — corpus PDFs → pgvector.

For each present PDF in the manifest: extract text → chunk (fixed 500/50) →
embed via Voyage → upsert into `documents` + `chunks`. Idempotent: unchanged
documents are skipped by content-hash short-circuit, so re-running is cheap.

CLI::

  python -m scripts.ingest [--only DOC_PATH] [--force] [--dry-run]
                           [--batch-size N] [--log-level LEVEL]

Design notes worth remembering:
  - Idempotency key is `documents.content_hash` (sha256 of joined page text).
    Match → skip embed + upsert entirely; miss (or --force) → re-embed all
    the doc's chunks. Keeps the "unchanged corpus → zero API calls" invariant.
  - One VectorStore, one VoyageEmbedder, one loop. No parallelism — the
    Voyage SDK is sync, the DB writes are cheap, and the total budget is
    a few hundred chunks. Parallelism would be complexity without payoff.
  - Missing REQUIRED corpus files fail-fast at doc time (log + surface in
    summary + non-zero exit). Missing OPTIONAL files (per MANIFEST.optional)
    are warned + skipped.
  - Per-doc summary rows land in `logs/ingest_run_<utc_iso>.jsonl` for the
    blog post's cost/latency story. Per-Voyage-call records land in
    `logs/llm_calls.jsonl` via `log_llm_call`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from scripts.corpus_manifest import MANIFEST, ManifestEntry
from src.config import get_settings
from src.observability import configure_logging, get_logger
from src.pipeline.chunk import PIPELINE_TAG, chunk_fixed
from src.pipeline.embed import VoyageEmbedder
from src.pipeline.extract import extract_pdf
from src.pipeline.store import ChunkRow, DocumentRow, VectorStore

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
    status: str  # "ingested" | "skipped_unchanged" | "missing" | "extract_failed" | "error"
    num_pages: int = 0
    num_chunks: int = 0
    tokens_embedded: int = 0
    cost_usd: float = 0.0
    elapsed_ms: float = 0.0
    error: str | None = None


def _entries_for_only(only: str | None) -> list[ManifestEntry]:
    """Manifest slice honoring --only (matches on doc_path suffix or source_id)."""
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


def _fetch_existing_hash(store: VectorStore, doc_path: str) -> str | None:
    """Read the persisted content_hash for `doc_path`, or None if unseen."""
    with store.conn.cursor() as cur:
        cur.execute(
            "SELECT content_hash FROM documents WHERE doc_path = %s",
            (doc_path,),
        )
        row = cur.fetchone()
    return row[0] if row else None


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
    store: VectorStore,
    embedder: VoyageEmbedder,
    force: bool,
    dry_run: bool,
    log_path: Path,
    logger,
) -> DocResult:
    """Extract + chunk + embed + upsert one manifest entry. Never raises."""
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

    # Extract.
    try:
        doc = extract_pdf(dest)
    except (ValueError, FileNotFoundError) as e:
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

    # Idempotency short-circuit — same content hash and not forced → skip.
    existing_hash = _fetch_existing_hash(store, entry.dest_path)
    if existing_hash == doc.content_hash and not force:
        result.status = "skipped_unchanged"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("skipped_unchanged", doc_path=entry.dest_path, content_hash=doc.content_hash)
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

    # Embed. Snapshot log size so we can attribute per-doc token counts.
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    try:
        embeddings = embedder.embed_documents([c.text for c in chunks])
    except Exception as e:
        # Third-party SDK exceptions surface into the run summary; we cannot
        # anticipate every subclass but must not let one bad doc wedge the run.
        result.status = "error"
        result.error = f"embed failed: {type(e).__name__}: {e}"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        logger.error("embed_failed", doc_path=entry.dest_path, error=str(e))
        return result

    tokens_billed, cost = _tokens_since(log_path, log_offset)
    result.tokens_embedded = tokens_billed
    result.cost_usd = cost

    # Upsert. `upsert_document` returns the id; `upsert_chunks` replaces on
    # (document_id, pipeline, chunk_index) conflict so re-runs are safe.
    doc_row = DocumentRow(
        source_id=entry.source_id,
        doc_path=entry.dest_path,
        title=doc.title,
        num_pages=doc.num_pages,
        content_hash=doc.content_hash,
    )
    try:
        document_id = store.upsert_document(doc_row)
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
        store.upsert_chunks(document_id, chunk_rows)
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
    """Tally + list of failures + total cost. Aggregate story for humans."""
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
        # Missing optional entries are demoted to a lower-noise line; missing
        # REQUIRED entries and every extract_failed / error is loud.
        required_only = status == "missing"
        if required_only:
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
        help="Extract + chunk only. No Voyage calls, no DB writes. Reports estimated tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override Voyage embed batch size (default: 64).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger = get_logger("ingest")
    settings = get_settings()

    entries = _entries_for_only(args.only)
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

    run_started = datetime.now(UTC)
    run_log = settings.log_dir / f"ingest_run_{run_started.strftime('%Y%m%dT%H%M%SZ')}.jsonl"

    with VectorStore(settings.database_url) as store:
        store.ensure_schema()

        # Instantiate the embedder even in --dry-run: cheap (no API call until
        # embed_documents is called) and keeps the code path single.
        embedder = VoyageEmbedder(
            api_key=settings.voyage_api_key,
            model=settings.embedding_model,
            log_path=settings.llm_call_log,
            **embedder_kwargs,
        )

        results: list[DocResult] = []
        for i, entry in enumerate(entries, start=1):
            result = _process_entry(
                entry,
                store=store,
                embedder=embedder,
                force=args.force,
                dry_run=args.dry_run,
                log_path=settings.llm_call_log,
                logger=logger,
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

    # Non-zero exit iff any REQUIRED entry failed. Optional-missing does NOT
    # fail the run — mirrors fetch_corpus semantics.
    any_required_failed = any(
        r.status in {"error", "extract_failed"}
        or (r.status == "missing" and _is_required(r.doc_path))
        for r in results
    )
    return 1 if any_required_failed else 0


if __name__ == "__main__":
    sys.exit(main())
