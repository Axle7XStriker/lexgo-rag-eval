"""Ingest corpus PDFs into the chunk set selected by a pipeline.

Equal chunker configurations share one stored chunk set. A document is skipped
only when its text hash matches and that chunk set already has rows. Each
document replacement is one transaction, so failed writes remain retryable.

CLI::

  python -m scripts.ingest [--pipeline p1] [--only DOC_PATH] [--force]
                           [--dry-run] [--batch-size N] [--log-level LEVEL]

Required corpus failures make the command fail. Optional corpus failures are
reported but do not change the exit status. A dry run opens no database and
makes no paid embedding call.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import psycopg
import tiktoken

from scripts.corpus_manifest import MANIFEST, ManifestEntry
from src.config import get_settings
from src.observability import configure_logging, get_logger
from src.pipeline.chunk import Chunk, chunk_fixed
from src.pipeline.chunk_semantic import chunk_semantic
from src.pipeline.embed import VoyageEmbedder
from src.pipeline.extract import ExtractedDoc, extract_pdf
from src.pipeline.hashing import join_pages
from src.pipeline.pipeline_config import PIPELINES, PipelineConfig, get_pipeline
from src.pipeline.store import ChunkRow, DocumentRow, VectorStore

DocStatus = Literal["ingested", "skipped_unchanged", "missing", "extract_failed", "error"]

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "corpus"
MIN_EXTRACTED_CHARS = 200  # Shorter text is a failed extract, not an empty document.


@dataclass
class DocResult:
    """Outcome and accounting for one manifest entry."""

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
    """Return the manifest entries to process. The full manifest when `only` is None."""
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


def _fetch_existing_retrying(
    store: VectorStore, doc_path: str, chunk_set_id: str
) -> tuple[str, bool] | None:
    """`_fetch_existing`, opening a new connection if Neon has dropped the socket.

    The retry runs the same SELECT. A second `OperationalError` propagates
    so the caller can record a doc-level error without calling `rollback()`
    on a connection that is already gone.
    """
    try:
        return _fetch_existing(store, doc_path, chunk_set_id)
    except psycopg.OperationalError:
        store.connect()
        return _fetch_existing(store, doc_path, chunk_set_id)


def _fetch_existing(
    store: VectorStore, doc_path: str, chunk_set_id: str
) -> tuple[str, bool] | None:
    """Read `(content_hash, has_chunks_for_chunk_set_id)` for `doc_path`, or None if unseen.

    Combined into one call because both pieces are load-bearing for the
    idempotency short-circuit: same hash AND chunks present = safe skip.
    Same hash but zero chunks means a prior ingest half-failed, and we
    must re-embed.
    """
    with store.conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                d.content_hash,
                EXISTS(
                    SELECT 1 FROM chunks c
                    WHERE c.document_id = d.id AND c.chunk_set_id = %s
                )
            FROM documents d
            WHERE d.doc_path = %s
            """,
            (chunk_set_id, doc_path),
        )
        row = cur.fetchone()
    store.conn.commit()
    if row is None:
        return None
    return row[0], bool(row[1])


def _tokens_since(log_path: Path, offset_bytes: int) -> tuple[int, float]:
    """Sum input_tokens and cost_usd from `log_path` starting at `offset_bytes`."""
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
    cfg: PipelineConfig,
    chunker: Callable[[ExtractedDoc], list[Chunk]],
    force: bool,
    dry_run: bool,
    log_path: Path,
    logger,
    run_id: str | None = None,
) -> DocResult:
    """Ingest (extract + chunk + embed + upsert) one manifest entry and return failures as data."""
    pipeline_tag = cfg.tag
    chunk_set_id = cfg.chunk_set_id
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

    # A matching document with no chunks is a partial ingest and must retry.
    if store is not None:
        try:
            existing = _fetch_existing_retrying(store, entry.dest_path, chunk_set_id)
        except psycopg.Error as e:
            # Reset recoverable state so later documents can continue.
            store._rollback_if_alive()
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
                pipeline_tag=pipeline_tag,
                chunk_set_id=chunk_set_id,
            )
            return result

    # Avoid the semantic chunker's paid sentence embeddings during a dry run.
    if dry_run and cfg.chunker.algorithm == "semantic":
        encoder = tiktoken.get_encoding(cfg.chunker.encoding)
        estimated_tokens = len(encoder.encode(join_pages([p.text for p in doc.pages])))
        result.status = "ingested"
        result.num_chunks = 0
        result.tokens_embedded = estimated_tokens
        result.cost_usd = 0.0
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "dry_run_ok_embedding_chunker",
            doc_path=entry.dest_path,
            algorithm=cfg.chunker.algorithm,
            estimated_sentence_embed_tokens=estimated_tokens,
        )
        return result

    chunks = chunker(doc)
    result.num_chunks = len(chunks)
    if not chunks:
        result.status = "extract_failed"
        result.error = "chunker returned zero chunks"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning("zero_chunks", doc_path=entry.dest_path)
        return result

    if dry_run:
        estimated_tokens = sum(c.num_tokens for c in chunks)
        result.status = "ingested"
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

    # Replace the document and its chunk set together to avoid partial ingests.
    doc_row = DocumentRow(
        source_id=entry.source_id,
        doc_path=entry.dest_path,
        title=doc.title,
        num_pages=doc.num_pages,
        content_hash=doc.content_hash,
    )
    chunk_rows = [
        ChunkRow(
            chunk_set_id=chunk_set_id,
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
        store.replace_document_chunks(doc_row, chunk_rows, chunk_set_id=chunk_set_id)
    except Exception as e:
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
    return True


def _build_chunker(
    cfg: PipelineConfig,
    *,
    embedder: VoyageEmbedder,
    run_id: str,
) -> Callable[[ExtractedDoc], list[Chunk]]:
    algorithm = cfg.chunker.algorithm
    if algorithm == "fixed":
        return lambda doc: chunk_fixed(doc, cfg.chunker)
    if algorithm == "semantic":
        return lambda doc: chunk_semantic(
            doc,
            embedder=embedder,
            config=cfg.chunker,
            run_id=run_id,
        )
    raise ValueError(f"no chunker registered for algorithm={algorithm!r}")


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        choices=tuple(PIPELINES.keys()),
        default="p1",
        help="Which retrieval pipeline's chunks to ingest. Selects a chunker "
        "+ knobs from src.pipeline.pipeline_config and derives the chunk_set_id. "
        "Pipelines sharing a ChunkerConfig share one chunk set; subsequent "
        "runs for a pipeline whose chunker already ingested are no-ops.",
    )
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
        "--log-level",
        default=settings.log_level,
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger = get_logger("ingest")

    cfg = get_pipeline(args.pipeline)
    pipeline_tag = cfg.tag
    chunk_set_id = cfg.chunk_set_id

    entries = _select_manifest_entries(args.only)
    logger.info(
        "ingest_start",
        n_entries=len(entries),
        pipeline_tag=pipeline_tag,
        pipeline_key=cfg.key,
        chunk_set_id=chunk_set_id,
        dry_run=args.dry_run,
        force=args.force,
    )

    embedder_kwargs = {}
    if args.batch_size is not None:
        embedder_kwargs["batch_size"] = args.batch_size

    run_started = datetime.now(UTC)
    run_id = f"ingest_{run_started.strftime('%Y%m%dT%H%M%SZ')}"  # Joins both log streams.
    run_log = settings.log_dir / f"{run_id}.jsonl"

    embedder = VoyageEmbedder(
        api_key=settings.voyage_api_key,
        model=settings.embedding_model,
        log_path=settings.llm_call_log,
        **embedder_kwargs,
    )

    chunker = _build_chunker(cfg, embedder=embedder, run_id=run_id)

    results: list[DocResult] = []
    store_ctx = _NullContext() if args.dry_run else VectorStore(settings.database_url)
    with store_ctx as store:
        if store is not None:
            store.ensure_schema()

        for i, entry in enumerate(entries, start=1):
            result = _process_entry(
                entry,
                store=store,
                embedder=embedder,
                cfg=cfg,
                chunker=chunker,
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
    return any(
        r.status in {"error", "extract_failed", "missing"} and _is_required(r.doc_path)
        for r in results
    )


class _NullContext:
    """Context manager used when a dry run must not open the database."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


if __name__ == "__main__":
    sys.exit(main())
