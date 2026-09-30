"""Postgres+pgvector store for the RAG pipelines.

Design notes:
- Sync psycopg 3. Eval loop is a batch job over 100 Q&As, not a serving
  path — an async pool would be complexity without a payoff.
- pgvector's psycopg adapter registered per-connection so `vector` columns
  become Python lists on read and accept lists on write.
- Schema is applied idempotently by `ensure_schema()` reading schema.sql;
  callers can invoke on every ingest run without special-casing "first
  time" vs "subsequent" runs.
- All chunks tagged with a `pipeline` column (P1..P4-derived, e.g.
  "p1_fixed_500_50"). Pipelines coexist in one table — an A/B eval
  across P2/P3/P4 is a WHERE-clause change, not DDL.
- One global HNSW index over `chunks.embedding` for cosine similarity.
  Queries compose `WHERE pipeline = $1` with the index scan; at our
  scale (a few thousand chunks per pipeline) that's cheaper than
  maintaining a separate HNSW index per pipeline and keeps the schema
  simpler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import psycopg
from pgvector.psycopg import register_vector

from src.db import __file__ as _db_pkg_file

SCHEMA_PATH = Path(_db_pkg_file).parent / "schema.sql"
EMBEDDING_DIM = 1024  # voyage-3-large


# ── Typed rows ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DocumentRow:
    """A row destined for the `documents` table (pre-insert form)."""

    source_id: str  # A1..A4 / B1..B5
    doc_path: str  # relative to corpus/, matches Citation.doc_path
    title: str | None
    num_pages: int | None
    content_hash: str  # sha256 of extracted text


@dataclass(frozen=True)
class ChunkRow:
    """A chunk destined for the `chunks` table (pre-insert form)."""

    pipeline: str  # e.g. "p1_fixed_500_50"
    chunk_index: int
    text: str
    num_tokens: int
    page_start: int
    page_end: int
    content_hash: str  # sha256 of chunk text
    embedding: list[float]  # length must equal EMBEDDING_DIM


@dataclass(frozen=True)
class RetrievedChunk:
    """A row returned from `dense_search` — chunk + document context + score."""

    chunk_id: int
    document_id: int
    doc_path: str
    source_id: str
    pipeline: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    score: float  # cosine similarity in [-1, 1] (higher is closer)


# ── Row + fusion helpers (module-level for testability) ──────────────
#
# Pure functions kept out of the class body so the fusion math is
# testable without a live psycopg connection. `_row_to_retrieved`
# adapts a psycopg row tuple (matching `VectorStore._SELECT_COLUMNS`
# + a trailing score column) into a `RetrievedChunk`.


def _row_to_retrieved(row: tuple) -> RetrievedChunk:
    """Adapt a psycopg row (columns per `VectorStore._SELECT_COLUMNS + score`)
    into a `RetrievedChunk`. Keeps dense + lexical branches in lockstep."""
    return RetrievedChunk(
        chunk_id=row[0],
        document_id=row[1],
        doc_path=row[2],
        source_id=row[3],
        pipeline=row[4],
        chunk_index=row[5],
        text=row[6],
        page_start=row[7],
        page_end=row[8],
        score=row[9],
    )


def _rrf_fuse(
    dense: list[RetrievedChunk],
    lexical: list[RetrievedChunk],
    *,
    rrf_k: int,
    top_k: int,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion of two ranked lists → top-`top_k` fused.

    For each chunk (identified by `chunk_id`), sum `1 / (rrf_k + rank)`
    across the two lists it appears in (1-indexed rank). Sort by fused
    score DESC, break ties by chunk_id ASC for determinism.

    The returned `RetrievedChunk.score` is the RRF score (small float,
    upper-bounded by `2 / (rrf_k + 1)`), not either source's original
    score. Provenance from the dense list wins on tie (dense chunks
    tend to carry a cosine score that's more informative for display
    than a `ts_rank_cd` value).

    Cost: O(N + M) construction, O((N+M) log (N+M)) sort. N and M ≤ top_k
    for us (~10 each), so this is free.
    """
    # Sum inverse ranks per chunk_id. Rank is 1-indexed per RRF spec.
    scores: dict[int, float] = {}
    for rank, chunk in enumerate(dense, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, chunk in enumerate(lexical, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    # Prefer the dense list's RetrievedChunk for provenance — cosine
    # score displays better than ts_rank_cd when the UI ignores the
    # fused score and reads the original. Fall back to lexical for
    # chunks not present in the dense list.
    by_id: dict[int, RetrievedChunk] = {c.chunk_id: c for c in lexical}
    for chunk in dense:
        by_id[chunk.chunk_id] = chunk

    # Sort by fused score DESC, break ties by chunk_id ASC (deterministic
    # so two eval runs return the exact same fused ordering).
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    fused: list[RetrievedChunk] = []
    for chunk_id, fused_score in ranked[:top_k]:
        base = by_id[chunk_id]
        fused.append(
            RetrievedChunk(
                chunk_id=base.chunk_id,
                document_id=base.document_id,
                doc_path=base.doc_path,
                source_id=base.source_id,
                pipeline=base.pipeline,
                chunk_index=base.chunk_index,
                text=base.text,
                page_start=base.page_start,
                page_end=base.page_end,
                score=fused_score,
            )
        )
    return fused


# ── Store ─────────────────────────────────────────────────────────────


class VectorStore:
    """Thin sync wrapper around psycopg + pgvector.

    Must be used as a context manager. The connection is opened in
    __enter__; any method call on an unopened store raises RuntimeError.
        with VectorStore(dsn) as store:
            store.ensure_schema()
            ...
    """

    def __init__(self, dsn: str) -> None:
        # Postgres data source name (connection string).
        self._dsn = dsn
        self._conn: psycopg.Connection | None = None

    def __enter__(self) -> Self:
        self._conn = psycopg.connect(self._dsn, autocommit=False)
        # register_vector needs the `vector` type to exist, but on a fresh
        # DB nothing has installed it yet — ensure_schema() would, but the
        # caller can't reach it until __enter__ returns. Break the cycle by
        # creating the extension here. Idempotent (IF NOT EXISTS) + cheap.
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.commit()
        register_vector(self._conn)
        return self

    def __exit__(
        self,
        # Required by the context-manager protocol, unused here — underscored
        # to signal that. We just close the connection on any exit path.
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None:
            raise RuntimeError("VectorStore not opened; use `with VectorStore(dsn) as store: ...`.")
        return self._conn

    # ── Schema ────────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        """Apply schema.sql. Idempotent — every DDL statement is guarded.

        schema.sql is the source of truth for the schema; this method is
        a thin execute-the-file wrapper. Additive changes (new tables,
        new indexes) are picked up on the next call because every DDL is
        guarded with IF NOT EXISTS. Destructive changes (drop column,
        change column type) need explicit migration handling, which is
        out of scope for P1.
        """
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text())
        self.conn.commit()

    # ── Documents ─────────────────────────────────────────────────────

    def upsert_document(self, doc: DocumentRow) -> int:
        """Insert or refresh a document by `doc_path`; return its id.

        Does NOT commit — caller controls the transaction boundary so
        this can be composed with `upsert_chunks` atomically (see
        `replace_document_chunks`).
        """
        # ON CONFLICT DO UPDATE / EXCLUDED: on doc_path collision, update
        # the row from the values that would have been inserted. EXCLUDED
        # is a Postgres pseudo-row bound to that pending insert, only
        # available inside the DO UPDATE clause.
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (source_id, doc_path, title, num_pages, content_hash)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (doc_path) DO UPDATE
                    SET source_id    = EXCLUDED.source_id,
                        title        = EXCLUDED.title,
                        num_pages    = EXCLUDED.num_pages,
                        content_hash = EXCLUDED.content_hash
                RETURNING id
                """,
                (doc.source_id, doc.doc_path, doc.title, doc.num_pages, doc.content_hash),
            )
            row = cur.fetchone()
            assert row is not None  # RETURNING guarantees a row on INSERT/UPDATE
            return row[0]

    # ── Chunks ────────────────────────────────────────────────────────

    def upsert_chunks(self, document_id: int, chunks: list[ChunkRow]) -> None:
        """Bulk-insert chunks; on `(document_id, pipeline, chunk_index)` conflict, replace.

        Does NOT commit — caller controls the transaction. Does NOT delete
        stale higher-index chunks left over from a previous ingest with more
        chunks; use `replace_document_chunks` when re-embedding a document.

        Raises `ValueError` up front if any embedding is the wrong dim,
        so a single bad chunk fails fast instead of mid-transaction.
        """
        if not chunks:
            return
        for c in chunks:
            if len(c.embedding) != EMBEDDING_DIM:
                raise ValueError(
                    f"embedding dim {len(c.embedding)} ≠ expected {EMBEDDING_DIM} "
                    f"(document_id={document_id}, chunk_index={c.chunk_index})"
                )
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (document_id, pipeline, chunk_index, text, num_tokens,
                     page_start, page_end, content_hash, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id, pipeline, chunk_index) DO UPDATE
                    SET text         = EXCLUDED.text,
                        num_tokens   = EXCLUDED.num_tokens,
                        page_start   = EXCLUDED.page_start,
                        page_end     = EXCLUDED.page_end,
                        content_hash = EXCLUDED.content_hash,
                        embedding    = EXCLUDED.embedding
                """,
                [
                    (
                        document_id,
                        c.pipeline,
                        c.chunk_index,
                        c.text,
                        c.num_tokens,
                        c.page_start,
                        c.page_end,
                        c.content_hash,
                        c.embedding,
                    )
                    for c in chunks
                ],
            )

    def delete_chunks(self, document_id: int, pipeline: str) -> int:
        """Delete all chunks for `(document_id, pipeline)`; return the row count.

        Does NOT commit. Used by `replace_document_chunks` to purge stale
        chunks (including higher-index orphans a plain UPSERT would miss)
        before re-inserting a fresh chunk set.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE document_id = %s AND pipeline = %s",
                (document_id, pipeline),
            )
            return cur.rowcount

    def replace_document_chunks(
        self,
        doc: DocumentRow,
        chunks: list[ChunkRow],
        *,
        pipeline: str,
    ) -> int:
        """Atomic: upsert `doc`, wipe existing chunks for `(doc, pipeline)`, insert `chunks`.

        Returns the document_id. The document row's content_hash is only
        visible to other readers after all chunks land — a mid-flight failure
        rolls back both writes, so the idempotency short-circuit
        (`skip if content_hash matches`) can trust that a committed doc row
        implies its chunks are present.

        Also purges any orphan chunks with `chunk_index` past the current
        batch's length — the ON CONFLICT UPDATE in `upsert_chunks` alone
        would leave those stranded when a re-ingest produces fewer chunks.
        """
        try:
            document_id = self.upsert_document(doc)
            self.delete_chunks(document_id, pipeline)
            self.upsert_chunks(document_id, chunks)
            self.conn.commit()
            return document_id
        except Exception:
            self.conn.rollback()
            raise

    # ── Retrieval ─────────────────────────────────────────────────────

    # Column list shared by both retrievers so their SELECT bodies stay
    # in lockstep — any new field on `RetrievedChunk` needs updating in
    # one place. `{score_expr}` is the only per-caller substitution.
    _SELECT_COLUMNS = (
        "c.id, c.document_id, d.doc_path, d.source_id, "
        "c.pipeline, c.chunk_index, c.text, c.page_start, c.page_end"
    )

    def dense_search(
        self,
        pipeline: str,
        query_embedding: list[float],
        k: int,
    ) -> list[RetrievedChunk]:
        """Top-k cosine-similarity search over chunks of a given pipeline.

        Uses pgvector's `<=>` cosine-distance operator (0 = identical
        direction, 1 = orthogonal, 2 = opposite). Returned `score`
        is `1 - distance` = cosine similarity in [-1, 1] (higher is
        closer), which matches the more common downstream convention.
        """
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"query embedding dim {len(query_embedding)} ≠ expected {EMBEDDING_DIM}"
            )
        # Explicit ::vector cast: on INSERT the destination column type
        # tells psycopg to adapt the Python list as a vector, but as a
        # bare `%s` parameter the type is inferred as double precision[]
        # and pgvector's <=> operator has no such overload.
        sql = f"""
            SELECT
                {self._SELECT_COLUMNS},
                1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.pipeline = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (query_embedding, pipeline, query_embedding, k))
            rows = cur.fetchall()
        return [_row_to_retrieved(r) for r in rows]

    def lexical_search(
        self,
        pipeline: str,
        query_text: str,
        k: int,
    ) -> list[RetrievedChunk]:
        """Top-k BM25-alike search via Postgres FTS (ts_rank_cd + GIN index).

        `plainto_tsquery` (not `to_tsquery`) accepts free-form English
        text — a user question with punctuation or a stopword-only
        phrase won't error, it'll just return an empty result set.
        `ts_rank_cd` weights term proximity (BM25-inspired); the `32`
        normalization flag divides by log(unique_words) so long docs
        aren't unfairly boosted.

        Returned `score` is `ts_rank_cd` (small positive float, unbounded
        upper end but typically < 1.0); NOT comparable to `dense_search`'s
        cosine score. Callers that want a unified score across both use
        `hybrid_search`, which fuses by rank via RRF instead.
        """
        sql = f"""
            SELECT
                {self._SELECT_COLUMNS},
                ts_rank_cd(c.text_tsv, query, 32) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id,
                 plainto_tsquery('english', %s) AS query
            WHERE c.pipeline = %s
              AND c.text_tsv @@ query
            ORDER BY score DESC
            LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (query_text, pipeline, k))
            rows = cur.fetchall()
        return [_row_to_retrieved(r) for r in rows]

    def hybrid_search(
        self,
        pipeline: str,
        query_embedding: list[float],
        query_text: str,
        k: int,
        rrf_k: int,
    ) -> list[RetrievedChunk]:
        """Fetch top-k from dense + lexical, RRF-fuse, return top-k fused.

        Reciprocal Rank Fusion (Cormack et al. 2009): a chunk's fused score
        is the sum of `1 / (rrf_k + rank)` across the source lists it
        appears in (1-indexed rank). Rank-based, so the two sources'
        wildly-different score scales (cosine in [-1, 1] vs ts_rank_cd
        unbounded) don't need normalization — the algorithm is robust
        by construction.

        Returned `RetrievedChunk.score` is the RRF score (small float,
        roughly bounded by `2 / (rrf_k + 1)`), NOT cosine or ts_rank_cd.
        Downstream renders it verbatim; document this in any UI that
        surfaces the number.
        """
        # Both source pulls fetch `k` candidates each (matches the "top-N
        # from each" recipe). Fetching more would trade DB cost for
        # marginal recall gain — not worth it at our scale.
        dense = self.dense_search(pipeline, query_embedding, k=k)
        lexical = self.lexical_search(pipeline, query_text, k=k)
        return _rrf_fuse(dense, lexical, rrf_k=rrf_k, top_k=k)

    # ── Smoke / observability helpers ─────────────────────────────────

    def count_chunks(self, pipeline: str | None = None) -> int:
        """Total chunks in the store, optionally filtered by pipeline."""
        with self.conn.cursor() as cur:
            if pipeline is None:
                cur.execute("SELECT COUNT(*) FROM chunks")
            else:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE pipeline = %s", (pipeline,))
            row = cur.fetchone()
            return row[0] if row else 0
